"""Paired GNCA and Boids rollouts, with optional agent removal or insertion.

Without an intervention this is the evaluation protocol of the main results:
every start is rolled out by the GNCA and by its Boids expert from the same
initial state and goal sequence, and task completion and the flock metrics
are recorded for both.  With an intervention, the evaluator keeps the
population change matched between the two.  Each rollout receives at most
one intervention:

* removal deletes the same randomly selected agent indices, or, with
  ``--removal_pattern slab``, the agents in a central slab of the flock
  perpendicular to a shared random direction, which can split the flock;
* insertion uses the same unit-box samples and parent-agent indices, mapped
  into each controller's live flock bounding box.

Expert settings and evaluation defaults come from the experiment's preset. The script
writes a per-run CSV, a human-readable aggregate report, the plots of evaluation/plots.py,
and, with ``--save_radius_series``, the flock radius and neighbor count over time with
their report (evaluation/series_report.py).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf

from models.gnca import load_gnca
from modules.experts import fixed_waypoints, reference_expert
from modules.presets import NAMES, goal_conditioned, load_preset
from modules.regions import region_bounds, region_count
from modules.waypoints import goal_conditioned_state, sample_waypoint_sequence
from evaluation import parallel, plots, series_report


TEMPORAL_STATS = ("mean", "min", "max", "median", "std")
METRIC_NAMES = ("radius", "polarization", "speed_cv", "neighbors")
# Share of the final frames of each rollout over which the late neighbor count is averaged.
LATE_FRACTION = 0.2
# The box the evaluation starts are drawn from, per axis.
CANVAS = (-5.0, 5.0)


@dataclass(frozen=True)
class EventPlan:
    """Random choices shared by GNCA and Boids for one matched rollout."""

    step: int
    removal_order: np.ndarray
    insertion_unit_positions: np.ndarray
    insertion_parent_indices: np.ndarray
    removal_direction: np.ndarray


def build_parser():
    parser = argparse.ArgumentParser(
        description="Evaluate a GNCA against its matched Boids expert, with optional one-shot agent "
        "removal or insertion."
    )
    parser.add_argument("--experiment", choices=NAMES, required=True,
                        help="Selects the task and the expert settings from its preset.")
    parser.add_argument("--weights", required=True, help="Checkpoint prefix of the model.")
    parser.add_argument("--n_boids", type=int, default=None, help="Swarm size (default: the preset's).")
    parser.add_argument("--init_scatter", type=float, default=None,
                        help="Side of the starting box around each sampled center (default: the preset's).")
    parser.add_argument("--max_steps", type=int, default=None, help="Horizon (default: the preset's).")
    parser.add_argument("--centers_per_region", type=int, default=None,
                        help="Starts per quadrant or octant (default: the preset's).")
    parser.add_argument("--seed", type=int, default=None, help="Seed of the starts (default: the preset's).")
    parser.add_argument(
        "--conditions",
        choices=("baseline", "removal", "insertion"),
        nargs="+",
        default=("baseline", "removal", "insertion"),
    )
    parser.add_argument(
        "--population_changes", type=int, nargs="+", default=(10, 25, 50)
    )
    parser.add_argument(
        "--removal_pattern",
        choices=("random", "slab"),
        default="random",
        help="random: uniformly random agents; slab: the agents closest to a "
        "plane through the flock centroid, perpendicular to a random direction.",
    )
    parser.add_argument("--event_step_min", type=int, default=500)
    parser.add_argument("--event_step_max", type=int, default=1000)
    parser.add_argument("--verbose_every", type=int, default=500)
    parser.add_argument("--output_dir", default="results/paired_rollouts")
    parser.add_argument("--skip_plots", action="store_true",
                        help="Save metrics and time series without summary figures.")
    parser.add_argument("--gnca_only", action="store_true",
                        help="Roll out the GNCA alone, without the matched Boids expert.")
    parser.add_argument(
        "--save_radius_series",
        action="store_true",
        help="Also save the per-step radius r(t) and neighbor count of every rollout to "
        "radius_series.npz, and write the over-time report series_report.txt.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Evaluate the starts in this many parallel processes (one per visible GPU "
        "in turn, or on the CPU) and merge their results. Every start is seeded by its "
        "own index.",
    )
    parser.add_argument("--run_start", type=int, default=0,
                        help="First start index to evaluate (sharding).")
    parser.add_argument("--run_stop", type=int, default=None,
                        help="One past the last start index to evaluate (sharding).")
    # Set by the parent of a sharded evaluation: write raw rows instead of reports.
    parser.add_argument("--_shard", action="store_true", default=False, help=argparse.SUPPRESS)
    return parser


def apply_preset(args):
    """Fill the settings an evaluation takes from the experiment's preset."""
    preset = load_preset(args.experiment)
    expert, evaluation = preset["expert"], preset["evaluation"]
    args.dimension = preset["dimension"]
    args.task = preset["task"]
    args.goal_conditioned = goal_conditioned(preset)
    args.run_tag = os.path.basename(args.weights)
    args.perception = expert["perception"]
    args.crowding = expert["crowding"]
    args.n_boids = args.n_boids if args.n_boids is not None else expert["n_boids"]
    args.init_scatter = args.init_scatter if args.init_scatter is not None else expert["init_scatter"]
    args.max_steps = args.max_steps if args.max_steps is not None else evaluation["max_steps"]
    args.centers_per_region = (args.centers_per_region if args.centers_per_region is not None
                               else evaluation["centers_per_region"])
    args.seed = args.seed if args.seed is not None else evaluation["seed"]
    args.success_threshold = evaluation["success_threshold"]
    args.online_goal_count = evaluation["online_goal_count"]
    args.goal_min_distance = evaluation["goal_min_distance"]
    args.goal_bounds = expert.get("goal_bounds")
    args.exclusion = expert.get("goal_exclusion_size")
    args.regions = list(range(region_count(args.dimension)))
    return args


def parse_args(argv=None):
    args = apply_preset(build_parser().parse_args(argv))
    _validate_args(args)
    return args


def _validate_args(args):
    if args.n_boids < 2:
        raise ValueError("--n_boids must be at least 2.")
    if args.max_steps < 1 or args.centers_per_region < 1:
        raise ValueError("Step and center counts must be positive.")
    if any(condition != "baseline" for condition in args.conditions) and not (
        1 <= args.event_step_min <= args.event_step_max <= args.max_steps
    ):
        raise ValueError(
            "Require 1 <= event_step_min <= event_step_max <= max_steps."
        )
    changes = sorted(set(args.population_changes))
    if not changes or changes[0] < 1:
        raise ValueError("--population_changes must contain positive integers.")
    # Population changes only matter for the removal condition; a baseline-only
    # evaluation of a small swarm must not be rejected by the default counts.
    if "removal" in args.conditions and changes[-1] >= args.n_boids:
        raise ValueError("Removal counts must be smaller than --n_boids.")
    args.population_changes = changes
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    if args.run_start < 0 or (args.run_stop is not None and args.run_stop <= args.run_start):
        raise ValueError("Require 0 <= run_start < run_stop.")


def _condition_specs(args):
    specs = []
    if "baseline" in args.conditions:
        specs.append(("baseline", 0, "baseline"))
    if "removal" in args.conditions:
        specs.extend(
            ("removal", count, f"remove_{count}")
            for count in args.population_changes
        )
    if "insertion" in args.conditions:
        specs.extend(
            ("insertion", count, f"insert_{count}")
            for count in args.population_changes
        )
    return specs


def _sample_centers(args, rng, fixed_goals):
    centers, labels = [], []

    for region in args.regions:
        bounds = np.asarray(region_bounds(CANVAS * args.dimension, region), dtype=np.float64).reshape(-1, 2)
        made = 0
        while made < args.centers_per_region:
            center = rng.uniform(bounds[:, 0], bounds[:, 1]).astype(np.float32)
            if args.task == "fixed_waypoints" and _inside_fixed_exclusion(center, fixed_goals, args.exclusion):
                continue
            centers.append(center)
            labels.append(region)
            made += 1
    return centers, labels


def _inside_fixed_exclusion(point, goals, radius):
    """Return whether a point lies within a fixed-waypoint path capsule."""
    point = np.asarray(point, dtype=np.float64)
    goals = np.asarray(goals, dtype=np.float64)
    for start, end in zip(goals[:-1], goals[1:]):
        segment = end - start
        denom = float(np.dot(segment, segment))
        fraction = 0.0 if denom == 0 else np.clip(
            np.dot(point - start, segment) / denom, 0.0, 1.0
        )
        if np.linalg.norm(point - (start + fraction * segment)) < radius:
            return True
    return False


def _initial_state(center, n_boids, dimension, rng, init_scatter):
    positions = np.asarray(center, dtype=np.float32) + np.float32(init_scatter) * rng.random(
        (n_boids, dimension), dtype=np.float32
    )
    velocities = np.zeros((n_boids, dimension), dtype=np.float32)
    velocities[:, 0] = 0.01
    return np.concatenate((positions, velocities), axis=-1)


def _event_plan(args, run_index):
    rng = np.random.default_rng(
        np.random.SeedSequence([args.seed, run_index, 0xA6E17])
    )
    max_count = max(args.population_changes)
    step = int(rng.integers(args.event_step_min, args.event_step_max + 1))
    removal_order = rng.permutation(args.n_boids)
    insertion_unit_positions = rng.random((max_count, args.dimension))
    insertion_parent_indices = rng.integers(
        0, args.n_boids, size=max_count, endpoint=False
    )
    # Drawn last so the draws above match earlier random-removal runs.
    direction = rng.normal(size=args.dimension)
    return EventPlan(
        step=step,
        removal_order=removal_order,
        insertion_unit_positions=insertion_unit_positions,
        insertion_parent_indices=insertion_parent_indices,
        removal_direction=direction / np.linalg.norm(direction),
    )


def _slab_indices(state, count, direction, dimension):
    """Indices of the ``count`` agents closest to the central plane."""
    positions = state[:, :dimension]
    offsets = (positions - positions.mean(axis=0)) @ direction
    return np.argsort(np.abs(offsets), kind="stable")[:count]


def _apply_event(state, mode, count, plan, dimension, pattern="random"):
    if mode == "baseline":
        return state
    if mode == "removal":
        if pattern == "slab":
            removed = set(
                _slab_indices(state, count, plan.removal_direction, dimension).tolist()
            )
        else:
            removed = set(plan.removal_order[:count].tolist())
        keep = np.asarray(
            [index not in removed for index in range(len(state))], dtype=bool
        )
        return state[keep].copy()
    if mode != "insertion":
        raise ValueError(f"Unknown intervention mode: {mode}")

    positions = state[:, :dimension]
    lower = positions.min(axis=0)
    upper = positions.max(axis=0)
    unit = plan.insertion_unit_positions[:count]
    inserted_positions = lower + unit * (upper - lower)
    parents = plan.insertion_parent_indices[:count] % len(state)
    inserted_velocities = state[parents, dimension : 2 * dimension].copy()
    inserted = np.concatenate((inserted_positions, inserted_velocities), axis=-1)
    return np.concatenate((state, inserted.astype(state.dtype)), axis=0)


def _mean_neighbors(positions, perception):
    """Mean number of agents within the perception radius of each agent (itself excluded)."""
    if not np.all(np.isfinite(positions)):
        return np.nan
    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
    return float(((distances < perception).sum(axis=1) - 1).mean())


def _frame_metrics(state, dimension, perception):
    positions = state[:, :dimension]
    velocity = state[:, dimension : 2 * dimension]
    centroid = positions.mean(axis=0)
    radius = float(np.percentile(np.linalg.norm(positions - centroid, axis=-1), 95))
    speed = np.linalg.norm(velocity, axis=-1)
    moving = speed > 1e-12
    if np.any(moving):
        unit_velocity = velocity[moving] / speed[moving, None]
        polarization = float(np.linalg.norm(unit_velocity.mean(axis=0)))
        mean_speed = float(speed[moving].mean())
        speed_cv = float(speed[moving].std() / mean_speed) if mean_speed else np.nan
    else:
        polarization = np.nan
        speed_cv = np.nan
    return radius, polarization, speed_cv, _mean_neighbors(positions, perception)


GOAL_DISTANCE_STATS = ("mean", "max", "min", "median", "std")


def _summarize(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0, **{name: np.nan for name in TEMPORAL_STATS}}
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "median": float(np.median(values)),
        "std": float(values.std()),
    }


def _motion_summary(metric_series):
    result = {}
    for metric_index, metric_name in enumerate(METRIC_NAMES):
        stats = _summarize([values[metric_index] for values in metric_series])
        result[f"valid_{metric_name}_frames"] = stats.pop("n")
        result.update(
            {f"{stat}_{metric_name}": value for stat, value in stats.items()}
        )
    return result


def _goal_tube_radii(radius_series, arrival_steps):
    radius_series = np.asarray(radius_series, dtype=np.float64)
    segment_start = 0
    radii = []
    final_frame = len(radius_series) - 1
    for arrival_step in arrival_steps:
        stop = min(max(int(arrival_step), segment_start), final_frame)
        radii.append(float(np.percentile(radius_series[segment_start : stop + 1], 99)))
        segment_start = stop
    if segment_start < final_frame or not radii:
        radii.append(float(np.percentile(radius_series[segment_start:], 99)))
    return radii


def _rollout(
    *, controller, model, boids, initial_state, goals, plan, mode, count, args,
    record_every=0, stop_step=None,
):
    """Run one rollout.

    ``record_every > 0`` stores agent positions every that many steps (and at
    the event) for visualization; ``stop_step`` ends the rollout early.
    """
    dimension = args.dimension
    state = np.asarray(initial_state, dtype=np.float32).copy()
    active_goal = 0
    arrival_steps = []
    # Closest approach of the centroid to each waypoint while it is the active goal.
    closest = [float(np.linalg.norm(state[:, :dimension].mean(axis=0) - goals[0]))]
    event_applied = False
    failure_type = "timeout"
    metric_series = [_frame_metrics(state, dimension, args.perception)]
    frames = []
    pattern = args.removal_pattern
    last_step = args.max_steps if stop_step is None else min(stop_step, args.max_steps)

    for step in range(1, last_step + 1):
        if mode != "baseline" and step == plan.step:
            if record_every:
                removed_mask = np.ones(len(state), dtype=bool)
                kept = _apply_event(
                    np.column_stack((state, np.arange(len(state)))),
                    mode, count, plan, dimension, pattern,
                )[:, -1].astype(int)
                removed_mask[kept] = False
                frames.append(
                    (step, state[:, :dimension].copy(), removed_mask)
                )
            state = _apply_event(state, mode, count, plan, dimension, pattern)
            event_applied = True

        if controller == "gnca":
            model_input = (
                goal_conditioned_state(state, goals[active_goal])
                if args.goal_conditioned
                else state
            )
            adjacency_matrix = boids.get_neighbors(state[:, :dimension])
            adjacency = to_tf_sparse(adjacency_matrix)
            state_next = model(
                [
                    tf.constant(model_input, dtype=tf.float32),
                    adjacency,
                    tf.constant(0),
                ],
                training=False,
            ).numpy()
        else:
            boids.goal_positions = goals[active_goal : active_goal + 1]
            boids.current_goal = 0
            positions, velocities, _ = boids.update_boids(
                state[:, :dimension].copy(),
                state[:, dimension : 2 * dimension].copy(),
            )
            state_next = np.concatenate((positions, velocities), axis=-1)

        if not np.all(np.isfinite(state_next)):
            failure_type = "instability"
            break
        state = np.asarray(state_next, dtype=np.float32)
        metric_series.append(_frame_metrics(state, dimension, args.perception))
        if record_every and step % record_every == 0:
            frames.append((step, state[:, :dimension].copy(), None))
        centroid = state[:, :dimension].mean(axis=0)
        centroid_distance = float(np.linalg.norm(centroid - goals[active_goal]))
        closest[active_goal] = min(closest[active_goal], centroid_distance)
        if centroid_distance <= args.success_threshold:
            arrival_steps.append(step)
            active_goal += 1
            if active_goal == len(goals):
                failure_type = "none"
                break
            closest.append(float(np.linalg.norm(centroid - goals[active_goal])))
        if args.verbose_every and step % args.verbose_every == 0:
            print(
                f"      {controller}: step {step}/{args.max_steps}, "
                f"goals {active_goal}/{len(goals)}, agents={len(state)}"
            )

    radius_series = [values[0] for values in metric_series]
    neighbor_series = [values[3] for values in metric_series]
    late_start = int(len(neighbor_series) * (1 - LATE_FRACTION))
    goal_radii = _goal_tube_radii(radius_series, arrival_steps)
    return {
        "reached": active_goal,
        "requested": len(goals),
        "completed": active_goal == len(goals),
        "failure_type": failure_type,
        "steps": len(metric_series) - 1,
        "event_applied": event_applied,
        "final_agents": len(state),
        "arrival_steps": arrival_steps,
        "goal_closest_distances": closest,
        "goal_tube_radii": goal_radii,
        "max_goal_tube_radius": max(goal_radii),
        "motion": _motion_summary(metric_series),
        "late_mean_neighbors": _summarize(neighbor_series[late_start:])["mean"],
        "radius_series": np.asarray(radius_series, dtype=np.float32),
        "neighbor_series": np.asarray(neighbor_series, dtype=np.float32),
        "frames": frames,
    }


def to_tf_sparse(matrix):
    indices = np.stack((matrix.row, matrix.col), axis=1)
    sparse = tf.SparseTensor(
        indices=indices,
        values=matrix.data.astype(np.float32),
        dense_shape=matrix.shape,
    )
    return tf.sparse.reorder(sparse)


def _load_checkpoint(args):
    print(f"Loading weights: {args.weights}")
    model = load_gnca(args.weights, args.dimension, args.goal_conditioned, args.n_boids)
    print(f"Weights loaded (summed term scale {float(model.sum_scale.numpy()):.4f}).")
    return model


def _boids(args):
    return reference_expert(args.dimension, n_boids=args.n_boids, perception=args.perception,
                            crowding=args.crowding, init_scatter=args.init_scatter)


def _goals_for_run(args, initial_state, run_index, fixed_goals):
    if args.task == "fixed_waypoints":
        return fixed_goals.copy()
    rng = np.random.default_rng(
        np.random.SeedSequence([args.seed, run_index, 0x60A15])
    )
    return sample_waypoint_sequence(
        rng,
        initial_state[:, : args.dimension].mean(axis=0),
        n_waypoints=args.online_goal_count,
        bounds=tuple(args.goal_bounds),
        min_distance=args.goal_min_distance,
    )


def _flatten_row(
    run, region, center, goals, mode, count, key, plan, gnca, expert
):
    row = {
        "run": run,
        "region": region,
        "center": json.dumps(np.asarray(center).tolist()),
        "goals": json.dumps(np.asarray(goals).tolist()),
        "condition": key,
        "intervention": mode,
        "population_change": count,
        "event_step": "" if mode == "baseline" else plan.step,
    }
    for prefix, result in (("gnca", gnca), ("boids", expert)):
        if result is None:
            continue
        row.update(
            {
                f"{prefix}_event_applied": result["event_applied"],
                f"{prefix}_final_agents": result["final_agents"],
                f"{prefix}_goals_reached": result["reached"],
                f"{prefix}_completed": result["completed"],
                f"{prefix}_failure_type": result["failure_type"],
                f"{prefix}_steps": result["steps"],
                f"{prefix}_arrival_steps": json.dumps(result["arrival_steps"]),
                f"{prefix}_goal_radii": json.dumps(result["goal_tube_radii"]),
                f"{prefix}_max_goal_tube_radius": result[
                    "max_goal_tube_radius"
                ],
                f"{prefix}_goal_closest_distances": json.dumps(result["goal_closest_distances"]),
            }
        )
        distance_stats = _summarize(result["goal_closest_distances"])
        row.update({f"{prefix}_goal_distance_{stat}": distance_stats[stat] for stat in GOAL_DISTANCE_STATS})
        row.update(
            {f"{prefix}_{name}": value for name, value in result["motion"].items()}
        )
        row[f"{prefix}_late_mean_neighbors"] = result["late_mean_neighbors"]
    return row


def _write_csv(rows, output_dir):
    path = os.path.join(output_dir, "per_run.csv")
    fieldnames = list(rows[0])
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _controllers(rows):
    """Controllers present in the rows: the GNCA, and Boids unless it was skipped."""
    return [controller for controller in ("gnca", "boids") if f"{controller}_completed" in rows[0]]


def _condition_order(rows):
    seen = []
    for row in rows:
        if row["condition"] not in seen:
            seen.append(row["condition"])
    return seen


def _aggregate_rows(rows, condition):
    selected = [row for row in rows if row["condition"] == condition]
    output = {"n": len(selected)}
    for controller in _controllers(selected):
        completed = np.asarray(
            [row[f"{controller}_completed"] for row in selected], dtype=bool
        )
        output[f"{controller}_completed"] = int(completed.sum())
        output[f"{controller}_completion_rate"] = 100.0 * completed.mean()
        output[f"{controller}_mean_goals"] = float(
            np.mean([row[f"{controller}_goals_reached"] for row in selected])
        )
        output[f"{controller}_instability"] = sum(
            row[f"{controller}_failure_type"] == "instability" for row in selected
        )
        output[f"{controller}_timeout"] = sum(
            row[f"{controller}_failure_type"] == "timeout" for row in selected
        )
        output[f"{controller}_event_applied"] = sum(
            bool(row[f"{controller}_event_applied"]) for row in selected
        )
        # Rollouts that diverged count as failures above and are left out of the metrics below.
        stable = [row for row in selected if row[f"{controller}_failure_type"] != "instability"]
        for metric in METRIC_NAMES:
            values = [row[f"{controller}_mean_{metric}"] for row in stable]
            output[f"{controller}_mean_{metric}"] = _summarize(values)["mean"]
        output[f"{controller}_late_mean_neighbors"] = _summarize(
            [row[f"{controller}_late_mean_neighbors"] for row in stable])["mean"]
        pooled = [distance for row in stable
                  for distance in json.loads(row[f"{controller}_goal_closest_distances"])]
        output[f"{controller}_goal_distance"] = _summarize(pooled)
        output[f"{controller}_run_goal_distance_mean"] = _summarize(
            [row[f"{controller}_goal_distance_mean"] for row in stable])
    return output


def _goal_distance_lines(agg, controller, label):
    pooled, per_run = agg[f"{controller}_goal_distance"], agg[f"{controller}_run_goal_distance_mean"]
    return [
        f"  {label} closest approach to each activated waypoint, pooled over {pooled['n']} waypoints: "
        + ", ".join(f"{stat} {pooled[stat]:.4f}" for stat in GOAL_DISTANCE_STATS),
        f"  {label} per-run mean closest approach, over {per_run['n']} runs: "
        + ", ".join(f"{stat} {per_run[stat]:.4f}" for stat in GOAL_DISTANCE_STATS),
    ]


def _write_report(args, rows, output_dir):
    conditions = _condition_order(rows)
    aggregates = {
        condition: _aggregate_rows(rows, condition) for condition in conditions
    }
    baseline_rate = (
        aggregates["baseline"]["gnca_completion_rate"]
        if "baseline" in aggregates
        else np.nan
    )
    lines = [
        "GNCA single-event population ablation",
        f"run_tag: {args.run_tag}",
        f"task: {args.task}",
        f"dimension: {args.dimension}D",
        f"expert: perception {args.perception:g}, crowding {args.crowding:g}, "
        f"starting box {args.init_scatter:g}, {args.n_boids} agents",
        f"rollouts per condition: {len(rows) // len(conditions)}",
        f"maximum evaluation horizon: {args.max_steps}",
        (
            "task completion: every waypoint reached sequentially by the "
            f"flock centroid within {args.success_threshold:g} units"
        ),
        (
            "diverged rollouts: count as failures in task completion and mean goals reached, and are "
            "left out of closest approach, radius, polarization, speed CV and neighbor count"
        ),
        (
            "closest approach: the smallest centroid distance to each waypoint while it is the "
            "active goal; waypoints never activated are not scored"
        ),
        (
            f"neighbor count: agents within the perception radius of each agent, averaged over agents "
            f"and frames; 'late' averages the last {LATE_FRACTION:.0%} of each rollout's frames"
        ),
        (
            f"intervention: one event sampled uniformly from steps "
            f"[{args.event_step_min}, {args.event_step_max}]"
        ),
        (
            "removal: the agents closest to a plane through the flock centroid, "
            "perpendicular to a shared random direction, are deleted (slab)"
            if args.removal_pattern == "slab"
            else "removal: uniformly random active agent indices are deleted"
        ),
        (
            "insertion: positions are sampled within the live flock's axis-aligned "
            "bounding box and velocities are copied from random live agents"
        ),
        (
            "matching: GNCA and Boids share starts, goals, event steps, "
            + (
                "slab directions (each slab is taken from that controller's own flock), "
                if args.removal_pattern == "slab"
                else "removal indices, "
            )
            + "insertion unit-box samples, and parent-agent indices"
            if "boids" in _controllers(rows)
            else "controller: GNCA only (no Boids rollouts)"
        ),
        "",
    ]
    for condition in conditions:
        agg = aggregates[condition]
        delta = agg["gnca_completion_rate"] - baseline_rate
        if "boids_completed" not in agg:
            lines.extend(
                [
                    f"[{condition}]",
                    f"  GNCA task completion: {agg['gnca_completed']}/{agg['n']} = {agg['gnca_completion_rate']:.2f}%",
                    f"  mean goals reached: {agg['gnca_mean_goals']:.3f}",
                    f"  GNCA failures (instability / timeout): {agg['gnca_instability']} / {agg['gnca_timeout']}",
                    *_goal_distance_lines(agg, "gnca", "GNCA"),
                    f"  mean temporal radius: {agg['gnca_mean_radius']:.6f}",
                    f"  mean temporal polarization: {agg['gnca_mean_polarization']:.6f}",
                    f"  mean temporal speed CV: {agg['gnca_mean_speed_cv']:.6f}",
                    f"  mean neighbor count (whole rollout / late): {agg['gnca_mean_neighbors']:.2f} / "
                    f"{agg['gnca_late_mean_neighbors']:.2f}",
                    "",
                ]
            )
            continue
        lines.extend(
            [
                f"[{condition}]",
                (
                    f"  GNCA task completion: {agg['gnca_completed']}/{agg['n']} "
                    f"= {agg['gnca_completion_rate']:.2f}%"
                ),
                (
                    f"  Boids task completion: {agg['boids_completed']}/{agg['n']} "
                    f"= {agg['boids_completion_rate']:.2f}%"
                ),
                (
                    f"  GNCA completion-rate change from baseline: {delta:+.2f} "
                    "percentage points"
                    if np.isfinite(delta)
                    else "  GNCA completion-rate change from baseline: unavailable"
                ),
                (
                    f"  mean goals reached (GNCA / Boids): "
                    f"{agg['gnca_mean_goals']:.3f} / {agg['boids_mean_goals']:.3f}"
                ),
                (
                    f"  GNCA failures (instability / timeout): "
                    f"{agg['gnca_instability']} / {agg['gnca_timeout']}"
                ),
                *_goal_distance_lines(agg, "gnca", "GNCA"),
                *_goal_distance_lines(agg, "boids", "Boids"),
                (
                    f"  event applied (GNCA / Boids): "
                    f"{agg['gnca_event_applied']}/{agg['n']} / "
                    f"{agg['boids_event_applied']}/{agg['n']}"
                ),
                (
                    "  mean temporal radius (GNCA / Boids): "
                    f"{agg['gnca_mean_radius']:.6f} / "
                    f"{agg['boids_mean_radius']:.6f}"
                ),
                (
                    "  mean temporal polarization (GNCA / Boids): "
                    f"{agg['gnca_mean_polarization']:.6f} / "
                    f"{agg['boids_mean_polarization']:.6f}"
                ),
                (
                    "  mean temporal speed CV (GNCA / Boids): "
                    f"{agg['gnca_mean_speed_cv']:.6f} / "
                    f"{agg['boids_mean_speed_cv']:.6f}"
                ),
                (
                    "  mean neighbor count (GNCA / Boids): "
                    f"{agg['gnca_mean_neighbors']:.2f} / {agg['boids_mean_neighbors']:.2f}"
                ),
                (
                    "  late mean neighbor count (GNCA / Boids): "
                    f"{agg['gnca_late_mean_neighbors']:.2f} / {agg['boids_late_mean_neighbors']:.2f}"
                ),
                "",
            ]
        )
    path = os.path.join(output_dir, "summary.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")
    return path, aggregates


def _stable_series(result, key):
    """A saved series, all NaN for a rollout that diverged so that it stays out of the medians over time."""
    if result["failure_type"] == "instability":
        return np.full_like(result[key], np.nan)
    return result[key]


def _evaluate_runs(args, model, centers, regions, run_indices, specs, fixed_goals):
    """Roll out every condition for the given start indices; returns rows and radius series."""
    rows = []
    radius_series = {}
    for mode, count, key in specs:
        print(f"\n=== Condition: {key} ===")
        for run_index in run_indices:
            center, region = centers[run_index], regions[run_index]
            init_rng = np.random.default_rng(
                np.random.SeedSequence([args.seed, run_index, 0x1A117])
            )
            initial_state = _initial_state(
                center, args.n_boids, args.dimension, init_rng, args.init_scatter,
            )
            goals = _goals_for_run(
                args, initial_state, run_index, fixed_goals
            )
            plan = _event_plan(args, run_index)
            print(
                f"  Run {run_index + 1}/{len(centers)} | region={region} | "
                f"event_step={plan.step if mode != 'baseline' else 'none'}"
            )
            gnca = _rollout(
                controller="gnca",
                model=model,
                boids=_boids(args),
                initial_state=initial_state,
                goals=goals,
                plan=plan,
                mode=mode,
                count=count,
                args=args,
            )
            expert = None if args.gnca_only else _rollout(
                controller="boids",
                model=None,
                boids=_boids(args),
                initial_state=initial_state,
                goals=goals,
                plan=plan,
                mode=mode,
                count=count,
                args=args,
            )
            print(
                f"    GNCA goals {gnca['reached']}/{gnca['requested']}"
                + ("" if expert is None else f" | Boids goals {expert['reached']}/{expert['requested']}")
            )
            if args.save_radius_series:
                prefix = f"{key}_run{run_index:03d}"
                radius_series[f"{prefix}_gnca"] = _stable_series(gnca, "radius_series")
                radius_series[f"{prefix}_gnca_neighbors"] = _stable_series(gnca, "neighbor_series")
                if expert is not None:
                    radius_series[f"{prefix}_boids"] = _stable_series(expert, "radius_series")
                    radius_series[f"{prefix}_boids_neighbors"] = _stable_series(expert, "neighbor_series")
                radius_series[f"{prefix}_event_step"] = np.asarray(
                    plan.step if mode != "baseline" else -1
                )
            rows.append(
                _flatten_row(
                    run_index,
                    region,
                    center,
                    goals,
                    mode,
                    count,
                    key,
                    plan,
                    gnca,
                    expert,
                )
            )
    return rows, radius_series


def _write_outputs(args, rows, radius_series):
    csv_path = _write_csv(rows, args.output_dir)
    report_path, aggregates = _write_report(args, rows, args.output_dir)
    print(f"Saved per-run metrics: {csv_path}")
    print(f"Saved summary: {report_path}")
    if not args.skip_plots:
        completion_path, metrics_path = plots.rollout_plots(rows, aggregates, args.output_dir)
        print(f"Saved completion figure: {completion_path}")
        print(f"Saved metric figure: {metrics_path}")
    if args.save_radius_series:
        series_path = os.path.join(args.output_dir, "radius_series.npz")
        np.savez_compressed(series_path, **radius_series)
        print(f"Saved radius series: {series_path}")
        recovery_path = None if args.skip_plots else plots.recovery_plot(radius_series, args.output_dir)
        if recovery_path:
            print(f"Saved recovery figure: {recovery_path}")
        report_path = series_report.write_report(radius_series, args.output_dir)
        if report_path:
            print(f"Saved over-time report: {report_path}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    fixed_goals = fixed_waypoints(args.dimension)
    centers, regions = _sample_centers(
        args, np.random.default_rng(args.seed), fixed_goals
    )
    if not args._shard:
        np.save(os.path.join(args.output_dir, "test_centers.npy"), np.asarray(centers))
    if args.workers > 1:
        parallel.run_with_workers(
            "evaluation.paired_rollouts", args, len(centers),
            [key for _, _, key in _condition_specs(args)], _write_outputs,
        )
        return

    run_stop = len(centers) if args.run_stop is None else min(args.run_stop, len(centers))
    run_indices = list(range(args.run_start, run_stop))
    if not run_indices:
        raise ValueError(f"No starts in range [{args.run_start}, {run_stop}).")
    model = _load_checkpoint(args)
    specs = _condition_specs(args)
    print(
        f"Evaluating {len(run_indices)} of {len(centers)} starts across {len(specs)} conditions "
        + (f"({len(run_indices) * len(specs)} GNCA rollouts)." if args.gnca_only
           else f"({len(run_indices) * len(specs)} matched GNCA--Boids pairs).")
    )
    rows, radius_series = _evaluate_runs(args, model, centers, regions, run_indices, specs, fixed_goals)
    if args._shard:
        parallel.save_shard(args.output_dir, rows, radius_series)
        return
    _write_outputs(args, rows, radius_series)


if __name__ == "__main__":
    main()
