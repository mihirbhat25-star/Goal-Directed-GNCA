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

The script supports fixed-waypoint and online-goal checkpoints in 2D and 3D.
It writes a per-run CSV, a human-readable aggregate report, and the plots of evaluation/plots.py.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from dataclasses import dataclass

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from models.gnn_ca_simple_boids import GNNCASimpleBoids
from models.gnn_ca_simple_boids_3d import GNNCASimpleBoids3D
from models.gnn_ca_goal_conditioned_boids import GoalConditionedGNNCABoids
from models.gnn_ca_goal_conditioned_boids_3d import GoalConditionedGNNCABoids3D
from modules.boids import BOIDS_GOAL_POSITIONS, Boids
from modules.boids_3d import (
    BOIDS_GOAL_POSITIONS_3D,
    Boids3D,
)
from modules.waypoints import goal_conditioned_state, sample_waypoint_sequence
from evaluation import parallel, plots


TEMPORAL_STATS = ("mean", "min", "max", "median", "std")
METRIC_NAMES = ("radius", "polarization", "speed_cv")
QUADRANT_BOUNDS = {
    0: (0, 5, 0, 5),
    1: (-5, 0, 0, 5),
    2: (-5, 0, -5, 0),
    3: (0, 5, -5, 0),
}
OCTANT_BOUNDS = {
    0: (0, 5, 0, 5, 0, 5),
    1: (-5, 0, 0, 5, 0, 5),
    2: (-5, 0, -5, 0, 0, 5),
    3: (0, 5, -5, 0, 0, 5),
    4: (0, 5, 0, 5, -5, 0),
    5: (-5, 0, 0, 5, -5, 0),
    6: (-5, 0, -5, 0, -5, 0),
    7: (0, 5, -5, 0, -5, 0),
}


@dataclass(frozen=True)
class EventPlan:
    """Random choices shared by GNCA and Boids for one matched rollout."""

    step: int
    removal_order: np.ndarray
    insertion_unit_positions: np.ndarray
    insertion_parent_indices: np.ndarray
    removal_direction: np.ndarray


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate one-shot agent removal and insertion ablations."
    )
    parser.add_argument("--dimension", type=int, choices=(2, 3), required=True)
    parser.add_argument(
        "--task",
        choices=("fixed_waypoints", "online_goals"),
        required=True,
    )
    parser.add_argument("--run_tag", required=True,
                        help="Name of the evaluated model; selects saved_models/best_weights_<run_tag> "
                        "unless --weights_path is given.")
    parser.add_argument("--weights_path", default="")
    parser.add_argument("--n_boids", type=int, default=100)
    parser.add_argument("--perception", type=float, default=0.1,
                        help="Neighbor radius of the matched Boids expert (and of the GNCA's graph).")
    parser.add_argument("--crowding", type=float, default=0.02,
                        help="Separation radius of the matched Boids expert.")
    parser.add_argument("--init_scatter", type=float, default=0.325,
                        help="Side of the starting box around each sampled center.")
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--success_threshold", type=float, default=0.5)
    parser.add_argument("--centers_per_region", type=int, default=10)
    parser.add_argument("--regions", type=int, nargs="+", default=None)
    parser.add_argument("--exclusion", type=float, default=0.5)
    parser.add_argument("--online_goal_count", type=int, default=5)
    parser.add_argument("--goal_min_distance", type=float, default=2.5)
    parser.add_argument(
        "--goal_bounds",
        type=float,
        nargs="+",
        default=None,
        help="Flat lower/upper pairs; defaults to [-4.5, 4.5] per axis.",
    )
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
    parser.add_argument("--seed", type=int, default=460)
    parser.add_argument("--verbose_every", type=int, default=500)
    parser.add_argument("--output_dir", default="results/paired_rollouts")
    parser.add_argument("--skip_plots", action="store_true",
                        help="Save metrics and time series without summary figures.")
    parser.add_argument("--gnca_only", action="store_true",
                        help="Roll out the GNCA alone, without the matched Boids expert.")
    parser.add_argument(
        "--save_radius_series",
        action="store_true",
        help="Also save per-step radius r(t) of every GNCA and Boids rollout "
        "to radius_series.npz.",
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
    return parser.parse_args()


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
    expected_bounds = 2 * args.dimension
    if args.goal_bounds is None:
        args.goal_bounds = [-4.5, 4.5] * args.dimension
    if len(args.goal_bounds) != expected_bounds:
        raise ValueError(
            f"--goal_bounds needs {expected_bounds} values in {args.dimension}D."
        )
    valid_regions = set(range(4 if args.dimension == 2 else 8))
    if args.regions is None:
        args.regions = sorted(valid_regions)
    if not set(args.regions).issubset(valid_regions):
        raise ValueError(f"--regions must be selected from {sorted(valid_regions)}.")
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
    bounds_by_region = QUADRANT_BOUNDS if args.dimension == 2 else OCTANT_BOUNDS
    centers, labels = [], []

    for region in args.regions:
        bounds = np.asarray(bounds_by_region[region], dtype=np.float64).reshape(-1, 2)
        made = 0
        while made < args.centers_per_region:
            center = rng.uniform(bounds[:, 0], bounds[:, 1]).astype(np.float32)
            if args.task == "fixed_waypoints":
                if args.dimension == 2 and _inside_fixed_exclusion_2d(
                    center, fixed_goals, args.exclusion
                ):
                    continue
                if args.dimension == 3 and _inside_fixed_exclusion(
                    center, fixed_goals, args.exclusion
                ):
                    continue
            centers.append(center)
            labels.append(region)
            made += 1
    return centers, labels


def _inside_fixed_exclusion_2d(point, goals, radius):
    """Match the fixed-center exclusion with a union of goal-path capsules."""
    return _inside_fixed_exclusion(point, goals, radius)


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


def _initial_state(center, n_boids, dimension, rng, init_scatter=0.325):
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


def _frame_metrics(state, dimension):
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
    return radius, polarization, speed_cv


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
    event_applied = False
    failure_type = "timeout"
    metric_series = [_frame_metrics(state, dimension)]
    frames = []
    pattern = getattr(args, "removal_pattern", "random")
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
                if args.task == "online_goals"
                else state
            )
            adjacency_matrix = boids.get_neighbors(state[:, :dimension])
            adjacency = _to_tf_sparse(adjacency_matrix)
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
        metric_series.append(_frame_metrics(state, dimension))
        if record_every and step % record_every == 0:
            frames.append((step, state[:, :dimension].copy(), None))
        centroid_distance = float(
            np.linalg.norm(state[:, :dimension].mean(axis=0) - goals[active_goal])
        )
        if centroid_distance <= args.success_threshold:
            arrival_steps.append(step)
            active_goal += 1
            if active_goal == len(goals):
                failure_type = "none"
                break
        if args.verbose_every and step % args.verbose_every == 0:
            print(
                f"      {controller}: step {step}/{args.max_steps}, "
                f"goals {active_goal}/{len(goals)}, agents={len(state)}"
            )

    radius_series = [values[0] for values in metric_series]
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
        "goal_tube_radii": goal_radii,
        "max_goal_tube_radius": max(goal_radii),
        "motion": _motion_summary(metric_series),
        "radius_series": np.asarray(radius_series, dtype=np.float32),
        "frames": frames,
    }


def _to_tf_sparse(matrix):
    indices = np.stack((matrix.row, matrix.col), axis=1)
    sparse = tf.SparseTensor(
        indices=indices,
        values=matrix.data.astype(np.float32),
        dense_shape=matrix.shape,
    )
    return tf.sparse.reorder(sparse)


def _checkpoint_prefix(path):
    path = str(path)
    for extension in (".index", ".data-00000-of-00001"):
        if path.endswith(extension):
            return path[: -len(extension)]
    return path


def _resolve_weights(args):
    if args.weights_path:
        prefix = _checkpoint_prefix(args.weights_path)
        if not (os.path.exists(prefix) or os.path.isfile(prefix + ".index")):
            raise FileNotFoundError(f"Weights not found: {prefix}")
        return prefix

    stems = (f"best_weights_{args.run_tag}", f"gnca_model_{args.run_tag}")
    candidates = []
    for stem in stems:
        for pattern in (os.path.join("saved_models", stem),
                        os.path.join("saved_models", stem + "*")):
            candidates.extend(glob.glob(pattern))
    prefixes = []
    for candidate in candidates:
        prefix = _checkpoint_prefix(candidate)
        if prefix not in prefixes:
            prefixes.append(prefix)
    if not prefixes:
        raise FileNotFoundError(
            f"No checkpoint found for run tag '{args.run_tag}' in saved_models/."
        )
    prefixes.sort(
        key=lambda path: (
            0 if os.path.basename(path).startswith("best_weights_") else 1,
            path,
        )
    )
    return prefixes[0]


def _checkpoint_tensor_names():
    return [
        "mp/pre/mlp/layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE",
        "mp/pre/mlp/layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE",
        "mp/pre/mlp/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE",
        "mp/pre/mlp/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE",
        "mp/gnn/0/kernel/.ATTRIBUTES/VARIABLE_VALUE",
        "mp/gnn/0/bias/.ATTRIBUTES/VARIABLE_VALUE",
        "mp/post/mlp/layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE",
        "mp/post/mlp/layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE",
        "mp/post/mlp/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE",
        "mp/post/mlp/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE",
        "mp_diff/mlp/layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE",
        "mp_diff/mlp/layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE",
        "mp_diff/mlp/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE",
        "mp_diff/mlp/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE",
        "limits_model/mlp/layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE",
        "limits_model/mlp/layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE",
        "limits_model/mlp/layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE",
        "limits_model/mlp/layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE",
    ]


def _restore_2d_checkpoint(model, weights_path, n_boids, input_features):
    dummy = tf.SparseTensor(
        indices=np.zeros((0, 2), dtype=np.int64),
        values=np.zeros((0,), dtype=np.float32),
        dense_shape=(n_boids, n_boids),
    )
    model(
        [
            tf.zeros((n_boids, input_features), dtype=tf.float32),
            tf.sparse.reorder(dummy),
            tf.constant(0),
        ],
        training=False,
    )
    names = _checkpoint_tensor_names()
    available = {name for name, _ in tf.train.list_variables(weights_path)}
    if len(model.weights) != len(names):
        raise ValueError(
            f"Expected {len(names)} 2D model tensors, got {len(model.weights)}."
        )
    for variable, name in zip(model.weights, names):
        if name not in available:
            raise ValueError(f"Checkpoint tensor is missing: {name}")
        value = tf.train.load_variable(weights_path, name)
        if tuple(variable.shape) != tuple(value.shape):
            raise ValueError(
                f"Checkpoint tensor {name} has shape {value.shape}; "
                f"expected {tuple(variable.shape)}."
            )
        variable.assign(value)


def _load_checkpoint(args):
    weights = _resolve_weights(args)
    print(f"Loading weights: {weights}")
    if args.dimension == 2:
        model_class = (
            GoalConditionedGNNCABoids
            if args.task == "online_goals"
            else GNNCASimpleBoids
        )
        input_features = 6 if args.task == "online_goals" else 4
    else:
        model_class = (
            GoalConditionedGNNCABoids3D
            if args.task == "online_goals"
            else GNNCASimpleBoids3D
        )
        input_features = 9 if args.task == "online_goals" else 6
    model = model_class(
        activation="linear",
        batch_norm=False,
        hidden=256,
        hidden_activation="relu",
        connectivity="cat",
        aggregate="mean",
    )
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse", run_eagerly=True)
    if args.dimension == 2 and os.path.isfile(weights + ".index"):
        _restore_2d_checkpoint(model, weights, args.n_boids, input_features)
    else:
        dummy = tf.SparseTensor(
            indices=np.zeros((0, 2), dtype=np.int64),
            values=np.zeros((0,), dtype=np.float32),
            dense_shape=(args.n_boids, args.n_boids),
        )
        model(
            [
                tf.zeros((args.n_boids, input_features), dtype=tf.float32),
                tf.sparse.reorder(dummy),
                tf.constant(0),
            ],
            training=False,
        )
        model.load_weights(weights).expect_partial()
    print("Weights loaded.")
    return model


def _boids(args):
    cls = Boids if args.dimension == 2 else Boids3D
    expert = cls(n_boids=args.n_boids, perception=args.perception,
                 crowding=getattr(args, "crowding", 0.02))
    expert.init_scatter = getattr(args, "init_scatter", 0.325)
    return expert


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
            }
        )
        row.update(
            {f"{prefix}_{name}": value for name, value in result["motion"].items()}
        )
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
        for metric in METRIC_NAMES:
            values = [row[f"{controller}_mean_{metric}"] for row in selected]
            output[f"{controller}_mean_{metric}"] = _summarize(values)["mean"]
    return output


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
        f"expert: perception {args.perception:g}, crowding {getattr(args, 'crowding', 0.02):g}, "
        f"starting box {getattr(args, 'init_scatter', 0.325):g}, {args.n_boids} agents",
        f"rollouts per condition: {len(rows) // len(conditions)}",
        f"maximum evaluation horizon: {args.max_steps}",
        (
            "task completion: every waypoint reached sequentially by the "
            f"flock centroid within {args.success_threshold:g} units"
        ),
        (
            f"intervention: one event sampled uniformly from steps "
            f"[{args.event_step_min}, {args.event_step_max}]"
        ),
        (
            "removal: the agents closest to a plane through the flock centroid, "
            "perpendicular to a shared random direction, are deleted (slab)"
            if getattr(args, "removal_pattern", "random") == "slab"
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
                if getattr(args, "removal_pattern", "random") == "slab"
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
                    f"  mean temporal radius: {agg['gnca_mean_radius']:.6f}",
                    f"  mean temporal polarization: {agg['gnca_mean_polarization']:.6f}",
                    f"  mean temporal speed CV: {agg['gnca_mean_speed_cv']:.6f}",
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
                "",
            ]
        )
    path = os.path.join(output_dir, "summary.txt")
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")
    return path, aggregates


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
                center, args.n_boids, args.dimension, init_rng,
                getattr(args, "init_scatter", 0.325),
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
                radius_series[f"{prefix}_gnca"] = gnca["radius_series"]
                if expert is not None:
                    radius_series[f"{prefix}_boids"] = expert["radius_series"]
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


def main():
    args = _parse_args()
    _validate_args(args)
    os.makedirs(args.output_dir, exist_ok=True)

    fixed_goals = (
        BOIDS_GOAL_POSITIONS.copy()
        if args.dimension == 2
        else BOIDS_GOAL_POSITIONS_3D.copy()
    )
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
