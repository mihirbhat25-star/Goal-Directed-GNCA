"""Teacher-forced one-step imitation error of a GNCA on held-out expert rollouts.

The experiment has two stages.

``generate`` samples new starts with the evaluation protocol (balanced over
quadrants in 2D and octants in 3D, fixed-waypoint starts outside the goal-path
exclusion zone, starting box ``--init_scatter``) from a seed that differs from the
evaluation starts, and lets the Boids expert drive every rollout: the next
waypoint becomes active when the flock centroid is within 0.5 of the current
one, for the fixed route as for the five random dynamic goals.  No GNCA is
involved.  Each rollout is saved as ``run_XXX.npz``.

``score`` applies the GNCA one step ahead at every ``--stride``-th state of
each saved expert rollout, using the expert's state and neighborhood graph,
and compares the prediction with the expert's actual next state.  The GNCA
never drives the trajectory, so the errors do not accumulate.  Dynamic-waypoint
models receive the displacement to the waypoint the expert was pursuing at
that step.  The report opens with the one-step root mean squared error over the
full next state, the angle between predicted and true next velocity (direction
error) and the relative speed error, each as the mean over all agent-steps with
the standard deviation across rollouts; it then gives medians, the position
error, and the share of agent-steps with a neighbor inside the crowding radius,
with the direction error split by that condition.

``run`` performs both stages.  Run from the repository root, for example:

    python -m evaluation.imitation_error run --dimension 2 --task fixed_waypoints \
        --weights_path saved_models/best_weights_<run tag> \
        --output_dir results/imitation/fixed2d
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys
from types import SimpleNamespace

import numpy as np

from evaluation import paired_rollouts as rollouts
from evaluation.metrics.flock_metrics import OrderedWaypointProgress
from modules.boids import BOIDS_GOAL_POSITIONS, Boids
from modules.boids_3d import BOIDS_GOAL_POSITIONS_3D, Boids3D
from modules.waypoints import goal_conditioned_state

GOAL_THRESHOLD = 0.5
# The held-out expert rollouts draw their starts, goals and agent scatter from a
# stream derived from --seed and this tag, so they never coincide with the
# evaluation starts of paired_rollouts at the same seed.
HELD_OUT_STREAM = 0x1417


# ------------------------------------------------------------------ generate

def _round_robin(centers, regions, count):
    """First ``count`` starts, interleaving regions so the split stays balanced."""
    by_region = {}
    for center, region in zip(centers, regions):
        by_region.setdefault(region, []).append(center)
    order = sorted(by_region)
    picked_centers, picked_regions = [], []
    while len(picked_centers) < count:
        for region in order:
            if by_region[region] and len(picked_centers) < count:
                picked_centers.append(by_region[region].pop(0))
                picked_regions.append(region)
    return picked_centers, picked_regions


def _arrival_steps(trajectory, goals, dimension, threshold=GOAL_THRESHOLD):
    """Steps at which the centroid reaches each goal in order."""
    centroids = trajectory[:, :, :dimension].mean(axis=1)
    steps, active = [], 0
    for step in range(1, len(centroids)):
        if active < len(goals) and np.linalg.norm(centroids[step] - goals[active]) <= threshold:
            steps.append(step)
            active += 1
    return steps


def _fixed_expert_rollout(initial, goals, dimension, options):
    """Fixed-waypoint expert rollout; the expert switches on the centroid distance."""
    expert_class = Boids if dimension == 2 else Boids3D
    expert = expert_class(n_boids=options.n_boids, perception=options.perception,
                          crowding=options.crowding, goal_arrival_radius=GOAL_THRESHOLD)
    expert.goal_positions = goals.copy()
    expert.current_goal, expert.goals_completed, expert.reached_goal = 0, 0, False
    state = np.asarray(initial, dtype=np.float32).copy()
    frames = [state.copy()]
    progress = OrderedWaypointProgress(goals, GOAL_THRESHOLD)
    progress.update(state[:, :dimension].mean(axis=0), 0)
    for step in range(1, options.max_steps):
        if progress.completed:
            break
        positions, velocities, _ = expert.update_boids(
            state[:, :dimension].copy(), state[:, dimension:].copy()
        )
        state = np.concatenate((positions, velocities), axis=-1).astype(np.float32)
        frames.append(state.copy())
        progress.update(state[:, :dimension].mean(axis=0), step)
        expert.update_goal(state[:, :dimension])
    trajectory = np.asarray(frames)
    if progress.completed:
        trajectory = trajectory[: int(progress.first_hit_steps.max()) + 1]
    return trajectory, list(progress.arrival_steps)


def _online_expert_rollout(initial, goals, dimension, options):
    """Dynamic-waypoint expert rollout; the expert switches on the centroid distance."""
    if dimension == 2:
        from evaluation.run_inference import _run_matched_online_expert as run_online
    else:
        from evaluation.run_inference_3d import run_matched_online_expert as run_online
    args = SimpleNamespace(n_boids=options.n_boids, perception=options.perception,
                           crowding=options.crowding, max_steps=options.max_steps,
                           success_threshold=GOAL_THRESHOLD)
    result = run_online(initial, goals, args)
    trajectory = np.asarray(result["trajectory"], dtype=np.float32)
    return trajectory, _arrival_steps(trajectory, goals, dimension)


def held_out_seed(seed):
    """Integer seed of the held-out stream for a given --seed."""
    return int(np.random.SeedSequence([int(seed), HELD_OUT_STREAM]).generate_state(1)[0])


def generate(options):
    dimension, task = options.dimension, options.task
    stream = held_out_seed(options.seed)
    regions = list(range(4 if dimension == 2 else 8))
    sampler_args = SimpleNamespace(
        dimension=dimension, task=task, regions=regions,
        centers_per_region=math.ceil(options.n_rollouts / len(regions)),
        exclusion=0.5, seed=stream, online_goal_count=5,
        goal_min_distance=2.5, goal_bounds=[-4.5, 4.5] * dimension,
    )
    fixed_goals = (BOIDS_GOAL_POSITIONS if dimension == 2 else BOIDS_GOAL_POSITIONS_3D).copy()
    centers, labels = rollouts._sample_centers(sampler_args, np.random.default_rng(stream), fixed_goals)
    centers, labels = _round_robin(centers, labels, options.n_rollouts)

    rollout_dir = os.path.join(options.output_dir, "expert_rollouts")
    os.makedirs(rollout_dir, exist_ok=True)
    with open(os.path.join(rollout_dir, "generation_settings.json"), "w", encoding="utf-8") as handle:
        json.dump({"dimension": dimension, "task": task, "n_rollouts": options.n_rollouts,
                   "seed": options.seed, "held_out_stream_seed": stream,
                   "max_steps": options.max_steps, "n_boids": options.n_boids,
                   "perception": options.perception, "crowding": options.crowding,
                   "init_scatter": options.init_scatter, "goal_threshold": GOAL_THRESHOLD,
                   "fixed_switch_rule": "centroid", "fixed_switch_radius": GOAL_THRESHOLD},
                  handle, indent=1)

    for run_index, (center, region) in enumerate(zip(centers, labels)):
        stem = os.path.join(rollout_dir, f"run_{run_index:03d}")
        if os.path.isfile(stem + ".npz"):
            continue
        init_rng = np.random.default_rng(np.random.SeedSequence([stream, run_index, 0x1A117]))
        initial = rollouts._initial_state(center, options.n_boids, dimension, init_rng,
                                          options.init_scatter)
        goals = np.asarray(rollouts._goals_for_run(sampler_args, initial, run_index, fixed_goals), dtype=np.float32)
        rollout = _fixed_expert_rollout if task == "fixed_waypoints" else _online_expert_rollout
        trajectory, arrival_steps = rollout(initial, goals, dimension, options)
        np.savez_compressed(stem + ".npz", center=np.asarray(center), goals=goals, initial_state=initial,
                            boids_trajectory=trajectory,
                            boids_arrival_steps=np.asarray(arrival_steps, dtype=np.int64))
        with open(stem + ".json", "w", encoding="utf-8") as handle:
            json.dump({"run": run_index, "region": int(region), "center": [float(v) for v in center],
                       "goals": goals.tolist(), "requested": len(goals),
                       "boids_goals_reached": len(arrival_steps), "boids_steps": int(len(trajectory) - 1)},
                      handle, indent=1)
        print(f"run {run_index + 1}/{options.n_rollouts} | region {region} | "
              f"Boids goals {len(arrival_steps)}/{len(goals)} | {len(trajectory) - 1} steps", flush=True)
    return rollout_dir


# --------------------------------------------------------------------- score

def _load(options):
    if options.dimension == 2:
        from evaluation.run_inference import _load_model
        return _load_model(options.weights_path, task=options.task, n_boids=options.n_boids)
    from evaluation.run_inference_3d import build_model
    return build_model(options.n_boids, options.weights_path, task=options.task)


def _predictor(model):
    import tensorflow as tf

    def predict(state, adjacency):
        coo = adjacency.tocoo()
        sparse = tf.SparseTensor(
            indices=np.column_stack((coo.row, coo.col)).astype(np.int64),
            values=coo.data.astype(np.float32), dense_shape=coo.shape,
        )
        return model([tf.constant(state, dtype=tf.float32), tf.sparse.reorder(sparse),
                      tf.constant(0)], training=False).numpy()

    return predict


def _goal_indices(trajectory, goals, dimension, task, arrival_steps):
    """Index of the expert's active waypoint for the transition out of each state."""
    frames = len(trajectory)
    if task == "online_goals":
        arrivals = np.asarray(arrival_steps, dtype=int)
        return np.minimum(np.searchsorted(arrivals, np.arange(frames), side="right"), len(goals) - 1)
    index = np.zeros(frames, dtype=int)
    active = 0
    for step in range(1, frames):
        distance = np.linalg.norm(goals[active] - trajectory[step, :, :dimension].mean(axis=0))
        if distance <= GOAL_THRESHOLD:
            active = (active + 1) % len(goals)
        index[step] = active
    return index


def _angle(a, b):
    cosine = (a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-15)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _iqr(values):
    q1, median, q3 = np.percentile(values, [25, 50, 75])
    return f"{median:.4g} [{q1:.4g}, {q3:.4g}]"


def score(options, rollout_dir):
    dimension, task = options.dimension, options.task
    paths = sorted(glob.glob(os.path.join(rollout_dir, "run_*.npz")))
    if not paths:
        raise FileNotFoundError(f"No run_*.npz under {rollout_dir}")
    predict = _predictor(_load(options))
    expert = (Boids if dimension == 2 else Boids3D)(n_boids=options.n_boids, perception=options.perception,
                                                    crowding=options.crowding)
    crowding_radius = expert.crowding

    mse, position_error, direction_error, speed_error, crowded = [], [], [], [], []
    per_rollout = []  # (rmse, mean direction error, mean relative speed error) of each rollout
    for path in paths:
        data = np.load(path)
        trajectory, goals = data["boids_trajectory"], data["goals"]
        indices = _goal_indices(trajectory, goals, dimension, task, data["boids_arrival_steps"])
        first = len(mse)
        for step in range(0, len(trajectory) - 1, options.stride):
            state = trajectory[step].astype(np.float32)
            truth = trajectory[step + 1].astype(np.float32)
            positions = state[:, :dimension]
            adjacency = expert.get_neighbors(positions)
            model_input = (goal_conditioned_state(state, goals[indices[step]])
                           if task == "online_goals" else state)
            predicted = np.asarray(predict(model_input, adjacency), dtype=np.float32)[:, : 2 * dimension]
            error = predicted - truth
            mse.append(np.mean(error ** 2, axis=-1))
            position_error.append(np.linalg.norm(error[:, :dimension], axis=-1))
            predicted_velocity, true_velocity = predicted[:, dimension:], truth[:, dimension:]
            direction_error.append(_angle(predicted_velocity, true_velocity))
            true_speed = np.linalg.norm(true_velocity, axis=-1)
            speed_error.append(np.abs(np.linalg.norm(predicted_velocity, axis=-1) - true_speed)
                               / (true_speed + 1e-12))
            distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
            np.fill_diagonal(distances, np.inf)
            crowded.append(distances.min(axis=1) < crowding_radius)
        per_rollout.append((
            float(np.sqrt(np.mean(np.concatenate(mse[first:])))),
            float(np.mean(np.concatenate(direction_error[first:]))),
            float(np.mean(np.concatenate(speed_error[first:]))),
        ))
        print(f"scored {os.path.basename(path)}", flush=True)

    mse, position_error = np.concatenate(mse), np.concatenate(position_error)
    direction_error, speed_error, crowded = (np.concatenate(direction_error),
                                             np.concatenate(speed_error), np.concatenate(crowded))
    per_rollout = np.asarray(per_rollout, dtype=np.float64)
    rmse = float(np.sqrt(mse.mean()))
    # Means over all agent-steps; the spread is the standard deviation of the
    # per-rollout means across rollouts (the quantity the paper reports).
    spread = per_rollout.std(axis=0, ddof=1) if len(per_rollout) > 1 else np.zeros(3)
    lines = [
        f"=== {task} {dimension}D: {len(paths)} expert rollouts, every {options.stride}th state, "
        f"{len(mse)} agent-steps, checkpoint {options.weights_path}",
        f"one-step RMSE (full next state): {rmse:.4g} (s.d. across rollouts {spread[0]:.3g})",
        f"direction error (deg): mean {direction_error.mean():.4g} (s.d. across rollouts {spread[1]:.3g})",
        f"relative speed error:  mean {speed_error.mean():.4g} (s.d. across rollouts {spread[2]:.3g})",
        f"one-step MSE (full next state): mean {mse.mean():.4g} | median [IQR] {_iqr(mse)}",
        f"position error:        median [IQR] {_iqr(position_error)}",
        f"direction error (deg): median [IQR] {_iqr(direction_error)} | mean {direction_error.mean():.3f} "
        f"| 90th pct {np.percentile(direction_error, 90):.3f}",
        f"relative speed error:  median [IQR] {_iqr(speed_error)}",
        f"agent-steps with a neighbor inside the crowding radius {crowding_radius}: {100 * crowded.mean():.2f}%",
        f"  direction error, separation inactive: {_iqr(direction_error[~crowded])} "
        f"| active: {_iqr(direction_error[crowded]) if crowded.any() else 'n/a'}",
        "",
    ]
    os.makedirs(options.output_dir, exist_ok=True)
    with open(os.path.join(options.output_dir, "imitation_error_report.txt"), "w", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
    q1, q3 = np.percentile(direction_error, [25, 75])
    with open(os.path.join(options.output_dir, "imitation_error_summary.csv"), "w", newline="",
              encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["dimension", "task", "weights_path", "rollouts", "agent_steps",
                         "rmse", "rmse_sd_rollouts", "direction_error_mean_deg", "direction_error_sd_rollouts",
                         "speed_error_mean", "speed_error_sd_rollouts",
                         "mse_mean", "mse_median", "position_error_median", "direction_error_median_deg",
                         "direction_error_q1", "direction_error_q3", "speed_error_median", "crowded_share"])
        writer.writerow([dimension, task, options.weights_path, len(paths), len(mse),
                         rmse, spread[0], direction_error.mean(), spread[1], speed_error.mean(), spread[2],
                         mse.mean(), np.median(mse), np.median(position_error), np.median(direction_error),
                         q1, q3, np.median(speed_error), crowded.mean()])
    with open(os.path.join(options.output_dir, "imitation_error_per_rollout.csv"), "w", newline="",
              encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["rollout", "rmse", "direction_error_mean_deg", "speed_error_mean"])
        for index, row in enumerate(per_rollout):
            writer.writerow([os.path.basename(paths[index]), *row])
    print("\n".join(lines), flush=True)


# ---------------------------------------------------------------------- main

def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("generate", "score", "run"))
    parser.add_argument("--dimension", type=int, choices=(2, 3), required=True)
    parser.add_argument("--task", choices=("fixed_waypoints", "online_goals"), required=True)
    parser.add_argument("--output_dir", required=True,
                        help="Expert rollouts go to <output_dir>/expert_rollouts; the report to <output_dir>.")
    parser.add_argument("--weights_path", default=None, help="Checkpoint prefix (needed by score and run).")
    parser.add_argument("--rollout_dir", default=None,
                        help="Existing expert rollouts to score (default: <output_dir>/expert_rollouts).")
    parser.add_argument("--n_rollouts", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1,
                        help="Seed of the held-out expert rollouts; their starts come from a stream separate "
                             "from the evaluation starts of the same seed.")
    parser.add_argument("--n_boids", type=int, default=100)
    parser.add_argument("--perception", type=float, default=0.1,
                        help="Neighbor radius of the Boids expert and of the GNCA's graph.")
    parser.add_argument("--crowding", type=float, default=0.02, help="Separation radius of the Boids expert.")
    parser.add_argument("--init_scatter", type=float, default=0.325,
                        help="Side of the starting box around each sampled center.")
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--stride", type=int, default=10)
    options = parser.parse_args(argv)
    if options.command in ("score", "run") and not options.weights_path:
        parser.error("--weights_path is required for score and run.")
    if options.n_rollouts < 1 or options.stride < 1 or options.max_steps < 2:
        parser.error("--n_rollouts and --stride must be positive and --max_steps at least 2.")
    return options


def main(argv=None):
    options = _parse_args(argv)
    rollout_dir = options.rollout_dir or os.path.join(options.output_dir, "expert_rollouts")
    if options.command in ("generate", "run"):
        rollout_dir = generate(options)
    if options.command in ("score", "run"):
        score(options, rollout_dir)


if __name__ == "__main__":
    sys.exit(main())
