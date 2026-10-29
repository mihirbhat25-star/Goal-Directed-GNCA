"""Standalone 2D GNCA inference and reporting.

Loads saved GNCA weights, generates fresh test centers, runs inference, and plots.

Canonical usage: ``python -m evaluation.run_inference``.
The original ``python -m boids.run_inference`` command remains compatible.
"""
import glob
import os
import argparse
import re

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from modules.boids import Boids
from models.gnn_ca_simple_boids import GNNCASimpleBoids
from models.gnn_ca_goal_conditioned_boids import GoalConditionedGNNCABoids
from modules.waypoints import goal_conditioned_state, sample_waypoint_sequence
from evaluation.visualization.visualize_boids import (
    _get_tube_exterior,
    _plot_individual_ranked as _plot_individual_ranked_shared,
)
from evaluation.metrics.flock_metrics import (
    OrderedWaypointProgress,
    ordered_waypoint_progress,
    goal_segment_tube_radii,
    robust_tube_radius,
    trajectory_motion_metrics,
    write_online_benchmark_files,
)
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from shapely.geometry import LineString, Point, Polygon

QUADRANT_BOUNDS = {
    0: ( 0,  5,  0,  5),
    1: (-5,  0,  0,  5),
    2: (-5,  0, -5,  0),
    3: ( 0,  5, -5,  0),
}

def _count_collision_events(traj, threshold, warmup_steps=0):
    """Count post-warmup collision onsets and unique pairs involved."""
    n_agents = traj.shape[1]
    upper = np.triu(np.ones((n_agents, n_agents), dtype=bool), k=1)
    start = min(max(0, warmup_steps), len(traj))

    def collision_mask(positions):
        delta = positions[:, None, :] - positions[None, :, :]
        return (np.linalg.norm(delta, axis=-1) < threshold) & upper

    previous = (
        collision_mask(traj[start - 1, :, :2])
        if start > 0 else np.zeros((n_agents, n_agents), dtype=bool)
    )
    total = 0
    event_pairs = np.zeros_like(previous)
    for positions in traj[start:, :, :2]:
        colliding = collision_mask(positions)
        onsets = colliding & ~previous
        total += int(np.count_nonzero(onsets))
        event_pairs |= onsets
        previous = colliding
    return total, int(np.count_nonzero(event_pairs))


def _plot_individual_ranked(trajs, goals, n_boids, centers, run_tag, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    _plot_individual_ranked_shared(
        trajs,
        goals,
        n_boids,
        selected_centers=centers,
        run_tag=run_tag,
        output_dir=output_dir,
    )


def _plot_multi_tubular(trajs, goals, run_tag, output_dir):
    colors = cm.hsv(np.linspace(0, 0.9, len(trajs)))
    radii = []
    fig, ax = plt.subplots(figsize=(7, 7))
    for color, traj in zip(colors, trajs):
        centroid = traj[:, :, :2].mean(axis=1)
        dists = np.linalg.norm(traj[:, :, :2] - centroid[:, None, :], axis=-1)
        r = np.percentile(np.percentile(dists, 95, axis=1), 99)
        radii.append(r)
        line = LineString(centroid)
        tube = line.buffer(r)
        tx, ty = _get_tube_exterior(tube)
        ax.fill(tx, ty, color=color, alpha=0.25)
        ax.plot(centroid[:, 0], centroid[:, 1], color=color, lw=1.2)
        ax.scatter([centroid[0, 0]], [centroid[0, 1]], c='blue', marker='*', s=80, zorder=6)
    ax.scatter(goals[:, 0], goals[:, 1], c='red', marker='*', s=200, zorder=5, label='Goals')
    ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
    ax.set_aspect('equal')
    ax.set_title(f"GNCA Inference - {len(trajs)} runs")
    if radii:
        ax.set_title(f"GNCA Inference - {len(trajs)} tubes | mean r={np.mean(radii):.3f}, max r={np.max(radii):.3f}")
    else:
        ax.set_title(f"GNCA Inference - {len(trajs)} tubes")
    ax.legend()
    plt.tight_layout()
    fname = os.path.join(output_dir, f"inference_{run_tag}_multi.pdf")
    plt.savefig(fname)
    plt.close()
    print(f"Saved {fname}")


def _trajectory_tube_radius(traj):
    """Return the same robust flock radius used by the tubular plots."""
    return robust_tube_radius(traj, spatial_dims=2)


def _plot_tube_on_axis(ax, traj, color="#8B0000", label="centroid path"):
    centroid = traj[:, :, :2].mean(axis=1)
    r = _trajectory_tube_radius(traj)
    line = LineString(centroid)
    tube = line.buffer(r)
    tx, ty = _get_tube_exterior(tube)
    ax.fill(tx, ty, color=color, alpha=0.3, label=f"99% tube (r={r:.3f})")
    ax.plot(centroid[:, 0], centroid[:, 1], lw=1.5, color=color, label=label)
    ax.scatter([centroid[0, 0]], [centroid[0, 1]], c='blue', marker='*', s=120, zorder=6, label='Start')
    return centroid, r


def _clean_checkpoint_prefix(path):
    path = str(path)
    for ext in ('.index', '.data-00000-of-00001'):
        path = path.replace(ext, '')
    return path


def _safe_model_name(path):
    base = os.path.basename(_clean_checkpoint_prefix(path))
    base = re.sub(r"^(best_weights_|gnca_model_)", "", base)
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", base)


def _model_size_suffix(model_name):
    match = re.search(r"\d+x\d+", model_name)
    return match.group(0) if match else "model"


def _find_weight_prefixes(weights_dir):
    prefixes = []
    for pattern in (os.path.join(weights_dir, "best_weights_*"),
                    os.path.join(weights_dir, "gnca_model_*")):
        for match in sorted(glob.glob(pattern)):
            prefix = _clean_checkpoint_prefix(match)
            if prefix not in prefixes:
                prefixes.append(prefix)
    by_tag = {}
    for prefix in prefixes:
        name = os.path.basename(prefix)
        tag = re.sub(r"^(best_weights_|gnca_model_)", "", name)
        current = by_tag.get(tag)
        # Prefer best_weights checkpoint pairs; gnca_model entries may be SavedModel dirs.
        if current is None or name.startswith("best_weights_"):
            by_tag[tag] = prefix
    return [by_tag[tag] for tag in sorted(by_tag)]


def _find_single_weights(run_tag):
    search_patterns = [
        f"best_weights_{run_tag}*",
        f"saved_models/best_weights_{run_tag}*",
    ]
    print("2D inference weight search:")
    for pattern in search_patterns:
        matches = glob.glob(pattern)
        print(f"  pattern: {pattern}")
        print(f"  matches: {matches if matches else 'NONE'}")
    candidates = glob.glob(search_patterns[0]) or glob.glob(search_patterns[1])
    if not candidates:
        raise FileNotFoundError(f"No files matching 'best_weights_{run_tag}*' in cwd: {os.getcwd()}")
    return _clean_checkpoint_prefix(candidates[0])


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


def _restore_checkpoint_by_name(model, weights_path, n_boids, input_features):
    index_path = weights_path + ".index"
    data_path = weights_path + ".data-00000-of-00001"
    if not os.path.isfile(index_path) or not os.path.isfile(data_path):
        raise FileNotFoundError(
            "TensorFlow checkpoint requires matching .index and .data files: "
            f"{weights_path}"
        )

    dummy_adjacency = tf.SparseTensor(
        indices=np.zeros((0, 2), dtype=np.int64),
        values=np.zeros((0,), dtype=np.float32),
        dense_shape=(n_boids, n_boids),
    )
    model(
        [
            tf.zeros((n_boids, input_features), dtype=tf.float32),
            tf.sparse.reorder(dummy_adjacency),
            tf.constant(0),
        ],
        training=False,
    )
    checkpoint_names = _checkpoint_tensor_names()
    if len(model.weights) != len(checkpoint_names):
        raise ValueError(
            f"Expected {len(checkpoint_names)} model tensors, got {len(model.weights)}."
        )
    available = {name for name, _ in tf.train.list_variables(weights_path)}
    for variable, checkpoint_name in zip(model.weights, checkpoint_names):
        if checkpoint_name not in available:
            raise ValueError(f"Checkpoint tensor is missing: {checkpoint_name}")
        value = tf.train.load_variable(weights_path, checkpoint_name)
        if tuple(variable.shape) != tuple(value.shape):
            raise ValueError(
                f"Checkpoint does not match this model: {checkpoint_name} has "
                f"shape {tuple(value.shape)}, expected {tuple(variable.shape)}."
            )
        variable.assign(value)


def _load_model(weights_path, task="fixed_waypoints", n_boids=100):
    weights_path = _clean_checkpoint_prefix(weights_path)
    print(f"Loading weights: {weights_path}")
    model_class = (
        GoalConditionedGNNCABoids
        if task == "online_goals"
        else GNNCASimpleBoids
    )
    model = model_class(
        activation="linear", batch_norm=False, hidden=256,
        hidden_activation="relu", connectivity="cat", aggregate="mean",
    )
    model.compile(optimizer=Adam(learning_rate=1e-3), loss="mse", run_eagerly=True)
    if os.path.isfile(weights_path + ".index"):
        input_features = 6 if task == "online_goals" else 4
        _restore_checkpoint_by_name(
            model, weights_path, n_boids, input_features
        )
    else:
        model.load_weights(weights_path).expect_partial()
    print("Weights loaded.")
    return model


def _manual_goal_array(values):
    if values is None:
        return None
    values = np.asarray(values, dtype=np.float32)
    if len(values) < 4 or len(values) % 2:
        raise ValueError("--manual_goals needs at least two x y pairs.")
    return values.reshape(-1, 2)


def _run_online_goal_trajectory(model, boids, center, args, rng, manual_goals=None):
    """Roll out a goal-conditioned GNCA under an external waypoint manager."""
    position, velocity, _, _ = boids.get_random_init(
        args.n_boids, save_config=False, center=np.asarray(center)
    )
    physical = np.concatenate((position, velocity), axis=-1).astype(np.float32)
    frames = [physical.copy()]
    switch_steps = []
    closest_distances = []
    reached = 0
    diverged = False
    divergence_step = None

    if manual_goals is None:
        sampled_goals = sample_waypoint_sequence(
            rng,
            position.mean(axis=0),
            n_waypoints=args.online_goal_count,
            bounds=tuple(args.goal_bounds),
            min_distance=args.goal_min_distance,
        )
    else:
        sampled_goals = np.asarray(manual_goals, dtype=np.float32)
    active_goal = sampled_goals[0]

    closest_current = float(np.linalg.norm(position.mean(axis=0) - active_goal))
    for step in range(1, args.max_steps + 1):
        conditioned = goal_conditioned_state(physical, active_goal)
        adjacency = _to_tf_sparse(boids.get_neighbors(physical[:, :2]))
        next_physical = model(
            [tf.constant(conditioned), adjacency, tf.constant(0)],
            training=False,
        ).numpy()
        if not np.all(np.isfinite(next_physical)):
            diverged = True
            divergence_step = step
            print(
                f"    non-finite state at step {step}; "
                "stopping this rollout and marking it unsuccessful"
            )
            break
        physical = next_physical
        frames.append(physical.copy())
        centroid = physical[:, :2].mean(axis=0)
        centroid_distance = float(np.linalg.norm(centroid - active_goal))
        closest_current = min(closest_current, centroid_distance)
        if step % 500 == 0:
            print(
                f"    step {step}/{args.max_steps} | "
                f"centroid=({centroid[0]:.2f},{centroid[1]:.2f}) | "
                f"active_goal=({active_goal[0]:.2f},{active_goal[1]:.2f})"
            )

        if centroid_distance > args.success_threshold:
            continue

        closest_distances.append(closest_current)
        reached += 1
        switch_steps.append(step)
        if reached >= len(sampled_goals):
            break
        active_goal = sampled_goals[reached]
        closest_current = float(np.linalg.norm(centroid - active_goal))

    requested = len(sampled_goals)
    trajectory = np.asarray(frames)
    goal_tube_radii = goal_segment_tube_radii(
        trajectory, switch_steps, spatial_dims=2
    )
    max_goal_tube_radius = max(goal_tube_radii, default=float("inf"))
    reached_all_goals = reached == requested
    return {
        "trajectory": trajectory,
        "initial_state": frames[0].copy(),
        "goals": np.asarray(sampled_goals),
        "switch_steps": switch_steps,
        "closest_distances": closest_distances,
        "reached": reached,
        "requested": requested,
        "goal_tube_radii": goal_tube_radii,
        "max_goal_tube_radius": max_goal_tube_radius,
        "motion_metrics": trajectory_motion_metrics(trajectory, 2),
        "reached_all_goals": reached_all_goals,
        "diverged": diverged,
        "divergence_step": divergence_step,
    }


def _run_matched_online_expert(initial_state, goals, args):
    """Evaluate Boids from the exact GNCA initial state and goal sequence."""
    expert = Boids(n_boids=args.n_boids, perception=args.perception,
                   crowding=getattr(args, "crowding", 0.02),
                   init_scatter=getattr(args, "init_scatter", 0.325))
    initial_state = np.asarray(initial_state, dtype=np.float32)
    positions = initial_state[:, :2].copy()
    velocities = initial_state[:, 2:4].copy()
    goals = np.asarray(goals, dtype=np.float32)
    frames = [initial_state.copy()]
    arrival_steps = []
    active_goal = 0

    for step in range(1, args.max_steps + 1):
        expert.goal_positions = goals[active_goal : active_goal + 1]
        expert.current_goal = 0
        positions, velocities, _ = expert.update_boids(positions, velocities)
        frames.append(
            np.concatenate((positions, velocities), axis=-1).astype(
                np.float32, copy=False
            )
        )
        centroid_distance = float(
            np.linalg.norm(positions.mean(axis=0) - goals[active_goal])
        )
        if centroid_distance > args.success_threshold:
            continue
        arrival_steps.append(step)
        active_goal += 1
        if active_goal == len(goals):
            break

    trajectory = np.asarray(frames)
    goal_radii = goal_segment_tube_radii(
        trajectory, arrival_steps, spatial_dims=2
    )
    return {
        "trajectory": trajectory,
        "goal_tube_radii": goal_radii,
        "max_goal_tube_radius": max(goal_radii, default=float("inf")),
        "motion_metrics": trajectory_motion_metrics(trajectory, 2),
        "reached": active_goal,
        "requested": len(goals),
        "completed": active_goal == len(goals),
        "steps": len(trajectory) - 1,
    }


def _attach_online_benchmarks(results, args):
    """Attach one exact-start, exact-goal Boids benchmark to every run."""
    for result in results:
        result["expert_benchmark"] = _run_matched_online_expert(
            result["initial_state"], result["goals"], args
        )


def _save_online_goal_pdf(result, center, run_index, run_tag, output_dir):
    trajectory = result["trajectory"]
    goals = result["goals"]
    fig, axis = plt.subplots(figsize=(7, 7))
    _plot_tube_on_axis(axis, trajectory, label="GNCA centroid")
    radius = result["max_goal_tube_radius"]
    colors = cm.viridis(np.linspace(0.1, 0.9, len(goals)))
    for goal_index, (goal, color) in enumerate(zip(goals, colors)):
        axis.scatter(
            goal[0], goal[1], color=color, marker="*", s=180, zorder=7
        )
        axis.annotate(
            f"G{goal_index + 1}", goal, xytext=(6, 6),
            textcoords="offset points",
        )
    axis.set_xlim(-5, 5)
    axis.set_ylim(-5, 5)
    axis.set_aspect("equal")
    axis.set_title(
        f"Online GNCA | start=({center[0]:.2f},{center[1]:.2f}) | "
        f"goals={result['reached']}/{result['requested']} | "
        f"max goal r={radius:.3f}"
    )
    axis.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(
        output_dir, f"online_{run_tag}_run{run_index:03d}.pdf"
    )
    plt.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def _save_online_goal_summary(results, run_tag, output_dir):
    fig, axis = plt.subplots(figsize=(8, 8))
    colors = cm.hsv(np.linspace(0, 0.9, len(results)))
    for result, color in zip(results, colors):
        _plot_tube_on_axis(
            axis,
            result["trajectory"],
            color=color,
            label=None,
        )
        axis.scatter(
            result["goals"][:, 0], result["goals"][:, 1],
            color=color, marker="x", s=30,
        )
    axis.set_xlim(-5, 5)
    axis.set_ylim(-5, 5)
    axis.set_aspect("equal")
    axis.set_title(f"Online GNCA - {len(results)} unseen runs")
    plt.tight_layout()
    path = os.path.join(output_dir, f"online_{run_tag}_all.pdf")
    plt.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def _sample_online_start_centers(quadrants, centers_per_quadrant, seed):
    if centers_per_quadrant < 1:
        raise ValueError("--centers_per_quadrant must be at least 1.")
    rng = np.random.default_rng(seed)
    centers = []
    center_quadrants = []
    for quadrant in quadrants:
        x_min, x_max, y_min, y_max = QUADRANT_BOUNDS[quadrant]
        for _ in range(centers_per_quadrant):
            centers.append(np.array([
                rng.uniform(x_min, x_max), rng.uniform(y_min, y_max)
            ], dtype=np.float32))
            center_quadrants.append(quadrant)
    return centers, center_quadrants


def _center_quadrant(center):
    x, y = np.asarray(center, dtype=np.float64)
    if x >= 0 and y >= 0:
        return 0
    if x < 0 and y >= 0:
        return 1
    if x < 0 and y < 0:
        return 2
    return 3


def _load_online_centers(path):
    """Load explicit 2D centers, with an optional quadrant column."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Centers file not found: {path}")
    if path.lower().endswith(".npy"):
        values = np.load(path)
    else:
        try:
            values = np.loadtxt(path, comments="#", delimiter=",")
        except ValueError:
            values = np.loadtxt(path, comments="#")
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] not in (2, 3) or len(values) == 0:
        raise ValueError(
            f"Centers file must have shape (N, 2) or (N, 3); got {values.shape}."
        )
    centers = [row[:2].copy() for row in values]
    if values.shape[1] == 3:
        quadrants = [int(row[2]) for row in values]
        if any(quadrant not in QUADRANT_BOUNDS for quadrant in quadrants):
            raise ValueError("Every explicit quadrant label must be between 0 and 3.")
    else:
        quadrants = [_center_quadrant(center) for center in centers]
    print(
        f"Loaded {len(centers)} explicit online test centers from {path} "
        f"with counts {dict((q, quadrants.count(q)) for q in sorted(set(quadrants)))}"
    )
    return centers, quadrants


def _run_online_goal_inference(args, model):
    if args.mode != "single":
        raise ValueError("online_goals currently uses --mode single.")
    manual_goals = _manual_goal_array(args.manual_goals)
    if args.centers_file:
        centers, center_quadrants = _load_online_centers(args.centers_file)
    else:
        centers, center_quadrants = _sample_online_start_centers(
            args.quadrants, args.centers_per_quadrant, args.seed
        )
    if manual_goals is not None and len(centers) != 1:
        print(
            ">>> Manual goal sequence will be evaluated from each sampled start center."
        )
    print(
        f"Using {len(centers)} online test centers."
    )
    np.save(os.path.join(args.output_dir, "online_test_centers.npy"), centers)

    boids = Boids(n_boids=args.n_boids, perception=args.perception,
                  crowding=getattr(args, "crowding", 0.02),
                  init_scatter=getattr(args, "init_scatter", 0.325))
    individual_dir = os.path.join(args.output_dir, "individual")
    os.makedirs(individual_dir, exist_ok=True)
    results = []
    for run_index, (center, quadrant) in enumerate(
        zip(centers, center_quadrants)
    ):
        print(
            f"  Online inference {run_index + 1}/{len(centers)} | "
            f"quadrant={quadrant} | center=({center[0]:.2f},{center[1]:.2f})"
        )
        result = _run_online_goal_trajectory(
            model,
            boids,
            center,
            args,
            np.random.default_rng(args.seed + 10_000 + run_index),
            manual_goals=manual_goals,
        )
        results.append(result)
        _save_online_goal_pdf(
            result, center, run_index, args.run_tag, individual_dir
        )

    _attach_online_benchmarks(results, args)
    for result in results:
        expert = result["expert_benchmark"]
        outcome = (
            "completed"
            if result["reached_all_goals"] and not result["diverged"]
            else ("instability" if result["diverged"] else "timeout")
        )
        print(
            f"    reached {result['reached']}/{result['requested']} goals | "
            f"goal_r={[round(r, 4) for r in result['goal_tube_radii']]} | "
            f"max_r={result['max_goal_tube_radius']:.4f} | "
            f"outcome={outcome} | matched_boids_max_r="
            f"{expert['max_goal_tube_radius']:.4f}"
        )

    _save_online_goal_summary(results, args.run_tag, args.output_dir)
    write_online_benchmark_files(
        output_dir=args.output_dir,
        run_tag=args.run_tag,
        centers=centers,
        regions=center_quadrants,
        region_name="quadrant",
        results=results,
        goal_threshold=args.success_threshold,
        max_steps=args.max_steps,
    )


def _sample_centers_per_quadrant(goals, n_per_quadrant=10, quadrants=(0, 1, 2, 3), exclusion=0.5, seed=1):
    rng = np.random.default_rng(seed)
    excl = Polygon(goals.tolist()).buffer(exclusion)
    centers, center_q_map = [], []
    for q_idx in quadrants:
        xmn, xmx, ymn, ymx = QUADRANT_BOUNDS[q_idx]
        picked = 0
        while picked < n_per_quadrant:
            c = np.array([rng.uniform(xmn, xmx), rng.uniform(ymn, ymx)])
            if not excl.contains(Point(c)):
                centers.append(c)
                center_q_map.append(q_idx)
                picked += 1
    return centers, center_q_map


def _to_tf_sparse(a):
    indices = np.stack([a.row, a.col], axis=1)
    sp = tf.SparseTensor(indices=indices, values=a.data.astype(np.float32), dense_shape=a.shape)
    return tf.sparse.reorder(sp)


def _run_one_trajectory_2d(model, boids, center, max_steps, n_boids, goals,
                           success_threshold, verbose_prefix="",
                           near_goal_verbose=False, near_goal_radius=1.0):
    """Roll out until every fixed goal is reached in order, or the budget ends."""
    pos, vel, _, _ = boids.get_random_init(n_boids, save_config=False, center=center)
    frames = [np.concatenate([pos, vel], axis=-1).astype(np.float32)]
    progress = OrderedWaypointProgress(goals, success_threshold)
    progress.update(frames[0][:, :2].mean(axis=0), 0)
    step_i = tf.constant(0)

    for step in range(max_steps - 1):
        if progress.completed:
            break
        x = frames[-1]
        a = _to_tf_sparse(boids.get_neighbors(x[:, :2]))
        x_next = model([tf.constant(x, dtype=tf.float32), a, step_i], training=False).numpy()
        frames.append(x_next)
        if not np.all(np.isfinite(x_next)):
            print(
                f"{verbose_prefix}  non-finite state at step {step + 1}; "
                "stopping rollout"
            )
            break
        centroid = x_next[:, :2].mean(axis=0)
        progress.update(centroid, step + 1)
        if (step + 1) % 500 == 0:
            print(f"{verbose_prefix}  step {step+1}/{max_steps}")
        if near_goal_verbose:
            mean_vel = x_next[:, 2:].mean(axis=0)
            dist_g0 = np.linalg.norm(centroid - goals[0])
            if dist_g0 < near_goal_radius:
                angle = np.degrees(np.arctan2(mean_vel[1], mean_vel[0]))
                print(
                    f"{verbose_prefix}  [step {step+1}] near goal0 | "
                    f"centroid=({centroid[0]:.3f},{centroid[1]:.3f}) "
                    f"dist={dist_g0:.3f} | mean_vel=({mean_vel[0]:.4f},"
                    f"{mean_vel[1]:.4f}) angle={angle:.1f} deg"
                )
        if progress.completed:
            print(f"{verbose_prefix}  all goals reached in order at step {step + 1}; stopping rollout")
            break

    return np.array(frames)


def _run_gnca_trajs(model, boids, centers, max_steps, n_boids,
                    success_threshold=0.5, verbose_prefix=""):
    trajs = []
    goals = boids.goal_positions
    for k, center in enumerate(centers):
        print(f"{verbose_prefix}Inference {k+1}/{len(centers)} | center=({center[0]:.2f}, {center[1]:.2f})")
        traj = _run_one_trajectory_2d(
            model, boids, center, max_steps, n_boids, goals,
            success_threshold, verbose_prefix=verbose_prefix,
        )
        trajs.append(traj)
        progress = ordered_waypoint_progress(traj, goals, success_threshold)
        for g_idx, g in enumerate(goals):
            distance = progress.closest_distances[g_idx]
            print(f"{verbose_prefix}  goal {g_idx} ({g[0]:.1f},{g[1]:.1f}): closest active-goal centroid dist = {distance:.4f}")
    return trajs


def _closest_goal_distances_2d(traj, goals, threshold=0.5):
    return ordered_waypoint_progress(traj, goals, threshold).closest_distances


def _tube_radius_2d(traj):
    positions = np.asarray(traj)[:, :, :2]
    centroid = positions.mean(axis=1, keepdims=True)
    distances = np.linalg.norm(positions - centroid, axis=-1)
    return float(np.percentile(np.percentile(distances, 95, axis=1), 99))


def _run_matched_fixed_expert_2d(initial_state, args):
    """Run fixed-waypoint Boids from the exact GNCA initial state."""
    # The expert switches waypoints when the flock centroid is within 0.5 of
    # the active one, the same rule task completion is scored with.
    expert = Boids(
        n_boids=args.n_boids,
        perception=args.perception,
        crowding=getattr(args, "crowding", 0.02),
        init_scatter=getattr(args, "init_scatter", 0.325),
        goal_arrival_radius=0.5,
    )
    initial_state = np.asarray(initial_state, dtype=np.float32)
    positions = initial_state[:, :2].copy()
    velocities = initial_state[:, 2:4].copy()
    frames = [initial_state.copy()]
    progress = OrderedWaypointProgress(expert.goal_positions, args.success_threshold)
    progress.update(positions.mean(axis=0), 0)

    for step in range(args.expert_calibration_max_steps):
        if progress.completed:
            break
        positions, velocities, _ = expert.update_boids(positions, velocities)
        frames.append(
            np.concatenate((positions, velocities), axis=-1).astype(
                np.float32, copy=False
            )
        )
        expert.update_goal(positions)
        if progress.update(positions.mean(axis=0), step + 1):
            break

    if not progress.completed:
        raise RuntimeError(
            "Matched fixed-waypoint Boids calibration did not finish within "
            f"{args.expert_calibration_max_steps} steps. Increase "
            "--expert_calibration_max_steps."
        )
    return _tube_radius_2d(np.asarray(frames))


def _fixed_r_thresholds_2d(trajs, args):
    if args.r_threshold_mode == "fixed":
        return (
            np.full(len(trajs), args.max_tube_radius, dtype=np.float64),
            None,
        )

    expert_radii = np.asarray(
        [
            _run_matched_fixed_expert_2d(traj[0], args)
            for traj in trajs
        ],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(expert_radii)):
        raise RuntimeError("Matched fixed-waypoint Boids produced a non-finite r.")
    if args.r_threshold_mode == "expert_per_trajectory":
        thresholds = expert_radii + args.expert_r_padding
    elif args.r_threshold_mode == "expert_global":
        thresholds = np.full(
            len(trajs), expert_radii.max() + args.expert_r_padding,
            dtype=np.float64,
        )
    else:
        raise ValueError(f"Unknown r threshold mode: {args.r_threshold_mode}")
    return thresholds, expert_radii


def _fixed_r_threshold_description_2d(args):
    if args.r_threshold_mode == "fixed":
        return f"fixed r <= {args.max_tube_radius:g}"
    if args.r_threshold_mode == "expert_per_trajectory":
        return f"r <= matched Boids r + {args.expert_r_padding:g} per trajectory"
    return (
        "r <= maximum matched-Boids r over this evaluation set + "
        f"{args.expert_r_padding:g}"
    )


def _write_success_rate_2d(
    trajs, goals, centers, center_q_map, output_path, radius=0.5,
    r_thresholds=None, expert_radii=None, r_description="",
):
    if r_thresholds is None:
        r_thresholds = np.full(len(trajs), np.inf, dtype=np.float64)
    successes = []
    lines = [
        f"2D success criterion: centroid distance <= {radius:g} for every goal in the prescribed order and {r_description or 'no compactness limit'}",
        "Goal distances are measured only while that goal is active; inactive-goal visits do not count.",
        f"Total trajectories: {len(trajs)}",
        "",
    ]
    for idx, traj in enumerate(trajs):
        progress = ordered_waypoint_progress(traj, goals, radius)
        dists = progress.closest_distances
        model_r = _tube_radius_2d(traj)
        applied_r = float(r_thresholds[idx])
        success = (
            progress.completed
            and np.all(np.isfinite(traj))
            and np.isfinite(model_r)
            and model_r <= applied_r
        )
        successes.append(success)
        center = centers[idx]
        quadrant = center_q_map[idx] if center_q_map is not None else "NA"
        dist_str = ", ".join(f"goal{g_idx}={dist:.4f}" for g_idx, dist in enumerate(dists))
        expert_text = (
            f" | expert_r={float(expert_radii[idx]):.4f}"
            if expert_radii is not None else ""
        )
        lines.append(
            f"run {idx:02d} | quadrant={quadrant} | center=({center[0]:.4f},{center[1]:.4f}) | "
            f"success={success} | ordered_goals={progress.reached}/{len(goals)} | arrival_steps={progress.arrival_steps} | r={model_r:.4f} | r_threshold={applied_r:.4f}"
            f"{expert_text} | {dist_str}"
        )

    n_success = int(sum(successes))
    rate = 100.0 * n_success / len(successes) if successes else 0.0
    lines.insert(2, f"Successes: {n_success}/{len(successes)} ({rate:.2f}%)")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return n_success, len(successes), rate


def _run_comparison(args):
    boids = Boids(n_boids=args.n_boids, perception=args.perception,
                  crowding=getattr(args, "crowding", 0.02),
                  init_scatter=getattr(args, "init_scatter", 0.325))
    goals = boids.goal_positions
    weights = args.models if args.models else _find_weight_prefixes(args.weights_dir)
    if not weights:
        raise FileNotFoundError(f"No checkpoints found in {args.weights_dir}")
    print("Comparison mode weights:")
    for w in weights:
        prefix = _clean_checkpoint_prefix(w)
        print(f"  {prefix} | index={os.path.exists(prefix + '.index')} data={os.path.exists(prefix + '.data-00000-of-00001')} bare={os.path.exists(prefix)}")

    centers, center_q_map = _sample_centers_per_quadrant(
        goals,
        n_per_quadrant=args.centers_per_quadrant,
        quadrants=args.quadrants,
        exclusion=args.exclusion,
        seed=args.seed,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, "comparison_centers.npy"), np.array(centers))
    np.save(os.path.join(args.output_dir, "comparison_center_quadrants.npy"), np.array(center_q_map))
    print(f"Generated {len(centers)} shared unseen centers ({args.centers_per_quadrant} per quadrant).")

    summary_lines = [
        f"2D comparison success criterion: centroid distance <= {args.success_threshold:g} for every goal in the prescribed order",
        f"Shared centers: {len(centers)}",
        "",
    ]
    for weights_path in weights:
        model_name = _safe_model_name(weights_path)
        model_dir = os.path.join(args.output_dir, model_name)
        individual_dir = os.path.join(model_dir, "individual")
        os.makedirs(model_dir, exist_ok=True)
        print(f"\n=== Running comparison model: {model_name} ===")
        model = _load_model(weights_path, n_boids=args.n_boids)
        trajs = _run_gnca_trajs(
            model,
            boids,
            centers,
            args.max_steps,
            args.n_boids,
            success_threshold=args.success_threshold,
            verbose_prefix="  ",
        )
        _plot_individual_ranked(trajs, goals, args.n_boids, centers, model_name, individual_dir)
        _plot_multi_tubular(trajs, goals, model_name, model_dir)
        r_thresholds, expert_radii = _fixed_r_thresholds_2d(trajs, args)
        n_success, total, rate = _write_success_rate_2d(
            trajs,
            goals,
            centers,
            center_q_map,
            os.path.join(model_dir, f"success_rate_{_model_size_suffix(model_name)}.txt"),
            radius=args.success_threshold,
            r_thresholds=r_thresholds,
            expert_radii=expert_radii,
            r_description=_fixed_r_threshold_description_2d(args),
        )
        summary_lines.append(f"{model_name}: {n_success}/{total} successful ({rate:.2f}%)")
        print(f"Saved comparison outputs for {model_name} under {model_dir}")

    with open(os.path.join(args.output_dir, "success_rate_summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")



def _parse_args(argv=None):
    parser = argparse.ArgumentParser(description="2D GNCA inference")
    parser.add_argument(
        "--task",
        choices=["fixed_waypoints", "online_goals"],
        default="fixed_waypoints",
    )
    parser.add_argument("--mode", choices=["single", "comparison"], default="single")
    parser.add_argument("--run_tag", default="")
    parser.add_argument("--weights_path", default="",
                        help="Explicit checkpoint prefix, .index, or .data file.")
    parser.add_argument("--weights_dir", default="important_weights_2d")
    parser.add_argument("--models", nargs="*", default=None,
                        help="Checkpoint prefixes/files to compare. Defaults to all models in --weights_dir.")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--centers_per_quadrant",
        type=int,
        default=10,
        help="Online-goal and comparison starts per selected quadrant.",
    )
    parser.add_argument(
        "--centers_file",
        default="",
        help=(
            "Optional NPY/text/CSV file containing x y or x y quadrant per "
            "row. For online evaluation, explicit centers replace sampling."
        ),
    )
    parser.add_argument("--quadrants", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--n_centers", type=int, default=15,
                        help="Fixed-waypoint single-mode total random centers.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--success_threshold", type=float, default=0.5)
    parser.add_argument(
        "--max_tube_radius",
        type=float,
        default=1.0,
        help=(
            "Fixed-waypoint compactness limit. Online-goal evaluation reports "
            "continuous GNCA and matched-Boids metrics instead."
        ),
    )
    parser.add_argument(
        "--r_threshold_mode",
        choices=["fixed", "expert_per_trajectory", "expert_global"],
        default="fixed",
        help=(
            "How fixed-waypoint compactness is judged. Online-goal evaluation "
            "does not threshold r."
        ),
    )
    parser.add_argument(
        "--expert_r_padding",
        type=float,
        default=0.1,
        help="Additive fixed-waypoint r tolerance for expert-calibrated modes.",
    )
    parser.add_argument(
        "--expert_calibration_max_steps",
        type=int,
        default=10000,
        help="Maximum steps for fixed-waypoint matched-Boids calibration.",
    )
    parser.add_argument("--n_boids", type=int, default=100)
    parser.add_argument(
        "--crowding", type=float, default=0.02,
        help="Separation radius of the Boids expert.")
    parser.add_argument(
        "--init_scatter", type=float, default=0.325,
        help="Side of the starting box around each sampled center.")
    parser.add_argument(
        "--perception", type=float, default=0.1,
        help="Neighbor radius used to build the inference graph.",
    )
    parser.add_argument("--exclusion", type=float, default=0.5)
    parser.add_argument(
        "--viz_mode",
        choices=["individual", "multi_tubular"],
        default="individual",
    )
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--skip_quadrant_inference", action="store_true")
    parser.add_argument("--near_goal_verbose", action="store_true")
    parser.add_argument("--near_goal_radius", type=float, default=1.0)
    parser.add_argument("--collision_warmup_steps", type=int, default=100)
    parser.add_argument("--print_collisions", action="store_true", default=False)
    parser.add_argument("--compare_ground_truth", action="store_true", default=False)
    parser.add_argument("--online_goal_count", type=int, default=5)
    parser.add_argument(
        "--goal_bounds", nargs=4, type=float,
        default=[-4.5, 4.5, -4.5, 4.5],
    )
    parser.add_argument("--goal_min_distance", type=float, default=2.5)
    parser.add_argument(
        "--manual_goals",
        nargs="*",
        type=float,
        default=None,
        help="Optional online goal sequence: x1 y1 x2 y2 ...",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if args.expert_r_padding < 0:
        raise ValueError("--expert_r_padding cannot be negative.")
    if args.expert_calibration_max_steps < 1:
        raise ValueError("--expert_calibration_max_steps must be positive.")
    if args.output_dir is None:
        args.output_dir = (
            "comparison_inference_2d" if args.mode == "comparison" else "."
        )
    os.makedirs(args.output_dir, exist_ok=True)
    if args.mode == "comparison":
        _run_comparison(args)
        return

    # -- Load weights --------------------------------------------------------
    if not args.run_tag:
        raise ValueError("Single mode requires --run_tag.")
    np.random.seed(args.seed)
    model = _load_model(
        args.weights_path or _find_single_weights(args.run_tag),
        task=args.task,
        n_boids=args.n_boids,
    )
    if args.task == "online_goals":
        _run_online_goal_inference(args, model)
        return

    # -- Build boids + exclusion zone -----------------------------------------
    boids = Boids(n_boids=args.n_boids, perception=args.perception,
                  crowding=getattr(args, "crowding", 0.02),
                  init_scatter=getattr(args, "init_scatter", 0.325))
    goals = boids.goal_positions
    excl  = Polygon(goals.tolist()).buffer(args.exclusion)

    # -- Sample test centers --------------------------------------------------
    n_per_q = max(1, args.n_centers // len(args.quadrants))
    test_centers = []
    for q_idx in args.quadrants:
        xmn, xmx, ymn, ymx = QUADRANT_BOUNDS[q_idx]
        q_centers = []
        while len(q_centers) < n_per_q:
            c = np.array([np.random.uniform(xmn, xmx), np.random.uniform(ymn, ymx)])
            if not excl.contains(Point(c)):
                q_centers.append(c)
        test_centers.extend(q_centers)
    print(f"Generated {len(test_centers)} test centers ({n_per_q} per quadrant, quadrants={args.quadrants})")

    # -- Run inference --------------------------------------------------------
    collision_counts = []
    q_labels = {0: "Q1(top-right)", 1: "Q2(top-left)", 2: "Q3(bot-left)", 3: "Q4(bot-right)"}
    center_q_map = [args.quadrants[k // n_per_q] for k in range(len(test_centers))]
    trajs = []

    if not args.skip_quadrant_inference:
        for k, center in enumerate(test_centers):
            q_idx = center_q_map[k]
            print(f"  Inference {k+1}/{len(test_centers)} | {q_labels[q_idx]} | center=({center[0]:.2f}, {center[1]:.2f})")
            traj = _run_one_trajectory_2d(
                model,
                boids,
                center,
                args.max_steps,
                args.n_boids,
                goals,
                args.success_threshold,
                verbose_prefix="  ",
                near_goal_verbose=args.near_goal_verbose,
                near_goal_radius=args.near_goal_radius,
            )
            trajs.append(traj)
            collisions, unique_pairs = _count_collision_events(
                traj, boids.crowding, args.collision_warmup_steps
            )
            collision_counts.append((collisions, unique_pairs))
            if args.print_collisions:
                print(f"    collision events after step {args.collision_warmup_steps} (< {boids.crowding:g}): {collisions}")
                print(f"    unique agent pairs involved: {unique_pairs}")
            progress = ordered_waypoint_progress(traj, goals, args.success_threshold)
            for g_idx, g in enumerate(goals):
                dist = progress.closest_distances[g_idx]
                print(f"    goal {g_idx} ({g[0]:.1f},{g[1]:.1f}): closest active-goal centroid dist = {dist:.4f}")
        print(f"Done. {len(trajs)} trajectories.")
        if args.viz_mode == "individual":
            _plot_individual_ranked(trajs, goals, args.n_boids, test_centers, args.run_tag, args.output_dir)
        else:
            _plot_multi_tubular(trajs, goals, args.run_tag, args.output_dir)
        r_thresholds, expert_radii = _fixed_r_thresholds_2d(trajs, args)
        report_path = os.path.join(
            args.output_dir, f"success_rate_{args.run_tag}.txt"
        )
        n_success, total, rate = _write_success_rate_2d(
            trajs,
            goals,
            test_centers,
            center_q_map,
            report_path,
            radius=args.success_threshold,
            r_thresholds=r_thresholds,
            expert_radii=expert_radii,
            r_description=_fixed_r_threshold_description_2d(args),
        )
        print(
            f"Saved success report: {report_path} "
            f"({n_success}/{total}, {rate:.2f}%)"
        )

    # -- Interactive mode ------------------------------------------------------
    if args.interactive:
        print(f"\n>>> Interactive mode - canvas is [-5,5]x[-5,5], goals at {goals.tolist()}")
        print("    Type 'q' to quit.\n")
        while True:
            try:
                raw = input("  Enter start center (x y): ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if raw.lower() in ('q', 'quit', 'exit'):
                break
            parts = raw.split()
            if len(parts) != 2:
                print("  Please enter exactly two numbers, e.g.: -3 2")
                continue
            try:
                cx, cy = float(parts[0]), float(parts[1])
            except ValueError:
                print("  Invalid numbers, try again.")
                continue
            center = np.array([cx, cy])
            print(f"  Running GNCA from ({cx:.2f}, {cy:.2f}) for {args.max_steps} steps...")
            traj = _run_one_trajectory_2d(
                model,
                boids,
                center,
                args.max_steps,
                args.n_boids,
                goals,
                args.success_threshold,
            )
            collisions, unique_pairs = _count_collision_events(
                traj, boids.crowding, args.collision_warmup_steps
            )
            collision_counts.append((collisions, unique_pairs))
            progress = ordered_waypoint_progress(traj, goals, args.success_threshold)
            for g_idx, g in enumerate(goals):
                dist = progress.closest_distances[g_idx]
                print(f"    goal {g_idx} ({g[0]:.1f},{g[1]:.1f}): closest active-goal centroid dist = {dist:.4f}")
            if args.print_collisions:
                print(f"    collision events after step {args.collision_warmup_steps} (< {boids.crowding:g}): {collisions}")
                print(f"    unique agent pairs involved: {unique_pairs}")
            # Plot
            fig, ax = plt.subplots(figsize=(7, 7))
            centroid, r = _plot_tube_on_axis(ax, traj, label="GNCA centroid")
            ax.scatter(goals[:, 0], goals[:, 1], c='red', marker='*', s=200, zorder=5)
            for g_idx, g in enumerate(goals):
                ax.annotate(f"G{g_idx}", g, textcoords="offset points", xytext=(6, 6), fontsize=10)
            ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
            ax.set_aspect('equal')
            ax.set_title(f"GNCA from ({cx:.2f},{cy:.2f}) | r={r:.3f}")
            ax.legend()
            plt.tight_layout()
            fname = os.path.join(args.output_dir, f"interactive_{cx:.1f}_{cy:.1f}.pdf")
            plt.savefig(fname); plt.close()
            print(f"  Saved {fname}\n")

    # -- Ground-truth boids on same centers -----------------------------------
    if args.compare_ground_truth and not args.skip_quadrant_inference:
        print("\n>>> Running ground-truth boids on same centers for comparison...")
        gt_trajs = []
        for k, center in enumerate(test_centers):
            q_idx = center_q_map[k]
            print(f"  GT {k+1}/{len(test_centers)} | {q_labels[q_idx]} | center=({center[0]:.2f}, {center[1]:.2f})")
            gt_boids = Boids(
                n_boids=args.n_boids,
                perception=args.perception,
                crowding=getattr(args, "crowding", 0.02),
                init_scatter=getattr(args, "init_scatter", 0.325),
            )
            history = gt_boids.generate_trajectory(save_config=False, random_init=center)
            pos_arr = history["positions"]
            vel_arr = history["velocities"]
            traj_gt = np.concatenate([pos_arr, vel_arr], axis=-1)
            gt_trajs.append(traj_gt)
            if args.near_goal_verbose:
                goal0 = goals[0]
                for step in range(len(traj_gt)):
                    centroid_now = traj_gt[step, :, :2].mean(axis=0)
                    mean_vel     = traj_gt[step, :, 2:].mean(axis=0)
                    dist_g0 = np.linalg.norm(centroid_now - goal0)
                    if dist_g0 < args.near_goal_radius:
                        angle = np.degrees(np.arctan2(mean_vel[1], mean_vel[0]))
                        print(f"    [GT step {step}] near goal0 | centroid=({centroid_now[0]:.3f},{centroid_now[1]:.3f}) dist={dist_g0:.3f} | mean_vel=({mean_vel[0]:.4f},{mean_vel[1]:.4f}) angle={angle:.1f} deg")
            progress = ordered_waypoint_progress(traj_gt, goals, args.success_threshold)
            for g_idx, g in enumerate(goals):
                dist = progress.closest_distances[g_idx]
                print(f"    goal {g_idx} ({g[0]:.1f},{g[1]:.1f}): closest active-goal centroid dist = {dist:.4f}")

        for k, (center, traj_gnca, traj_gt) in enumerate(zip(test_centers, trajs, gt_trajs)):
            fig, axes = plt.subplots(1, 2, figsize=(14, 7))
            for ax, traj, title in [(axes[0], traj_gnca, "GNCA"), (axes[1], traj_gt, "Ground Truth")]:
                centroid, r = _plot_tube_on_axis(ax, traj, label=f"{title} centroid")
                ax.scatter(goals[:, 0], goals[:, 1], c='red', marker='*', s=200, zorder=5, label='Goals')
                for g_idx, g in enumerate(goals):
                    ax.annotate(f"G{g_idx}", g, textcoords="offset points", xytext=(6, 6), fontsize=9)
                ax.set_xlim(-5, 5); ax.set_ylim(-5, 5)
                ax.set_aspect('equal')
                ax.set_title(f"{title} | r={r:.3f}")
                ax.legend(fontsize=8)
            fig.suptitle(f"Center ({center[0]:.2f},{center[1]:.2f}) | {q_labels[center_q_map[k]]}")
            plt.tight_layout()
            fname = os.path.join(args.output_dir, f"compare_gnca_vs_gt_center{k}.pdf")
            plt.savefig(fname); plt.close()
            print(f"Saved {fname}")

    if args.print_collisions:
        print(f"\nTotal post-warmup collision events across all GNCA runs: {sum(c[0] for c in collision_counts)}")
        print(f"Total unique-pair involvements across runs: {sum(c[1] for c in collision_counts)}")


if __name__ == "__main__":
    main()
