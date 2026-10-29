"""Standalone 3D GNCA inference and reporting.

Examples:
  python -m evaluation.run_inference_3d \
    --run_tag fixed3d --weights_path saved_models/best_weights_<run tag> \
    --octants 0 1 2 3 4 5 6 7 \
    --centers_per_octant 5 \
    --output_dir results/inference_3d_fixed3d \
    --save_multi \
    --save_individual
"""

import argparse
import glob
import os
import re
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["GRPC_VERBOSITY"] = "ERROR"
os.environ["GLOG_minloglevel"] = "3"
os.environ["ABSL_MIN_LOG_LEVEL"] = "3"

from runtime.common.subprocess_utils import install_filtered_stderr

install_filtered_stderr()
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from models.gnn_ca_simple_boids_3d import GNNCASimpleBoids3D
from models.gnn_ca_goal_conditioned_boids_3d import GoalConditionedGNNCABoids3D
from modules.boids_3d import Boids3D
from modules.waypoints import goal_conditioned_state, sample_waypoint_sequence
from evaluation.metrics.flock_metrics import (
    OrderedWaypointProgress,
    ordered_waypoint_progress,
    goal_segment_tube_radii,
    robust_tube_radius,
    trajectory_motion_metrics,
    write_online_benchmark_files,
)
from boids.generate_boids_cache_3d import (
    build_exclusion_zone_3d,
    _in_exclusion_zone_3d,
)


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


def parse_args():
    parser = argparse.ArgumentParser(description="Run 3D GNCA inference on unseen centers.")
    parser.add_argument(
        "--task",
        choices=("fixed_waypoints", "online_goals"),
        default="fixed_waypoints",
        help="Select the matching fixed- or online-goal GNCA architecture.",
    )
    parser.add_argument("--run_tag", required=True, help="Checkpoint run-tag suffix.")
    parser.add_argument("--weights_path", default="", help="Optional explicit checkpoint prefix.")
    parser.add_argument("--octants", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--centers_per_octant", type=int, default=5)
    parser.add_argument(
        "--centers_file",
        default="",
        help=(
            "Optional text/CSV/NPY file of explicit test centers. Text files must "
            "contain x y z or x y z octant per row. When provided, random center "
            "sampling arguments are ignored."
        ),
    )
    parser.add_argument("--exclusion", type=float, default=0.5)
    parser.add_argument("--n_boids", type=int, default=100)
    parser.add_argument(
        "--perception",
        type=float,
        default=0.1,
        help="Neighbor radius used to reconstruct the graph at every step.",
    )
    parser.add_argument("--crowding", type=float, default=0.02,
                        help="Separation radius of the Boids expert.")
    parser.add_argument("--init_scatter", type=float, default=0.325,
                        help="Side of the starting box around each sampled center.")
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--success_threshold", type=float, default=0.5)
    parser.add_argument(
        "--max_success_r",
        type=float,
        default=2.0,
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
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output_dir", default="inference_3d_outputs")
    parser.add_argument("--individual_dir", default="", help="Defaults to <output_dir>/individual.")
    parser.add_argument("--save_multi", action="store_true", default=False)
    parser.add_argument("--save_individual", action="store_true", default=False)
    parser.add_argument("--view_elev", type=float, default=24.0)
    parser.add_argument("--view_azim", type=float, default=-58.0)
    parser.add_argument("--debug_weight_paths", action="store_true", default=False)
    parser.add_argument("--online_goal_count", type=int, default=5)
    parser.add_argument(
        "--goal_bounds",
        nargs=6,
        type=float,
        default=[-4.5, 4.5, -4.5, 4.5, -4.5, 4.5],
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX", "Z_MIN", "Z_MAX"),
    )
    parser.add_argument("--goal_min_distance", type=float, default=2.5)
    return parser.parse_args()


def checkpoint_prefix(path):
    for ext in (".index", ".data-00000-of-00001"):
        if path.endswith(ext):
            return path[: -len(ext)]
    return path


def resolve_weights_path(run_tag, explicit_path="", debug=False, task="fixed_waypoints"):
    if explicit_path:
        prefix = checkpoint_prefix(explicit_path)
        print(f"Loading explicit weights prefix: {prefix}")
        return prefix

    patterns = [
        f"saved_models/best_weights_{run_tag}",
        f"saved_models/gnca_model_{run_tag}",
        f"saved_models/best_weights_{run_tag}*",
        f"saved_models/gnca_model_{run_tag}*",
    ]
    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern))

    prefixes = []
    for candidate in candidates:
        prefix = checkpoint_prefix(candidate)
        if prefix not in prefixes:
            prefixes.append(prefix)

    if debug:
        print("Weight search patterns:")
        for pattern in patterns:
            print(f"  {pattern}")
        print("Weight candidates:")
        for prefix in prefixes:
            print(f"  {prefix}")

    if not prefixes:
        raise FileNotFoundError(
            f"No weights found for run_tag '{run_tag}' in saved_models/."
        )

    prefixes.sort(key=lambda p: (0 if os.path.basename(p).startswith("gnca_model_") else 1, p))
    print(f"Loading weights: {prefixes[0]}")
    return prefixes[0]


def build_model(n_boids, weights_path, task="fixed_waypoints"):
    def custom_loss(y_true, y_pred):
        n = tf.shape(y_pred)[-1]
        next_state = y_true[..., n:2 * n]
        return tf.reduce_mean(tf.square(next_state - y_pred), axis=-1)

    model_class = (
        GoalConditionedGNNCABoids3D
        if task == "online_goals"
        else GNNCASimpleBoids3D
    )
    model = model_class(
        activation="linear",
        batch_norm=False,
        hidden=256,
        hidden_activation="relu",
        connectivity="cat",
        aggregate="mean",
    )
    model.compile(optimizer=Adam(learning_rate=1e-3), loss=custom_loss, run_eagerly=True)

    input_features = 9 if task == "online_goals" else 6
    x_dummy = tf.zeros((n_boids, input_features), dtype=tf.float32)
    a_dummy = tf.SparseTensor(
        indices=tf.zeros((0, 2), dtype=tf.int64),
        values=tf.zeros((0,), dtype=tf.float32),
        dense_shape=(n_boids, n_boids),
    )
    model([x_dummy, tf.sparse.reorder(a_dummy), tf.constant(0)], training=False)
    model.load_weights(weights_path).expect_partial()
    print("Weights loaded.")
    return model


def center_octant(center):
    for octant, (xmn, xmx, ymn, ymx, zmn, zmx) in OCTANT_BOUNDS.items():
        if xmn <= center[0] < xmx and ymn <= center[1] < ymx and zmn <= center[2] < zmx:
            return octant
    return -1


def sample_centers(octants, centers_per_octant, goals, exclusion):
    centers, labels = [], []
    exclusion_zone = build_exclusion_zone_3d(goals, exclusion)
    for octant in octants:
        xmn, xmx, ymn, ymx, zmn, zmx = OCTANT_BOUNDS[octant]
        made = 0
        while made < centers_per_octant:
            center = np.array([
                np.random.uniform(xmn, xmx),
                np.random.uniform(ymn, ymx),
                np.random.uniform(zmn, zmx),
            ], dtype=np.float32)
            if not _in_exclusion_zone_3d(center, exclusion_zone):
                centers.append(center)
                labels.append(octant)
                made += 1
    print(f"Generated {len(centers)} test centers with counts {dict((o, labels.count(o)) for o in octants)}")
    return centers, labels


def load_centers_file(path):
    """Load centers from numeric data or a run_inference_3d success report."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Centers file not found: {path}")

    if path.lower().endswith(".npy"):
        values = np.load(path)
    else:
        with open(path, "r", encoding="utf-8") as handle:
            report_text = handle.read()
        report_matches = re.findall(
            r"run\s+\d+\s+\|\s+octant\s+(-?\d+)\s+\|\s+"
            r"center=\(([^)]+)\)",
            report_text,
        )
        if report_matches:
            centers = []
            labels = []
            for octant_text, center_text in report_matches:
                center = np.asarray(
                    [float(value.strip()) for value in center_text.split(",")],
                    dtype=np.float32,
                )
                if center.shape != (3,):
                    raise ValueError(
                        f"Invalid center in success report '{path}': {center_text}"
                    )
                centers.append(center)
                labels.append(int(octant_text))
            print(
                f"Loaded {len(centers)} explicit test centers from success report "
                f"{path} with counts "
                f"{dict((o, labels.count(o)) for o in sorted(set(labels)))}"
            )
            return centers, labels

        try:
            values = np.loadtxt(path, comments="#", delimiter=",")
        except ValueError:
            values = np.loadtxt(path, comments="#")

    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] not in (3, 4) or len(values) == 0:
        raise ValueError(
            f"Centers file must have shape (N, 3) or (N, 4); got {values.shape}."
        )

    centers = [row[:3].copy() for row in values]
    if values.shape[1] == 4:
        labels = [int(row[3]) for row in values]
        if any(label not in OCTANT_BOUNDS for label in labels):
            raise ValueError("Every explicit octant label must be between 0 and 7.")
    else:
        labels = [center_octant(center) for center in centers]

    print(
        f"Loaded {len(centers)} explicit test centers from {path} with counts "
        f"{dict((o, labels.count(o)) for o in sorted(set(labels)))}"
    )
    return centers, labels


def to_tf_sparse(a):
    indices = np.stack([a.row, a.col], axis=1)
    sparse = tf.SparseTensor(
        indices=indices,
        values=a.data.astype(np.float32),
        dense_shape=a.shape,
    )
    return tf.sparse.reorder(sparse)


def run_one_trajectory(model, boids, center, n_boids, max_steps, goals,
                       success_threshold, max_success_r):
    """Stop at ordered goal completion; r limits affect reporting only."""
    pos, vel, _, _ = boids.get_random_init(n_boids, save_config=False, center=center)
    frames = [np.concatenate([pos, vel], axis=-1).astype(np.float32)]
    step_i = tf.constant(0)
    progress = OrderedWaypointProgress(goals, success_threshold)
    progress.update(frames[0][:, :3].mean(axis=0), 0)

    for step in range(max_steps - 1):
        if progress.completed:
            break
        x = frames[-1]
        a = to_tf_sparse(boids.get_neighbors(x[:, :3]))
        x_next = model([tf.constant(x, dtype=tf.float32), a, step_i], training=False).numpy()
        frames.append(x_next)
        if not np.all(np.isfinite(x_next)):
            print(f"    non-finite state at step {step + 1}; stopping rollout")
            break
        progress.update(x_next[:, :3].mean(axis=0), step + 1)
        if (step + 1) % 500 == 0:
            centroid = x_next[:, :3].mean(axis=0)
            print(f"    step {step+1} | centroid=({centroid[0]:.2f},{centroid[1]:.2f},{centroid[2]:.2f})")
        if progress.completed:
            print(
                f"    all goals reached in order at step {step + 1}; stopping rollout"
            )
            break
    return np.array(frames)


def goal_distances(traj, goals, threshold=0.5):
    return ordered_waypoint_progress(traj, goals, threshold).closest_distances


def tube_radius(traj):
    return robust_tube_radius(traj, spatial_dims=3)


def run_matched_fixed_expert_3d(initial_state, args):
    """Run fixed-waypoint 3D Boids from the exact GNCA initial state."""
    # The expert switches waypoints when the flock centroid is within 0.5 of
    # the active one, the same rule task completion is scored with.
    expert = Boids3D(
        n_boids=args.n_boids,
        perception=args.perception,
        crowding=getattr(args, "crowding", 0.02),
        goal_arrival_radius=0.5,
    )
    expert.init_scatter = getattr(args, "init_scatter", 0.325)
    initial_state = np.asarray(initial_state, dtype=np.float32)
    positions = initial_state[:, :3].copy()
    velocities = initial_state[:, 3:6].copy()
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
            "Matched fixed-waypoint 3D Boids calibration did not finish "
            f"within {args.expert_calibration_max_steps} steps. Increase "
            "--expert_calibration_max_steps."
        )
    return tube_radius(np.asarray(frames))


def fixed_r_thresholds_3d(trajs, args):
    if args.r_threshold_mode == "fixed":
        return (
            np.full(len(trajs), args.max_success_r, dtype=np.float64),
            None,
        )

    expert_radii = np.asarray(
        [run_matched_fixed_expert_3d(traj[0], args) for traj in trajs],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(expert_radii)):
        raise RuntimeError("Matched fixed-waypoint 3D Boids produced a non-finite r.")
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


def fixed_r_threshold_description_3d(args):
    if args.r_threshold_mode == "fixed":
        return f"fixed r <= {args.max_success_r:g}"
    if args.r_threshold_mode == "expert_per_trajectory":
        return (
            "r <= matched 3D Boids r + "
            f"{args.expert_r_padding:g} per trajectory"
        )
    return (
        "r <= maximum matched-3D-Boids r over this evaluation set + "
        f"{args.expert_r_padding:g}"
    )


def draw_tube_circles(ax, centroid, radius, color, n_sections=30, n_points=16,
                      lw=0.7, alpha=0.35):
    """Draw circular tube cross-sections around a centroid trajectory."""
    if not np.isfinite(radius) or radius <= 0:
        return
    step = max(1, len(centroid) // n_sections)
    theta = np.linspace(0, 2 * np.pi, n_points)
    for i in range(0, len(centroid), step):
        x_circ = centroid[i, 0] + radius * np.cos(theta)
        y_circ = centroid[i, 1] + radius * np.sin(theta)
        z_circ = np.ones_like(theta) * centroid[i, 2]
        ax.plot(x_circ, y_circ, z_circ, color=color, lw=lw, alpha=alpha)


def setup_3d_axes(ax, goals, elev, azim):
    ax.scatter(goals[:, 0], goals[:, 1], goals[:, 2], c="red", marker="*", s=200, zorder=5, label="Goals")
    for g_idx, goal in enumerate(goals):
        ax.text(goal[0], goal[1], goal[2], f"G{g_idx}", fontsize=9)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=elev, azim=azim)


def plot_multi(trajs, goals, centers, octants, run_tag, output_dir, elev, azim):
    os.makedirs(output_dir, exist_ok=True)
    fig = plt.figure(figsize=(9, 9))
    ax = fig.add_subplot(111, projection="3d")
    colors = cm.hsv(np.linspace(0, 0.9, len(trajs)))
    for color, traj, center, octant in zip(colors, trajs, centers, octants):
        centroid = traj[:, :, :3].mean(axis=1)
        r = tube_radius(traj)
        ax.plot(centroid[:, 0], centroid[:, 1], centroid[:, 2], color=color, lw=1.4, alpha=0.85)
        draw_tube_circles(
            ax, centroid, r, color,
            n_sections=14,
            n_points=12,
            lw=0.45,
            alpha=0.22,
        )
        ax.scatter([center[0]], [center[1]], [center[2]], c=[color], marker="o", s=35, zorder=6)
        ax.text(center[0], center[1], center[2], f"O{octant}", fontsize=6)
    setup_3d_axes(ax, goals, elev, azim)
    ax.set_title(f"3D GNCA Inference | {len(trajs)} runs | {run_tag}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(output_dir, f"inference3d_{run_tag}_multi.pdf")
    plt.savefig(path)
    plt.close()
    print(f"Saved combined PDF: {path}")
    return path


def plot_individual_trajectory(
    traj, goals, center, octant, run_idx, run_tag, output_dir, elev, azim
):
    """Save one tubular trajectory PDF immediately after its rollout finishes."""
    os.makedirs(output_dir, exist_ok=True)
    radius = tube_radius(traj)
    centroid = traj[:, :, :3].mean(axis=1)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        centroid[:, 0], centroid[:, 1], centroid[:, 2],
        color="#8B0000", lw=1.5, label="Centroid path",
    )
    draw_tube_circles(
        ax, centroid, radius, "#8B0000",
        n_sections=30,
        n_points=16,
        lw=0.8,
        alpha=0.4,
    )
    ax.plot([], [], [], color="#8B0000", lw=0.8, alpha=0.4, label="Flock tube")
    ax.scatter(
        [center[0]], [center[1]], [center[2]],
        c="blue", marker="*", s=100, zorder=6, label="Start",
    )
    setup_3d_axes(ax, goals, elev, azim)
    ax.set_title(
        f"Run {run_idx} | Octant {octant} | "
        f"start=({center[0]:.2f},{center[1]:.2f},{center[2]:.2f}) | "
        f"r={radius:.3f}"
    )
    ax.legend(fontsize=8)
    plt.tight_layout()
    path = os.path.join(
        output_dir,
        f"inference3d_{run_tag}_run{run_idx:03d}_oct{octant}.pdf",
    )
    plt.savefig(path)
    plt.close()
    print(f"  Saved individual PDF: {path}", flush=True)
    return path


def write_success_report(
    path, run_tag, centers, octants, dists, radii, diverged, threshold,
    r_thresholds, r_description, expert_radii=None,
):
    reaches_all_goals = np.all(dists <= threshold, axis=1)
    r_thresholds = np.asarray(r_thresholds, dtype=np.float64)
    stable = np.isfinite(radii) & (radii <= r_thresholds)
    diverged = np.asarray(diverged, dtype=bool)
    successes = reaches_all_goals & stable & ~diverged
    lines = [
        f"3D GNCA success report",
        f"run_tag: {run_tag}",
        f"success definition: centroid distance <= {threshold:g} to every goal in the prescribed order and {r_description}",
        "Goal distances are measured only while that goal is active; inactive-goal visits do not count.",
        f"overall: {int(successes.sum())}/{len(successes)} = {successes.mean() * 100:.2f}%",
        "",
        "per octant:",
    ]
    for octant in sorted(set(octants)):
        mask = np.array(octants) == octant
        n = int(mask.sum())
        s = int(successes[mask].sum())
        lines.append(f"  octant {octant}: {s}/{n} = {(s / n * 100 if n else 0):.2f}%")
    lines.extend(["", "per run:"])
    for i, (center, octant, run_dists, r, applied_r, hit_goal, is_stable, success) in enumerate(
        zip(
            centers, octants, dists, radii, r_thresholds,
            reaches_all_goals, stable, successes,
        )
    ):
        dist_str = ", ".join(f"g{j}={d:.4f}" for j, d in enumerate(run_dists))
        expert_text = (
            f" | expert_r={float(expert_radii[i]):.4f}"
            if expert_radii is not None else ""
        )
        lines.append(
            f"  run {i:03d} | octant {octant} | center=({center[0]:.4f},{center[1]:.4f},{center[2]:.4f}) "
            f"| success={bool(success)} | reaches_all_goals={bool(hit_goal)} "
            f"| stable_r={bool(is_stable)} | r={r:.4f} "
            f"| r_threshold={applied_r:.4f}{expert_text} | {dist_str}"
        )
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved success report: {path}")


def sample_online_centers(octants, centers_per_octant, rng):
    """Sample independent continuous test starts without fixed-goal exclusion."""
    centers, labels = [], []
    for octant in octants:
        if octant not in OCTANT_BOUNDS:
            raise ValueError(f"Unknown octant {octant}; expected an integer from 0 to 7.")
        # OCTANT_BOUNDS is stored as interleaved lower/upper values.
        lower = np.asarray(
            [OCTANT_BOUNDS[octant][0], OCTANT_BOUNDS[octant][2], OCTANT_BOUNDS[octant][4]],
            dtype=np.float32,
        )
        upper = np.asarray(
            [OCTANT_BOUNDS[octant][1], OCTANT_BOUNDS[octant][3], OCTANT_BOUNDS[octant][5]],
            dtype=np.float32,
        )
        for _ in range(centers_per_octant):
            centers.append(rng.uniform(lower, upper).astype(np.float32))
            labels.append(int(octant))
    print(
        f"Generated {len(centers)} online-goal test centers with counts "
        f"{dict((o, labels.count(o)) for o in octants)}"
    )
    return centers, labels


def run_online_goal_trajectory(model, boids, center, args, rng):
    """Roll out the conditioned 3D GNCA under an external waypoint manager."""
    position, velocity, _, _ = boids.get_random_init(
        args.n_boids, save_config=False, center=np.asarray(center)
    )
    physical = np.concatenate((position, velocity), axis=-1).astype(np.float32)
    frames = [physical.copy()]
    arrival_steps = []
    closest_distances = []
    reached = 0
    diverged = False
    divergence_step = None

    sampled_goals = sample_waypoint_sequence(
        rng,
        position.mean(axis=0),
        n_waypoints=args.online_goal_count,
        bounds=tuple(args.goal_bounds),
        min_distance=args.goal_min_distance,
    )
    active_goal = sampled_goals[0]
    closest_current = float(np.linalg.norm(position.mean(axis=0) - active_goal))

    for step in range(1, args.max_steps + 1):
        conditioned = goal_conditioned_state(physical, active_goal)
        adjacency = to_tf_sparse(boids.get_neighbors(physical[:, :3]))
        next_physical = model(
            [tf.constant(conditioned), adjacency, tf.constant(0)],
            training=False,
        ).numpy()

        if not np.all(np.isfinite(next_physical)):
            print(f"    non-finite state encountered at step {step}; stopping rollout")
            diverged = True
            divergence_step = step
            break
        physical = next_physical
        frames.append(physical.copy())
        centroid = physical[:, :3].mean(axis=0)
        centroid_distance = float(np.linalg.norm(centroid - active_goal))
        closest_current = min(closest_current, centroid_distance)
        if step % 500 == 0:
            print(
                f"    step {step}/{args.max_steps} | "
                f"centroid=({centroid[0]:.2f},{centroid[1]:.2f},{centroid[2]:.2f}) | "
                f"active_goal=({active_goal[0]:.2f},{active_goal[1]:.2f},{active_goal[2]:.2f})"
            )

        if centroid_distance > args.success_threshold:
            continue

        closest_distances.append(float(closest_current))
        reached += 1
        arrival_steps.append(step)
        if reached >= len(sampled_goals):
            break
        active_goal = sampled_goals[reached]
        closest_current = float(np.linalg.norm(centroid - active_goal))

    if reached < args.online_goal_count:
        closest_distances.append(float(closest_current))
        closest_distances.extend(
            [float("inf")] * (args.online_goal_count - len(closest_distances))
        )

    trajectory = np.asarray(frames)
    goal_radii = goal_segment_tube_radii(
        trajectory, arrival_steps, spatial_dims=3
    )
    max_goal_radius = max(goal_radii, default=float("inf"))
    reached_all = reached == args.online_goal_count
    return {
        "trajectory": trajectory,
        "initial_state": frames[0].copy(),
        "goals": np.asarray(sampled_goals, dtype=np.float32),
        "arrival_steps": arrival_steps,
        "closest_distances": np.asarray(closest_distances, dtype=np.float64),
        "reached": reached,
        "requested": args.online_goal_count,
        "goal_tube_radii": goal_radii,
        "max_goal_tube_radius": max_goal_radius,
        "motion_metrics": trajectory_motion_metrics(trajectory, 3),
        "reached_all_goals": reached_all,
        "diverged": diverged,
        "divergence_step": divergence_step,
    }


def run_matched_online_expert(initial_state, goals, args):
    """Evaluate 3D Boids from the GNCA's exact start and goal sequence."""
    expert = Boids3D(n_boids=args.n_boids, perception=args.perception,
                     crowding=getattr(args, "crowding", 0.02))
    expert.init_scatter = getattr(args, "init_scatter", 0.325)
    initial_state = np.asarray(initial_state, dtype=np.float32)
    positions = initial_state[:, :3].copy()
    velocities = initial_state[:, 3:6].copy()
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
        trajectory, arrival_steps, spatial_dims=3
    )
    return {
        "trajectory": trajectory,
        "goal_tube_radii": goal_radii,
        "max_goal_tube_radius": max(goal_radii, default=float("inf")),
        "motion_metrics": trajectory_motion_metrics(trajectory, 3),
        "reached": active_goal,
        "requested": len(goals),
        "completed": active_goal == len(goals),
        "steps": len(trajectory) - 1,
    }


def attach_online_benchmarks(results, args):
    """Attach one exact-start, exact-goal 3D Boids benchmark to every run."""
    for result in results:
        result["expert_benchmark"] = run_matched_online_expert(
            result["initial_state"], result["goals"], args
        )


def plot_online_individual(result, center, octant, run_index, run_tag,
                           output_dir, elev, azim):
    """Save one online-goal trajectory immediately after rollout completion."""
    trajectory = result["trajectory"]
    centroid = trajectory[:, :, :3].mean(axis=1)
    goals = result["goals"]
    radius = result["max_goal_tube_radius"]
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(*centroid.T, color="#8B0000", lw=1.6, label="Centroid path")
    draw_tube_circles(ax, centroid, radius, "#8B0000")
    colors = cm.viridis(np.linspace(0.08, 0.92, max(len(goals), 1)))
    for goal_index, (goal, color) in enumerate(zip(goals, colors), start=1):
        ax.scatter(*goal, color=color, marker="*", s=180, depthshade=False)
        ax.text(*(goal + 0.12), f"G{goal_index}", fontsize=8)
    ax.scatter(*center, color="blue", marker="*", s=100, label="Start", depthshade=False)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(
        f"Online 3D GNCA | Octant {octant} | "
        f"start=({center[0]:.2f},{center[1]:.2f},{center[2]:.2f})\n"
        f"goals={result['reached']}/{result['requested']} | max goal r={radius:.3f}"
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(
        output_dir, f"online3d_{run_tag}_run{run_index:03d}_oct{octant}.pdf"
    )
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved individual PDF: {path}", flush=True)


def plot_online_summary(results, centers, run_tag, output_dir, elev, azim):
    """Save all online-goal test trajectories in one PDF."""
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(111, projection="3d")
    colors = cm.hsv(np.linspace(0, 0.9, len(results)))
    for result, center, color in zip(results, centers, colors):
        centroid = result["trajectory"][:, :, :3].mean(axis=1)
        ax.plot(*centroid.T, color=color, lw=1.0, alpha=0.8)
        ax.scatter(*center, color=color, marker="o", s=18, depthshade=False)
        if len(result["goals"]):
            ax.scatter(*result["goals"].T, color=color, marker="x", s=14, alpha=0.65)
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)
    ax.set_zlim(-5, 5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(f"Online 3D GNCA | {len(results)} unseen runs | {run_tag}")
    fig.tight_layout()
    path = os.path.join(output_dir, f"online3d_{run_tag}_all.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved combined PDF: {path}")




def run_online_goal_inference(args, model):
    rng = np.random.default_rng(args.seed)
    if args.centers_file:
        centers, octant_labels = load_centers_file(args.centers_file)
    else:
        centers, octant_labels = sample_online_centers(
            args.octants, args.centers_per_octant, rng
        )
    np.save(os.path.join(args.output_dir, "online_test_centers.npy"), centers)

    boids = Boids3D(n_boids=args.n_boids, perception=args.perception,
                    crowding=getattr(args, "crowding", 0.02))
    boids.init_scatter = getattr(args, "init_scatter", 0.325)
    individual_dir = args.individual_dir or os.path.join(args.output_dir, "individual")
    if args.save_individual:
        os.makedirs(individual_dir, exist_ok=True)

    results = []
    for run_index, (center, octant) in enumerate(zip(centers, octant_labels)):
        print(
            f"  Online inference {run_index + 1}/{len(centers)} | "
            f"octant={octant} | "
            f"center=({center[0]:.2f},{center[1]:.2f},{center[2]:.2f})"
        )
        result = run_online_goal_trajectory(
            model,
            boids,
            center,
            args,
            np.random.default_rng(int(args.seed or 0) + 10_000 + run_index),
        )
        results.append(result)
        if args.save_individual:
            plot_online_individual(
                result, center, octant, run_index, args.run_tag,
                individual_dir, args.view_elev, args.view_azim,
            )

    attach_online_benchmarks(results, args)
    for result in results:
        expert = result["expert_benchmark"]
        outcome = (
            "completed"
            if result["reached_all_goals"] and not result["diverged"]
            else ("instability" if result["diverged"] else "timeout")
        )
        print(
            f"    reached {result['reached']}/{result['requested']} goals | "
            f"goal_r={np.round(result['goal_tube_radii'], 4).tolist()} | "
            f"max_r={result['max_goal_tube_radius']:.4f} | "
            f"outcome={outcome} | matched_boids_max_r="
            f"{expert['max_goal_tube_radius']:.4f}"
        )

    if args.save_multi or not args.save_individual:
        plot_online_summary(
            results, centers, args.run_tag,
            args.output_dir, args.view_elev, args.view_azim,
        )
    write_online_benchmark_files(
        output_dir=args.output_dir,
        run_tag=args.run_tag,
        centers=centers,
        regions=octant_labels,
        region_name="octant",
        results=results,
        goal_threshold=args.success_threshold,
        max_steps=args.max_steps,
    )


def main():
    args = parse_args()
    if args.expert_r_padding < 0:
        raise ValueError("--expert_r_padding cannot be negative.")
    if args.expert_calibration_max_steps < 1:
        raise ValueError("--expert_calibration_max_steps must be positive.")
    if args.online_goal_count < 1:
        raise ValueError("--online_goal_count must be positive.")
    if args.goal_min_distance < 0:
        raise ValueError("--goal_min_distance cannot be negative.")
    if args.seed is not None:
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    individual_dir = args.individual_dir or os.path.join(args.output_dir, "individual")

    weights_path = resolve_weights_path(
        args.run_tag,
        args.weights_path,
        args.debug_weight_paths,
        task=args.task,
    )
    model = build_model(args.n_boids, weights_path, task=args.task)

    if args.task == "online_goals":
        run_online_goal_inference(args, model)
        return

    boids = Boids3D(n_boids=args.n_boids, perception=args.perception,
                    crowding=getattr(args, "crowding", 0.02))
    boids.init_scatter = getattr(args, "init_scatter", 0.325)
    goals = boids.goal_positions
    if args.centers_file:
        centers, octant_labels = load_centers_file(args.centers_file)
    else:
        centers, octant_labels = sample_centers(
            args.octants, args.centers_per_octant, goals, args.exclusion
        )

    trajs, dist_rows, radii, divergences = [], [], [], []
    for idx, (center, octant) in enumerate(zip(centers, octant_labels), start=1):
        print(
            f"  Inference {idx}/{len(centers)} | octant={octant} | "
            f"center=({center[0]:.2f},{center[1]:.2f},{center[2]:.2f})"
        )
        traj = run_one_trajectory(
            model,
            boids,
            center,
            args.n_boids,
            args.max_steps,
            goals,
            args.success_threshold,
            args.max_success_r,
        )
        trajs.append(traj)
        diverged = not np.all(np.isfinite(traj))
        divergences.append(diverged)
        dists = goal_distances(traj, goals, args.success_threshold)
        r = tube_radius(traj)
        dist_rows.append(dists)
        radii.append(r)
        for goal_idx, dist in enumerate(dists):
            print(f"    goal {goal_idx} {goals[goal_idx].tolist()}: closest active-goal centroid dist = {dist:.4f}")
        print(f"    r = {r:.4f}")
        if args.save_individual:
            plot_individual_trajectory(
                traj,
                goals,
                center,
                octant,
                idx - 1,
                args.run_tag,
                individual_dir,
                args.view_elev,
                args.view_azim,
            )

    dists = np.stack(dist_rows, axis=0)
    radii = np.array(radii, dtype=np.float32)
    r_thresholds, expert_radii = fixed_r_thresholds_3d(trajs, args)
    successes = (
        np.all(dists <= args.success_threshold, axis=1)
        & np.isfinite(radii)
        & (radii <= r_thresholds)
        & ~np.asarray(divergences, dtype=bool)
    )
    for idx, success in enumerate(successes):
        expert_text = (
            f", expert_r={expert_radii[idx]:.4f}"
            if expert_radii is not None else ""
        )
        print(
            f"  Run {idx + 1} success: {bool(success)} "
            f"(r={radii[idx]:.4f}, r_threshold={r_thresholds[idx]:.4f}"
            f"{expert_text})"
        )

    if args.save_multi or not args.save_individual:
        plot_multi(trajs, goals, centers, octant_labels, args.run_tag, args.output_dir, args.view_elev, args.view_azim)

    report_path = os.path.join(args.output_dir, f"success_rate_{args.run_tag}.txt")
    write_success_report(
        report_path,
        args.run_tag,
        centers,
        octant_labels,
        dists,
        radii,
        divergences,
        args.success_threshold,
        r_thresholds,
        fixed_r_threshold_description_3d(args),
        expert_radii,
    )


if __name__ == "__main__":
    main()
