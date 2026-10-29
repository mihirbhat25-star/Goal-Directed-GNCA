"""Reusable visualization and offline analysis for 3D GNCA evaluation.

Standalone visualization loads a saved 3D GNCA model and Boids object.
Loads a saved 3D GNCA model and boids object, then generates 3D visualizations.

Canonical usage:
    python -m evaluation.visualization.visualize_boids_3d [options]

The original ``python -m boids.visualize_boids_3d`` entry point remains
compatible.
"""
import argparse
import os
import sys

import joblib
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from evaluation.metrics.flock_metrics import (
    load_statistics,
    nearest_runs,
    print_summary,
    write_blowup_csv,
)
from modules.boids_3d import BOIDS_GOAL_POSITIONS_3D

@tf.function(experimental_relax_shapes=True)
def forward_3d(model, x, a, i, training=None):
    """Computes one forward pass of the 3D GNCA"""
    x_pred = model((x, a, i[:, None]), training=training)
    return x_pred


def to_tf_sparse(a):
    """Convert scipy sparse matrix to TensorFlow sparse tensor."""
    indices = np.stack([a.row, a.col], axis=1)
    a_tf = tf.SparseTensor(
        indices=indices,
        values=a.data.astype(np.float32),
        dense_shape=a.shape
    )
    return tf.sparse.reorder(a_tf)


def run_autoregressive_3d(model, boids, positions, velocities, n_boids, max_trajectory_len):
    """Runs the 3D GNCA autoregressively from a given start position."""
    x = np.concatenate([positions, velocities], axis=-1)
    a_scipy = boids.get_neighbors(positions)
    a = to_tf_sparse(a_scipy)
    trajectory = [x.astype(np.float32)]
    for _ in range(max_trajectory_len - 1):
        x_last = trajectory[-1]
        a_scipy = boids.get_neighbors(x_last[:, :3])
        a = to_tf_sparse(a_scipy)
        x_next = forward_3d(model, x_last, a, np.zeros((n_boids, 1)), training=False)
        trajectory.append(x_next.numpy())
    return np.array(trajectory)


def _compute_tube_stats_3d(trajs):
    """
    Compute tube radius for 3D trajectories.
    Returns per-run tube radii and centroid curves.

    trajs: list of (T, n_boids, 6) arrays, one per run.
    Returns: list of (centroid_curve, tube_radius, boid_dists) tuples
    """
    stats = []
    for traj in trajs:
        centroid = traj[:, :, :3].mean(axis=1)  # (T, 3)
        boid_dists = np.linalg.norm(traj[:, :, :3] - centroid[:, None, :], axis=-1)  # (T, n_boids)
        r = np.percentile(np.percentile(boid_dists, 95, axis=1), 99)
        stats.append((centroid, r, boid_dists))
    return stats


def _plot_tubular_mean_3d(trajs, goals, filename="boids_tube_mean_3d.pdf"):
    """
    Mean tubular visualization: average trajectory with 3D tube representation.
    """
    stats = _compute_tube_stats_3d(trajs)
    centroids = np.array([s[0] for s in stats])  # (N_runs, T, 3)
    mean_curve = centroids.mean(axis=0)  # (T, 3)
    r_99 = np.percentile([s[1] for s in stats], 99)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot mean curve as centerline
    ax.plot(mean_curve[:, 0], mean_curve[:, 1], mean_curve[:, 2],
            color='#8B0000', lw=3, label='Mean path', zorder=10)

    # Create tube surface by plotting circular cross-sections
    step = max(1, len(mean_curve) // 30)
    for i in range(0, len(mean_curve), step):
        theta = np.linspace(0, 2*np.pi, 16)
        # Create points in a circle perpendicular to the curve
        x_circ = mean_curve[i, 0] + r_99 * np.cos(theta)
        y_circ = mean_curve[i, 1] + r_99 * np.sin(theta)
        z_circ = np.ones_like(theta) * mean_curve[i, 2]
        ax.plot(x_circ, y_circ, z_circ, color='#8B0000', lw=0.8, alpha=0.4)

    ax.scatter(goals[:, 0], goals[:, 1], goals[:, 2],
               c='red', marker='*', s=300, zorder=15, label='Goals', edgecolors='darkred', linewidth=1)
    ax.set_xlabel("X", fontsize=11)
    ax.set_ylabel("Y", fontsize=11)
    ax.set_zlabel("Z", fontsize=11)
    ax.set_title(f"Mean Trajectory (r={r_99:.2f})", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=100)
    plt.close()
    print(f"Saved {filename}")


def _plot_single_tubular_3d(trajs, goals, filename="boids_tube_single_3d.pdf"):
    """
    Single run tubular visualization with continuous 3D tube.
    """
    run_idx = np.random.randint(len(trajs))
    traj = trajs[run_idx]
    centroid = traj[:, :, :3].mean(axis=1)
    boid_dists = np.linalg.norm(traj[:, :, :3] - centroid[:, None, :], axis=-1)
    r = np.percentile(np.percentile(boid_dists, 95, axis=1), 99)

    print(f"Single-tube viz using run {run_idx} of {len(trajs)}")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot curve as centerline
    ax.plot(centroid[:, 0], centroid[:, 1], centroid[:, 2],
            color='#8B0000', lw=3, label=f'Run {run_idx}', zorder=10)

    # Create tube surface
    step = max(1, len(centroid) // 30)
    for i in range(0, len(centroid), step):
        theta = np.linspace(0, 2*np.pi, 16)
        x_circ = centroid[i, 0] + r * np.cos(theta)
        y_circ = centroid[i, 1] + r * np.sin(theta)
        z_circ = np.ones_like(theta) * centroid[i, 2]
        ax.plot(x_circ, y_circ, z_circ, color='#8B0000', lw=0.8, alpha=0.4)

    ax.scatter(goals[:, 0], goals[:, 1], goals[:, 2],
               c='red', marker='*', s=300, zorder=15, label='Goals', edgecolors='darkred', linewidth=1)
    ax.set_xlabel("X", fontsize=11)
    ax.set_ylabel("Y", fontsize=11)
    ax.set_zlabel("Z", fontsize=11)
    ax.set_title(f"Run {run_idx} (r={r:.2f})", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=100)
    plt.close()
    print(f"Saved {filename}")


def _plot_multi_tubular_3d(trajs, goals, n_show=10, filename="boids_multi_tube_3d.pdf", specific_runs=None):
    """
    Multiple runs as separate tubes in a single 3D view.
    """
    import matplotlib.cm as cm

    if specific_runs is not None:
        indices = [i for i in specific_runs if i < len(trajs)]
    else:
        indices = np.random.choice(len(trajs), min(n_show, len(trajs)), replace=False)

    colors = cm.hsv(np.linspace(0, 0.9, len(indices)))

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for color, run_idx in zip(colors, indices):
        traj = trajs[run_idx]
        centroid = traj[:, :, :3].mean(axis=1)
        boid_dists = np.linalg.norm(traj[:, :, :3] - centroid[:, None, :], axis=-1)
        r = np.percentile(np.percentile(boid_dists, 95, axis=1), 99)

        # Plot centerline
        ax.plot(centroid[:, 0], centroid[:, 1], centroid[:, 2],
                color=color, lw=2.5, label=f'Run {run_idx}', zorder=10)

        # Create tube surface
        step = max(1, len(centroid) // 20)
        for i in range(0, len(centroid), step):
            theta = np.linspace(0, 2*np.pi, 12)
            x_circ = centroid[i, 0] + r * np.cos(theta)
            y_circ = centroid[i, 1] + r * np.sin(theta)
            z_circ = np.ones_like(theta) * centroid[i, 2]
            ax.plot(x_circ, y_circ, z_circ, color=color, lw=0.6, alpha=0.3)

    ax.scatter(goals[:, 0], goals[:, 1], goals[:, 2],
               c='red', marker='*', s=300, zorder=15, label='Goals', edgecolors='darkred', linewidth=1)
    ax.set_xlabel("X", fontsize=11)
    ax.set_ylabel("Y", fontsize=11)
    ax.set_zlabel("Z", fontsize=11)
    ax.set_title(f"{len(indices)} Runs", fontsize=12)
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=100)
    plt.close()
    print(f"Saved {filename}")


def _plot_per_boid_3d(trajs, goals, n_boids, filename="boids_per_boid_3d.pdf"):
    """
    Individual boid paths visualization for one random run.
    """
    run_idx = np.random.randint(len(trajs))
    traj = trajs[run_idx]

    print(f"Per-boid viz using run {run_idx} of {len(trajs)}")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    for b in range(n_boids):
        ax.plot(traj[:, b, 0], traj[:, b, 1], traj[:, b, 2],
                color='#8B0000', lw=0.5, alpha=0.5)

    ax.scatter(goals[:, 0], goals[:, 1], goals[:, 2],
               c='red', marker='*', s=300, zorder=15, label='Goals', edgecolors='darkred', linewidth=1)
    ax.set_xlabel("X", fontsize=11)
    ax.set_ylabel("Y", fontsize=11)
    ax.set_zlabel("Z", fontsize=11)
    ax.set_title(f"Individual Boid Paths (run {run_idx})", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=100)
    plt.close()
    print(f"Saved {filename}")


def _plot_individual_ranked_3d(trajs, goals, selected_centers=None, run_tag="", output_dir="."):
    """
    Generate individual PDFs for each run, ranked by tube radius.
    Each PDF shows a single 3D visualization with tube.
    """
    stats = _compute_tube_stats_3d(trajs)

    # Create tuples (run_idx, tube_radius, centroid, boid_dists, center)
    run_stats = []
    for idx, (centroid, r, boid_dists) in enumerate(stats):
        center = selected_centers[idx] if selected_centers is not None and len(selected_centers) > idx else None
        run_stats.append((idx, r, centroid, boid_dists, center))

    # Sort by tube radius (greatest to least)
    run_stats.sort(key=lambda x: x[1], reverse=True)

    print("\n--- Runs sorted by tube radius r (greatest -> least) ---")
    for rank, (run_idx, r, centroid, boid_dists, center) in enumerate(run_stats):
        print(f"Rank {rank+1:02d} | Run {run_idx:02d} | r={r:.4f} | center: {np.round(center, 4) if center is not None else 'unknown'}")

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot centerline
        ax.plot(centroid[:, 0], centroid[:, 1], centroid[:, 2],
                color='#8B0000', lw=3, zorder=10)

        # Create tube surface
        step = max(1, len(centroid) // 30)
        for i in range(0, len(centroid), step):
            theta = np.linspace(0, 2*np.pi, 16)
            x_circ = centroid[i, 0] + r * np.cos(theta)
            y_circ = centroid[i, 1] + r * np.sin(theta)
            z_circ = np.ones_like(theta) * centroid[i, 2]
            ax.plot(x_circ, y_circ, z_circ, color='#8B0000', lw=0.8, alpha=0.4)

        ax.scatter(goals[:, 0], goals[:, 1], goals[:, 2],
                   c='red', marker='*', s=300, zorder=15, label='Goals', edgecolors='darkred', linewidth=1)
        ax.set_xlabel("X", fontsize=11)
        ax.set_ylabel("Y", fontsize=11)
        ax.set_zlabel("Z", fontsize=11)
        ax.set_title(f"Rank {rank+1:02d} | Run {run_idx:02d} | r={r:.3f}", fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        tag = f"_{run_tag}" if run_tag else ""
        fname = os.path.join(output_dir, f"boids_tube_run{tag}_rank{rank+1:02d}_run{run_idx:02d}_3d.pdf")
        plt.savefig(fname, dpi=100)
        plt.close()
        print(f"  -> Saved {fname}")


def _save_tubular_triplet_3d(trajs, goals, tag, label, output_dir="."):
    """Save mean, single, and all-runs 3D tubular PDFs for a trajectory set."""
    _plot_tubular_mean_3d(trajs, goals,
                          filename=os.path.join(output_dir, f"boids_tube_mean_{label}{tag}.pdf"))
    _plot_single_tubular_3d(trajs, goals,
                            filename=os.path.join(output_dir, f"boids_tube_single_{label}{tag}.pdf"))
    # Cap all-runs to 50 max, randomly selected
    n_runs_to_show = min(50, len(trajs))
    selected_runs = list(np.random.choice(len(trajs), n_runs_to_show, replace=False))
    _plot_multi_tubular_3d(trajs, goals, n_show=n_runs_to_show,
                           filename=os.path.join(output_dir, f"boids_tube_50_runs_{label}{tag}.pdf"),
                           specific_runs=selected_runs)

def _model_hash(model):
    """Compute a short hash of the model's trainable weights for cache keying."""
    import hashlib
    h = hashlib.md5()
    for w in model.trainable_variables:
        h.update(w.numpy().tobytes())
    return h.hexdigest()[:12]


def generate_trajectories(model, boids, max_trajectory_len, n_boids,
                         use_saved_config, traj_cache_path, max_viz_runs=50):
    """
    Generate or load cached 3D trajectories.
    """
    np.random.seed(1)

    model_tag = _model_hash(model)
    trajs = None
    selected_centers = None

    # Check cache
    if os.path.exists(traj_cache_path):
        try:
            cache = np.load(traj_cache_path, allow_pickle=True)
            if cache.get("model_tag", "") == model_tag and cache.get("n_runs", 0) == max_viz_runs:
                print(f"Loaded cached 3D trajectories ({max_viz_runs} runs) from '{traj_cache_path}'")
                trajs = list(cache["trajs"])
                if "centers" in cache:
                    selected_centers = cache["centers"]
        except Exception as e:
            print(f"Warning: Cache load failed: {e}")

    if trajs is None:
        print(f"Computing {max_viz_runs} 3D trajectories (model_tag={model_tag})...")
        trajs = []
        all_centers = []

        if use_saved_config and len(boids.rand_configs) > 0:
            all_center_configs = np.array(boids.rand_configs)
            n_runs = min(max_viz_runs, len(all_center_configs))
            rand_indices = np.random.choice(len(all_center_configs), size=n_runs, replace=False)
            selected_centers = all_center_configs[rand_indices]
        else:
            n_runs = max_viz_runs
            selected_centers = None

        for i in range(n_runs):
            if selected_centers is not None:
                center = selected_centers[i]
            else:
                center = np.array([-2.5 + np.random.rand() * 5,
                                   -2.5 + np.random.rand() * 5,
                                   np.random.rand()])

            positions = center + 0.325 * np.random.rand(n_boids, 3)
            direction = np.array([1.0, 0.0, 0.0])
            velocity = direction * boids.max_speed
            velocities = np.tile(velocity, (n_boids, 1))

            traj = run_autoregressive_3d(model, boids, positions, velocities, n_boids, max_trajectory_len)
            trajs.append(traj)
            all_centers.append(center)

        np.savez(traj_cache_path,
                 trajs=np.array(trajs),
                 model_tag=model_tag,
                 n_runs=len(trajs),
                 centers=np.array(all_centers) if all_centers else np.array([]))
        print(f"Cached 3D trajectories to '{traj_cache_path}'")

    np.random.seed(None)
    return trajs, selected_centers


def visualize_3d(model, boids, max_trajectory_len, n_boids, viz_mode='tubular',
                 traj_cache_path="viz_trajectories_3d.npz", run_tag="", n_show=50, specific_runs=None, output_dir="."):
    """
    Main 3D visualization function.

    viz_mode: 'tubular'       - mean over all runs with tubular band
              'per_boid'      - individual boid paths from one run
              'multi_tubular' - multiple runs as separate tubes
              'individual'    - individual PDFs ranked by tube radius
    """
    trajs, selected_centers = generate_trajectories(
        model, boids, max_trajectory_len, n_boids,
        use_saved_config=True,
        traj_cache_path=traj_cache_path,
        max_viz_runs=50
    )

    goals = boids.goal_positions

    if viz_mode == 'tubular':
        # Mean tubular
        tag = f"_{run_tag}" if run_tag else ""
        mean_fname = os.path.join(output_dir, f"boids_tube_mean{tag}.pdf")
        _plot_tubular_mean_3d(trajs, goals, filename=mean_fname)

        # Single tubular
        single_fname = os.path.join(output_dir, f"boids_tube_single{tag}.pdf")
        _plot_single_tubular_3d(trajs, goals, filename=single_fname)

        # All runs
        all_fname = os.path.join(output_dir, f"boids_tube_all_runs{tag}.pdf")
        _plot_multi_tubular_3d(trajs, goals, n_show=len(trajs),
                              filename=all_fname, specific_runs=list(range(len(trajs))))

    elif viz_mode == 'per_boid':
        tag = f"_{run_tag}" if run_tag else ""
        _plot_per_boid_3d(trajs, goals, n_boids, filename=os.path.join(output_dir, f"boids_per_boid{tag}.pdf"))

    elif viz_mode == 'multi_tubular':
        tag = f"_{run_tag}" if run_tag else ""
        _plot_multi_tubular_3d(trajs, goals, n_show=n_show,
                              filename=os.path.join(output_dir, f"boids_multi_tube{tag}.pdf"),
                              specific_runs=specific_runs)

    elif viz_mode == 'individual':
        _plot_individual_ranked_3d(trajs, goals, selected_centers, run_tag, output_dir)

    else:
        raise ValueError(f"Unknown viz_mode '{viz_mode}'. Choose 'tubular', 'per_boid', 'multi_tubular', or 'individual'.")


def run_individual_viz_3d(run_tag, cache_path=None):
    """Load a cached 3D run and return trajectory/tube statistics."""
    cache_path = cache_path or (
        f"saved_trajectory/viz_trajectories_{run_tag}.npz"
    )
    return load_statistics(
        cache_path,
        run_tag,
        dimensions=3,
        goals=BOIDS_GOAL_POSITIONS_3D,
    )


def generate_individual_pdfs_3d(stats, run_tag, output_dir=None):
    """Render one ranked 3D trajectory PDF per cached run."""
    output_dir = output_dir or f"results/analysis_3d/{run_tag}/individual"
    os.makedirs(output_dir, exist_ok=True)
    centers = [run[4] for run in sorted(stats["run_stats"])]
    _plot_individual_ranked_3d(
        stats["trajs"],
        stats["goals"],
        centers,
        run_tag,
        output_dir,
    )
    return output_dir


def generate_tubular_triplet_3d(stats, run_tag, output_dir=None):
    """Render mean, single-run, and multi-run 3D tubular summaries."""
    output_dir = output_dir or f"results/analysis_3d/{run_tag}"
    os.makedirs(output_dir, exist_ok=True)
    _save_tubular_triplet_3d(
        stats["trajs"],
        stats["goals"],
        run_tag,
        "cached",
        output_dir,
    )
    return output_dir


def generate_blowup_neighbor_pdfs_3d(
    stats, run_tag, threshold=2.0, k=5, output_dir=None
):
    """Plot each high-radius 3D path with its nearest starting states."""
    output_dir = output_dir or f"results/analysis_3d/{run_tag}/blowups"
    os.makedirs(output_dir, exist_ok=True)
    blowups = [run for run in stats["run_stats"] if run[1] >= threshold]
    for run in blowups:
        run_index, radius, centroid, _, _ = run
        neighbors = nearest_runs(run, stats["run_stats"], count=k)
        figure = plt.figure(figsize=(9, 9))
        axis = figure.add_subplot(111, projection="3d")
        axis.plot(
            centroid[:, 0],
            centroid[:, 1],
            centroid[:, 2],
            color="#8B0000",
            linewidth=2,
            label=f"blowup run {run_index} (r={radius:.3f})",
        )
        colors = plt.cm.tab10(np.linspace(0, 1, len(neighbors)))
        for color, neighbor in zip(colors, neighbors):
            distance, index, neighbor_radius, path, _, _ = neighbor
            axis.plot(
                path[:, 0],
                path[:, 1],
                path[:, 2],
                color=color,
                label=(
                    f"nearest d={distance:.3f}: run {index} "
                    f"(r={neighbor_radius:.3f})"
                ),
            )
        axis.scatter(
            stats["goals"][:, 0],
            stats["goals"][:, 1],
            stats["goals"][:, 2],
            c="red",
            marker="*",
            s=200,
            label="Goals",
        )
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.set_zlabel("Z")
        axis.legend(fontsize=8)
        axis.set_title(f"{run_tag}: blowup run {run_index}")
        figure.tight_layout()
        output_path = os.path.join(output_dir, f"run_{run_index:03d}.pdf")
        figure.savefig(output_path)
        plt.close(figure)
    print(f"Saved {len(blowups)} blowup-neighbor PDFs to {output_dir}")
    return output_dir


def write_blowup_table_3d(stats, run_tag, threshold=2.0, output_dir=None):
    """Write the cached 3D run's high-radius nearest-neighbor table."""
    output_dir = output_dir or f"results/analysis_3d/{run_tag}"
    return write_blowup_csv(
        stats,
        os.path.join(output_dir, f"blowups_{run_tag}.csv"),
        threshold=threshold,
    )


def analysis_main(argv=None):
    """Compatibility CLI for the former ``vis_analysis_3d.py`` script."""
    parser = argparse.ArgumentParser(description="Analyze cached 3D trajectories.")
    parser.add_argument("--run_tag", required=True)
    parser.add_argument("--cache_path", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--threshold", type=float, default=2.0)
    parser.add_argument("--gen_csv", action="store_true")
    parser.add_argument("--gen_blowup_pdfs", action="store_true")
    parser.add_argument("--gen_individual", action="store_true")
    parser.add_argument("--gen_triplet", action="store_true")
    arguments = parser.parse_args(argv)
    stats = run_individual_viz_3d(arguments.run_tag, arguments.cache_path)
    print_summary(stats, arguments.threshold)
    if arguments.gen_csv:
        write_blowup_table_3d(
            stats, arguments.run_tag, arguments.threshold, arguments.output_dir
        )
    if arguments.gen_blowup_pdfs:
        generate_blowup_neighbor_pdfs_3d(
            stats,
            arguments.run_tag,
            arguments.threshold,
            output_dir=arguments.output_dir,
        )
    if arguments.gen_individual:
        generate_individual_pdfs_3d(
            stats, arguments.run_tag, arguments.output_dir
        )
    if arguments.gen_triplet:
        generate_tubular_triplet_3d(
            stats, arguments.run_tag, arguments.output_dir
        )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--viz_mode", default="tubular",
                       choices=["tubular", "per_boid", "multi_tubular", "individual"],
                       help="Visualization style")
    parser.add_argument("--run_tag", default=None,
                       help="Training run tag, e.g. '50x20'. If set, infers model and boids paths automatically.")
    parser.add_argument("--model_path", default="gnca_model",
                       help="Path to the saved TF model directory")
    parser.add_argument("--boids_path", default="training_boids.pkl",
                       help="Path to the saved training_boids pickle")
    parser.add_argument("--traj_cache_path", default="viz_trajectories_3d.npz",
                       help="Path to the trajectory cache .npz file")
    parser.add_argument("--max_trajectory_len", type=int, default=1850)
    parser.add_argument("--n_boids", type=int, default=100)
    parser.add_argument("--n_show", type=int, default=50,
                       help="Number of tubes to show in multi_tubular mode")
    parser.add_argument("--runs", type=int, nargs="+", default=None,
                       help="Specific run indices to plot")
    args = parser.parse_args(argv)

    if args.run_tag:
        model_path = f"saved_models/gnca_model_{args.run_tag}"
        boids_path = f"saved_training_data/training_boids_{args.run_tag}.pkl"
        traj_cache_path = f"saved_trajectory/viz_trajectories_{args.run_tag}.npz"
        output_dir = "."
    else:
        model_path = args.model_path
        boids_path = args.boids_path
        traj_cache_path = args.traj_cache_path
        output_dir = "."

    print(f"Loading 3D model from '{model_path}'...")
    model = tf.saved_model.load(model_path)

    print(f"Loading 3D boids from '{boids_path}'...")
    training_boids = joblib.load(boids_path)

    print(f"Running 3D visualization with viz_mode='{args.viz_mode}'...")
    visualize_3d(
        model=model,
        boids=training_boids,
        max_trajectory_len=args.max_trajectory_len,
        n_boids=args.n_boids,
        viz_mode=args.viz_mode,
        traj_cache_path=traj_cache_path,
        run_tag=args.run_tag or "",
        n_show=args.n_show,
        specific_runs=args.runs,
        output_dir=output_dir,
    )

    print("3D visualization complete!")


def cli(argv=None):
    """Dispatch visualization or cached-trajectory analysis from one script."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if "--analysis" in arguments:
        arguments.remove("--analysis")
        analysis_main(arguments)
    else:
        main(arguments)


if __name__ == "__main__":
    cli()
