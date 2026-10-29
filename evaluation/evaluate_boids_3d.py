"""Evaluate a trained 3D GNCA against the Boids expert."""
import numpy as np
import tensorflow as tf
import os
from modules.boids_3d import Boids3D
from evaluation.visualization.visualize_boids_3d import (
    _plot_individual_ranked_3d,
    _plot_multi_tubular_3d,
    _plot_per_boid_3d,
    _save_tubular_triplet_3d,
)


@tf.function(experimental_relax_shapes=True)
def forward(model, x, a, i, training=None):
    """Computes one forward pass of the 3D GNCA"""
    x_pred = model((x, a, i[:, None]), training=training)
    return x_pred


def to_tf_sparse(a):
    indices = np.stack([a.row, a.col], axis=1)
    a_tf = tf.SparseTensor(
        indices=indices,
        values=a.data.astype(np.float32),
        dense_shape=a.shape
    )
    return tf.sparse.reorder(a_tf)


def _model_hash_3d(model):
    """Compute a short hash of the model's trainable weights for cache keying."""
    import hashlib
    h = hashlib.md5()
    for w in model.trainable_variables:
        h.update(w.numpy().tobytes())
    return h.hexdigest()[:12]


def _make_gnca_runner_3d(model, boids, n_boids, max_trajectory_len):
    """Returns a callable that runs the 3D GNCA autoregressively."""
    i = tf.zeros(n_boids, dtype=tf.int64)

    def run_gnca_traj(positions, velocities):
        traj = [np.concatenate([positions, velocities], axis=-1).astype(np.float32)]
        for _ in range(max_trajectory_len - 1):
            x_last = traj[-1]
            a = to_tf_sparse(boids.get_neighbors(x_last[:, :3]))
            x_next = model([tf.constant(x_last, dtype=tf.float32), a, i], training=False)
            traj.append(x_next.numpy())
        return np.array(traj)

    return run_gnca_traj


def _load_or_compute_trained_trajs_3d(boids, n_boids, run_gnca_traj, traj_cache_path, model_tag):
    """Select up to 50 trained centers, load from cache or compute 3D trajectories."""
    if len(boids.rand_configs) > 0:
        all_centers = np.array(boids.rand_configs)
        n_runs = min(50, len(all_centers))
        rand_indices = np.random.choice(len(all_centers), size=n_runs, replace=False)
        selected_centers = all_centers[rand_indices]
    else:
        selected_centers = None
        n_runs = 50

    trajs = None
    if os.path.exists(traj_cache_path):
        cache = np.load(traj_cache_path, allow_pickle=True)
        if cache.get("model_tag", "") == model_tag and cache.get("n_runs", 0) == n_runs:
            print(f"Loaded cached 3D trajectories ({n_runs} runs) from '{traj_cache_path}'")
            trajs = list(cache["trajs"])
            if "centers" in cache:
                selected_centers = cache["centers"]

    if trajs is None:
        print(f"Computing {n_runs} 3D trajectories from trained centers...")
        trajs = []
        for idx in range(n_runs):
            if selected_centers is not None:
                center = selected_centers[idx]
                pos = center + boids.init_scatter * np.random.rand(n_boids, 3)
            else:
                pos, _, _, center = boids.get_random_init(n_boids, save_config=False)
            vel = np.tile(np.array([1.0, 0.0, 0.0]) * boids.max_speed, (n_boids, 1))
            trajs.append(run_gnca_traj(pos, vel))
        np.savez(traj_cache_path, trajs=np.array(trajs), model_tag=model_tag, n_runs=n_runs,
                 centers=selected_centers if selected_centers is not None else np.array([]))
        print(f"Cached 3D trajectories to '{traj_cache_path}'")

    return trajs, selected_centers


def _compute_test_trajs_3d(boids, n_boids, test_centers, run_gnca_traj, selected_centers):
    """Generate 50 test trajectories and check overlap with training centers."""
    print(f"\nGenerating 50 3D test-center trajectories...")
    trajs_test = []
    centers_collected = []

    if test_centers is not None:
        test_centers_arr = np.array(test_centers)
        n_test = min(50, len(test_centers_arr))
        print(f"   Using {n_test} provided test centers")
        for idx in range(n_test):
            center = test_centers_arr[idx]
            pos = center + boids.init_scatter * np.random.rand(n_boids, 3)
            vel = np.tile(np.array([1.0, 0.0, 0.0]) * boids.max_speed, (n_boids, 1))
            centers_collected.append(center)
            trajs_test.append(run_gnca_traj(pos, vel))
    else:
        for _ in range(50):
            pos, vel, _, center = boids.get_random_init(n_boids, save_config=False)
            centers_collected.append(center)
            trajs_test.append(run_gnca_traj(pos, vel))

    print(f"Generated {len(trajs_test)} test trajectories")

    if selected_centers is not None and len(selected_centers) > 0:
        matches = sum(1 for rc in centers_collected if any(np.array_equal(rc, tc) for tc in selected_centers))
        print(f"Overlap check: {matches}/{len(trajs_test)} test centers matched training centers")

    return trajs_test


def evaluate_3d(model, max_trajectory_len, n_boids, saved_boids,
                run_tag="", viz_mode='tubular', test_centers=None,
                traj_cache_path="viz_trajectories_3d.npz",
                n_show=50, specific_runs=None, output_dir=".",
                skip_trained_viz=False):
    """
    Evaluate 3D GNCA trajectories and produce visualization PDFs.
    test_centers: if provided, use these for the test-set visualization instead of fresh random centers.
    Matches 2D evaluate() signature for consistent CLI handling.
    """
    np.random.seed(1)
    boids = saved_boids if saved_boids is not None else Boids3D(n_boids=n_boids)
    goals = boids.goal_positions

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(traj_cache_path) or ".", exist_ok=True)

    run_gnca_traj = _make_gnca_runner_3d(model, boids, n_boids, max_trajectory_len)
    model_tag = _model_hash_3d(model)

    if not skip_trained_viz:
        trajs, selected_centers = _load_or_compute_trained_trajs_3d(
            boids, n_boids, run_gnca_traj, traj_cache_path, model_tag)
    else:
        trajs, selected_centers = [], None

    np.random.seed(None)

    tag = f"_{run_tag}" if run_tag else ""

    trajs_test = _compute_test_trajs_3d(boids, n_boids, test_centers, run_gnca_traj, selected_centers)

    if viz_mode == 'tubular':
        if trajs:
            _save_tubular_triplet_3d(trajs, goals, tag, 'trained', output_dir)
        _save_tubular_triplet_3d(trajs_test, goals, tag, 'test', output_dir)

    elif viz_mode == 'per_boid':
        _plot_per_boid_3d(trajs_test, goals, n_boids,
                         filename=os.path.join(output_dir, f"boids_per_boid{tag}.pdf"))

    elif viz_mode == 'multi_tubular':
        _plot_multi_tubular_3d(trajs_test, goals, n_show=n_show,
                              filename=os.path.join(output_dir, f"boids_multi_tube{tag}.pdf"),
                              specific_runs=specific_runs if specific_runs else list(range(min(n_show, len(trajs_test)))))

    elif viz_mode == 'individual':
        _plot_individual_ranked_3d(trajs_test, goals, test_centers, run_tag, output_dir)

    else:
        raise ValueError(f"Unknown viz_mode '{viz_mode}'. Choose 'tubular', 'per_boid', 'multi_tubular', or 'individual'.")
