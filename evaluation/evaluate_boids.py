"""Evaluate a trained 2D GNCA against the Boids expert."""

import nolds
import numpy as np
import os
import tensorflow as tf
from spektral.data import DisjointLoader
from modules.boids import make_dataset
from modules.boids import Boids

from evaluation.visualization.visualize_boids import (
    _plot_individual_ranked,
    _plot_multi_tubular,
    _plot_per_boid,
    save_tubular_triplet,
)

@tf.function(experimental_relax_shapes=True)
def forward(model, x, a, i, training=None):
    """Computes one forward pass of the GNCA"""
    x_pred = model((x, a, i[:, None]), training=training)
    return x_pred

def avg_measure(trajectory, measure_fn, n_boids=None, coord=0, **kwargs):
    n_boids_total = trajectory.shape[-2]
    measures = []
    # Ensure we don't try to sample more boids than exist
    sample_size = min(n_boids if n_boids else 5, n_boids_total)
    for i in np.random.permutation(n_boids_total)[:sample_size]:
        measures.append(measure_fn(trajectory[:, i, coord], **kwargs))

    mn, std = np.mean(measures), np.std(measures)
    print(f"{measure_fn.__name__} {mn} +- {std}")
    return np.array(measures)

def convert_to_tf_sparse(a):
    """Safe conversion from scipy sparse to TF SparseTensor using standard API"""
    # 1. Get the indices where the connections are (row, col)
    # a.row and a.col come from the Scipy COO matrix
    indices = np.stack([a.row, a.col], axis=1)

    # 2. Create the SparseTensor
    a_tf = tf.SparseTensor(
        indices=indices,
        values=a.data.astype(np.float32),
        dense_shape=a.shape
    )

    # 3. Always reorder to ensure the sparse indices are in canonical order
    return tf.sparse.reorder(a_tf)

def _model_hash(model):
    """Compute a short hash of the model's trainable weights for cache keying."""
    import hashlib
    h = hashlib.md5()
    for w in model.trainable_variables:
        h.update(w.numpy().tobytes())
    return h.hexdigest()[:12]


def _make_gnca_runner(model, boids, n_boids, max_trajectory_len):
    """Returns a callable that runs the 2D GNCA autoregressively."""
    i = tf.zeros(n_boids, dtype=tf.int64)

    def to_tf_sparse(a):
        indices = np.stack([a.row, a.col], axis=1)
        a_tf = tf.SparseTensor(indices=indices, values=a.data.astype(np.float32), dense_shape=a.shape)
        return tf.sparse.reorder(a_tf)

    def run_gnca_traj(positions, velocities):
        traj = [np.concatenate([positions, velocities], axis=-1).astype(np.float32)]
        for _ in range(max_trajectory_len - 1):
            x_last = traj[-1]
            a = to_tf_sparse(boids.get_neighbors(x_last[:, :2]))
            x_next = model([tf.constant(x_last, dtype=tf.float32), a, i], training=False)
            traj.append(x_next.numpy())
        return np.array(traj)

    return run_gnca_traj


def _load_or_compute_trained_trajs(boids, n_boids, run_gnca_traj, traj_cache_path, model_tag):
    """Select up to 50 trained centers, load from cache or compute trajectories."""
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
            print(f"Loaded cached trajectories ({n_runs} runs) from '{traj_cache_path}'")
            trajs = list(cache["trajs"])
            # Only use cached centers if we don't already have them from boids.rand_configs
            if selected_centers is None and "centers" in cache:
                selected_centers = cache["centers"]

    if trajs is None:
        print(f"Computing {n_runs} trajectories from trained centers...")
        trajs = []
        for idx in range(n_runs):
            if selected_centers is not None:
                center = selected_centers[idx]
                pos = center + boids.init_scatter * np.random.rand(n_boids, 2)
                vel = np.tile(np.array([1.0, 0.0]) * boids.max_speed, (n_boids, 1))
            else:
                pos, vel, _, _ = boids.get_random_init(n_boids, save_config=False)
            trajs.append(run_gnca_traj(pos, vel))
        np.savez(traj_cache_path, trajs=np.array(trajs), model_tag=model_tag, n_runs=n_runs,
                 centers=selected_centers if selected_centers is not None else np.array([]))
        print(f"Cached trajectories to '{traj_cache_path}'")

    return trajs, selected_centers


def _compute_test_trajs(boids, n_boids, test_centers, run_gnca_traj, selected_centers):
    """Generate test trajectories from provided or random centers."""
    n_test = len(test_centers) if test_centers is not None else 50
    print(f"\nComputing {n_test} test-center trajectories...")
    trajs_test = []
    centers_collected = []

    if test_centers is not None:
        test_centers_arr = np.array(test_centers)
        n_test = min(50, len(test_centers_arr))
        print(f"   Using {n_test} provided test centers")
        for idx in range(n_test):
            center = test_centers_arr[idx]
            pos = center + boids.init_scatter * np.random.rand(n_boids, 2)
            vel = np.tile(np.array([1.0, 0.0]) * boids.max_speed, (n_boids, 1))
            centers_collected.append(center)
            trajs_test.append(run_gnca_traj(pos, vel))
            print(f"   [{idx+1}/{n_test}] center=({center[0]:.2f}, {center[1]:.2f})")
    else:
        for i in range(50):
            pos, vel, _, center = boids.get_random_init(n_boids, save_config=False)
            centers_collected.append(center)
            trajs_test.append(run_gnca_traj(pos, vel))
            print(f"   [{i+1}/50] center=({center[0]:.2f}, {center[1]:.2f})")

    print(f"Generated {len(trajs_test)} test trajectories")

    if selected_centers is not None and len(selected_centers) > 0:
        matches = sum(1 for rc in centers_collected if any(np.array_equal(rc, tc) for tc in selected_centers))
        print(f"Overlap check: {matches}/{len(trajs_test)} test centers matched training centers")

    return trajs_test


def evaluate(model, max_trajectory_len, n_boids, saved_boids,
             viz_mode='tubular',
             traj_cache_path="viz_trajectories.npz", run_tag="", n_show=50, specific_runs=None, output_dir=".",
             test_centers=None, viz_trained=False):
    """
    Evaluate GNCA trajectories and produce visualization PDFs.
    test_centers: if provided, use these for the test-set visualization instead of fresh random centers.
    """
    np.random.seed(1)
    boids = saved_boids if saved_boids is not None else Boids(n_boids=n_boids)
    goals = boids.goal_positions

    run_gnca_traj = _make_gnca_runner(model, boids, n_boids, max_trajectory_len)
    model_tag = _model_hash(model)
    if viz_trained:
        trajs, selected_centers = _load_or_compute_trained_trajs(
            boids, n_boids, run_gnca_traj, traj_cache_path, model_tag)
    else:
        trajs, selected_centers = [], (np.array(boids.rand_configs) if boids.rand_configs else None)
    np.random.seed(None)

    tag = f"_{run_tag}" if run_tag else ""

    # Compute test trajectories when test_centers are provided or as fallback
    trajs_test = _compute_test_trajs(boids, n_boids, test_centers, run_gnca_traj, selected_centers) if (test_centers is not None or not viz_trained) else []

    # Save test trajectories to disk so they can be reloaded for re-visualization
    if trajs_test:
        test_cache_path = traj_cache_path.replace('.npz', '_test.npz')
        centers_arr = np.array(test_centers) if test_centers is not None else np.array([])
        np.savez(test_cache_path, trajs=np.array(trajs_test, dtype=object),
                 centers=centers_arr, model_tag=model_tag)
        print(f"Cached test trajectories to '{test_cache_path}'")

    # Primary trajectories for visualization: test if viz_trained=False, else trained
    trajs_viz    = trajs_test if not viz_trained else trajs
    centers_viz  = list(test_centers) if (not viz_trained and test_centers is not None) else (list(selected_centers) if selected_centers is not None else [])

    if viz_mode == 'tubular':
        if viz_trained and trajs:
            save_tubular_triplet(trajs, goals, tag, 'trained', output_dir)
        save_tubular_triplet(trajs_test, goals, tag, 'test', output_dir)

    elif viz_mode == 'per_boid':
        _plot_per_boid(trajs_viz, goals, n_boids, filename=os.path.join(output_dir, f"boids_auto_rand{tag}.pdf"))

    elif viz_mode == 'multi_tubular':
        _plot_multi_tubular(trajs_viz, goals, n_show=n_show,
                           filename=os.path.join(output_dir, f"boids_multi_tubular{tag}.pdf"),
                           specific_runs=specific_runs if specific_runs else list(range(min(n_show, len(trajs_viz)))))

    elif viz_mode == 'individual':
        _plot_individual_ranked(trajs_viz, goals, n_boids, centers_viz, run_tag, output_dir)

    else:
        raise ValueError(f"Unknown viz_mode '{viz_mode}'. Choose 'tubular', 'per_boid', 'multi_tubular', or 'individual'.")

def evaluate_complexity(model, forward, te_set_size, trajectory_len, n_boids):
    """
    Runs multiple randomized test trajectories to calculate average SampEn and CorrDim.
    """
    np.random.seed(1)
    all_measures = []

    for _ in range(te_set_size):
        # We rely on get_random_init() for the 'clump', so we keep init=None
        data_te, boids_te = make_dataset(
            unique_reps=1,
            repeat_reps=1,
            save_config=False,
            random_init=True,
            return_boids=True,
            n_boids=n_boids,
        )
        loader_te = DisjointLoader(data_te, node_level=True, epochs=1, shuffle=False)

        boid_trajectory_true = []
        boid_trajectory_auto = []

        for sample in loader_te:
            inputs, x_next = sample

            if len(boid_trajectory_auto) == 0:
                # Synchronize start: use the first prediction as the seed
                x_start_pred = forward(model, *inputs, training=False)
                boid_trajectory_auto.append(x_start_pred)
            else:
                x_last = boid_trajectory_auto[-1]
                # Re-calculate neighbors for the GNN's current position
                a_scipy = boids_te.get_neighbors(x_last[:, :2])
                a = convert_to_tf_sparse(a_scipy)

                # Forward pass using GNN's own previous output
                inputs_auto = [x_last, a, inputs[-1]]
                x_next_auto = forward(model, *inputs_auto, training=False)
                boid_trajectory_auto.append(x_next_auto)

            # Targets also carry loss-only goal-transition metadata; compare
            # only the expert's next physical state against model rollouts.
            boid_trajectory_true.append(x_next[:, 4:8])
            if len(boid_trajectory_true) >= trajectory_len:
                break

        # Convert to numpy for complexity analysis
        traj_true = np.array(boid_trajectory_true)
        traj_auto = np.array(boid_trajectory_auto)

        # Calculate metrics for this specific trajectory
        m_true_samp = avg_measure(traj_true, nolds.sampen)
        m_auto_samp = avg_measure(traj_auto, nolds.sampen)
        m_true_corr = avg_measure(traj_true, nolds.corr_dim, emb_dim=10)
        m_auto_corr = avg_measure(traj_auto, nolds.corr_dim, emb_dim=10)

        all_measures.append((m_true_samp, m_auto_samp, m_true_corr, m_auto_corr))

    # Convert to array and calculate final averages/stds
    measures = np.array(all_measures)
    measures_mean = np.mean(measures, (0, -1))
    measures_std = np.std(measures, (0, -1))

    print(f"\nFINAL COMPLEXITY STATS OVER {te_set_size} RUNS:")
    print(f"SampEn True: {measures_mean[0]:.6f} +- {measures_std[0]:.6f}")
    print(f"SampEn GNCA: {measures_mean[1]:.6f} +- {measures_std[1]:.6f}")
    print(f"CorrDim True: {measures_mean[2]:.6f} +- {measures_std[2]:.6f}")
    print(f"CorrDim GNCA: {measures_mean[3]:.6f} +- {measures_std[3]:.6f}")
    return measures_mean, measures_std
