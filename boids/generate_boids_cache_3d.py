"""
Generates 3D boids trajectories, converts them to training graph tuples,
and saves to an HDF5 file for memory-efficient loading during training.
Writes to disk incrementally per trajectory - no large RAM accumulation.

The HDF5 contains:
  - 'x':            (N, n_boids, 6)  - [pos_x, pos_y, pos_z, vel_x, vel_y, vel_z]
  - 'a_row':        (N, max_edges)   - COO row indices (padded with -1)
  - 'a_col':        (N, max_edges)   - COO col indices (padded with -1)
  - 'a_len':        (N,)             - actual edges per timestep
  - 'y':            (N, n_boids, 16) - [cur_state(6), next_state(6), goal(3), active_mask(1)]
  - 'centers':      (unique_reps, 3) - flock start centers
  - 'traj_lengths': (unique_reps * repeats,) - samples per trajectory

Usage:
    python -m boids.generate_boids_cache_3d --unique 50 --repeats 5 --sample_mode octant --goal_exclusion_size 0.5
"""
import argparse
import os
import numpy as np
import h5py
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

from modules.boids_3d import (
    BOIDS_3D_ACTIVE_MASK_INDEX,
    BOIDS_3D_MASKED_TARGET_FEATURES,
    Boids3D,
    history_to_samples_3d,
)
from modules.perturbations import (
    add_perturbation_arguments,
    config_for_mode,
    perturbation_counts,
    validate_perturbation_args,
)

# 8 octants: (x_min, x_max, y_min, y_max, z_min, z_max, label)
OCTANTS = [
    ( 0.0,  5.0,  0.0,  5.0,  0.0,  5.0, "O0"),
    (-5.0,  0.0,  0.0,  5.0,  0.0,  5.0, "O1"),
    (-5.0,  0.0, -5.0,  0.0,  0.0,  5.0, "O2"),
    ( 0.0,  5.0, -5.0,  0.0,  0.0,  5.0, "O3"),
    ( 0.0,  5.0,  0.0,  5.0, -5.0,  0.0, "O4"),
    (-5.0,  0.0,  0.0,  5.0, -5.0,  0.0, "O5"),
    (-5.0,  0.0, -5.0,  0.0, -5.0,  0.0, "O6"),
    ( 0.0,  5.0, -5.0,  0.0, -5.0,  0.0, "O7"),
]


def build_exclusion_zone_3d(goal_positions, radius):
    """Capsule exclusion: exclude any center within `radius` of the line segment
    connecting all goals. For 2 goals this is a cylinder with hemispherical caps."""
    if radius <= 0:
        return None
    return (goal_positions, radius)


def _in_exclusion_zone_3d(center, exclusion_zone):
    """Return True if center is within the capsule exclusion zone."""
    if exclusion_zone is None:
        return False
    goals, radius = exclusion_zone
    # Check distance from center to each pair of adjacent goals (line segment)
    for i in range(len(goals)):
        a = goals[i]
        b = goals[(i + 1) % len(goals)]
        ab = b - a
        ab_len_sq = np.dot(ab, ab)
        if ab_len_sq < 1e-10:
            if np.linalg.norm(center - a) < radius:
                return True
            continue
        t = np.clip(np.dot(center - a, ab) / ab_len_sq, 0.0, 1.0)
        closest = a + t * ab
        if np.linalg.norm(center - closest) < radius:
            return True
    return False


def main():
    parser = argparse.ArgumentParser(description="Generate and cache 3D boids graph training data")
    parser.add_argument("--unique",   type=int,   default=50)
    parser.add_argument("--repeats",  type=int,   default=5)
    parser.add_argument("--n_boids",  type=int,   default=100)
    parser.add_argument("--perception", type=float, default=0.1)
    parser.add_argument("--crowding", type=float, default=0.02)
    parser.add_argument("--init_scatter", type=float, default=0.325)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output",   type=str,   default=None)
    parser.add_argument(
        "--run_tag_suffix",
        type=str,
        default="",
        help="Optional descriptive suffix appended to the generated cache filename.",
    )
    parser.add_argument("--skip_centers_plot", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--sample_mode", default="octant",
                        choices=["octant", "full_canvas"],
                        help="'octant' = --unique per octant; 'full_canvas' = [-5,5]^3 with exclusion; "
                             "Centers are sampled outside the goal exclusion region.")
    parser.add_argument("--goal_exclusion_size", type=float, default=0.5,
                        help="Sphere radius around each goal to exclude from center sampling.")
    parser.add_argument("--octants", type=int, nargs="+", default=None,
                        help="Which octants to sample from (0-7). Default: all 8.")
    parser.add_argument("--goal_arrival_radius", type=float, default=0.5)
    add_perturbation_arguments(parser, 3)
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    boids = Boids3D(
        n_boids=args.n_boids,
        perception=args.perception,
        crowding=args.crowding,
        goal_arrival_radius=args.goal_arrival_radius,
    )
    boids.init_scatter = args.init_scatter
    exclusion_zone = build_exclusion_zone_3d(boids.goal_positions, args.goal_exclusion_size)

    if args.sample_mode == "octant":
        selected = args.octants if args.octants is not None else list(range(8))
        total_unique = len(selected) * args.unique
        center_tasks = []
        for o_idx in selected:
            xmn, xmx, ymn, ymx, zmn, zmx, _ = OCTANTS[o_idx]
            center_tasks.extend([(xmn, xmx, ymn, ymx, zmn, zmx)] * args.unique)
        oct_str = "".join(str(o) for o in selected)
        mode_suffix = f"_o{oct_str}_ex{args.goal_exclusion_size:g}"
    elif args.sample_mode == "full_canvas":
        total_unique = args.unique
        center_tasks = [(-5.0, 5.0, -5.0, 5.0, -5.0, 5.0)] * total_unique
        mode_suffix = f"_fc_ex{args.goal_exclusion_size:g}"
    args.train_set_unique = total_unique * args.repeats
    validate_perturbation_args(args, 3, trajectory_total=args.train_set_unique)
    requested_modes = perturbation_counts(
        args, trajectory_total=args.train_set_unique
    )
    modes = [mode for mode, count in requested_modes.items() for _ in range(count)]
    mode_rng = np.random.default_rng(args.seed)
    mode_rng.shuffle(modes)
    trajectory_index = 0

    run_tag_suffix = f"_{args.run_tag_suffix}" if args.run_tag_suffix else ""
    if args.output is None:
        args.output = f"boids_cache_3d_{total_unique}x{args.repeats}{mode_suffix}{run_tag_suffix}.h5"

    print(f"\n>>> Sample mode: {args.sample_mode} | {total_unique} unique x {args.repeats} repeats = {total_unique * args.repeats} trajectories")
    if exclusion_zone:
        print(f">>> Exclusion: capsule of radius {args.goal_exclusion_size} along goal-to-goal segment")
    print(f">>> Output: {args.output}")
    print(f">>> Perturbation mixture: {requested_modes}\n")

    max_edges = args.n_boids * (args.n_boids - 1)
    n_feat_x, n_feat_y = 6, BOIDS_3D_MASKED_TARGET_FEATURES
    all_centers, all_traj_lengths = [], []

    with h5py.File(args.output, 'w') as f:
        ds_x    = f.create_dataset('x',     shape=(0, args.n_boids, n_feat_x), maxshape=(None, args.n_boids, n_feat_x), dtype='float32', chunks=(256, args.n_boids, n_feat_x), compression='gzip', compression_opts=4)
        ds_y    = f.create_dataset('y',     shape=(0, args.n_boids, n_feat_y), maxshape=(None, args.n_boids, n_feat_y), dtype='float32', chunks=(256, args.n_boids, n_feat_y), compression='gzip', compression_opts=4)
        ds_arow = f.create_dataset('a_row', shape=(0, max_edges), maxshape=(None, max_edges), dtype='int32', chunks=(256, max_edges), fillvalue=-1, compression='gzip', compression_opts=4)
        ds_acol = f.create_dataset('a_col', shape=(0, max_edges), maxshape=(None, max_edges), dtype='int32', chunks=(256, max_edges), fillvalue=-1, compression='gzip', compression_opts=4)
        ds_alen = f.create_dataset('a_len', shape=(0,), maxshape=(None,), dtype='int32', chunks=(1024,), compression='gzip', compression_opts=4)

        for bounds in tqdm(center_tasks, desc="Unique centers", disable=args.quiet):
            _, _, _, center = boids.get_random_init(
                args.n_boids, save_config=False,
                bounds=bounds, exclusion_zone=exclusion_zone,
            )
            all_centers.append(center)

            for _ in range(args.repeats):
                # Match the older cache-generation behavior: pass the center
                # directly and let Boids3D.generate_trajectory create the
                # repeated clump initialization internally.
                history = boids.generate_trajectory(
                    random_init=center,
                    save_config=False,
                    perturbation=config_for_mode(args, modes[trajectory_index]),
                    perturbation_rng=(args.seed or 0) + trajectory_index + 1,
                )
                trajectory_index += 1
                samples = history_to_samples_3d(history)
                n = len(samples)
                all_traj_lengths.append(n)

                x_buf    = np.zeros((n, args.n_boids, n_feat_x), dtype=np.float32)
                y_buf    = np.zeros((n, args.n_boids, n_feat_y), dtype=np.float32)
                arow_buf = np.full((n, max_edges), -1, dtype=np.int32)
                acol_buf = np.full((n, max_edges), -1, dtype=np.int32)
                alen_buf = np.zeros(n, dtype=np.int32)

                for k, (x, _, y) in enumerate(samples):
                    x_buf[k] = x
                    y_buf[k] = y
                    # Canonicalize the graph from the exact float32 state that
                    # is written to the cache.  The simulator evolves float64
                    # positions, so reusing its graph can disagree at the
                    # perception boundary after x is quantized to float32.
                    a = boids.get_neighbors(
                        x_buf[k, :, :3],
                        active_mask=y_buf[k, :, BOIDS_3D_ACTIVE_MASK_INDEX],
                    )
                    nnz = a.nnz
                    arow_buf[k, :nnz] = a.row
                    acol_buf[k, :nnz] = a.col
                    alen_buf[k] = nnz

                cur = ds_x.shape[0]
                for ds, buf in [(ds_x, x_buf), (ds_y, y_buf), (ds_arow, arow_buf), (ds_acol, acol_buf), (ds_alen, alen_buf)]:
                    ds.resize(cur + n, axis=0)
                    ds[cur:cur + n] = buf

                del history, samples, x_buf, y_buf, arow_buf, acol_buf, alen_buf

        centers_arr = np.stack(all_centers, axis=0)
        f.create_dataset('centers',      data=centers_arr)
        f.create_dataset('traj_lengths', data=np.array(all_traj_lengths, dtype=np.int32))
        f.attrs['unique_reps']  = total_unique
        f.attrs['repeats']      = args.repeats
        f.attrs['goal_arrival_radius'] = args.goal_arrival_radius
        f.attrs['n_boids']      = args.n_boids
        f.attrs['max_edges']    = max_edges
        f.attrs['perception']   = boids.perception
        f.attrs['adjacency_alignment'] = 'current_state'
        f.attrs['sample_mode']  = args.sample_mode
        f.attrs['target_features'] = n_feat_y
        for mode, count in requested_modes.items():
            f.attrs[f'{mode}_trajectories'] = count
        if args.seed is not None:
            f.attrs['generation_seed'] = args.seed
        if exclusion_zone:
            f.attrs['goal_exclusion_size'] = args.goal_exclusion_size
        total = ds_x.shape[0]

    print(f"\nSaved {total} samples to '{args.output}'")
    print(f"   centers: {centers_arr.shape}  |  traj lengths: min={min(all_traj_lengths)} max={max(all_traj_lengths)} mean={int(np.mean(all_traj_lengths))}")
    print(f"   File size: ~{os.path.getsize(args.output) / 1e6:.1f} MB")

    if args.skip_centers_plot:
        return

    # Centers scatter plot (XY projection)
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(centers_arr[:, 0], centers_arr[:, 1], centers_arr[:, 2], c='blue', marker='o', s=20, alpha=0.5, label='Centers')
    ax.scatter(boids.goal_positions[:, 0], boids.goal_positions[:, 1], boids.goal_positions[:, 2], c='red', marker='*', s=200, zorder=5, label='Goals')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.set_title(f"{total_unique} centers | {args.sample_mode}")
    ax.legend()
    plot_path = args.output.replace('.h5', '_centers.pdf')
    plt.savefig(plot_path); plt.close()
    print(f"   Centers plot: {plot_path}")


if __name__ == "__main__":
    main()
