"""
Generates boids trajectories, converts them to training graph tuples,
and saves to an HDF5 file for memory-efficient loading during training.
Writes to disk incrementally per trajectory - no large RAM accumulation.
"""

import argparse
import os
import numpy as np
import h5py
from shapely.geometry import Polygon
from tqdm import tqdm
import matplotlib.pyplot as plt

from modules.boids import (
    BOIDS_MASKED_TARGET_FEATURES,
    BOIDS_STATE_FEATURES,
    Boids,
    history_to_samples,
)
from modules.perturbations import (
    add_perturbation_arguments,
    config_for_mode,
    perturbation_counts,
    validate_perturbation_args,
)

# (xmin, xmax, ymin, ymax, label) for each quadrant
QUADRANTS = [
    ( 0.0,  5.0,  0.0,  5.0, "Q1"),
    (-5.0,  0.0,  0.0,  5.0, "Q2"),
    (-5.0,  0.0, -5.0,  0.0, "Q3"),
    ( 0.0,  5.0, -5.0,  0.0, "Q4"),
]


def build_exclusion_zone(goal_positions, size):
    """Triangle through goal vertices, buffered outward by `size` units."""
    if size <= 0:
        return None
    return Polygon(goal_positions.tolist()).buffer(size)



def main():
    parser = argparse.ArgumentParser(description="Generate and cache boids graph training data")
    parser.add_argument("--unique",   type=int,   default=50)
    parser.add_argument("--repeats",  type=int,   default=20)
    parser.add_argument("--n_boids",  type=int,   default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output",   type=str,   default=None)
    parser.add_argument(
        "--run_tag_suffix",
        type=str,
        default="",
        help="Optional descriptive suffix appended to the generated cache filename.",
    )
    parser.add_argument("--sample_mode", default="q3",
                        choices=["q3", "full_canvas", "quadrant"],
                        help="'q3' = [-5,-2.5]^2; 'full_canvas' = [-5,5]^2 with exclusion; "
                             "'quadrant' = --unique per quadrant (4xunique total) with exclusion.")
    parser.add_argument("--goal_exclusion_size", type=float, default=0.2,
                        help="Buffer radius around the goal triangle to exclude from sampling. "
                             "Used with full_canvas and quadrant modes. Default: 0.2")
    parser.add_argument("--quadrants", type=int, nargs="+", default=None,
                        help="Which quadrants to sample from (0=Q1, 1=Q2, 2=Q3, 3=Q4). "
                             "Default: all 4. E.g. '--quadrants 2 3' for Q3 and Q4 only.")
    parser.add_argument("--goal_arrival_radius", type=float, default=0.5)
    parser.add_argument("--perception", type=float, default=0.1)
    parser.add_argument("--crowding", type=float, default=0.02)
    parser.add_argument("--init_scatter", type=float, default=0.325)
    add_perturbation_arguments(parser, 2)
    args = parser.parse_args()
    np.random.seed(args.seed)
    boids = Boids(
        n_boids=args.n_boids,
        goal_arrival_radius=args.goal_arrival_radius,
        perception=args.perception,
        crowding=args.crowding,
        init_scatter=args.init_scatter,
    )

    # Determine total unique centers and per-center bounds list
    if args.sample_mode == "quadrant":
        selected_quads = args.quadrants if args.quadrants is not None else [0, 1, 2, 3]
        total_unique = len(selected_quads) * args.unique
        center_tasks = []
        for q_idx in selected_quads:
            xmn, xmx, ymn, ymx, _ = QUADRANTS[q_idx]
            center_tasks.extend([(xmn, xmx, ymn, ymx)] * args.unique)
    elif args.sample_mode == "full_canvas":
        total_unique = args.unique
        center_tasks = [(-5.0, 5.0, -5.0, 5.0)] * total_unique
    else:  # q3
        total_unique = args.unique
        center_tasks = [(-5.0, -2.5, -5.0, -2.5)] * total_unique

    args.train_set_unique = total_unique * args.repeats
    validate_perturbation_args(args, 2, trajectory_total=args.train_set_unique)
    requested_modes = perturbation_counts(
        args, trajectory_total=args.train_set_unique
    )
    modes = [mode for mode, count in requested_modes.items() for _ in range(count)]
    mode_rng = np.random.default_rng(args.seed)
    mode_rng.shuffle(modes)
    trajectory_index = 0

    exclusion_zone = None
    if args.sample_mode in ("full_canvas", "quadrant"):
        exclusion_zone = build_exclusion_zone(boids.goal_positions, args.goal_exclusion_size)

    if args.output is None:
        run_tag_suffix = f"_{args.run_tag_suffix}" if args.run_tag_suffix else ""
        if args.sample_mode == "quadrant":
            quad_str = "".join(str(q) for q in (args.quadrants if args.quadrants is not None else [0, 1, 2, 3]))
            mode_suffix = f"_q{quad_str}_ex{args.goal_exclusion_size:g}"
        elif args.sample_mode == "full_canvas":
            mode_suffix = f"_fc_ex{args.goal_exclusion_size:g}"
        else:
            mode_suffix = ""
        args.output = f"boids_cache_{total_unique}x{args.repeats}{mode_suffix}{run_tag_suffix}.h5"

    print(f"\n>>> Sample mode: {args.sample_mode} | "
          f"{total_unique} unique x {args.repeats} repeats = {total_unique * args.repeats} trajectories")
    if exclusion_zone is not None:
        print(f">>> Exclusion: goal triangle buffered by {args.goal_exclusion_size} units")
    print(f">>> Output: {args.output}")
    print(f">>> Perturbation mixture: {requested_modes}\n")

    max_edges = args.n_boids * (args.n_boids - 1)
    n_feat_x = BOIDS_STATE_FEATURES
    n_feat_y = BOIDS_MASKED_TARGET_FEATURES
    all_centers, all_traj_lengths = [], []

    with h5py.File(args.output, 'w') as f:
        ds_x    = f.create_dataset('x',     shape=(0, args.n_boids, n_feat_x),
                                   maxshape=(None, args.n_boids, n_feat_x), dtype='float32',
                                   chunks=(256, args.n_boids, n_feat_x), compression='gzip', compression_opts=4)
        ds_y    = f.create_dataset('y',     shape=(0, args.n_boids, n_feat_y),
                                   maxshape=(None, args.n_boids, n_feat_y), dtype='float32',
                                   chunks=(256, args.n_boids, n_feat_y), compression='gzip', compression_opts=4)
        ds_arow = f.create_dataset('a_row', shape=(0, max_edges), maxshape=(None, max_edges),
                                   dtype='int32', chunks=(256, max_edges), fillvalue=-1,
                                   compression='gzip', compression_opts=4)
        ds_acol = f.create_dataset('a_col', shape=(0, max_edges), maxshape=(None, max_edges),
                                   dtype='int32', chunks=(256, max_edges), fillvalue=-1,
                                   compression='gzip', compression_opts=4)
        ds_alen = f.create_dataset('a_len', shape=(0,), maxshape=(None,), dtype='int32',
                                   chunks=(1024,), compression='gzip', compression_opts=4)

        for xmn, xmx, ymn, ymx in tqdm(center_tasks, desc="Unique centers"):
            # Sample one center; get_random_init handles bounds + exclusion + initialisation
            _, _, _, center = boids.get_random_init(
                args.n_boids, save_config=False,
                bounds=(xmn, xmx, ymn, ymx),
                exclusion_zone=exclusion_zone,
            )
            all_centers.append(center)

            for _ in range(args.repeats):
                # Reuse the same center with fresh boid scatter for each repeat
                positions, velocities, _, _ = boids.get_random_init(
                    args.n_boids, save_config=False, center=center
                )
                history = boids.generate_trajectory(
                    random_init=(positions, velocities),
                    save_config=False,
                    perturbation=config_for_mode(args, modes[trajectory_index]),
                    perturbation_rng=args.seed + trajectory_index + 1,
                )
                trajectory_index += 1
                samples = history_to_samples(history)
                n = len(samples)
                all_traj_lengths.append(n)

                x_buf    = np.zeros((n, args.n_boids, n_feat_x), dtype=np.float32)
                y_buf    = np.zeros((n, args.n_boids, n_feat_y), dtype=np.float32)
                arow_buf = np.full((n, max_edges), -1, dtype=np.int32)
                acol_buf = np.full((n, max_edges), -1, dtype=np.int32)
                alen_buf = np.zeros(n, dtype=np.int32)

                for k, (x, a, y) in enumerate(samples):
                    x_buf[k] = x
                    y_buf[k] = y
                    nnz = a.nnz
                    arow_buf[k, :nnz] = a.row
                    acol_buf[k, :nnz] = a.col
                    alen_buf[k] = nnz

                cur = ds_x.shape[0]
                for ds, buf in [(ds_x, x_buf), (ds_y, y_buf),
                                (ds_arow, arow_buf), (ds_acol, acol_buf), (ds_alen, alen_buf)]:
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
        f.attrs['sample_mode']  = args.sample_mode
        f.attrs['target_features'] = n_feat_y
        for mode, count in requested_modes.items():
            f.attrs[f'{mode}_trajectories'] = count
        f.attrs['loss_transition_metadata'] = (
            'active_goal,previous_goal,max_previous_goal_centroid_distance'
        )
        if exclusion_zone is not None:
            f.attrs['goal_exclusion_size'] = args.goal_exclusion_size
        total = ds_x.shape[0]

    print(f"\nSaved {total} samples to '{args.output}'")
    print(f"   centers: {centers_arr.shape}  |  traj lengths: "
          f"min={min(all_traj_lengths)} max={max(all_traj_lengths)} mean={int(np.mean(all_traj_lengths))}")
    print(f"   File size: ~{os.path.getsize(args.output) / 1e6:.1f} MB")

    _, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(centers_arr[:, 0], centers_arr[:, 1],
               c='blue', marker='*', s=80, alpha=0.7, label=f'{total_unique} centers')
    ax.scatter(boids.goal_positions[:, 0], boids.goal_positions[:, 1],
               c='red', marker='*', s=200, zorder=5, label='Goals')
    if exclusion_zone is not None:
        ex_x, ex_y = exclusion_zone.exterior.xy
        ax.fill(ex_x, ex_y, alpha=0.15, color='red',
                label=f'Exclusion (r={args.goal_exclusion_size})')
        ax.plot(ex_x, ex_y, color='red', lw=1.0)
    ax.set_xlim(-5.5, 5.5); ax.set_ylim(-5.5, 5.5)
    ax.axhline(0, color='gray', lw=0.5, ls='--')
    ax.axvline(0, color='gray', lw=0.5, ls='--')
    ax.set_xlabel("X"); ax.set_ylabel("Y")
    ax.set_title(f"{args.sample_mode} - {total_unique} unique centers")
    ax.set_aspect('equal'); ax.legend(fontsize=8)
    plt.tight_layout()
    plot_path = args.output.replace('.h5', '_centers.pdf')
    plt.savefig(plot_path); plt.close()
    print(f"   Center plot: {plot_path}")


if __name__ == "__main__":
    main()
