"""Command-line interface for 2D GNCA training."""

import argparse
from modules.perturbations import add_perturbation_arguments


def build_parser():
    parser = argparse.ArgumentParser(description="Train fixed- or dynamic-waypoint 2D GNCA.")
    parser.add_argument(
        "--task",
        choices=["fixed_waypoints", "online_goals"],
        default="fixed_waypoints",
        help="Experiment semantics. Existing commands use fixed_waypoints.",
    )
    parser.add_argument(
        "--backend",
        choices=["local", "cloud"],
        default="local",
        help=(
            "local: disk-cached training on CPU or GPU; "
            "cloud: compiled in-memory data and synchronous GPU training."
        ),
    )
    parser.add_argument("--lr", default=3e-4, type=float, help="Initial LR")
    parser.add_argument(
        "--batch_size", default=30, type=int, help="Size of the mini-batches"
    )
    parser.add_argument(
        "--epochs", default=1000000, type=int, help="Number of training epochs"
    )
    parser.add_argument(
        "--es_patience", default=12, type=int, help="Patience for early stopping"
    )
    parser.add_argument(
        "--lr_patience", default=10, type=int, help="Patience for LR annealing"
    )
    parser.add_argument(
        "--best_after_epoch", default=50, type=int,
        help="Save best val_loss checkpoint only after this epoch (default: 50)."
    )
    parser.add_argument(
        "--lr_red_factor", default=0.1, type=float, help="Rate for LR annealing"
    )
    parser.add_argument("--min_lr", default=1e-6, type=float)
    parser.add_argument("--early_stopping_min_delta", default=1e-8, type=float)
    parser.add_argument(
        "--init_weights",
        default=None,
        help=(
            "Optional TensorFlow checkpoint prefix used to initialize training. "
            "Intended for warm-start experiments."
        ),
    )
    parser.add_argument(
        "--steps_per_execution",
        default=100,
        type=int,
        help="Compiled distributed training steps per Python dispatch in cloud mode.",
    )
    parser.add_argument(
        "--n_boids", default=100, type=int, help="N. of boids in simulation"
    )
    parser.add_argument(
        "--train_set_unique", default=50, type=int, help="N. of unique training trajectories"
    )
    parser.add_argument(
        "--train_set_repeats", default=20, type=int, help="N. of repeats for each training trajectory"
    )
    add_perturbation_arguments(parser, 2)
    parser.add_argument(
        "--val_set_size", default=10, type=int, help="N. of valid. trajectories"
    )
    parser.add_argument(
        "--te_set_size", default=30, type=int, help="N. of test trajectories"
    )
    parser.add_argument(
        "--test_complexity_every",
        default=-1,
        type=int,
        help="How often to test for complexity (-1 for never)",
    )
    parser.add_argument(
        "--init_centers_npz",
        default="",
        type=str,
        help="Optional NPZ path containing key 'centers' (N,2). If provided, training uses these centers and repeats each by train_set_repeats.",
    )
    parser.add_argument(
        "--loss_type",
        default="mse",
        choices=["mse", "goal_weighted"],
        help="Loss mode: 'mse' = plain MSE, 'goal_weighted' = near-goal upweighted MSE.",
    )
    parser.add_argument(
        "--critical_distance",
        default=2.5,
        type=float,
        help=(
            "Radius for upweighting near the active goal and, immediately after a "
            "switch, the previous goal until that radius is first exited."
        ),
    )
    parser.add_argument(
        "--distance_weight",
        default=2.5,
        type=float,
        help="Weight multiplier used inside the critical distance when using loss_type=goal_weighted.",
    )
    parser.add_argument(
        "--run_tag_suffix",
        default="",
        type=str,
        help="Optional descriptive suffix appended to generated run and artifact names.",
    )
    parser.add_argument(
        "--boids_cache",
        default="",
        type=str,
        help="Path to a precomputed boids cache HDF5 file (from generate_boids_cache.py). If set, skips simulation entirely.",
    )
    parser.add_argument("--cache_dir", default="cached_datasets",
                        help="Directory for reusable online-goal local caches.")
    parser.add_argument(
        "--timestep_stride",
        default=1,
        type=int,
        help="Load every Nth timestep from cache (default 1 = all). Increase to reduce RAM usage (e.g. 10 = 10x fewer samples).",
    )
    parser.add_argument(
        "--near_goal_radius",
        default=1.0,
        type=float,
        help="When timestep_stride>1, always include timesteps where the flock mean position "
             "is within this distance of the active or still-eligible previous goal "
             "(default: 1.0). "
             "Prevents goal-transition frames from being skipped by the stride. "
             "Set to 0 to disable.",
    )
    parser.add_argument(
        "--generation_workers",
        default=0,
        type=int,
        help="Expert CPU workers (0: cloud uses CPU count minus two; online local uses four).",
    )
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--select_best_over_run", action="store_true", default=True,
        help="Default. Validate every chunk on the same episodes and export the weights of the chunk "
             "with the lowest validation loss over the whole run.")
    parser.add_argument("--last_chunk_selection", dest="select_best_over_run", action="store_false",
        help="Export the best state of the last chunk instead, each chunk drawing its own validation episodes.")
    parser.add_argument(
        "--weight_init_seed",
        default=0,
        type=int,
        help="TensorFlow seed used when initializing a fresh fixed-waypoint model.",
    )
    parser.add_argument(
        "--goal_waypoints_per_episode",
        default=2,
        type=int,
        help="Online-goal training is defined using exactly two terminations.",
    )
    parser.add_argument(
        "--goal_bounds",
        nargs=4,
        type=float,
        default=[-4.5, 4.5, -4.5, 4.5],
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
    )
    parser.add_argument(
        "--start_bounds",
        nargs=4,
        type=float,
        default=[-5.0, 5.0, -5.0, 5.0],
        metavar=("X_MIN", "X_MAX", "Y_MIN", "Y_MAX"),
    )
    parser.add_argument("--goal_min_distance", default=2.5, type=float)
    parser.add_argument("--goal_arrival_radius", default=0.5, type=float)
    parser.add_argument("--expert_max_steps", default=10000, type=int)
    parser.add_argument(
        "--eval_online_goal_count",
        default=5,
        type=int,
        help="Number of sequential random goals per post-training online rollout.",
    )
    parser.add_argument("--eval_max_steps", default=5000, type=int)
    parser.add_argument("--eval_seed", default=1, type=int)
    parser.add_argument(
        "--eval_max_tube_radius",
        default=1.0,
        type=float,
        help=(
            "Maximum per-goal tube radius allowed for online evaluation success."
        ),
    )
    parser.add_argument(
        "--eval_r_threshold_mode",
        choices=["fixed", "expert_per_trajectory", "expert_global"],
        default="fixed",
        help="Compactness threshold mode used by post-training online evaluation.",
    )
    parser.add_argument(
        "--eval_expert_r_padding",
        default=0.1,
        type=float,
        help="Additive matched-Boids r tolerance for calibrated evaluation.",
    )
    parser.add_argument(
        "--eval_expert_calibration_max_steps",
        default=10000,
        type=int,
        help="Step limit for each matched Boids calibration rollout.",
    )
    parser.add_argument(
        "--perception",
        default=0.1,
        type=float,
        help="Neighbor radius used when rebuilding cached adjacency matrices.",
    )
    parser.add_argument(
        "--crowding",
        default=0.02,
        type=float,
        help="Separation radius of the Boids expert.",
    )
    parser.add_argument(
        "--init_scatter",
        default=0.325,
        type=float,
        help="Side of the starting box around each sampled center.",
    )
    parser.add_argument(
        "--shuffle_buffer_size",
        default=4096,
        type=int,
        help="Number of individual timesteps mixed before packed batching.",
    )
    parser.add_argument(
        "--viz_n_centers",
        default=50,
        type=int,
        help="Number of centers to use for the test/robustness visualization (default: 50).",
    )
    parser.add_argument(
        "--viz_centers_source",
        default="trained",
        choices=["trained", "unseen_cache", "random", "fresh_quadrant"],
        help="Source of centers for the test visualization: "
             "'trained' = from training set; "
             "'unseen_cache' = cache centers not used in training; "
             "'random' = fresh random; "
             "'fresh_quadrant' = newly sampled centers per quadrant, confirmed not in training set.",
    )
    parser.add_argument(
        "--selected_quadrants",
        type=int,
        nargs="+",
        default=None,
        help="Which quadrants to restrict fresh_quadrant sampling to (0=Q1, 1=Q2, 2=Q3, 3=Q4). "
             "Default: all 4. E.g. '--selected_quadrants 2 3' for Q3 and Q4 only.",
    )
    parser.add_argument(
        "--viz_mode",
        default="tubular",
        choices=["tubular", "per_boid", "multi_tubular", "individual"],
        help="Visualization mode: 'tubular' (mean + tube), 'per_boid' (one run, all boids), 'multi_tubular' (multiple tubes), 'individual' (ranked PDFs).",
    )
    # Private worker protocol, not user-facing experiment settings.
    parser.add_argument("--_chunk_worker", action="store_true", default=False, help=argparse.SUPPRESS)
    parser.add_argument("--_chunk_indices_file", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_cache_path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_checkpoint_path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_n_boids_cache", default=None, type=int, help=argparse.SUPPRESS)
    parser.add_argument("--_boundaries_file", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_val_npz_file", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_lr_state_file", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_run_best_path", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_chunk_number", default=0, type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--viz_n_show",
        default=None,
        type=int,
        help="Number of tubes to show in multi_tubular mode (default: use all training runs).",
    )
    parser.add_argument(
        "--viz_runs",
        type=int,
        nargs="+",
        default=None,
        help="Specific run indices to plot in multi_tubular mode (e.g., 0 5 10 15).",
    )
    parser.add_argument(
        "--viz_trained",
        action="store_true",
        default=False,
        help="Also run GNCA inference on training centers and include them in visualization. "
             "Off by default - only test centers are visualized.",
    )
    parser.add_argument(
        "--chunk_size",
        default=0,
        type=int,
        help="Trajectories per training chunk (0 = disabled, load all at once). "
             "Enable to train on large expert caches without loading them fully into RAM.",
    )
    parser.add_argument(
        "--chunk_patience",
        default=3,
        type=int,
        help="Per-chunk early-stopping patience (default: 3). Training on each chunk "
             "stops when validation loss doesn't improve for this many epochs.",
    )
    parser.add_argument(
        "--train_quadrants",
        type=int,
        nargs="+",
        default=None,
        help="Which quadrants to use for training (0=Q1, 1=Q2, 2=Q3, 3=Q4). "
             "Default: all 4. E.g. '--train_quadrants 2 3' for Q3 and Q4 only. "
             "Only applicable when loading from --boids_cache.",
    )
    parser.add_argument(
        "--exclusion_size",
        type=float,
        default=0.2,
        help="Buffer radius around the goal triangle for excluding test center sampling. "
             "Should match the exclusion size used when generating the cache (default: 0.2).",
    )
    parser.add_argument(
        "--skip_evaluation",
        action="store_true",
        default=False,
        help="Skip post-training inference and visualization after saving outputs.",
    )
    return parser
