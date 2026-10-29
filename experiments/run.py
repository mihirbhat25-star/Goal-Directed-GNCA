"""Named paper experiments; prints ordinary CLI commands before running them.

The launcher uses only the Python standard library and the dependency-free
``modules.run_tags`` helper, so ``--dry-run`` works before TensorFlow is
installed. Commands are executed without a shell.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
NAMES = ("fixed2d", "fixed3d", "online2d", "online3d")
# Seed of the evaluation and removal test starts when --seed is not given.
TEST_SEED = 460
sys.path.insert(0, str(ROOT))
from modules.run_tags import PAPER_EXPERT, expert_tag, run_tag as build_run_tag  # noqa: E402


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def cli_arguments(values):
    result = []
    for key, value in values.items():
        if value is None or value is False:
            continue
        result.append("--" + key)
        if value is not True:
            result.extend(str(item) for item in (value if isinstance(value, list) else [value]))
    return result


def checkpoint_tag(experiment, preset, values):
    """Run tag the training script will use, so the launcher knows its output path."""
    online = preset["task"] == "online_goals"
    return build_run_tag(experiment, values["train_set_unique"],
                         1 if online else values["train_set_repeats"], values["loss_type"],
                         values["critical_distance"], values["distance_weight"],
                         values.get("run_tag_suffix", ""),
                         expert=expert_tag(**expert_settings(values)))


def expert_settings(source):
    """Perception, crowding and starting box from a mapping or the parsed arguments."""
    get = source.get if isinstance(source, dict) else (lambda key, default: getattr(source, key, None) or default)
    return {key: float(get(key, default) if get(key, default) is not None else default)
            for key, default in PAPER_EXPERT.items()}


# Drift perturbation of the expert during data generation. Every ``interval`` steps
# each active agent is selected independently with probability ``probability``, at
# most ``max_agents`` of them per event, and the selected agents receive bounded
# uniform noise on their position (``position_bound`` per axis) and velocity
# (``velocity_bound`` per axis). These defaults apply when a preset does not set
# its own drift values; ``--drift-target`` chooses which part of the state is hit.
DRIFT_DEFAULTS = {"interval": 250, "probability": 0.1, "max_agents": 10,
                  "position_bound": 0.1, "velocity_bound": 0.005, "target": "both"}
# The 3D experiments perturb less often with a larger velocity bound (the dynamic 3D recipe).
DRIFT_DEFAULTS_3D = dict(DRIFT_DEFAULTS, interval=500, velocity_bound=0.01)
DRIFT_TARGETS = ("both", "position", "velocity")


# Training hyperparameters every trainer accepts; a value given on the command line
# replaces the preset.
TRAINING_OVERRIDES = ("generation_workers", "batch_size", "chunk_size", "epochs", "loss_type",
                      "perception", "crowding", "init_scatter",
                      "train_set_repeats", "chunk_patience", "lr_patience", "lr", "min_lr",
                      "lr_red_factor", "val_set_size", "critical_distance", "distance_weight",
                      "timestep_stride", "near_goal_radius", "early_stopping_min_delta")
# Expert settings that the cached fixed-waypoint datasets must share with training.
CACHE_EXPERT_KEYS = ("perception", "crowding", "init_scatter")


def training_plan(preset, args):
    backend = args.backend or preset["backend"]
    base = dict(preset["train"], task=preset["task"], backend=backend)
    explicit = {key: getattr(args, key) for key in TRAINING_OVERRIDES if getattr(args, key) is not None}
    base.update(explicit)
    if args.train_set_unique is not None:
        base["train_set_unique"] = args.train_set_unique
        if "octant_unique_counts" in base:
            # The unique trajectories are split evenly over the preset's octants.
            octants = list(base["train_octants"])
            per_octant, remainder = divmod(args.train_set_unique, len(octants))
            base["octant_unique_counts"] = [per_octant + (1 if index < remainder else 0)
                                            for index in range(len(octants))]
    if args.last_chunk_selection:
        base["last_chunk_selection"] = True
    if args.train_arrival_radius is not None:
        # Waypoint switch of the expert in the training data only; evaluation keeps
        # the centroid within 0.5 for both controllers.
        base["goal_arrival_radius"] = args.train_arrival_radius
    if args.train_set_unique is not None and "normal_trajectories" in base:
        # The online presets spell out the perturbation mix; keep it all-normal at the new size.
        if int(base.get("removal_trajectories", 0)):
            raise ValueError("--train-set-unique cannot be combined with a preset that mixes "
                             "removal trajectories; run the training module directly.")
        if int(base.get("drift_trajectories", 0)) and args.drift_trajectories is None:
            raise ValueError("This preset mixes drift-perturbed trajectories; with --train-set-unique, "
                             "also give --drift-trajectories (0 for none).")
        base["normal_trajectories"] = args.train_set_unique
    if args.drift_trajectories is not None:
        # Part of the unique trajectories come from a drift-perturbed expert. Each
        # drift setting comes from the command line, else from the preset, else
        # from DRIFT_DEFAULTS.
        fixed = preset["task"] == "fixed_waypoints"
        # Fixed tasks generate every unique center train_set_repeats times.
        unique = int(base["train_set_unique"]) * (int(base.get("train_set_repeats", 1)) if fixed else 1)
        if args.drift_trajectories > unique:
            raise ValueError("--drift-trajectories cannot exceed the number of generated trajectories.")
        dimension = preset["dimension"]
        defaults = DRIFT_DEFAULTS_3D if dimension == 3 else DRIFT_DEFAULTS

        def choose(option, preset_key, default):
            value = getattr(args, option)
            if value is not None:
                return value
            return base[preset_key] if preset_key in base else default

        interval = choose("drift_interval", "perturbation_interval", defaults["interval"])
        probability = choose("drift_probability", "perturbation_probability", defaults["probability"])
        max_agents = choose("drift_max_agents", "drift_max_per_event", defaults["max_agents"])
        preset_position = base.get("drift_bounds")
        position_bound = choose("drift_position_bound", "_", preset_position[0] if preset_position else
                                defaults["position_bound"])
        preset_velocity = base.get("drift_velocity_bounds")
        velocity_bound = choose("drift_velocity_bound", "_", preset_velocity[0] if preset_velocity else
                                defaults["velocity_bound"])
        if args.drift_target is not None:
            target = args.drift_target
        elif "drift_perturb_velocity" in base:
            target = "both" if base["drift_perturb_velocity"] else "position"
        else:
            target = defaults["target"]
        base.update({
            "perturbation_interval": int(interval), "perturbation_probability": float(probability),
            "drift_max_per_event": int(max_agents),
            "drift_bounds": [float(position_bound) if target != "velocity" else 0.0] * dimension,
            "drift_perturb_velocity": target != "position",
            "drift_velocity_bounds": [float(velocity_bound)] * dimension if target != "position" else None,
            "drift_trajectories": args.drift_trajectories,
            "removal_trajectories": int(base.get("removal_trajectories", 0)),
        })
        base["normal_trajectories"] = unique - args.drift_trajectories - base["removal_trajectories"]
    if (preset["task"] == "fixed_waypoints" and backend == "local"
            and int(base.get("drift_trajectories", 0) or 0) > 0):
        raise ValueError("Fixed-waypoint training on the local backend reads an unperturbed expert cache; "
                         "use --backend cloud for the drift-perturbed trajectories, or --drift-trajectories 0.")
    if args.seed is not None:
        base["seed" if preset["dimension"] == 2 else "generation_seed"] = args.seed
        if preset["task"] == "fixed_waypoints":
            base["weight_init_seed"] = args.seed
    cache = None
    if backend == "local" and "cache" in preset:
        cache = json.loads(json.dumps(preset["cache"]))
        # The cache holds as many starting centers per region, and repeats, as training uses.
        regions = len(cache["args"].get("octants") or cache["args"].get("quadrants") or [0])
        needed = math.ceil(int(base["train_set_unique"]) / regions)
        repeats = int(base.get("train_set_repeats", 1))
        if needed != int(cache["args"]["unique"]) or repeats != int(cache["args"]["repeats"]):
            cache["args"]["unique"], cache["args"]["repeats"] = needed, repeats
            old_path = Path(cache["args"]["output"])
            cache["args"]["output"] = str(old_path.with_name(old_path.stem + f"_{needed}x{repeats}.h5"))
        if args.seed is not None:
            cache["args"]["seed"] = args.seed
            old_path = Path(cache["args"]["output"])
            cache["args"]["output"] = str(old_path.with_name(old_path.stem + f"_seed{args.seed}.h5"))
        changed = [key for key in CACHE_EXPERT_KEYS if key in explicit]
        for key in changed:
            cache["args"][key] = explicit[key]
        if changed:
            old_path = Path(cache["args"]["output"])
            cache["args"]["output"] = str(old_path.with_name(
                old_path.stem + "_" + expert_tag(**expert_settings(explicit)) + ".h5"))
        if args.train_arrival_radius is not None and "goal_arrival_radius" in cache["args"]:
            cache["args"]["goal_arrival_radius"] = args.train_arrival_radius
            old_path = Path(cache["args"]["output"])
            cache["args"]["output"] = str(old_path.with_name(old_path.stem + f"_arrival{args.train_arrival_radius:g}.h5"))
        base["boids_cache"] = cache["args"]["output"]
    elif backend == "local":
        base["cache_dir"] = args.cache_dir
    if preset["dimension"] == 2 and preset["task"] == "fixed_waypoints" and backend == "cloud":
        base["exclusion_size"] = preset["cache"]["args"]["goal_exclusion_size"]
    module = "boids.run_boids" + ("_3d" if preset["dimension"] == 3 else "")
    values = dict(base)
    values["run_tag_suffix"] = "_".join(
        part for part in (values.get("run_tag_suffix", ""), args.label) if part)
    checkpoint = "saved_models/best_weights_" + checkpoint_tag(args.experiment, preset, values)
    return cache, [(module, values, checkpoint)]


def default_workers():
    """One evaluation process per visible GPU, otherwise one; no TensorFlow import needed."""
    configured = os.environ.get("CUDA_VISIBLE_DEVICES")
    if configured is not None:
        return max(1, len([item for item in configured.split(",") if item.strip()]))
    try:
        listing = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=20)
        gpus = [line for line in listing.stdout.splitlines() if line.startswith("GPU ")]
        return max(1, len(gpus)) if listing.returncode == 0 else 1
    except (OSError, subprocess.SubprocessError):
        return 1


def run_command(module, values, dry_run):
    command = [sys.executable, "-m", module] + cli_arguments(values)
    display = subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)
    print(display, flush=True)
    if not dry_run:
        environment = os.environ.copy()
        environment.setdefault("TF_USE_LEGACY_KERAS", "1")
        environment.setdefault("MPLBACKEND", "Agg")
        environment.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        subprocess.run(command, cwd=str(ROOT), env=environment, check=True)


def train(preset, args):
    cache, commands = training_plan(preset, args)
    if not args.dry_run:
        for _, _, checkpoint in commands:
            if (ROOT / (checkpoint + ".index")).exists():
                raise ValueError("Training output already exists. Choose a new --label; checkpoints are not silently overwritten.")
        for folder in ("cached_datasets", "saved_models", "saved_history", "saved_training_data", "results"):
            (ROOT / folder).mkdir(exist_ok=True)
    if cache:
        path = ROOT / cache["args"]["output"]
        recipe_path = Path(str(path) + ".recipe.json")
        if path.exists() and not args.dry_run:
            if not recipe_path.exists() or read_json(recipe_path) != cache:
                raise ValueError("Existing cache has no matching recipe. Move it aside before generating this experiment's cache.")
            print(f"Reusing {cache['args']['output']}")
        else:
            run_command(cache["module"], cache["args"], args.dry_run)
            if not args.dry_run:
                recipe_path.write_text(json.dumps(cache, indent=2) + "\n", encoding="utf-8")
    for module, values, checkpoint in commands:
        print(f"\nTraining; output checkpoint: {checkpoint}", flush=True)
        run_command(module, values, args.dry_run)
        if not args.dry_run and not (ROOT / (checkpoint + ".index")).is_file():
            raise RuntimeError("Training exited without its expected best-weights checkpoint.")
    if not args.dry_run:
        record = {"experiment": args.experiment, "label": args.label,
                  "cache": cache, "commands": commands, "checkpoint": commands[-1][2]}
        (ROOT / "saved_history" / f"{args.experiment}_{args.label}_recipe.json").write_text(
            json.dumps(record, indent=2) + "\n", encoding="utf-8")


def checked_weights(args):
    weights = args.weights
    prefix = weights[:-6] if weights.endswith(".index") else weights
    if not args.dry_run and not (ROOT / (prefix + ".index")).is_file():
        raise ValueError("Missing checkpoint; --weights takes a TensorFlow checkpoint prefix, not a model folder.")
    return weights


def default_output(args, expert, seed):
    return args.output_dir or "_".join(part for part in (
        f"results/{args.command}/{args.experiment}", expert_tag(**expert), f"seed{seed}") if part)


def paired_rollout_values(preset, args, weights, expert, output):
    """Arguments of evaluation.paired_rollouts shared by evaluate, removal and scaling."""
    values = {
        "dimension": preset["dimension"], "task": preset["task"],
        "run_tag": Path(weights).name,
        "weights_path": weights, "n_boids": 100, **expert,
        "max_steps": args.max_steps, "success_threshold": 0.5,
        "regions": list(range(4 if preset["dimension"] == 2 else 8)),
        "centers_per_region": args.centers_per_region or preset["evaluation_centers_per_region"],
        "online_goal_count": 5, "goal_min_distance": 2.5,
        "seed": args.seed if args.seed is not None else TEST_SEED,
        "conditions": ["baseline"] if args.command != "removal" else ["baseline", "removal"],
        "population_changes": [10, 25, 50], "removal_pattern": "slab",
        "event_step_min": 500, "event_step_max": 1000,
        "save_radius_series": True, "output_dir": output,
    }
    values["workers"] = args.workers if args.workers is not None else default_workers()
    return values


def evaluate(preset, args):
    weights = checked_weights(args)
    expert = expert_settings(args)
    output = default_output(args, expert, args.seed if args.seed is not None else TEST_SEED)
    if not args.dry_run and (ROOT / output / "per_run.csv").exists():
        raise ValueError("Evaluation output already exists. Choose a new --output-dir to preserve it.")
    values = paired_rollout_values(preset, args, weights, expert, output)
    run_command("evaluation.paired_rollouts", values, args.dry_run)
    if not args.dry_run:
        from evaluation.metrics.summarize import summarize_csv
        summarize_csv(ROOT / output / "per_run.csv")


def scaling(preset, args):
    """Evaluate a model trained on 100 agents at other swarm sizes, without retraining.

    ``--sizes`` start at the training density: the side of the starting box grows as
    ``init_scatter * (N / 100) ** (1 / d)``. ``--training-box-sizes`` place N agents in the
    training box itself, at N / 100 times the training density. Each size is written to
    ``<output>/constant_density/n<N>/`` or ``<output>/training_box/n<N>/``; sizes whose
    ``per_run.csv`` already exists are skipped, so an interrupted run can be resumed.
    Only the GNCA is rolled out. ``<output>/scaling.pdf`` plots its completion and flock
    radius against N.
    """
    weights = checked_weights(args)
    expert = expert_settings(args)
    output = default_output(args, expert, args.seed if args.seed is not None else TEST_SEED)
    dimension = preset["dimension"]
    box = expert["init_scatter"]
    runs = [("constant_density", n, box * (n / 100) ** (1 / dimension)) for n in args.sizes]
    runs += [("training_box", n, box) for n in args.training_box_sizes]
    for regime, n_agents, side in runs:
        folder = f"{output}/{regime}/n{n_agents}"
        if not args.dry_run and (ROOT / folder / "per_run.csv").exists():
            print(f"Skipping {folder}: already evaluated", flush=True)
            continue
        print(f"\n{regime.replace('_', ' ')}, N = {n_agents}, starting box side {side:.5f}", flush=True)
        values = paired_rollout_values(preset, args, weights, expert, folder)
        values.update({"n_boids": n_agents, "init_scatter": round(side, 6), "save_radius_series": False,
                       "skip_plots": True, "gnca_only": True})
        run_command("evaluation.paired_rollouts", values, args.dry_run)
    run_command("evaluation.plots", {"scaling": output}, args.dry_run)


def imitation(preset, args):
    weights = checked_weights(args)
    prefix = weights[:-6] if weights.endswith(".index") else weights
    seed = args.seed if args.seed is not None else 1
    expert = expert_settings(args)
    output = args.output_dir or "_".join(part for part in (
        f"results/imitation/{args.experiment}", expert_tag(**expert), f"seed{seed}") if part)
    if not args.dry_run and (ROOT / output / "imitation_error_report.txt").exists():
        raise ValueError("Imitation output already exists. Choose a new --output-dir to preserve it.")
    values = {
        "dimension": preset["dimension"], "task": preset["task"], "weights_path": prefix,
        "output_dir": output, "n_rollouts": args.n_rollouts, "seed": seed,
        "n_boids": 100, **expert, "max_steps": args.max_steps, "stride": 10,
    }
    command = [sys.executable, "-m", "evaluation.imitation_error", "run"] + cli_arguments(values)
    display = subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)
    print(display, flush=True)
    if not args.dry_run:
        environment = dict(os.environ, PYTHONPATH=str(ROOT))
        subprocess.run(command, cwd=str(ROOT), env=environment, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("list", "train", "evaluate", "removal", "scaling", "imitation"))
    parser.add_argument("experiment", choices=NAMES, nargs="?")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--backend", choices=("local", "cloud"))
    parser.add_argument("--cache-dir", default="cached_datasets",
                        help="Reusable online-goal cache directory with --backend local.")
    parser.add_argument("--label", default="retrain", help="Suffix for NEW training outputs; published weights remain read-only.")
    parser.add_argument("--generation-workers", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--chunk-size", type=int)
    parser.add_argument("--select-best-over-run", action="store_true", default=None,
                        help="The default: validate every chunk on the same episodes and export the weights of the "
                             "chunk with the lowest validation loss over the whole run. Kept for explicit commands.")
    parser.add_argument("--last-chunk-selection", action="store_true", default=None,
                        help="Export the best state of the last chunk instead, each chunk drawing its own "
                             "validation episodes (how the models reported in the paper were selected).")
    parser.add_argument("--train-arrival-radius", type=float,
                        help="Centroid distance at which the expert switches to its next waypoint in the training "
                             "data (preset: 0.5). Evaluation always uses 0.5.")
    parser.add_argument("--lr", type=float,
                        help="Learning rate every chunk starts (and restarts) at, replacing the preset's value. "
                             "")
    parser.add_argument("--min-lr", type=float, help="Floor of the learning-rate schedule.")
    parser.add_argument("--lr-red-factor", type=float,
                        help="Factor applied to the learning rate after --lr-patience epochs without improvement.")
    parser.add_argument("--val-set-size", type=int, help="Validation trajectories per chunk.")
    parser.add_argument("--critical-distance", type=float,
                        help="Goal-weighted loss: distance to the previous waypoint within which transitions are upweighted.")
    parser.add_argument("--distance-weight", type=float, help="Goal-weighted loss: the upweighting factor.")
    parser.add_argument("--timestep-stride", type=int,
                        help="Keep every n-th transition far from the goals (near-goal transitions are all kept).")
    parser.add_argument("--near-goal-radius", type=float,
                        help="Distance to a goal within which --timestep-stride does not subsample.")
    parser.add_argument("--early-stopping-min-delta", type=float,
                        help="Smallest validation improvement that resets the chunk and learning-rate patience.")
    parser.add_argument("--chunk-patience", type=int,
                        help="Epochs without validation improvement before a chunk stops, replacing the preset's value.")
    parser.add_argument("--lr-patience", type=int,
                        help="Epochs without validation improvement before the learning rate is reduced, replacing the preset's value.")
    parser.add_argument("--train-set-unique", type=int,
                        help="Unique expert trajectories (fixed tasks: unique starting centers) to train on, "
                             "replacing the preset's count. The run tag records the new size.")
    parser.add_argument("--train-set-repeats", type=int,
                        help="Repeats of every starting center (fixed-waypoint experiments only).")
    parser.add_argument("--drift-trajectories", type=int,
                        help="Number of the unique training trajectories generated with the drift-perturbed expert; "
                             "the rest are unperturbed. Every --drift-interval steps each agent is selected with "
                             "probability --drift-probability, at most --drift-max-agents per event, and the selected "
                             "agents receive bounded uniform noise on position and/or velocity (--drift-target). "
                             f"Defaults (unless the preset sets them): interval {DRIFT_DEFAULTS['interval']}, "
                             f"probability {DRIFT_DEFAULTS['probability']}, at most {DRIFT_DEFAULTS['max_agents']} "
                             f"agents, position bound {DRIFT_DEFAULTS['position_bound']}, velocity bound "
                             f"{DRIFT_DEFAULTS['velocity_bound']}, target {DRIFT_DEFAULTS['target']}; in 3D the "
                             f"interval is {DRIFT_DEFAULTS_3D['interval']} and the velocity bound "
                             f"{DRIFT_DEFAULTS_3D['velocity_bound']}. Not available for fixed2d, whose expert cache "
                             "is unperturbed.")
    parser.add_argument("--drift-target", choices=DRIFT_TARGETS,
                        help="Which part of a selected agent's state the drift noise perturbs.")
    parser.add_argument("--drift-interval", type=int, help="Steps between drift events.")
    parser.add_argument("--drift-probability", type=float, help="Selection probability of each agent at an event.")
    parser.add_argument("--drift-max-agents", type=int, help="Maximum number of agents perturbed per event.")
    parser.add_argument("--drift-position-bound", type=float, help="Uniform position noise bound per axis.")
    parser.add_argument("--drift-velocity-bound", type=float, help="Uniform velocity noise bound per axis.")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--loss-type", choices=("mse", "goal_weighted"),
                        help="Training loss: plain one-step MSE, or the near-goal weighted MSE of the presets.")
    parser.add_argument("--seed", type=int,
                        help="Training: data generation and weight initialization seed; defaults to the "
                             "preset's value (1). Evaluation and removal: seed of the test starts, default "
                             f"{TEST_SEED}. Imitation: seed of the held-out expert rollouts, default 1.")
    parser.add_argument("--weights", help="Checkpoint prefix of the model to evaluate, for example "
                        "saved_models/best_weights_<run tag>; required by evaluate, removal, scaling and imitation.")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--centers-per-region", type=int)
    parser.add_argument("--perception", type=float,
                        help="Neighbor radius of the Boids expert and of the GNCA's graph (paper: 0.1).")
    parser.add_argument("--crowding", type=float,
                        help="Separation radius of the Boids expert (paper: 0.02).")
    parser.add_argument("--init-scatter", type=float,
                        help="Side of the starting box around each sampled center (paper: 0.325).")
    parser.add_argument("--workers", type=int,
                        help="Evaluation processes to run in parallel (evaluate, removal and scaling); the starts are "
                             "split between them and the results merged. Default: one per visible GPU, "
                             "otherwise one.")
    parser.add_argument("--sizes", type=int, nargs="+", default=[25, 500],
                        help="scaling: swarm sizes started at the training density (default: 25 500).")
    parser.add_argument("--training-box-sizes", type=int, nargs="+", default=[200, 500],
                        help="scaling: swarm sizes placed in the training box (default: 200 500).")
    parser.add_argument("--n-rollouts", type=int, default=50,
                        help="Held-out expert rollouts for the imitation command (default: 50).")
    args = parser.parse_args()
    if args.n_rollouts < 1:
        parser.error("--n-rollouts must be positive")
    if min(args.sizes + args.training_box_sizes) < 2:
        parser.error("--sizes and --training-box-sizes must be at least 2")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.label):
        parser.error("--label must contain only letters, numbers, underscores or hyphens")
    for key in ("batch_size", "chunk_size", "epochs", "max_steps", "centers_per_region",
                "workers", "train_set_unique", "train_set_repeats", "val_set_size", "timestep_stride",
                "chunk_patience", "lr_patience"):
        value = getattr(args, key)
        if value is not None and value < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.generation_workers is not None and args.generation_workers < 0:
        parser.error("--generation-workers must be nonnegative")
    if args.drift_trajectories is not None and args.drift_trajectories < 0:
        parser.error("--drift-trajectories must be nonnegative")
    drift_options = ("drift_target", "drift_interval", "drift_probability", "drift_max_agents",
                     "drift_position_bound", "drift_velocity_bound")
    if args.drift_trajectories is None and any(getattr(args, key) is not None for key in drift_options):
        parser.error("The drift settings require --drift-trajectories")
    if args.drift_interval is not None and args.drift_interval < 1:
        parser.error("--drift-interval must be positive")
    if args.drift_probability is not None and not 0 < args.drift_probability <= 1:
        parser.error("--drift-probability must be in (0, 1]")
    if args.drift_max_agents is not None and args.drift_max_agents < 1:
        parser.error("--drift-max-agents must be positive")
    for key in ("drift_position_bound", "drift_velocity_bound"):
        value = getattr(args, key)
        if value is not None and value <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    for key in ("perception", "crowding", "init_scatter", "lr", "critical_distance", "distance_weight",
                "near_goal_radius", "train_arrival_radius"):
        value = getattr(args, key)
        if value is not None and value <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.command == "list":
        for name in NAMES:
            preset = read_json(ROOT / "experiments/presets" / (name + ".json"))
            print(f"{name}: {preset['dimension']}D {preset['task']}, default backend {preset['backend']}")
        return
    if not args.experiment:
        parser.error("This action requires an experiment name")
    if args.command in ("evaluate", "removal", "scaling", "imitation") and not args.weights:
        parser.error(f"{args.command} needs --weights, the checkpoint prefix of a trained model "
                     "(for example saved_models/best_weights_<run tag>)")
    if args.command != "train" and any(getattr(args, k) is not None for k in (
            "backend", "generation_workers", "batch_size", "chunk_size", "epochs", "loss_type",
            "train_set_unique", "train_set_repeats", "chunk_patience", "lr_patience",
            "select_best_over_run", "last_chunk_selection", "train_arrival_radius", "lr", "min_lr",
            "lr_red_factor", "val_set_size", "critical_distance", "distance_weight", "timestep_stride",
            "near_goal_radius", "early_stopping_min_delta",
            "drift_trajectories", "drift_target",
            "drift_interval", "drift_probability", "drift_max_agents", "drift_position_bound",
            "drift_velocity_bound")):
        parser.error("Training overrides cannot be used for evaluation")
    if args.command == "train" and (args.weights or args.output_dir or args.centers_per_region or args.workers):
        parser.error("--weights, --output-dir, --centers-per-region and --workers are evaluation options")
    if args.command == "imitation" and args.workers:
        parser.error("--workers applies to evaluate, removal and scaling")
    preset = read_json(ROOT / "experiments/presets" / (args.experiment + ".json"))
    if args.train_set_repeats is not None and preset["task"] != "fixed_waypoints":
        parser.error("--train-set-repeats applies to the fixed-waypoint experiments (fixed2d, fixed3d)")
    try:
        {"train": train, "imitation": imitation, "scaling": scaling}.get(args.command, evaluate)(preset, args)
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
