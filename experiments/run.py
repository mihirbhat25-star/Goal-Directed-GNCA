"""The experiments: train the two-stage recipe, and evaluate its models.

    python -m experiments.run list
    python -m experiments.run train fixed2d
    python -m experiments.run evaluate fixed2d --weights saved_models/fixed2d_s<S>

The launcher needs only NumPy before it starts a training or evaluation process, so
``--dry-run`` works before TensorFlow is installed. Commands are executed without a
shell. Hyperparameters and expert settings are in ``experiments/presets``.
"""
import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modules.presets import NAMES, load_preset  # noqa: E402
from training.train import describe, plan_stages, run_stages  # noqa: E402


def cli_arguments(values):
    result = []
    for key, value in values.items():
        if value is None or value is False:
            continue
        result.append("--" + key)
        if value is not True:
            result.extend(str(item) for item in (value if isinstance(value, list) else [value]))
    return result


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


def run_command(command, dry_run):
    display = subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)
    print(display, flush=True)
    if not dry_run:
        environment = os.environ.copy()
        environment.setdefault("TF_USE_LEGACY_KERAS", "1")
        environment.setdefault("MPLBACKEND", "Agg")
        environment.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
        subprocess.run(command, cwd=str(ROOT), env=environment, check=True)


def run_module(module, values, dry_run):
    run_command([sys.executable, "-m", module] + cli_arguments(values), dry_run)


def train(preset, args):
    stages = plan_stages(args.experiment, preset, args.stage, args.seed if args.seed is not None else 0, args.label,
                         args.init_weights or "", args.generation_workers or 0)
    if args.dry_run:
        for settings in stages:
            print(describe(settings))
        return
    run_stages(stages)


def checked_weights(args):
    prefix = args.weights[:-6] if args.weights.endswith(".index") else args.weights
    if not args.dry_run and not (ROOT / (prefix + ".index")).is_file():
        raise ValueError("Missing checkpoint; --weights takes a TensorFlow checkpoint prefix, not a model folder.")
    return prefix


def evaluation_seed(preset, args):
    return args.seed if args.seed is not None else preset["evaluation"]["seed"]


def default_output(preset, args):
    return args.output_dir or f"results/{args.command}/{args.experiment}_seed{evaluation_seed(preset, args)}"


def paired_rollout_values(preset, args, weights, output):
    """Arguments of evaluation.paired_rollouts shared by evaluate, removal and scaling."""
    return {
        "experiment": args.experiment, "weights": weights,
        "max_steps": args.max_steps, "centers_per_region": args.centers_per_region, "seed": args.seed,
        "conditions": ["baseline"] if args.command != "removal" else ["baseline", "removal"],
        "population_changes": [50], "removal_pattern": "slab",
        "event_step_min": 500, "event_step_max": 1000,
        "save_radius_series": True, "output_dir": output,
        "workers": args.workers if args.workers is not None else default_workers(),
    }


def evaluate(preset, args):
    weights = checked_weights(args)
    output = default_output(preset, args)
    if not args.dry_run and (ROOT / output / "per_run.csv").exists():
        raise ValueError("Evaluation output already exists. Choose a new --output-dir to preserve it.")
    run_module("evaluation.paired_rollouts", paired_rollout_values(preset, args, weights, output), args.dry_run)
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
    Only the GNCA is rolled out unless ``--with-boids`` is given. ``<output>/scaling.pdf`` plots its completion and flock
    radius against N.
    """
    weights = checked_weights(args)
    output = default_output(preset, args)
    box = preset["expert"]["init_scatter"]
    runs = [("constant_density", n, box * (n / 100) ** (1 / preset["dimension"])) for n in args.sizes]
    runs += [("training_box", n, box) for n in args.training_box_sizes]
    for regime, n_agents, side in runs:
        folder = f"{output}/{regime}/n{n_agents}"
        if not args.dry_run and (ROOT / folder / "per_run.csv").exists():
            print(f"Skipping {folder}: already evaluated", flush=True)
            continue
        print(f"\n{regime.replace('_', ' ')}, N = {n_agents}, starting box side {side:.5f}", flush=True)
        values = paired_rollout_values(preset, args, weights, folder)
        values.update({"n_boids": n_agents, "init_scatter": round(side, 6), "save_radius_series": False,
                       "skip_plots": True, "gnca_only": not args.with_boids})
        run_module("evaluation.paired_rollouts", values, args.dry_run)
    run_module("evaluation.plots", {"scaling": output}, args.dry_run)


def imitation(preset, args):
    weights = checked_weights(args)
    seed = evaluation_seed(preset, args)
    output = args.output_dir or f"results/imitation/{args.experiment}_seed{seed}"
    if not args.dry_run and (ROOT / output / "imitation_error_report.txt").exists():
        raise ValueError("Imitation output already exists. Choose a new --output-dir to preserve it.")
    values = {"experiment": args.experiment, "weights": weights, "output_dir": output,
              "n_rollouts": args.n_rollouts, "seed": seed, "max_steps": args.max_steps}
    run_command([sys.executable, "-m", "evaluation.imitation_error", "run"] + cli_arguments(values), args.dry_run)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=("list", "train", "evaluate", "removal", "scaling", "imitation"))
    parser.add_argument("experiment", choices=NAMES, nargs="?")
    parser.add_argument("--dry-run", action="store_true", help="Print what would run without running it.")
    parser.add_argument("--seed", type=int,
                        help="Random seed.")
    parser.add_argument("--label", default="", help="train: suffix of the run name.")
    parser.add_argument("--generation-workers", type=int,
                        help="train: CPU threads of the expert generation (default: CPU count minus two).")
    parser.add_argument("--stage", choices=("both", "onestep", "bptt"), default="both",
                        help="train: run both stages (default), only one-step pretraining, or only the BPTT "
                             "fine-tune from --init-weights.")
    parser.add_argument("--init-weights", help="train --stage bptt: checkpoint prefix of the one-step model.")
    parser.add_argument("--weights", help="Checkpoint prefix of the model, for example saved_models/fixed2d_s<S>; "
                                          "required by evaluate, removal, scaling and imitation.")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-steps", type=int, help="Rollout horizon (default: the preset's).")
    parser.add_argument("--centers-per-region", type=int, help="Starts per quadrant or octant (default: the preset's).")
    parser.add_argument("--workers", type=int,
                        help="Evaluation processes to run in parallel (evaluate, removal and scaling); the starts are "
                             "split between them and the results merged. Default: one per visible GPU, otherwise one.")
    parser.add_argument("--sizes", type=int, nargs="+", default=[25, 200],
                        help="scaling: swarm sizes started at the training density (default: 25 200).")
    parser.add_argument("--training-box-sizes", type=int, nargs="+", default=[250],
                        help="scaling: swarm sizes placed in the training box (default: 250).")
    parser.add_argument("--with-boids", action="store_true",
                        help="scaling: also roll out the Boids expert from the same starts.")
    parser.add_argument("--n-rollouts", type=int, default=50,
                        help="imitation: held-out expert rollouts (default: 50).")
    args = parser.parse_args()
    if args.n_rollouts < 1:
        parser.error("--n-rollouts must be positive")
    if min(args.sizes + args.training_box_sizes) < 2:
        parser.error("--sizes and --training-box-sizes must be at least 2")
    if args.label and not re.fullmatch(r"[A-Za-z0-9_-]+", args.label):
        parser.error("--label must contain only letters, numbers, underscores or hyphens")
    for key in ("max_steps", "centers_per_region", "workers"):
        value = getattr(args, key)
        if value is not None and value < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.generation_workers is not None and args.generation_workers < 0:
        parser.error("--generation-workers must be nonnegative")
    if args.command == "list":
        for name in NAMES:
            preset = load_preset(name)
            print(f"{name}: {preset['dimension']}D {preset['task']}")
        return
    if not args.experiment:
        parser.error("This action requires an experiment name")
    if args.command == "train":
        if args.weights or args.output_dir or args.centers_per_region or args.workers or args.max_steps:
            parser.error("--weights, --output-dir, --centers-per-region, --workers and --max-steps are "
                         "evaluation options")
        if args.stage == "bptt" and not args.init_weights:
            parser.error("--stage bptt needs --init-weights, the checkpoint prefix of the one-step model")
        if args.stage != "bptt" and args.init_weights:
            parser.error("--init-weights applies to --stage bptt")
    else:
        if not args.weights:
            parser.error(f"{args.command} needs --weights, the checkpoint prefix of a trained model "
                         "(for example saved_models/fixed2d_s<S>)")
        if args.label or args.generation_workers is not None or args.init_weights or args.stage != "both":
            parser.error("--label, --generation-workers, --stage and --init-weights are training options")
    if args.command == "imitation" and args.workers:
        parser.error("--workers applies to evaluate, removal and scaling")
    preset = load_preset(args.experiment)
    try:
        {"train": train, "imitation": imitation, "scaling": scaling}.get(args.command, evaluate)(preset, args)
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
