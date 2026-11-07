"""The two-stage recipe: one-step pretraining, then a BPTT fine-tune.

Stage 1 trains on single expert transitions, with seed S for data and weights. Stage 2
starts from stage 1's weights with a fresh optimizer and trains on K-step rollout windows
of new, unperturbed expert trajectories, drawn with a data seed derived from S. Each stage is
a sequence of chunks, one subprocess per chunk (``training.worker``), that hand their best
weights and Adam state on through a temporary state directory. The hyperparameters of both
stages are in ``experiments/presets/<experiment>.json``.

This module imports no TensorFlow; it can plan a run before TensorFlow is installed.
"""

import json
import os
import sys
import tempfile

from modules.presets import ROOT, goal_conditioned
from modules.run_names import run_name
from training.data import chunk_plan

STAGES = ("onestep", "bptt")


def stage_settings(experiment, preset, stage, seed, label="", init_weights="", generation_workers=0):
    """Everything one training stage needs, flat except for the expert and drift settings."""
    name = run_name(experiment, seed, stage, label)
    settings = {
        "experiment": experiment,
        "stage": stage,
        "run_name": name,
        "weights_path": f"saved_models/{name}",
        "dimension": preset["dimension"],
        "task": preset["task"],
        "goal_conditioned": goal_conditioned(preset),
        "expert": preset["expert"],
        "drift": preset["drift"],
        "seed": int(seed) + int(preset[stage].get("data_seed_offset", 0)),
        "init_weights": init_weights,
        "generation_workers": int(generation_workers),
        **preset["training"],
        **preset[stage],
    }
    if settings["perturbed"] > settings["trajectories"]:
        raise ValueError("A stage cannot have more perturbed trajectories than trajectories.")
    if stage == "bptt" and not init_weights:
        raise ValueError("The BPTT stage starts from a one-step checkpoint.")
    return settings


def plan_stages(experiment, preset, which, seed, label="", init_weights="", generation_workers=0):
    """The stages of a ``train`` command; with both, stage 2 starts from stage 1's output."""
    stages = []
    if which in ("both", "onestep"):
        stages.append(stage_settings(experiment, preset, "onestep", seed, label,
                                     generation_workers=generation_workers))
    if which in ("both", "bptt"):
        start = stages[0]["weights_path"] if stages else init_weights
        stages.append(stage_settings(experiment, preset, "bptt", seed, label, init_weights=start,
                                     generation_workers=generation_workers))
    return stages


def describe(settings):
    """One paragraph of what a stage will do."""
    chunks = chunk_plan(settings)
    objective = (
        f"{settings['window']}-step rollout windows, batch {settings['batch_size']}, "
        f"gradient clipping {settings['grad_clip_norm']:g}"
        if settings["stage"] == "bptt" else
        f"single transitions, batch {settings['batch_size']}, timestep stride {settings['timestep_stride']}"
    )
    return (
        f"{settings['stage']}: {settings['trajectories']} trajectories ({settings['perturbed']} perturbed) in "
        f"{len(chunks)} chunks, data seed {settings['seed']}, {objective}, lr {settings['lr']:g}"
        + (f", from {settings['init_weights']}" if settings["init_weights"] else "")
        + f"; output {settings['weights_path']}"
    )


def _worker_environment():
    environment = os.environ.copy()
    environment.setdefault("TF_USE_LEGACY_KERAS", "1")
    environment.setdefault("MPLBACKEND", "Agg")
    # Override NVIDIA image defaults rather than inheriting a noisier value.
    environment["TF_CPP_MIN_LOG_LEVEL"] = "3"
    return environment


def run_stage(settings):
    """Train all chunks of one stage and save its record next to the other runs' records."""
    from training.native_output import run_with_filtered_stderr

    chunks = chunk_plan(settings)
    print(f"\n>>> {describe(settings)}", flush=True)
    print(f">>> Learning rate across chunks: schedule restarted at {settings['lr']:g} for every chunk, "
          "from the previous chunk's best weights and optimizer moments.", flush=True)
    (ROOT / "saved_models").mkdir(exist_ok=True)
    results = []
    with tempfile.TemporaryDirectory(prefix="gnca_training_") as temp_dir:
        state_dir = os.path.join(temp_dir, "training_state")
        os.makedirs(state_dir)
        settings_file = os.path.join(temp_dir, "settings.json")
        with open(settings_file, "w", encoding="utf-8") as handle:
            json.dump(settings, handle)
        for chunk_index, (region_counts, mode_counts) in enumerate(chunks):
            result_path = os.path.join(temp_dir, f"chunk_{chunk_index:04d}.json")
            print(
                f"\n{'=' * 64}\n  {settings['run_name']}: chunk {chunk_index + 1}/{len(chunks)} | "
                f"regions {region_counts} | {mode_counts}\n{'=' * 64}",
                flush=True,
            )
            returncode = run_with_filtered_stderr(
                [sys.executable, "-m", "training.worker", "--settings", settings_file, "--chunk", str(chunk_index),
                 "--state-dir", state_dir, "--result", result_path],
                str(ROOT), _worker_environment(),
            )
            if returncode != 0:
                raise RuntimeError(f"Training chunk {chunk_index + 1}/{len(chunks)} failed with exit code {returncode}.")
            with open(result_path, encoding="utf-8") as handle:
                results.append(json.load(handle))
    if not (ROOT / (settings["weights_path"] + ".index")).is_file():
        raise RuntimeError("Training exited without its expected checkpoint.")
    (ROOT / "saved_history").mkdir(exist_ok=True)
    record = {"settings": settings, "chunks": results}
    (ROOT / "saved_history" / f"{settings['run_name']}.json").write_text(json.dumps(record, indent=1) + "\n",
                                                                         encoding="utf-8")
    print(f"\nSaved weights to {settings['weights_path']}", flush=True)


def run_stages(stages):
    for settings in stages:
        if (ROOT / (settings["weights_path"] + ".index")).exists():
            raise ValueError(f"{settings['weights_path']} already exists. Choose a new --label; "
                             "checkpoints are not silently overwritten.")
    for settings in stages:
        run_stage(settings)
