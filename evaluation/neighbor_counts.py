"""Neighbor counts along GNCA rollouts.

Runs the baseline rollouts of ``evaluation.paired_rollouts`` for one checkpoint
and swarm size, samples the agent positions every ``--record_every`` steps and
counts, for every agent in every sampled frame, the other agents inside the
perception radius. The report gives the distribution of these counts, the share
of isolated agent-steps (no neighbor at all) and the per-frame means, next to
the completion count of the rollouts. Starts are sampled as in the paired
evaluation; the starting box side scales with ``(n_boids / 100) ** (1 / d)``
unless ``--init_scatter`` is given, so a smaller swarm starts at the training
density.

Example (online 2D checkpoint, 25 agents, 3 starts per quadrant):

    python -m evaluation.neighbor_counts --experiment online2d \
        --weights saved_models/online2d_s<S> --n_boids 25 --centers_per_region 3 \
        --output_dir results/neighbor_counts/online2d_n25
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np

from evaluation import paired_rollouts as pr
from modules.experts import fixed_waypoints
from modules.presets import NAMES, load_preset


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", choices=NAMES, required=True,
                        help="Selects the task and the expert settings from its preset.")
    parser.add_argument("--weights", required=True, help="TensorFlow checkpoint prefix.")
    parser.add_argument("--n_boids", type=int, default=100)
    parser.add_argument("--init_scatter", type=float, default=None,
                        help="Side of the starting box; default keeps the preset's density at 100 agents.")
    parser.add_argument("--centers_per_region", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=None, help="Horizon (default: the preset's).")
    parser.add_argument("--record_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=None, help="Seed of the starts (default: the preset's).")
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args()


def main():
    options = _parse_args()
    preset = load_preset(options.experiment)
    dimension = preset["dimension"]
    scatter = options.init_scatter
    if scatter is None:
        scatter = preset["expert"]["init_scatter"] * (options.n_boids / 100.0) ** (1.0 / dimension)
    argv = ["--experiment", options.experiment, "--weights", options.weights, "--n_boids", str(options.n_boids),
            "--init_scatter", f"{scatter:.6f}", "--conditions", "baseline", "--population_changes", "1",
            "--centers_per_region", str(options.centers_per_region), "--verbose_every", "0", "--skip_plots",
            "--output_dir", options.output_dir]
    for flag, value in (("--max_steps", options.max_steps), ("--seed", options.seed)):
        if value is not None:
            argv += [flag, str(value)]
    args = pr.parse_args(argv)
    fixed_goals = fixed_waypoints(dimension)
    centers, _ = pr._sample_centers(args, np.random.default_rng(args.seed), fixed_goals)
    model = pr._load_checkpoint(args)

    counts = []
    frame_means = []
    isolated = 0
    total = 0
    completed = 0
    per_run = []
    for run_index, center in enumerate(centers):
        init_rng = np.random.default_rng(np.random.SeedSequence([args.seed, run_index, 0x1A117]))
        initial_state = pr._initial_state(center, args.n_boids, dimension, init_rng, args.init_scatter)
        goals = pr._goals_for_run(args, initial_state, run_index, fixed_goals)
        plan = pr._event_plan(args, run_index)
        boids = pr._boids(args)
        out = pr._rollout(controller="gnca", model=model, boids=boids, initial_state=initial_state,
                          goals=goals, plan=plan, mode="baseline", count=0, args=args,
                          record_every=options.record_every)
        completed += int(out["completed"])
        run_counts = []
        for _, positions, _ in out["frames"]:
            adjacency = boids.get_neighbors(positions)
            adjacency = np.asarray(adjacency.todense()) if hasattr(adjacency, "todense") else np.asarray(adjacency)
            adjacency = adjacency.astype(bool)
            np.fill_diagonal(adjacency, False)
            per_agent = adjacency.sum(axis=1)
            run_counts.extend(per_agent.tolist())
            isolated += int((per_agent == 0).sum())
            total += len(per_agent)
            frame_means.append(float(per_agent.mean()))
        counts.extend(run_counts)
        per_run.append((run_index, int(out["completed"]), float(np.median(run_counts)), float(np.mean(run_counts))))
        print(f"run {run_index}: completed={bool(out['completed'])} median neighbors {np.median(run_counts):.0f}", flush=True)

    counts = np.asarray(counts)
    lines = [
        f"checkpoint: {options.weights}",
        f"task: {args.task} {dimension}D, agents: {options.n_boids}, starting box side: {scatter:.4f}",
        f"rollouts: {len(centers)}, completed: {completed}/{len(centers)}, positions sampled every {options.record_every} steps",
        f"neighbors per agent (within perception {args.perception:g}, self excluded): median {np.median(counts):.0f}, "
        f"mean {counts.mean():.2f}, IQR [{np.percentile(counts, 25):.0f}, {np.percentile(counts, 75):.0f}], "
        f"min {counts.min()}, max {counts.max()}",
        f"isolated agent-steps: {100.0 * isolated / total:.2f}%",
        f"per-frame mean neighbors: min {min(frame_means):.2f}, median {np.median(frame_means):.2f}, max {max(frame_means):.2f}",
    ]
    output = Path(options.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "neighbor_counts_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(output / "neighbor_counts_per_run.csv", "w", encoding="utf-8") as stream:
        stream.write("run,completed,median_neighbors,mean_neighbors\n")
        for row in per_run:
            stream.write(f"{row[0]},{row[1]},{row[2]:.3f},{row[3]:.3f}\n")
    np.save(output / "neighbor_counts.npy", counts)
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
