"""Filmstrip of slab removal in one 2D fixed-waypoint GNCA rollout.

Simulates one evaluation start of a trained fixed 2D model and draws the flock
at several times around the removal step, one row per removal count, with the
removed agents marked. The start, removal step and slab direction are drawn as
in ``evaluation.paired_rollouts`` with the same ``--seed`` and ``--run``.

    python -m evaluation.visualization.slab_filmstrip --weights saved_models/best_weights_<run tag>
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluation import paired_rollouts as rollouts
from modules.boids import BOIDS_GOAL_POSITIONS

OFFSETS = (-100, 0, 100, 300, 700)  # steps relative to the removal step
RECORD_EVERY = 5
HALF_WIDTH = 0.45  # half side of each panel, centered on the flock


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", required=True, help="Checkpoint prefix of a trained fixed 2D model.")
    parser.add_argument("--removed", type=int, nargs="+", default=[50], help="Agents removed, one row each.")
    parser.add_argument("--seed", type=int, default=123, help="Seed of the evaluation starts.")
    parser.add_argument("--run", type=int, default=0, help="Index of the start to simulate.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    return parser.parse_args()


def _rollout_args(options):
    return argparse.Namespace(
        dimension=2, task="fixed_waypoints", run_tag=Path(options.weights).name,
        weights_path=options.weights, n_boids=100, perception=0.1, crowding=0.02, init_scatter=0.325,
        max_steps=5000, success_threshold=0.5, centers_per_region=10, regions=[0, 1, 2, 3], exclusion=0.5,
        online_goal_count=5, goal_min_distance=2.5, goal_bounds=[-4.5, 4.5, -4.5, 4.5],
        population_changes=list(options.removed), removal_pattern="slab",
        event_step_min=500, event_step_max=1000, seed=options.seed, verbose_every=0,
    )


def simulate(args, model, run):
    goals = BOIDS_GOAL_POSITIONS.copy()
    centers, _ = rollouts._sample_centers(args, np.random.default_rng(args.seed), goals)
    init_rng = np.random.default_rng(np.random.SeedSequence([args.seed, run, 0x1A117]))
    initial_state = rollouts._initial_state(centers[run], args.n_boids, 2, init_rng)
    plan = rollouts._event_plan(args, run)
    frames = {}
    for count in args.population_changes:
        print(f"Simulating slab removal of {count} agents (removal step {plan.step})")
        result = rollouts._rollout(
            controller="gnca", model=model, boids=rollouts._boids(args), initial_state=initial_state,
            goals=goals, plan=plan, mode="removal", count=count, args=args,
            record_every=RECORD_EVERY, stop_step=plan.step + max(OFFSETS),
        )
        frames[count] = {frame[0]: frame for frame in result["frames"]}
    return plan, frames


def draw(plan, frames, output_dir):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
    counts = list(frames)
    fig, axes = plt.subplots(len(counts), len(OFFSETS), figsize=(6.5, 1.45 * len(counts)), squeeze=False)
    for row, count in enumerate(counts):
        for column, offset in enumerate(OFFSETS):
            axis = axes[row, column]
            target = plan.step + offset
            step = min(frames[count], key=lambda s: (abs(s - target), s != plan.step))
            positions, mask = frames[count][step][1], frames[count][step][2]
            if mask is None:
                mask = np.zeros(len(positions), dtype=bool)
            axis.scatter(*positions[~mask].T, s=4, c="#228833", edgecolors="none")
            axis.scatter(*positions[mask].T, s=9, c="#CC3311", marker="x", linewidths=0.7)
            center = positions[~mask].mean(axis=0)
            axis.set_xlim(center[0] - HALF_WIDTH, center[0] + HALF_WIDTH)
            axis.set_ylim(center[1] - HALF_WIDTH, center[1] + HALF_WIDTH)
            axis.set_aspect("equal")
            axis.set_xticks([])
            axis.set_yticks([])
            if row == 0:
                axis.set_title("removal" if offset == 0 else f"{offset:+d} steps", fontsize=8)
            if column == 0 and len(counts) > 1:
                axis.set_ylabel(f"{count} removed", fontsize=8)
    fig.tight_layout(pad=0.3, w_pad=0.3, h_pad=0.3)
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(output_dir / f"slab_removal_filmstrip.{extension}", dpi=200, bbox_inches="tight",
                    pad_inches=0.03)
    plt.close(fig)
    print(f"Saved {output_dir / 'slab_removal_filmstrip.pdf'}")


def main():
    options = _parse_args()
    args = _rollout_args(options)
    model = rollouts._load_checkpoint(args)
    plan, frames = simulate(args, model, options.run)
    draw(plan, frames, options.output_dir)


if __name__ == "__main__":
    main()
