"""Trajectory figures of a trained GNCA, in 2D and 3D.

Rolls the GNCA out from evaluation starts, with the expert settings of the experiment's
preset, and draws them:

* ``tubular``: the mean path over the rollouts with its 99% tube, one random rollout, and
  up to 50 rollouts together (fixed-waypoint experiments, whose rollouts share a route);
* ``per_boid``: the path of every agent in one random rollout;
* ``multi_tubular``: several rollouts as separate tubes;
* ``individual``: one figure per rollout, ranked by tube radius.

    python -m evaluation.visualization.trajectories fixed2d --weights saved_models/fixed2d_s<S> --mode individual
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
from shapely.geometry import LineString, MultiPolygon

from evaluation import paired_rollouts as rollouts
from modules.experts import fixed_waypoints
from modules.presets import NAMES

PATH_COLOR = "#8B0000"
MODES = ("tubular", "per_boid", "multi_tubular", "individual")


def tube_stats(trajectory, dimension):
    """Centroid path of one rollout and the 99th temporal percentile of its 95th agent radius."""
    positions = trajectory[:, :, :dimension]
    centroid = positions.mean(axis=1)
    distances = np.linalg.norm(positions - centroid[:, None, :], axis=-1)
    return centroid, float(np.percentile(np.percentile(distances, 95, axis=1), 99))


def _axes(dimension):
    figure = plt.figure(figsize=(7, 7) if dimension == 2 else (10, 8))
    if dimension == 2:
        axis = figure.add_subplot(111)
        axis.set_aspect("equal")
    else:
        axis = figure.add_subplot(111, projection="3d")
        axis.set_zlabel("Z")
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    return figure, axis


def _draw_tube(axis, centroid, radius, color, label=None, sections=30):
    """A path with its tube: a buffered band in 2D, circular cross-sections in 3D."""
    if centroid.shape[1] == 2:
        tube = LineString(centroid).buffer(radius)
        if isinstance(tube, MultiPolygon):
            tube = max(tube.geoms, key=lambda part: part.area)
        axis.fill(*tube.exterior.xy, color=color, alpha=0.3)
        axis.plot(centroid[:, 0], centroid[:, 1], color=color, lw=1.5, label=label)
        return
    axis.plot(*centroid.T, color=color, lw=2.5, label=label, zorder=10)
    theta = np.linspace(0, 2 * np.pi, 16)
    for point in centroid[:: max(1, len(centroid) // sections)]:
        circle = (point[0] + radius * np.cos(theta), point[1] + radius * np.sin(theta), np.full_like(theta, point[2]))
        axis.plot(*circle, color=color, lw=0.8, alpha=0.4)


def _finish(figure, axis, goals, path, legend=True):
    axis.scatter(*np.asarray(goals).T, c="red", marker="*", s=150, zorder=15, label="Goals")
    if legend:
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)
    print(f"Saved {path}", flush=True)


def plot_mean(trajectories, goals, dimension, path):
    """Mean path over rollouts, cropped to the shortest, with the 99th percentile tube of the rollouts."""
    length = min(len(trajectory) for trajectory in trajectories)
    centroids = np.stack([trajectory[:length, :, :dimension].mean(axis=1) for trajectory in trajectories])
    mean_path = centroids.mean(axis=0)
    radius = float(np.percentile([tube_stats(trajectory, dimension)[1] for trajectory in trajectories], 99))
    figure, axis = _axes(dimension)
    _draw_tube(axis, mean_path, radius, PATH_COLOR, f"Mean path (r={radius:.2f})")
    _finish(figure, axis, goals, path)


def plot_runs(trajectories, goals, dimension, path, indices, legend=True):
    """The chosen rollouts as separate tubes on one canvas."""
    colors = cm.hsv(np.linspace(0, 0.9, len(indices)))
    figure, axis = _axes(dimension)
    for color, index in zip(colors, indices):
        centroid, radius = tube_stats(trajectories[index], dimension)
        _draw_tube(axis, centroid, radius, color, f"Run {index}", sections=20)
    _finish(figure, axis, goals, path, legend)


def plot_per_boid(trajectory, goals, dimension, path):
    """The path of every agent of one rollout."""
    figure, axis = _axes(dimension)
    for agent in range(trajectory.shape[1]):
        axis.plot(*trajectory[:, agent, :dimension].T, color=PATH_COLOR, lw=0.6, alpha=0.4)
    _finish(figure, axis, goals, path)


def plot_individual(trajectories, goals_by_run, dimension, output_dir):
    """One figure per rollout, ranked by tube radius, greatest first."""
    stats = [tube_stats(trajectory, dimension) for trajectory in trajectories]
    order = sorted(range(len(trajectories)), key=lambda index: stats[index][1], reverse=True)
    for rank, index in enumerate(order):
        centroid, radius = stats[index]
        figure, axis = _axes(dimension)
        _draw_tube(axis, centroid, radius, PATH_COLOR, f"Run {index} (r={radius:.3f})")
        axis.set_title(f"Rank {rank + 1:02d} | Run {index:02d} | r={radius:.3f}")
        _finish(figure, axis, goals_by_run[index], os.path.join(output_dir, f"rank{rank + 1:02d}_run{index:02d}.pdf"))


def simulate(args, model):
    """GNCA rollouts from the evaluation starts, as (steps, agents, dimension) position arrays, with their goals."""
    fixed_goals = fixed_waypoints(args.dimension)
    centers, _ = rollouts._sample_centers(args, np.random.default_rng(args.seed), fixed_goals)
    trajectories, goals_by_run = [], []
    for run_index, center in enumerate(centers):
        init_rng = np.random.default_rng(np.random.SeedSequence([args.seed, run_index, 0x1A117]))
        initial_state = rollouts._initial_state(center, args.n_boids, args.dimension, init_rng, args.init_scatter)
        goals = rollouts._goals_for_run(args, initial_state, run_index, fixed_goals)
        result = rollouts._rollout(
            controller="gnca", model=model, boids=rollouts._boids(args), initial_state=initial_state, goals=goals,
            plan=rollouts._event_plan(args, run_index), mode="baseline", count=0, args=args, record_every=1,
        )
        frames = [initial_state[:, : args.dimension]] + [frame[1] for frame in result["frames"]]
        trajectories.append(np.stack(frames))
        goals_by_run.append(goals)
        print(f"Run {run_index + 1}/{len(centers)}: {result['reached']}/{result['requested']} goals, "
              f"{result['steps']} steps", flush=True)
    return trajectories, goals_by_run


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", choices=NAMES)
    parser.add_argument("--weights", required=True, help="Checkpoint prefix of the model.")
    parser.add_argument("--mode", choices=MODES, default="tubular")
    parser.add_argument("--centers-per-region", type=int, default=None,
                        help="Starts per quadrant or octant (default: the preset's).")
    parser.add_argument("--max-steps", type=int, default=None, help="Horizon (default: the preset's).")
    parser.add_argument("--seed", type=int, default=None, help="Seed of the starts (default: the preset's).")
    parser.add_argument("--n-show", type=int, default=10, help="Rollouts drawn by multi_tubular.")
    parser.add_argument("--output-dir", default=None)
    options = parser.parse_args(argv)
    argv = ["--experiment", options.experiment, "--weights", options.weights, "--conditions", "baseline",
            "--population_changes", "1", "--verbose_every", "0"]
    for flag, value in (("--centers_per_region", options.centers_per_region), ("--max_steps", options.max_steps),
                        ("--seed", options.seed)):
        if value is not None:
            argv += [flag, str(value)]
    args = rollouts.parse_args(argv)
    if options.mode == "tubular" and args.goal_conditioned:
        parser.error("tubular averages rollouts over one route; use another mode for an online experiment")
    output_dir = options.output_dir or f"results/figures/{options.experiment}_{options.mode}"
    os.makedirs(output_dir, exist_ok=True)

    trajectories, goals_by_run = simulate(args, rollouts._load_checkpoint(args))
    dimension = args.dimension
    rng = np.random.default_rng(args.seed)
    if options.mode == "individual":
        plot_individual(trajectories, goals_by_run, dimension, output_dir)
    elif options.mode == "per_boid":
        index = int(rng.integers(len(trajectories)))
        plot_per_boid(trajectories[index], goals_by_run[index], dimension, os.path.join(output_dir, "per_boid.pdf"))
    elif options.mode == "multi_tubular":
        indices = rng.permutation(len(trajectories))[: options.n_show].tolist()
        goals = np.concatenate([goals_by_run[index] for index in indices])
        plot_runs(trajectories, goals, dimension, os.path.join(output_dir, "multi_tube.pdf"), indices)
    else:
        plot_mean(trajectories, goals_by_run[0], dimension, os.path.join(output_dir, "tube_mean.pdf"))
        single = [int(rng.integers(len(trajectories)))]
        plot_runs(trajectories, goals_by_run[0], dimension, os.path.join(output_dir, "tube_single.pdf"), single)
        indices = rng.permutation(len(trajectories))[:50].tolist()
        plot_runs(trajectories, goals_by_run[0], dimension, os.path.join(output_dir, "tube_runs.pdf"), indices,
                  legend=False)


if __name__ == "__main__":
    main()
