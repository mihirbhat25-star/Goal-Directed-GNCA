"""Compiled, shared-memory expert generation for fixed- and online-goal 2D training."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import os
import time

import numpy as np

from runtime.cloud.compiled_expert_2d import (
    generate_fixed_trajectory,
    generate_online_trajectory,
    warm_up_compiled_expert_2d,
)
from modules.perturbations import NORMAL_MODE, config_for_mode


_QUADRANT_SIGNS = (
    (True, True),
    (False, True),
    (False, False),
    (True, False),
)




def _quadrant_bounds(bounds, quadrant):
    pairs = np.asarray(bounds, dtype=np.float64).reshape(2, 2)
    midpoints = pairs.mean(axis=1)
    selected = []
    for axis, positive in enumerate(_QUADRANT_SIGNS[int(quadrant)]):
        lower, upper = pairs[axis]
        middle = midpoints[axis]
        selected.extend((middle, upper) if positive else (lower, middle))
    return tuple(float(value) for value in selected)


def _timed_generate(config):
    started = time.perf_counter()
    generator = generate_fixed_trajectory if config.get("task") == "fixed_waypoints" else generate_online_trajectory
    trajectory = generator(config)
    return trajectory, time.perf_counter() - started


def build_chunk_configs(
    args, chunk_index, trajectory_count, *, validation=False, mode_counts=None
):
    """Describe identical expert episodes for disk and in-memory backends."""
    total = int(trajectory_count)
    if total < 1:
        raise ValueError("A compiled cloud chunk must contain a trajectory.")
    seed_offset = 50_000_021 if validation else 0
    fixed = args.task == "fixed_waypoints"
    repeats = int(args.train_set_repeats) if fixed and not validation else 1
    quadrants = (getattr(args, "train_quadrants", None) or [0, 1, 2, 3]) if fixed else [0, 1, 2, 3]

    mode_counts = (
        {NORMAL_MODE: total}
        if validation or mode_counts is None
        else {str(mode): int(count) for mode, count in mode_counts.items()}
    )
    modes = [mode for mode, count in mode_counts.items() for _ in range(count)]
    if len(modes) != total:
        raise ValueError("Perturbation mode counts must match the chunk size.")
    np.random.default_rng(args.seed + chunk_index).shuffle(modes)
    configs = []
    for task_index in range(total):
        episode_index = int(chunk_index) * int(args.chunk_size) + task_index
        center_index = episode_index // repeats
        quadrant = quadrants[center_index % len(quadrants)] if fixed else task_index % 4
        configs.append(
            {
                "task": args.task,
                "center_seed": int((args.seed + seed_offset + center_index) % (2**32 - 1)),
                "goal_exclusion_size": float(getattr(args, "exclusion_size", 0.5)),
                "seed": int(
                    (
                        args.seed
                        + seed_offset
                        + int(chunk_index) * 1_000_003
                        + task_index
                    )
                    % (2**32 - 1)
                ),
                "n_boids": int(args.n_boids),
                "n_waypoints": int(args.goal_waypoints_per_episode),
                "perception": float(args.perception),
                "crowding": float(getattr(args, "crowding", 0.02)),
                "init_scatter": float(getattr(args, "init_scatter", 0.325)),
                "start_bounds": _quadrant_bounds(args.start_bounds, quadrant),
                "goal_bounds": tuple(args.goal_bounds),
                "goal_min_distance": float(args.goal_min_distance),
                "goal_arrival_radius": float(args.goal_arrival_radius),
                "expert_max_steps": int(args.expert_max_steps),
                "start_quadrant": quadrant,
                "perturbation": asdict(config_for_mode(args, modes[task_index])),
            }
        )
    return configs


def generate_compiled_online_goal_chunk(
    args, chunk_index, trajectory_count, *, validation=False, mode_counts=None
):
    """Generate dynamically scheduled Numba trajectories in shared RAM."""
    configs = build_chunk_configs(args, chunk_index, trajectory_count,
                                  validation=validation, mode_counts=mode_counts)
    total = len(configs)
    available = max(1, (os.cpu_count() or 2) - 2)
    workers = max(1, min(int(args.generation_workers) or available, total))

    split = "validation" if validation else "training"
    counts = {
        quadrant: sum(c["start_quadrant"] == quadrant for c in configs)
        for quadrant in range(4)
    }
    print(
        f">>> Compiled 2D: generating {total} {split} trajectories "
        f"as dynamic tasks on {workers} CPU workers; start counts={counts}...",
        flush=True,
    )
    started = time.perf_counter()
    warmup_seconds = warm_up_compiled_expert_2d()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        generated = list(executor.map(_timed_generate, configs))
    trajectories = [item[0] for item in generated]
    centers = np.stack([item.sampled_center for item in trajectories]).astype(
        np.float32, copy=False
    )
    records = [
        {
            "seed": int(config["seed"]),
            "seconds": float(item[1]),
            "steps": int(item[0].steps),
            "start_quadrant": int(config["start_quadrant"]),
        }
        for config, item in zip(configs, generated)
    ]
    print(
        f">>> Compiled 2D: {split} trajectories ready in "
        f"{time.perf_counter() - started:.1f}s (Numba warm-up "
        f"{warmup_seconds:.1f}s), held in memory.",
        flush=True,
    )
    return trajectories, centers, records
