"""Compiled, shared-memory expert generation for the 3D cloud backend."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import os
import time

import numpy as np


from runtime.cloud.compiled_expert_3d import (
    generate_fixed_trajectory,
    generate_online_trajectory,
    warm_up_compiled_expert_3d,
)
from modules.perturbations import NORMAL_MODE, config_for_mode


_OCTANT_POSITIVE_AXES = (
    (True, True, True),
    (False, True, True),
    (False, False, True),
    (True, False, True),
    (True, True, False),
    (False, True, False),
    (False, False, False),
    (True, False, False),
)


def _octant_bounds(bounds, octant):
    """Split arbitrary 3D bounds at their midpoints using project octants."""
    pairs = np.asarray(bounds, dtype=np.float64).reshape(3, 2)
    midpoint = pairs.mean(axis=1)
    selected = []
    for axis, positive in enumerate(_OCTANT_POSITIVE_AXES[int(octant)]):
        lower, upper = pairs[axis]
        middle = midpoint[axis]
        selected.extend((middle, upper) if positive else (lower, middle))
    return tuple(float(value) for value in selected)


def _run_timed_generation(payload):
    generator, config = payload
    started = time.perf_counter()
    trajectory = generator(config)
    return trajectory, time.perf_counter() - started


def build_chunk_configs(
    args,
    chunk_index,
    octant_counts,
    *,
    task,
    validation=False,
    mode_counts=None,
):
    """Generate one fixed- or online-goal chunk without worker processes.

    Each trajectory is an independently scheduled task, so heterogeneous
    rollout lengths are balanced dynamically across all requested CPU
    workers.  Numba releases the GIL inside the simulation kernel; the thread
    pool therefore shares results directly in RAM instead of pickling them
    between hundreds of heavyweight Python processes.
    """
    if task not in ("fixed_waypoints", "online_goals"):
        raise ValueError(f"Unsupported compiled 3D task: {task}")

    total = int(sum(octant_counts.values()))
    if total < 1:
        raise ValueError("A compiled cloud chunk must contain a trajectory.")
    seed_offset = 50_000_021 if validation else 0
    mode_counts = (
        {NORMAL_MODE: total}
        if validation or mode_counts is None
        else {str(mode): int(count) for mode, count in mode_counts.items()}
    )
    modes = [mode for mode, count in mode_counts.items() for _ in range(count)]
    if len(modes) != total:
        raise ValueError("Perturbation mode counts must match the chunk size.")
    np.random.default_rng(args.generation_seed + chunk_index).shuffle(modes)

    configs = []
    task_index = 0
    for octant, count in sorted(octant_counts.items()):
        for _ in range(int(count)):
            seed = int(
                (
                    args.generation_seed
                    + seed_offset
                    + int(chunk_index) * 1_000_003
                    + task_index
                )
                % (2**32 - 1)
            )
            common = {
                "seed": seed,
                "octant": int(octant),
                "n_boids": int(args.n_boids),
                "perception": float(args.perception),
                "crowding": float(getattr(args, "crowding", 0.02)),
                "init_scatter": float(getattr(args, "init_scatter", 0.325)),
                "perturbation": asdict(
                    config_for_mode(args, modes[task_index])
                ),
            }
            if task == "online_goals":
                common.update(
                    {
                        "n_waypoints": int(args.goal_waypoints_per_episode),
                        "start_bounds": _octant_bounds(
                            args.start_bounds, octant
                        ),
                        "goal_bounds": tuple(args.goal_bounds),
                        "goal_min_distance": float(args.goal_min_distance),
                        "goal_arrival_radius": float(
                            args.goal_arrival_radius
                        ),
                        "expert_max_steps": int(args.expert_max_steps),
                    }
                )
            else:
                common.update(
                    {
                        "goal_exclusion_size": float(
                            args.goal_exclusion_size
                        ),
                        # The fixed-waypoint expert has no ordinary hard
                        # horizon.  This high guard only catches pathologies.
                        "expert_max_steps": 50_000,
                        "goal_arrival_radius": float(args.goal_arrival_radius),
                    }
                )
            configs.append(common)
            task_index += 1
    return configs


def generate_compiled_chunk(
    args, chunk_index, octant_counts, *, task, validation=False, mode_counts=None
):
    """Run the shared episode configurations concurrently without disk caching."""
    configs = build_chunk_configs(args, chunk_index, octant_counts, task=task,
                                  validation=validation, mode_counts=mode_counts)
    total = len(configs)
    available_workers = max(1, (os.cpu_count() or 2) - 2)
    workers = max(1, min(int(args.generation_workers) or available_workers, total))

    split = "validation" if validation else "training"
    print(
        f">>> Compiled 3D: generating {total} {split} trajectories "
        f"as {len(configs)} dynamic tasks on {workers} CPU workers...",
        flush=True,
    )
    started = time.perf_counter()
    compilation_seconds = warm_up_compiled_expert_3d()
    generator = (
        generate_online_trajectory
        if task == "online_goals"
        else generate_fixed_trajectory
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        generated = list(
            executor.map(
                _run_timed_generation,
                ((generator, config) for config in configs),
            )
        )

    trajectories = [item[0] for item in generated]
    durations = [item[1] for item in generated]
    centers = np.stack(
        [trajectory.sampled_center for trajectory in trajectories]
    ).astype(np.float32, copy=False)
    center_octants = np.asarray(
        [config["octant"] for config in configs], dtype=np.int8
    )
    records = [
        {
            "octant": int(config["octant"]),
            "seed": int(config["seed"]),
            "steps": int(trajectory.steps),
            "seconds": float(seconds),
        }
        for config, trajectory, seconds in zip(
            configs, trajectories, durations
        )
    ]
    elapsed = time.perf_counter() - started
    print(
        f">>> Compiled 3D: {split} trajectories ready in "
        f"{elapsed:.1f}s (Numba warm-up {compilation_seconds:.1f}s), held in memory.",
        flush=True,
    )
    return trajectories, centers, center_octants, records
