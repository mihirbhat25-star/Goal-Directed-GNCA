"""Drift perturbations of the expert during data generation.

Every ``interval`` steps each agent of a perturbed trajectory is selected
independently with probability ``probability``, at most ``max_agents`` of them per
event, and the selected agents receive bounded uniform noise on their position
(``position_bound`` per axis) and velocity (``velocity_bound`` per axis). The expert
then evolves the disturbed state, so the model also sees it recover. The compiled
experts apply the events; this module describes them and allocates exact trajectory
counts over the chunks.
"""

from __future__ import annotations

import math


NORMAL_MODE = "normal"
DRIFT_MODE = "drift"
PERTURBATION_MODES = (NORMAL_MODE, DRIFT_MODE)


def episode_perturbation(drift, mode, dimension):
    """The perturbation entry of one compiled expert's episode configuration."""
    return {
        "mode": mode,
        "interval": int(drift["interval"]),
        "probability": float(drift["probability"]),
        "drift_max_per_event": int(drift["max_agents"]),
        "drift_bounds": (float(drift["position_bound"]),) * int(dimension),
        "drift_perturb_velocity": True,
        "drift_velocity_bounds": (float(drift["velocity_bound"]),) * int(dimension),
    }


def allocate_mode_chunks(counts, chunk_sizes):
    """Split exact mode totals over predetermined chunk sizes proportionally."""
    remaining = {mode: int(counts.get(mode, 0)) for mode in PERTURBATION_MODES}
    allocations = []
    for requested_size in chunk_sizes:
        requested_size = int(requested_size)
        total_remaining = sum(remaining.values())
        if requested_size < 0 or requested_size > total_remaining:
            raise ValueError("Invalid perturbation chunk size.")
        if requested_size == 0:
            allocations.append({})
            continue
        exact = {
            mode: requested_size * count / total_remaining
            for mode, count in remaining.items()
        }
        allocation = {
            mode: min(remaining[mode], math.floor(exact[mode]))
            for mode in remaining
        }
        while sum(allocation.values()) < requested_size:
            candidates = [
                mode
                for mode in PERTURBATION_MODES
                if allocation[mode] < remaining[mode]
            ]
            selected = max(
                candidates,
                key=lambda mode: (
                    exact[mode] - allocation[mode],
                    remaining[mode],
                ),
            )
            allocation[selected] += 1
        allocation = {mode: count for mode, count in allocation.items() if count}
        allocations.append(allocation)
        for mode, count in allocation.items():
            remaining[mode] -= count
    if sum(remaining.values()):
        raise ValueError("Chunk sizes did not consume every perturbation trajectory.")
    return allocations
