"""Backend-independent chunk allocation helpers."""

import math


def proportional_chunk_allocations(total_counts, chunk_size):
    """Allocate proportional chunks while preserving exact octant totals."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    if any(int(count) < 0 for count in total_counts.values()):
        raise ValueError("Chunk allocation counts cannot be negative.")

    remaining = {int(octant): int(count) for octant, count in total_counts.items()}
    allocations = []
    while sum(remaining.values()) > 0:
        current_size = min(int(chunk_size), sum(remaining.values()))
        total_remaining = sum(remaining.values())
        exact = {
            octant: current_size * count / total_remaining
            for octant, count in remaining.items()
        }
        allocation = {
            octant: min(remaining[octant], math.floor(exact[octant]))
            for octant in remaining
        }
        unassigned = current_size - sum(allocation.values())
        while unassigned:
            candidates = [
                octant
                for octant in remaining
                if allocation[octant] < remaining[octant]
            ]
            chosen = max(
                candidates,
                key=lambda octant: (
                    exact[octant] - allocation[octant],
                    remaining[octant],
                ),
            )
            allocation[chosen] += 1
            unassigned -= 1
        allocation = {
            octant: count for octant, count in allocation.items() if count
        }
        allocations.append(allocation)
        for octant, count in allocation.items():
            remaining[octant] -= count
    return allocations
