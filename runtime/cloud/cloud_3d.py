"""Single supported 3D cloud pipeline.

Expert rollouts are Numba-compiled, adjacency is bit-packed in RAM, and the
shared synchronous trainer uses every visible GPU.
"""

from runtime.cloud.cloud_bitpacked_3d import dataset_from_bitpacked_trajectories
from runtime.cloud.cloud_compiled_3d import generate_compiled_chunk
from runtime.cloud.cloud_training import fit_distributed


def generate_fixed_waypoint_chunk(
    args,
    chunk_index,
    octant_counts,
    *,
    validation=False,
    mode_counts=None,
):
    return generate_compiled_chunk(
        args,
        chunk_index,
        octant_counts,
        task="fixed_waypoints",
        validation=validation,
        mode_counts=mode_counts,
    )


def generate_online_goal_chunk(
    args,
    chunk_index,
    octant_counts,
    *,
    validation=False,
    mode_counts=None,
):
    return generate_compiled_chunk(
        args,
        chunk_index,
        octant_counts,
        task="online_goals",
        validation=validation,
        mode_counts=mode_counts,
    )


dataset_from_trajectories = dataset_from_bitpacked_trajectories


__all__ = (
    "dataset_from_trajectories",
    "fit_distributed",
    "generate_fixed_waypoint_chunk",
    "generate_online_goal_chunk",
)
