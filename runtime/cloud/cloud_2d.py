"""Single supported 2D cloud pipeline.

Expert rollouts are Numba-compiled, adjacency is bit-packed in RAM, and the
shared synchronous trainer uses every visible GPU.
"""

from runtime.cloud.cloud_bitpacked_2d import dataset_from_bitpacked_trajectories
from runtime.cloud.cloud_compiled_2d import generate_compiled_online_goal_chunk
from runtime.cloud.cloud_training import fit_distributed


generate_online_goal_chunk = generate_compiled_online_goal_chunk
dataset_from_trajectories = dataset_from_bitpacked_trajectories


__all__ = (
    "dataset_from_trajectories",
    "fit_distributed",
    "generate_online_goal_chunk",
)
