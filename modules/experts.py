"""One interface over the 2D and 3D reference experts, which stay a pair of classes."""

from modules.boids import BOIDS_GOAL_POSITIONS, Boids
from modules.boids_3d import BOIDS_GOAL_POSITIONS_3D, Boids3D


def reference_expert(dimension, **settings):
    """A Boids expert of the given dimension; ``settings`` are its constructor arguments."""
    return (Boids if int(dimension) == 2 else Boids3D)(**settings)


def fixed_waypoints(dimension):
    """The fixed waypoint sequence of the fixed-waypoint task."""
    return (BOIDS_GOAL_POSITIONS if int(dimension) == 2 else BOIDS_GOAL_POSITIONS_3D).copy()
