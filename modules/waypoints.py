"""Waypoint sampling and transition state for goal-conditioned experiments."""

from dataclasses import dataclass, field

import numpy as np

def goal_arrival_distance(positions, goal):
    """Distance from the centroid of ``positions`` (N, d) to ``goal``.

    The expert switches to its next waypoint, and every evaluation counts a
    waypoint as reached, when this distance is at most the arrival radius.
    """
    positions = np.asarray(positions, dtype=np.float64)
    goal = np.asarray(goal, dtype=np.float64)
    return float(np.linalg.norm(positions.mean(axis=0) - goal))


def sample_separated_waypoint(
    rng,
    reference,
    *,
    bounds=(-4.5, 4.5, -4.5, 4.5),
    min_distance=2.5,
    max_attempts=10_000,
):
    """Sample a 2D or 3D waypoint separated from ``reference``.

    ``bounds`` is a flat sequence of lower/upper pairs, one pair per spatial
    dimension.  The existing 2D API therefore remains ``(x_min, x_max,
    y_min, y_max)``, while 3D callers append ``(z_min, z_max)``.
    """
    reference = np.asarray(reference, dtype=np.float32)
    if reference.ndim != 1 or reference.shape[0] not in (2, 3):
        raise ValueError(
            "reference must have shape (2,) or (3,), "
            f"got {reference.shape}"
        )

    bounds = tuple(map(float, bounds))
    expected_bounds = 2 * reference.shape[0]
    if len(bounds) != expected_bounds:
        raise ValueError(
            f"bounds must contain {expected_bounds} values for a "
            f"{reference.shape[0]}D reference, got {len(bounds)}"
        )
    bound_pairs = np.asarray(bounds, dtype=np.float64).reshape(-1, 2)
    if np.any(bound_pairs[:, 0] >= bound_pairs[:, 1]):
        raise ValueError("waypoint bounds must have positive width")
    if min_distance < 0:
        raise ValueError("min_distance cannot be negative")

    for _ in range(int(max_attempts)):
        waypoint = np.array(
            [rng.uniform(lower, upper) for lower, upper in bound_pairs],
            dtype=np.float32,
        )
        if np.linalg.norm(waypoint - reference) >= min_distance:
            return waypoint
    raise RuntimeError(
        "Could not sample a sufficiently separated waypoint. Reduce "
        "--goal_min_distance or enlarge --goal_bounds."
    )


def sample_waypoint_sequence(
    rng,
    initial_reference,
    *,
    n_waypoints,
    bounds,
    min_distance,
):
    """Pre-sample a waypoint sequence using the online manager's rule.

    The first waypoint is separated from the initial flock reference. Every
    later waypoint is separated from the preceding waypoint. Pre-sampling
    makes it possible to evaluate two controllers against exactly the same
    complete sequence, even if one controller fails before reaching a goal.
    """
    if int(n_waypoints) < 1:
        raise ValueError("n_waypoints must be positive")
    reference = np.asarray(initial_reference, dtype=np.float32)
    waypoints = []
    for _ in range(int(n_waypoints)):
        waypoint = sample_separated_waypoint(
            rng,
            reference,
            bounds=bounds,
            min_distance=min_distance,
        )
        waypoints.append(waypoint)
        reference = waypoint
    return np.asarray(waypoints, dtype=np.float32)


@dataclass
class OnlineWaypointManager:
    """Manage externally supplied goals without controlling agent motion."""

    rng: np.random.Generator
    n_waypoints: int = 2
    bounds: tuple = (-4.5, 4.5, -4.5, 4.5)
    min_distance: float = 2.5
    arrival_radius: float = 0.5
    waypoints: list = field(default_factory=list)
    completed: int = 0

    def __post_init__(self):
        if self.n_waypoints < 1:
            raise ValueError("n_waypoints must be positive")
        if self.arrival_radius <= 0:
            raise ValueError("arrival_radius must be positive")

    @property
    def current_goal(self):
        if not self.waypoints or self.completed >= len(self.waypoints):
            return None
        return self.waypoints[self.completed]

    @property
    def finished(self):
        return self.completed >= self.n_waypoints

    def start(self, flock_centroid):
        """Sample the initial goal, separated from the initial centroid."""
        self.waypoints = [
            sample_separated_waypoint(
                self.rng,
                flock_centroid,
                bounds=self.bounds,
                min_distance=self.min_distance,
            )
        ]
        self.completed = 0
        return self.current_goal.copy()

    def centroid_distance(self, positions):
        """Distance from the active-agent centroid to the active goal."""
        if self.current_goal is None:
            return np.inf
        positions = np.asarray(positions)
        if positions.ndim != 2 or positions.shape[0] == 0:
            return np.inf
        return float(np.linalg.norm(positions.mean(axis=0) - self.current_goal))

    def update(self, positions):
        """Advance after arrival and return ``(goal, switched, finished)``."""
        if self.current_goal is None:
            raise RuntimeError("Call start() before update().")
        if self.centroid_distance(positions) > self.arrival_radius:
            return self.current_goal.copy(), False, False

        previous = self.current_goal.copy()
        self.completed += 1
        if self.finished:
            return previous, False, True

        next_goal = sample_separated_waypoint(
            self.rng,
            previous,
            bounds=self.bounds,
            min_distance=self.min_distance,
        )
        self.waypoints.append(next_goal)
        return next_goal.copy(), True, False


def goal_conditioned_state(physical_state, goal):
    """Return per-agent ``[position, velocity, goal - position]`` features.

    The spatial dimension is inferred from ``goal``.  A 2D physical state has
    four columns and produces six conditioned features; a 3D physical state
    has six columns and produces nine.
    """
    physical_state = np.asarray(physical_state, dtype=np.float32)
    goal = np.asarray(goal, dtype=np.float32)
    if goal.ndim != 1 or goal.shape[0] not in (2, 3):
        raise ValueError(f"goal must have shape (2,) or (3,), got {goal.shape}")
    spatial_dims = goal.shape[0]
    expected_features = 2 * spatial_dims
    if (
        physical_state.ndim != 2
        or physical_state.shape[-1] != expected_features
    ):
        raise ValueError(
            "physical_state must have shape "
            f"(agents, {expected_features}) for a {spatial_dims}D goal, "
            f"got {physical_state.shape}"
        )
    relative_goal = goal[None, :] - physical_state[:, :spatial_dims]
    return np.concatenate((physical_state, relative_goal), axis=-1).astype(
        np.float32, copy=False
    )
