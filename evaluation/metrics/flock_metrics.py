"""Scoring of ordered waypoint visits by the flock centroid."""

from __future__ import annotations

import numpy as np


class OrderedWaypointProgress:
    """Count centroid visits only while the corresponding goal is active.

    At most one goal is credited per frame, so arrival steps are strictly
    increasing even when goal neighborhoods overlap. Passing an inactive
    waypoint does not give credit that can be redeemed later.
    """

    def __init__(self, goals, threshold=0.5):
        self.goals = np.asarray(goals, dtype=np.float64)
        if (self.goals.ndim != 2 or len(self.goals) == 0
                or not np.all(np.isfinite(self.goals))):
            raise ValueError("goals must be a nonempty, finite coordinate array")
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("goal threshold must be finite and positive")
        self.threshold = float(threshold)
        self.reached = 0
        self.arrival_steps = []
        self.first_hit_steps = np.full(len(self.goals), -1, dtype=int)
        self.closest_distances = np.full(len(self.goals), np.inf)
        self._last_step = -1

    @property
    def completed(self):
        return self.reached == len(self.goals)

    def update(self, centroid, step):
        if self.completed:
            return True
        if step <= self._last_step:
            raise ValueError("Waypoint progress needs strictly increasing frame indices")
        self._last_step = int(step)
        centroid = np.asarray(centroid, dtype=np.float64)
        if centroid.shape != self.goals.shape[1:]:
            raise ValueError("Centroid and goal dimensions differ")
        if not np.all(np.isfinite(centroid)):
            return False
        active = self.reached
        distance = float(np.linalg.norm(centroid - self.goals[active]))
        self.closest_distances[active] = min(self.closest_distances[active], distance)
        if distance <= self.threshold:
            self.first_hit_steps[active] = int(step)
            self.arrival_steps.append(int(step))
            self.reached += 1
        return self.completed
