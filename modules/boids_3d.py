"""The reference Boids expert in 3D: the equations the compiled expert implements."""
import numpy as np
import scipy.sparse as sp
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from spektral import utils

from modules.waypoints import goal_arrival_distance


BOIDS_GOAL_POSITIONS_3D = np.array(
    [[-4.0, -4.0, -4.0], [4.0, 4.0, 4.0]],
    dtype=np.float32,
)


class Boids3D:
    def __init__(
        self,
        min_speed=0.0001,
        max_speed=0.01,
        max_force=0.1,
        max_turn=15,
        perception=0.1,
        crowding=0.02,
        init_scatter=0.325,  # Spread of boids around the initial center
        n_boids=100,
        dt=1,
        canvas_scale=1,
        wrap=False,
        limits=True,
        show=False,
        goal_arrival_radius=0.5,  # centroid distance at which the next waypoint becomes active
    ):
        if goal_arrival_radius <= 0:
            raise ValueError('goal_arrival_radius must be positive')
        self.goal_arrival_radius = float(goal_arrival_radius)
        self.min_speed = min_speed
        self.max_speed = max_speed
        self.max_force = max_force
        self.max_turn = max_turn
        self.perception = perception
        self.crowding = crowding
        self.n_boids = n_boids
        self.dt = dt
        self.canvas_scale = canvas_scale
        self.wrap = wrap
        self.limits = limits
        self.init_scatter = init_scatter

        # 3D canvas: [xmin, ymin, zmin, xmax, ymax, zmax]
        self.borders = canvas_scale * np.array([-5, -5, -5, 5, 5, 5])
        self.canonical_goal_positions = BOIDS_GOAL_POSITIONS_3D.copy()
        self.goal_positions = self.canonical_goal_positions.copy()
        self.current_goal = 0
        self.reached_goal = False
        self.goals_completed = 0

        self.show = show
        self.figure = None

    def update_boids(
        self, positions, velocities, return_accel=False, active_mask=None
    ):
        if active_mask is None:
            active_mask = np.ones(len(positions), dtype=bool)
        active_mask = np.asarray(active_mask, dtype=bool)
        positions_before = positions.copy()
        accelerations = np.zeros_like(velocities)

        if self.wrap:
            positions = ((positions + 1) % 2) - 1

        neighbors = self.get_neighbors(positions, active_mask=active_mask)

        SEPARATION_WEIGHT = 0.35
        ALIGNMENT_WEIGHT = 0.35
        COHESION_WEIGHT = 0.001
        GOAL_WEIGHT = 0.01

        accelerations += SEPARATION_WEIGHT * self.get_separation(neighbors, positions)
        accelerations += ALIGNMENT_WEIGHT * self.get_alignment(neighbors, velocities)
        accelerations += COHESION_WEIGHT * self.get_cohesion(neighbors, positions)
        accelerations += GOAL_WEIGHT * self.get_goal(neighbors, positions)

        velocities_new = velocities + accelerations * self.dt
        if self.limits:
            velocities = self.enforce_limits(velocities, velocities_new)
        else:
            velocities = velocities_new

        positions = positions + velocities * self.dt

        # Clip to 3D borders
        for dim in range(3):
            positions[:, dim] = np.clip(positions[:, dim], self.borders[dim], self.borders[dim + 3])

        positions[~active_mask] = positions_before[~active_mask]
        velocities[~active_mask] = 0.0
        accelerations[~active_mask] = 0.0

        if self.show:
            self.plot(positions)

        return (positions, velocities, neighbors) + (
            (accelerations,) if return_accel else ()
        )

    def generate_trajectory(self, positions, velocities):
        """Run the expert through its goals from the given state and return the visited states."""
        self.current_goal = 0
        self.goals_completed = 0
        self.reached_goal = False
        self.goal_positions = self.canonical_goal_positions.copy()
        history = {"positions": [positions], "velocities": [velocities]}
        while not self.check_termination():
            positions, velocities, _ = self.update_boids(positions, velocities)
            history["positions"].append(positions)
            history["velocities"].append(velocities)
            self.update_goal(positions)
        return {key: np.array(values) for key, values in history.items()}

    def get_neighbors(self, positions, active_mask=None):
        neighbors_matrix = (
            np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=-1)
            < self.perception
        )
        np.fill_diagonal(neighbors_matrix, 0)
        if active_mask is not None:
            active_mask = np.asarray(active_mask, dtype=bool)
            neighbors_matrix &= active_mask[:, None] & active_mask[None, :]
        neighbors = sp.coo_matrix(neighbors_matrix, dtype=int)
        return neighbors

    def get_separation(self, neighbors, positions):
        self_idx, neig_idx = neighbors.row, neighbors.col
        distances = np.linalg.norm(positions[self_idx] - positions[neig_idx], axis=-1)
        mask = distances < self.crowding
        steering = np.zeros_like(positions)
        changes = -(positions[neig_idx[mask]] - positions[self_idx[mask]])
        np.add.at(steering, self_idx[mask], changes)
        steering = self.clamp(steering)
        return steering

    def get_alignment(self, neighbors, velocities):
        degree = utils.degree_power(neighbors, -1.0)
        steering = degree @ neighbors @ velocities
        steering -= velocities
        # With no neighbors there is no alignment/cohesion target. Do not
        # interpret the empty mean as zero velocity or the coordinate origin.
        has_neighbors = np.asarray(neighbors.sum(axis=1)).ravel() > 0
        steering[~has_neighbors] = 0.0
        steering = self.clamp(steering)
        return steering

    def get_cohesion(self, neighbors, positions):
        return self.get_alignment(neighbors, positions)

    def get_goal(self, neighbors, positions):
        neighbor_mask = neighbors.toarray().astype(float)  # (n, n)
        n_neighbors = neighbor_mask.sum(axis=1)  # (n,)
        neighbor_pos_sum = neighbor_mask @ positions  # (n, 3)
        centroid = (neighbor_pos_sum + positions) / (n_neighbors[:, None] + 1)
        goal_vec = self.goal_positions[self.current_goal] - centroid  # (n, 3)
        # The local centroid includes self, so an isolated agent can still
        # navigate using its own position and the externally supplied goal.
        steering = self.clamp(goal_vec)
        return steering

    def update_goal(self, positions, active_mask=None):
        if active_mask is not None:
            positions = positions[np.asarray(active_mask, dtype=bool)]
        if positions.shape[0] == 0:
            return
        arrival_distance = goal_arrival_distance(
            positions, self.goal_positions[self.current_goal]
        )
        if arrival_distance <= self.goal_arrival_radius and not self.reached_goal:
            self.reached_goal = True
        if self.reached_goal:
            self.current_goal = (self.current_goal + 1) % len(self.goal_positions)
            self.goals_completed += 1
            self.reached_goal = False

    def check_termination(self):
        return self.goals_completed == len(self.goal_positions)

    def enforce_limits(self, velocities_old, velocities_new):
        """Enforce speed AND turn limits in 3D."""
        velocities = velocities_new.copy()

        # -- Turn limit -------------------------------------------------------
        speed_old = np.linalg.norm(velocities_old, axis=-1, keepdims=True)
        speed_new = np.linalg.norm(velocities, axis=-1, keepdims=True)
        # Avoid division by zero
        safe_old = np.where(speed_old > 1e-8, velocities_old / speed_old, np.array([1., 0., 0.]))
        safe_new = np.where(speed_new > 1e-8, velocities / speed_new, safe_old)
        cos_angle = np.clip((safe_old * safe_new).sum(axis=-1), -1.0, 1.0)
        angle_rad = np.arccos(cos_angle)  # (n,)
        max_turn_rad = np.deg2rad(self.max_turn)
        mask = angle_rad > max_turn_rad
        if mask.any():
            # Rotate the old direction by exactly max_turn toward the new one, within
            # the plane of the two directions (the 2D turn limit applied in that plane).
            perpendicular = safe_new - cos_angle[:, None] * safe_old
            length = np.linalg.norm(perpendicular, axis=-1, keepdims=True)
            # Opposite directions span no unique plane: turn toward a fixed perpendicular.
            axis = np.where(np.abs(safe_old[:, :1]) < 0.9, np.array([1., 0., 0.]), np.array([0., 1., 0.]))
            fallback = axis - (axis * safe_old).sum(axis=-1, keepdims=True) * safe_old
            fallback /= np.linalg.norm(fallback, axis=-1, keepdims=True)
            perpendicular = np.where(length > 1e-8, perpendicular / np.maximum(length, 1e-8), fallback)
            capped = np.cos(max_turn_rad) * safe_old + np.sin(max_turn_rad) * perpendicular
            velocities = np.where(mask[:, None], capped * speed_new, velocities)

        # -- Speed limit ------------------------------------------------------
        speed = np.linalg.norm(velocities, axis=-1)
        velocities[speed < self.min_speed] = scale(velocities[speed < self.min_speed], self.min_speed)
        velocities[speed > self.max_speed] = scale(velocities[speed > self.max_speed], self.max_speed)
        return velocities

    def clamp(self, force):
        to_clamp = np.linalg.norm(force, axis=-1) > self.max_force
        force[to_clamp] = scale(force[to_clamp], length=self.max_force)
        return force

    def initial_state(self, center):
        """A clump of agents in the starting box at ``center``, all moving along +x at maximum speed."""
        positions = center + self.init_scatter * np.random.rand(self.n_boids, 3)
        velocities = np.tile(np.array([1.0, 0.0, 0.0]) * self.max_speed, (self.n_boids, 1))
        return positions, velocities

    def plot(self, positions):
        if self.figure is None:
            plt.ion()
            self.figure = plt.figure()
            self.ax = self.figure.add_subplot(111, projection='3d')
            self.scatter = self.ax.scatter(
                positions[:, 0], positions[:, 1], positions[:, 2],
                marker=".", edgecolor="k", lw=0.5
            )
            goal = self.goal_positions[self.current_goal]
            self.goal_marker = self.ax.scatter([goal[0]], [goal[1]], [goal[2]],
                                               c='red', marker='*', s=200, label='Goal')
            self.ax.set_xlim(self.borders[0], self.borders[3])
            self.ax.set_ylim(self.borders[1], self.borders[4])
            self.ax.set_zlim(self.borders[2], self.borders[5])
            self.ax.legend()
            plt.show()
        self.scatter._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
        goal = self.goal_positions[self.current_goal]
        self.goal_marker._offsets3d = ([goal[0]], [goal[1]], [goal[2]])
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()


def scale(x, length=1.0):
    return length * x / np.linalg.norm(x, axis=-1, keepdims=True)
