"""The reference Boids expert in 2D: the equations the compiled expert implements."""
import numpy as np
import scipy.sparse as sp
from matplotlib import pyplot as plt
from spektral import utils

from modules.waypoints import goal_arrival_distance


BOIDS_GOAL_POSITIONS = np.array(
    [[3.0, -3.0], [3.0, 3.0], [0.0, 0.0]],
    dtype=np.float32,
)


class Boids:
    def __init__(
        self,
        min_speed=0.0001,  # Min speed of the boids
        max_speed=0.01,  # Max speed of the boids
        max_force=0.1,  # Max amount of steering that any single update is allowed to add
        max_turn=5,  # How many degrees is a boid allowed to turn
        perception=0.1,  # How distant must two boids be in order to be neighbors
        crowding=0.02,  # How much groups are pushed apart (lower = tighter groups)
        init_scatter=0.325,  # Spread of boids around the initial center
        n_boids=100,  # How many boids in the environment
        dt=1,  # Size of a time step (lower = more precise simulation)
        canvas_scale=1,  # Canvas is rescaled by this amount (used to control size)
        wrap=False,  # If True, wrap around instead of avoiding boundary
        limits=True,  # If True, enforce speed and turn limits
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

        self.borders = canvas_scale * np.array([-5, -5, 5, 5])  # Hard borders of canvas
        self.canonical_goal_positions = BOIDS_GOAL_POSITIONS.copy()
        self.goal_positions = self.canonical_goal_positions.copy()
        self.current_goal = 0
        self.reached_goal = False  # Flag to track if boids have reached the current goal
        self.goals_completed = 0  # Counter to track how many goals have been completed
        self.init_scatter = init_scatter  # Spread of boids around the initial center

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
            positions = ((positions + 1) % 2) - 1  # Wrap around

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

        # Update positions
        positions = positions + velocities * self.dt

        positions[:, 0] = np.clip(positions[:, 0], self.borders[0], self.borders[2])
        positions[:, 1] = np.clip(positions[:, 1], self.borders[1], self.borders[3])

        positions[~active_mask] = positions_before[~active_mask]
        velocities[~active_mask] = 0.0
        accelerations[~active_mask] = 0.0

        # Plot if needed
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
        # Exclude self-neighbors before converting to sparse
        np.fill_diagonal(neighbors_matrix, 0)
        if active_mask is not None:
            active_mask = np.asarray(active_mask, dtype=bool)
            neighbors_matrix &= active_mask[:, None] & active_mask[None, :]

        # Convert to sparse matrix for return
        neighbors = sp.coo_matrix(neighbors_matrix, dtype=int)
        return neighbors

    def get_separation(self, neighbors, positions):
        """
        Get the steering component to separate boids that are too close
        """
        self_idx, neig_idx = neighbors.row, neighbors.col
        distances = np.linalg.norm(positions[self_idx] - positions[neig_idx], axis=-1)
        mask = distances < self.crowding
        steering = np.zeros_like(positions)
        changes = -(positions[neig_idx[mask]] - positions[self_idx[mask]])
        np.add.at(steering, self_idx[mask], changes)
        steering = self.clamp(steering)

        return steering

    def get_alignment(self, neighbors, velocities):
        """
        Get the steering component to align the velocities of neighbors
        """
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
        """
        Get the steering component to align the positions of neighbors
        """
        return self.get_alignment(neighbors, positions)

    def get_goal(self, neighbors, positions):
        """
        Get the steering component to move towards a goal position.
        """
        neighbor_mask = neighbors.toarray().astype(float)  # (n, n)
        n_neighbors = neighbor_mask.sum(axis=1)  # (n,)

        # Vectorized centroid: (sum of neighbor positions + self position) / (n_neighbors + 1)
        neighbor_pos_sum = neighbor_mask @ positions  # (n, 2)
        centroid = (neighbor_pos_sum + positions) / (n_neighbors[:, None] + 1)

        goal_vec = self.goal_positions[self.current_goal] - centroid  # (n, 2)
        # The local centroid includes self, so an isolated agent can still
        # navigate using its own position and the externally supplied goal.
        steering = self.clamp(goal_vec)
        return steering

    def update_goal(self, positions, active_mask=None):
        """
        Update the goal position after boids reach it and loiter for 1 step.
        """
        # Fetch current positions of boids and compute distance to goal
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
        """
        Terminate the trajectory if boids finish loitering at all goals for 1 step.
        """

        if self.goals_completed == len(self.goal_positions):  # Assuming multiple goals
            return True
        else:
            return False

    def enforce_limits(self, velocities_old, velocities_new):
        # Update velocities
        velocities_old_polar = to_polar(velocities_old)
        velocities_new_polar = to_polar(velocities_new)

        # Enforce turn limit
        phi_diff = (
            180 - (180 - velocities_new_polar[:, 1] + velocities_old_polar[:, 1]) % 360
        )
        mask = np.abs(phi_diff) > self.max_turn
        velocities_new_polar[mask, 1] = (
            velocities_old_polar[mask, 1] + np.sign(phi_diff[mask]) * self.max_turn
        )
        velocities = to_cartesian(velocities_new_polar)

        # Enforce speed limit
        speed = np.linalg.norm(velocities, axis=-1)
        velocities[speed < self.min_speed] = scale(
            velocities[speed < self.min_speed], self.min_speed
        )
        velocities[speed > self.max_speed] = scale(
            velocities[speed > self.max_speed], self.max_speed
        )

        return velocities

    def clamp(self, force):
        """
        Clamp a given force (steering) to the maximum value allowed (to make things
        more stable)
        """
        to_clamp = np.linalg.norm(force, axis=-1) > self.max_force
        force[to_clamp] = scale(force[to_clamp], length=self.max_force)

        return force

    def initial_state(self, center):
        """A clump of agents in the starting box at ``center``, all moving along +x at maximum speed."""
        positions = center + self.init_scatter * np.random.rand(self.n_boids, 2)
        velocities = np.tile(np.array([1.0, 0.0]) * self.max_speed, (self.n_boids, 1))
        return positions, velocities

    def plot(self, positions, **kwargs):
        if self.figure is None:
            plt.ion()
            self.figure = plt.figure()
            axes = plt.axes(xlim=self.borders[::2], ylim=self.borders[1::2])
            self.scatter = axes.scatter(
                positions[:, 0],
                positions[:, 1],
                marker=".",
                edgecolor="k",
                lw=0.5,
                **kwargs
            )
            self.goal_marker, = axes.plot([], [], 'r*', markersize=15, label='Goal')
            axes.legend()
            axes.set_title(
                f"Boids Evolution with Goal"
            )
            plt.show()
        # Always update the goal marker position
        self.goal_marker.set_data(
            [self.goal_positions[self.current_goal][0]],
            [self.goal_positions[self.current_goal][1]]
        )
        self.scatter.set_offsets(positions)
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
def to_polar(cartesian_coords):
    x, y = cartesian_coords.T
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x) * 180 / np.pi
    return np.stack((rho, phi), -1)

def to_cartesian(polar_coords):
    rho, phi = polar_coords.T
    phi *= np.pi / 180
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return np.stack((x, y), -1)

def scale(x, length=1.0):
    return length * x / np.linalg.norm(x, axis=-1, keepdims=True)
