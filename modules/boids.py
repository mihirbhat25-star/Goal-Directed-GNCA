import numpy as np
import scipy.sparse as sp
from matplotlib import pyplot as plt
try:
    from shapely.geometry import Point
except ImportError:  # Optional unless polygon-based center exclusion is requested.
    Point = None
from spektral import utils
from spektral.data import Dataset, Graph
from tqdm import tqdm
import h5py

from modules.waypoints import (
    OnlineWaypointManager,
    goal_arrival_distance,
    goal_conditioned_state,
)
from modules.perturbations import (
    DRIFT_MODE,
    REMOVAL_MODE,
    normalize_perturbation_config,
)


BOIDS_GOAL_POSITIONS = np.array(
    [[3.0, -3.0], [3.0, 3.0], [0.0, 0.0]],
    dtype=np.float32,
)
BOIDS_STATE_FEATURES = 4
ONLINE_GOAL_STATE_FEATURES = 6
BOIDS_BASE_TARGET_FEATURES = 10
BOIDS_TRANSITION_TARGET_FEATURES = 13
BOIDS_ACTIVE_MASK_INDEX = 13
BOIDS_MASKED_TARGET_FEATURES = 14
CURRENT_GOAL_SLICE = slice(8, 10)
PREVIOUS_GOAL_SLICE = slice(10, 12)
PREVIOUS_GOAL_MAX_DISTANCE_INDEX = 12


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
        boundary_size_pctg=0.2,  # Relative size of the soft boundary
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
        self.boundary_size_pctg = boundary_size_pctg
        self.wrap = wrap
        self.limits = limits

        self.borders = canvas_scale * np.array([-5, -5, 5, 5])  # Hard borders of canvas
        self.center = (self.borders[2:] + self.borders[:2]) / 2  # Center of the canvas
        self.canonical_goal_positions = BOIDS_GOAL_POSITIONS.copy()
        self.goal_positions = self.canonical_goal_positions.copy()
        self.current_goal = 0
        self.reached_goal = False  # Flag to track if boids have reached the current goal
        self.loiter_timer = 0  # Timer to track how long boids have been loitering at the current goal
        self.goals_completed = 0  # Counter to track how many goals have been completed
        self.rand_configs = []  # Random initial configurations used for training
        self.unseen_configs = []  # Centers from cache NOT used in training (for unseen viz testing)
        self.init_scatter = init_scatter  # Spread of boids around the initial center

        # Soft boundary inside which boids are pushed towards the center to avoid leaving the canvas
        self.boundary_margins = self.borders * boundary_size_pctg
        self.boundaries = self.borders - self.boundary_margins

        self.show = show
        self.figure = None
        self.print_neighbors_warning = True  # Flag to control printing neighbors warning

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
        # else:
        #     accelerations += self.avoid_borders(positions)  # Avoid edge collisions

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

    def generate_trajectory(
        self,
        save_config,
        random_init,
        return_accel=False,
        perturbation=None,
        perturbation_rng=None,
    ):
        # Reset goal state for each new trajectory
        self.current_goal = 0
        self.goals_completed = 0
        self.reached_goal = False

        if random_init is False:
            positions, velocities, _ = self.get_fixed_init(self.n_boids)
        elif random_init is True:
            positions, velocities, _, _ = self.get_random_init(self.n_boids, save_config)
        elif isinstance(random_init, np.ndarray) and random_init.shape == (2,):
            # Treat as a center for the flock
            center = random_init
            positions = center + 0.325 * np.random.rand(self.n_boids, 2)
            direction = np.array([1.0, 0.0])
            velocity = direction * self.max_speed
            velocities = np.tile(velocity, (self.n_boids, 1))
        else:
            assert (
                len(random_init) == 2
            ), "Expected random_init to have length 2 (positions, velocities) or a center (2,)"
            positions, velocities = random_init

        self.goal_positions = self.canonical_goal_positions.copy()
        perturbation = normalize_perturbation_config(perturbation, 2)
        perturbation_rng = self._perturbation_rng(perturbation_rng)
        active_mask = np.ones(self.n_boids, dtype=bool)
        removed_total = 0
        history = {
            "positions": [positions],
            "velocities": [velocities],
            "input_positions": [],
            "input_velocities": [],
            "neighbors": [],
            "active_masks": [],
            "goal_positions": [self.goal_positions[self.current_goal].copy()],
        }
        if return_accel:
            history["accelerations"] = []
        step = 0
        while True:
            (
                model_positions,
                model_velocities,
                active_mask,
                removed_total,
            ) = self._prepare_model_input(
                positions,
                velocities,
                active_mask,
                removed_total,
                step,
                perturbation,
                perturbation_rng,
            )
            history["input_positions"].append(model_positions)
            history["input_velocities"].append(model_velocities)
            history["neighbors"].append(
                self.get_neighbors(model_positions, active_mask=active_mask)
            )
            history["active_masks"].append(active_mask.copy())
            output = self.update_boids(
                model_positions,
                model_velocities,
                return_accel=return_accel,
                active_mask=active_mask,
            )
            positions, velocities, _ = output[:3]
            history["positions"].append(positions)
            history["velocities"].append(velocities)
            if return_accel:
                history["accelerations"].append(output[3])
            self.update_goal(positions, active_mask=active_mask)
            # The goal stored with state x_t must be the goal used to produce
            # x_{t+1}. Update the goal before labeling the newly appended state.
            history["goal_positions"].append(self.goal_positions[self.current_goal].copy())
            if self.check_termination():
                break
            step += 1

        history["positions"] = np.array(history["positions"])
        history["velocities"] = np.array(history["velocities"])
        history["goal_positions"] = np.array(history["goal_positions"])
        history["input_positions"] = np.asarray(history["input_positions"])
        history["input_velocities"] = np.asarray(history["input_velocities"])
        history["active_masks"] = np.asarray(history["active_masks"], dtype=np.float32)
        if return_accel:
            history["accelerations"] = np.array(history["accelerations"])

        return history

    def generate_online_goal_trajectory(
        self,
        *,
        save_config,
        random_init,
        rng=None,
        n_waypoints=2,
        goal_bounds=(-4.5, 4.5, -4.5, 4.5),
        start_bounds=(-5.0, 5.0, -5.0, 5.0),
        goal_min_distance=2.5,
        goal_arrival_radius=0.5,
        max_steps=10_000,
        perturbation=None,
        perturbation_rng=None,
    ):
        """Generate an expert episode with externally changing random goals.

        The goal label attached to state ``x_t`` is always the goal used to
        produce ``x_{t+1}``. Exactly ``n_waypoints`` successful arrivals are
        required; otherwise a bounded episode raises instead of silently
        returning incomplete supervision.
        """
        rng = np.random.default_rng() if rng is None else rng
        if random_init is False:
            positions, velocities, _ = self.get_fixed_init(self.n_boids)
        elif random_init is True:
            positions, velocities, _, _ = self.get_random_init(
                self.n_boids, save_config, bounds=tuple(start_bounds)
            )
        elif isinstance(random_init, np.ndarray) and random_init.shape == (2,):
            positions, velocities, _, _ = self.get_random_init(
                self.n_boids,
                save_config,
                center=random_init,
            )
        else:
            positions, velocities = random_init

        manager = OnlineWaypointManager(
            rng=rng,
            n_waypoints=n_waypoints,
            bounds=tuple(goal_bounds),
            min_distance=goal_min_distance,
            arrival_radius=goal_arrival_radius,
        )
        active_goal = manager.start(positions.mean(axis=0))
        self.goal_positions = active_goal[None, :]
        self.current_goal = 0

        perturbation = normalize_perturbation_config(perturbation, 2)
        perturbation_rng = self._perturbation_rng(
            rng if perturbation_rng is None else perturbation_rng
        )
        active_mask = np.ones(self.n_boids, dtype=bool)
        removed_total = 0
        history = {
            "positions": [positions],
            "velocities": [velocities],
            "input_positions": [],
            "input_velocities": [],
            "neighbors": [],
            "active_masks": [],
            "goal_positions": [active_goal.copy()],
        }
        for step in range(int(max_steps)):
            (
                model_positions,
                model_velocities,
                active_mask,
                removed_total,
            ) = self._prepare_model_input(
                positions,
                velocities,
                active_mask,
                removed_total,
                step,
                perturbation,
                perturbation_rng,
            )
            history["input_positions"].append(model_positions)
            history["input_velocities"].append(model_velocities)
            history["neighbors"].append(
                self.get_neighbors(model_positions, active_mask=active_mask)
            )
            history["active_masks"].append(active_mask.copy())
            positions, velocities, _ = self.update_boids(
                model_positions, model_velocities, active_mask=active_mask
            )
            active_goal, switched, finished = manager.update(positions[active_mask])
            if switched:
                self.goal_positions = active_goal[None, :]
            history["positions"].append(positions)
            history["velocities"].append(velocities)
            history["goal_positions"].append(active_goal.copy())
            if finished:
                break
        else:
            raise RuntimeError(
                f"Online expert did not reach {n_waypoints} waypoints within "
                f"{max_steps} steps."
            )

        history["positions"] = np.asarray(history["positions"])
        history["velocities"] = np.asarray(history["velocities"])
        history["goal_positions"] = np.asarray(history["goal_positions"])
        history["input_positions"] = np.asarray(history["input_positions"])
        history["input_velocities"] = np.asarray(history["input_velocities"])
        history["active_masks"] = np.asarray(history["active_masks"], dtype=np.float32)
        history["waypoints"] = np.asarray(manager.waypoints)
        return history

    @staticmethod
    def _perturbation_rng(rng):
        if isinstance(rng, np.random.Generator):
            return rng
        if rng is None:
            rng = int(np.random.randint(0, 2**32 - 1, dtype=np.uint32))
        return np.random.default_rng(rng)

    def _prepare_model_input(
        self, positions, velocities, active_mask, removed_total, step, config, rng
    ):
        """Apply an event to the state shared by the GNCA and expert.

        The caller evolves this disturbed state and keeps the resulting expert
        state for the next step. No independent noise is added to the label.
        """
        model_positions = positions.copy()
        model_velocities = velocities.copy()
        if step == 0 or step % config.interval or config.probability <= 0:
            return model_positions, model_velocities, active_mask, removed_total

        candidates = np.flatnonzero(active_mask)
        selected = candidates[rng.random(len(candidates)) < config.probability]
        if config.mode == REMOVAL_MODE:
            allowed = max(0, int(active_mask.sum()) - 1)
            if config.removal_max_per_event is not None:
                allowed = min(allowed, config.removal_max_per_event)
            if config.removal_max_total is not None:
                allowed = min(allowed, config.removal_max_total - removed_total)
            selected = self._cap_selected(selected, allowed, rng)
            active_mask = active_mask.copy()
            active_mask[selected] = False
            removed_total += len(selected)
        elif config.mode == DRIFT_MODE:
            selected = self._cap_selected(
                selected, config.drift_max_per_event, rng
            )
            if len(selected):
                bounds = np.asarray(config.drift_bounds, dtype=model_positions.dtype)
                model_positions[selected] += rng.uniform(
                    -bounds, bounds, size=(len(selected), 2)
                )
                if config.drift_perturb_velocity:
                    velocity_bounds = np.asarray(
                        config.drift_bounds if config.drift_velocity_bounds is None
                        else config.drift_velocity_bounds,
                        dtype=model_velocities.dtype,
                    )
                    model_velocities[selected] += rng.uniform(
                        -velocity_bounds, velocity_bounds, size=(len(selected), 2)
                    )
        return model_positions, model_velocities, active_mask, removed_total

    @staticmethod
    def _cap_selected(selected, cap, rng):
        if cap is None or len(selected) <= max(0, cap):
            return selected
        return rng.choice(selected, size=max(0, cap), replace=False)

    def avoid_borders(self, positions):
        """If a boid is within the external margins, steer it towards the centre"""
        in_margin = np.any(positions < self.boundaries[:2], -1) | np.any(
            positions > self.boundaries[2:], -1
        )
        steering = np.zeros_like(positions)

        steering[in_margin] += self.center - positions[in_margin]

        return steering

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

    def get_random_init(self, n_boids, save_config, bounds=None, exclusion_zone=None,
                        center=None, max_attempts=10000):
        """
        Spawns boids in a tight clump at a random location on the canvas.
        save_config=True:  Save center to rand_configs (training data collection)
        save_config=False: Generate fresh random center (testing/validation/robustness)
        bounds:          optional (x_min, x_max, y_min, y_max) to constrain sampling region.
        exclusion_zone:  optional Shapely polygon; sampled centers inside it are rejected.
        center:          if provided, skip sampling and use this center directly.
        Returns: positions, velocities, neighbors, center
        """
        if center is None:
            x_min, x_max, y_min, y_max = bounds if bounds is not None else (-5, -2.5, -5, -2.5)
            for _ in range(max_attempts):
                center = np.array([np.random.uniform(x_min, x_max),
                                   np.random.uniform(y_min, y_max)])
                if exclusion_zone is not None and Point is None:
                    raise ImportError(
                        "Shapely is required when an exclusion_zone is supplied."
                    )
                if exclusion_zone is None or not exclusion_zone.contains(Point(center)):
                    break
            else:
                raise RuntimeError(
                    f"Could not sample a valid center after {max_attempts} attempts. "
                    f"Decrease goal_exclusion_size."
                )
        if save_config:
            self.rand_configs.append(center)
        positions = center + self.init_scatter * np.random.rand(n_boids, 2)
        # Set all boids to have the same initial velocity
        # Example: all boids move right at max_speed
        direction = np.array([1.0, 0.0])  # unit vector to the right
        velocity = direction * self.max_speed
        velocities = np.tile(velocity, (n_boids, 1))
        neighbors = self.get_neighbors(positions)

        return positions, velocities, neighbors, center

    def get_fixed_init(self, n_boids):
        """
        Set a fixed initial position of (-4, -4).
        :param n_boids: int, number of boids
        """
        positions = np.full((n_boids, 2), -4.0) + self.init_scatter * np.random.rand(n_boids, 2)
        # Set all boids to have the same initial velocity
        # Example: all boids move right at max_speed
        direction = np.array([1.0, 0.0])  # unit vector to the right
        velocity = direction * self.max_speed
        velocities = np.tile(velocity, (n_boids, 1))
        neighbors = self.get_neighbors(positions)

        return positions, velocities, neighbors

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
class BoidsDataset(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.graphs = dataset

    def read(self):
        return []

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


def ensure_goal_transition_metadata(inputs, targets, active_masks=None):
    """Attach temporally correct previous-goal metadata to one trajectory.

    Targets contain current state, next state, and active goal. The appended
    fields are the immediately previous goal and the maximum centroid distance
    from that goal since the most recent switch. A value of -1 means that no
    previous goal exists. The running maximum makes eligibility irreversible
    after the flock first exits a chosen loss radius, even after timesteps are
    shuffled for training.
    """
    targets = np.asarray(targets)
    if targets.shape[-1] == BOIDS_MASKED_TARGET_FEATURES:
        if active_masks is not None:
            raise ValueError(
                "active_masks must not be supplied when targets already contain it."
            )
        active_masks = targets[..., BOIDS_ACTIVE_MASK_INDEX]
        targets = targets[..., :BOIDS_TRANSITION_TARGET_FEATURES]
    elif active_masks is not None:
        active_masks = np.asarray(active_masks, dtype=targets.dtype)
        if active_masks.shape != targets.shape[:2]:
            raise ValueError(
                "active_masks must match the target (timesteps, boids) axes, "
                f"got {active_masks.shape} for targets {targets.shape}."
            )
    if targets.shape[-1] == BOIDS_TRANSITION_TARGET_FEATURES:
        return (
            targets
            if active_masks is None
            else np.concatenate((targets, active_masks[..., None]), axis=-1)
        )
    if targets.shape[-1] != BOIDS_BASE_TARGET_FEATURES:
        raise ValueError(
            "Expected target feature dimension "
            f"{BOIDS_BASE_TARGET_FEATURES} or "
            f"{BOIDS_TRANSITION_TARGET_FEATURES}, got {targets.shape[-1]}"
        )
    if inputs.ndim != 3 or targets.ndim != 3 or inputs.shape[:2] != targets.shape[:2]:
        raise ValueError(
            "inputs and targets must have matching (timesteps, boids, features) "
            f"dimensions, got {inputs.shape} and {targets.shape}"
        )

    timesteps, n_boids = targets.shape[:2]
    active_goals = targets[:, 0, CURRENT_GOAL_SLICE]
    previous_goals = np.zeros((timesteps, 2), dtype=targets.dtype)
    max_previous_distances = np.full(timesteps, -1.0, dtype=targets.dtype)
    previous_goal = None
    running_max_distance = -1.0

    for timestep in range(timesteps):
        if timestep > 0 and not np.allclose(
            active_goals[timestep], active_goals[timestep - 1]
        ):
            previous_goal = active_goals[timestep - 1].copy()
            running_max_distance = 0.0

        if previous_goal is not None:
            if active_masks is None:
                centroid = targets[timestep, :, :2].mean(axis=0)
            else:
                active = active_masks[timestep] > 0.5
                centroid = targets[timestep, active, :2].mean(axis=0)
            distance = float(np.linalg.norm(centroid - previous_goal))
            running_max_distance = max(running_max_distance, distance)
            previous_goals[timestep] = previous_goal
            max_previous_distances[timestep] = running_max_distance

    previous_goal_features = np.broadcast_to(
        previous_goals[:, None, :], (timesteps, n_boids, 2)
    )
    previous_distance_features = np.broadcast_to(
        max_previous_distances[:, None, None], (timesteps, n_boids, 1)
    )
    enriched = np.concatenate(
        [targets, previous_goal_features, previous_distance_features], axis=-1
    ).astype(targets.dtype, copy=False)
    if active_masks is not None:
        enriched = np.concatenate((enriched, active_masks[..., None]), axis=-1)
    return enriched


def goal_transition_proximity(inputs, targets, radius):
    """Return timesteps near the active or still-eligible previous goal."""
    targets = ensure_goal_transition_metadata(inputs, targets)
    if targets.shape[-1] == BOIDS_MASKED_TARGET_FEATURES:
        active = targets[..., BOIDS_ACTIVE_MASK_INDEX]
        centroids = np.sum(targets[..., :2] * active[..., None], axis=1)
        centroids /= np.maximum(active.sum(axis=1, keepdims=True), 1.0)
    else:
        centroids = targets[..., :2].mean(axis=1)
    active_goals = targets[:, 0, CURRENT_GOAL_SLICE]
    active_distances = np.linalg.norm(centroids - active_goals, axis=-1)
    max_previous_distances = targets[
        :, 0, PREVIOUS_GOAL_MAX_DISTANCE_INDEX
    ]
    previous_is_eligible = (
        (max_previous_distances >= 0.0)
        & (max_previous_distances < radius)
    )
    return (active_distances < radius) | previous_is_eligible


def history_to_samples(history, accel=False):
    physical = np.concatenate(
        (history["positions"], history["velocities"]), axis=-1
    )
    input_positions = history.get("input_positions", history["positions"][:-1])
    input_velocities = history.get("input_velocities", history["velocities"][:-1])
    inputs = np.concatenate((input_positions, input_velocities), axis=-1)
    neighbors = history["neighbors"]
    n_boids = inputs.shape[1]
    if accel and "accelerations" in history:
        targets = history["accelerations"]
        return [(x, a, y) for x, a, y in zip(inputs, neighbors, targets)]
    else:
        # Inputs include any impulse at t; physical[t + 1] is its expert
        # response. Keep physical[t] intact as the preceding step's label.
        base_targets = np.concatenate([inputs, physical[1:]], axis=-1)
        active_masks = np.asarray(
            history.get(
                "active_masks",
                np.ones(inputs.shape[:2], dtype=np.float32),
            ),
            dtype=np.float32,
        )
        if "goal_positions" in history:
            # goal_positions: [T, 2] -> broadcast to [T-1, n_boids, 2]
            goal_pos = history["goal_positions"][:-1]  # [T-1, 2]
            goal_broadcast = np.broadcast_to(goal_pos[:, None, :], (len(goal_pos), n_boids, 2)).copy()
            targets = np.concatenate([base_targets, goal_broadcast], axis=-1)
            targets = ensure_goal_transition_metadata(
                inputs, targets, active_masks=active_masks
            )
        else:
            targets = np.concatenate(
                (base_targets, active_masks[..., None]), axis=-1
            )
        return [(x, a, y_) for x, a, y_ in zip(inputs, neighbors, targets)]


def ensure_active_mask_targets(targets):
    """Upgrade cached transition targets that lack the active-agent column by marking every agent active."""
    targets = np.asarray(targets)
    if targets.shape[-1] == BOIDS_MASKED_TARGET_FEATURES:
        return targets
    if targets.shape[-1] != BOIDS_TRANSITION_TARGET_FEATURES:
        raise ValueError(
            f"Expected {BOIDS_TRANSITION_TARGET_FEATURES} or "
            f"{BOIDS_MASKED_TARGET_FEATURES} target features, got "
            f"{targets.shape[-1]}."
        )
    active = np.ones(targets.shape[:-1] + (1,), dtype=targets.dtype)
    return np.concatenate((targets, active), axis=-1)


def online_goal_history_to_samples(history):
    """Convert an online-goal expert history into six-feature GNCA samples."""
    physical = np.concatenate(
        (history["positions"], history["velocities"]), axis=-1
    ).astype(np.float32, copy=False)
    goals = np.asarray(history["goal_positions"], dtype=np.float32)
    input_physical = np.concatenate(
        (
            history.get("input_positions", history["positions"][:-1]),
            history.get("input_velocities", history["velocities"][:-1]),
        ),
        axis=-1,
    ).astype(np.float32, copy=False)
    conditioned = np.stack(
        [
            goal_conditioned_state(state, goal)
            for state, goal in zip(input_physical, goals[:-1])
        ]
    )
    n_boids = physical.shape[1]
    goal_broadcast = np.broadcast_to(
        goals[:-1, None, :], (len(goals) - 1, n_boids, 2)
    ).copy()
    base_targets = np.concatenate(
        (input_physical, physical[1:], goal_broadcast), axis=-1
    )
    active_masks = np.asarray(
        history.get(
            "active_masks",
            np.ones(input_physical.shape[:2], dtype=np.float32),
        ),
        dtype=np.float32,
    )
    targets = ensure_goal_transition_metadata(
        conditioned, base_targets, active_masks=active_masks
    )
    return [
        (x, adjacency, target)
        for x, adjacency, target in zip(
            conditioned, history["neighbors"], targets
        )
    ]

def make_dataset(unique_reps, repeat_reps, save_config, random_init=True, return_boids=False, accel=False, **kwargs):
    boids_cache_npz = kwargs.pop("boids_cache_npz", None)
    timestep_stride = kwargs.pop("timestep_stride", 1)
    near_goal_radius = kwargs.pop("near_goal_radius", 1.0)
    perturbation_configs = kwargs.pop("perturbation_configs", None)

    boids = Boids(**kwargs)
    all_graphs = []
    if (
        perturbation_configs is not None
        and len(perturbation_configs) != unique_reps * repeat_reps
    ):
        raise ValueError(
            "perturbation_configs must contain one entry per generated trajectory."
        )

    # Fast path: load precomputed graph tuples from cache
    if boids_cache_npz is not None:

        print(f">>> Loading precomputed training data from '{boids_cache_npz}'...")
        with h5py.File(boids_cache_npz, 'r') as f:
            centers       = f['centers'][:]
            cache_unique  = int(f.attrs['unique_reps'])
            cache_repeats = int(f.attrs['repeats'])

            # Subsample unique_reps centers for training; rest go to unseen_configs
            n_train = min(unique_reps, cache_unique)
            train_indices  = np.random.choice(cache_unique, size=n_train, replace=False)
            unseen_indices = np.array([i for i in range(cache_unique) if i not in set(train_indices)])

            boids.rand_configs   = [np.array(centers[i], dtype=np.float32) for i in train_indices]
            boids.unseen_configs = [np.array(centers[i], dtype=np.float32) for i in unseen_indices]
            print(f"Loaded cache: {n_train} training centers, {len(unseen_indices)} unseen centers, timestep_stride={timestep_stride}")

            n_reps = min(repeat_reps, cache_repeats)

            # Use exact per-trajectory lengths if available, else fall back to estimate
            if 'traj_lengths' in f:
                traj_lengths = f['traj_lengths'][:]
                boundaries = np.concatenate([[0], np.cumsum(traj_lengths)])
            else:
                total_samples    = f['x'].shape[0]
                samples_per_traj = total_samples // (cache_unique * cache_repeats)
                boundaries = np.arange(cache_unique * cache_repeats + 1) * samples_per_traj

            for traj_i in train_indices:
                for rep_j in range(n_reps):
                    flat_idx = traj_i * cache_repeats + rep_j
                    start = int(boundaries[flat_idx])
                    end   = int(boundaries[flat_idx + 1])
                    x_chunk, y_chunk = \
                        _load_adaptive_stride(f, start, end, timestep_stride, near_goal_radius)
                    for k in range(len(x_chunk)):
                        # Caches store the adjacency offset by one step from
                        # the state. Recompute it from the current state so
                        # they remain usable without regeneration.
                        a = boids.get_neighbors(
                            x_chunk[k, :, :2],
                            active_mask=(
                                y_chunk[k, :, BOIDS_ACTIVE_MASK_INDEX] > 0.5
                            ),
                        )
                        all_graphs.append(Graph(
                            x=x_chunk[k],
                            a=a,
                            y=ensure_active_mask_targets(y_chunk[k]),
                        ))

        dataset = BoidsDataset(all_graphs)
        return (dataset, boids) if return_boids else dataset

    # Normal path: simulate trajectories
    print(f">>> Generating {unique_reps} unique trajectories with {repeat_reps} repeats each...")

    use_init_list = isinstance(random_init, list) and len(random_init) == unique_reps

    trajectory_index = 0
    for i in tqdm(range(unique_reps)):
        # Sample one center per unique rep, reuse for all repeats
        if use_init_list:
            center = random_init[i]
        elif random_init is True:
            _, _, _, center = boids.get_random_init(boids.n_boids, save_config=(save_config))
        else:
            center = None  # fixed init or other - handled inside generate_trajectory

        for j in range(repeat_reps):
            if center is not None:
                history = boids.generate_trajectory(
                    random_init=center,
                    return_accel=accel,
                    save_config=False,
                    perturbation=(
                        None if perturbation_configs is None
                        else perturbation_configs[trajectory_index]
                    ),
                )
            else:
                history = boids.generate_trajectory(
                    random_init=random_init,
                    return_accel=accel,
                    save_config=(save_config and j == 0),
                    perturbation=(
                        None if perturbation_configs is None
                        else perturbation_configs[trajectory_index]
                    ),
                )
            trajectory_index += 1
            samples = history_to_samples(history, accel=accel)
            for x, a, y in samples:
                all_graphs.append(Graph(x=x, a=a, y=y))

    dataset = BoidsDataset(all_graphs)
    return (dataset, boids) if return_boids else dataset


def _load_adaptive_stride(f, start, end, timestep_stride, near_goal_radius):
    """Load timesteps from an open HDF5 file with adaptive stride.

    Normal timesteps are loaded every `timestep_stride` steps.
    Timesteps where the flock's mean position is within `near_goal_radius`
    of the active goal or the still-eligible immediately previous goal are
    always included (stride=1 in that window), so transition frames are not
    skipped and unrelated waypoints do not affect sampling.

    Args:
        f:                Open h5py File object.
        start, end:       Absolute slice [start, end) into the dataset.
        timestep_stride:  Stride for non-near-goal timesteps.
        near_goal_radius: Distance threshold (in simulation units).
                          0 or stride==1 -> plain strided load, no detection.

    Returns:
        Tuple ``(x_c, y_c)`` containing selected states and targets. Adjacency
        is intentionally recomputed from each current state by the caller.
    """
    if timestep_stride <= 1 or near_goal_radius <= 0:
        selected_y = ensure_goal_transition_metadata(
            f['x'][start:end:timestep_stride],
            f['y'][start:end:timestep_stride],
        )
        selected_y = ensure_active_mask_targets(selected_y)
        return f['x'][start:end:timestep_stride], selected_y

    x_full  = f['x'][start:end]       # (T, n_boids, 4)
    y_full = ensure_active_mask_targets(
        ensure_goal_transition_metadata(x_full, f['y'][start:end])
    )

    T = end - start
    strided = np.zeros(T, dtype=bool)
    strided[::timestep_stride] = True
    if near_goal_radius > 0:
        near_transition_goal = goal_transition_proximity(
            x_full, y_full, near_goal_radius
        )
    else:
        near_transition_goal = np.zeros(T, dtype=bool)
    keep = np.where(strided | near_transition_goal)[0]

    return x_full[keep], y_full[keep]


def load_chunk_from_cache(cache_path, flat_indices, boundaries,
                          timestep_stride=1, near_goal_radius=1.0,
                          perception=0.1):
    """Load a subset of trajectories from an HDF5 cache into a BoidsDataset.

    Args:
        cache_path:      Path to the HDF5 cache file.
        flat_indices:    1-D array of flat trajectory indices to load
                         (flat_idx = center_i * cache_repeats + rep_j).
        boundaries:      Cumulative sample-count boundary array of length
                         (total_trajectories + 1), as returned by np.cumsum.
        timestep_stride: Load every Nth timestep (default 1 = all).

    Returns:
        BoidsDataset containing Graph objects for the requested trajectories.
    """
    graphs = []
    with h5py.File(cache_path, 'r') as f:
        for flat_idx in flat_indices:
            start = int(boundaries[flat_idx])
            end   = int(boundaries[flat_idx + 1])
            x_c, y_c = _load_adaptive_stride(
                f, start, end, timestep_stride, near_goal_radius
            )
            for k in range(len(x_c)):
                positions = x_c[k, :, :2]
                neighbors = (
                    np.linalg.norm(
                        positions[:, None, :] - positions[None, :, :], axis=-1
                    )
                    < perception
                )
                np.fill_diagonal(neighbors, False)
                active = y_c[k, :, BOIDS_ACTIVE_MASK_INDEX] > 0.5
                neighbors &= active[:, None] & active[None, :]
                a = sp.coo_matrix(neighbors, dtype=np.float32)
                graphs.append(Graph(x=x_c[k], a=a, y=y_c[k]))
    return BoidsDataset(graphs)
