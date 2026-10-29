import numpy as np
import scipy.sparse as sp
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from spektral import utils
from spektral.data import Dataset, Graph
from tqdm import tqdm
import h5py

from modules.waypoints import OnlineWaypointManager, goal_arrival_distance
from modules.perturbations import (
    DRIFT_MODE,
    REMOVAL_MODE,
    normalize_perturbation_config,
)


BOIDS_GOAL_POSITIONS_3D = np.array(
    [[-4.0, -4.0, -4.0], [4.0, 4.0, 4.0]],
    dtype=np.float32,
)
BOIDS_3D_FIXED_TARGET_FEATURES = 15
BOIDS_3D_ACTIVE_MASK_INDEX = 15
BOIDS_3D_MASKED_TARGET_FEATURES = 16


def _in_exclusion_zone_3d(center, exclusion_zone):
    """Return True if center is within capsule exclusion zone."""
    if exclusion_zone is None:
        return False
    goals, radius = exclusion_zone
    for i in range(len(goals)):
        a = goals[i]
        b = goals[(i + 1) % len(goals)]
        ab = b - a
        ab_len_sq = np.dot(ab, ab)
        if ab_len_sq < 1e-10:
            if np.linalg.norm(center - a) < radius:
                return True
            continue
        t = np.clip(np.dot(center - a, ab) / ab_len_sq, 0.0, 1.0)
        closest = a + t * ab
        if np.linalg.norm(center - closest) < radius:
            return True
    return False

class Boids3D:
    def __init__(
        self,
        min_speed=0.0001,
        max_speed=0.01,
        max_force=0.1,
        max_turn=15,
        perception=0.1,
        crowding=0.02,
        n_boids=100,
        dt=1,
        canvas_scale=1,
        boundary_size_pctg=0.2,
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
        self.boundary_size_pctg = boundary_size_pctg
        self.wrap = wrap
        self.limits = limits
        self.init_scatter = 0.325  # Spread of boids around the initial center

        # 3D canvas: [xmin, ymin, zmin, xmax, ymax, zmax]
        self.borders = canvas_scale * np.array([-5, -5, -5, 5, 5, 5])
        self.center = (self.borders[3:] + self.borders[:3]) / 2
        self.canonical_goal_positions = BOIDS_GOAL_POSITIONS_3D.copy()
        self.goal_positions = self.canonical_goal_positions.copy()
        self.current_goal = 0
        self.reached_goal = False
        self.loiter_timer = 0
        self.goals_completed = 0
        self.rand_configs = []
        self.unseen_configs = []  # Centers from cache NOT used in training

        self.boundary_margins = self.borders * boundary_size_pctg
        self.boundaries = self.borders - self.boundary_margins

        self.show = show
        self.figure = None
        self.print_neighbors_warning = True

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
        # else:
        #     accelerations += self.avoid_borders(positions)

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

    def generate_trajectory(
        self, save_config, random_init, return_accel=False,
        perturbation=None, perturbation_rng=None,
    ):
        # Reset goal state for each new trajectory
        self.current_goal = 0
        self.goals_completed = 0
        self.reached_goal = False

        if random_init is False:
            positions, velocities, _ = self.get_fixed_init(self.n_boids)
        elif random_init is True:
            positions, velocities, _, _ = self.get_random_init(self.n_boids, save_config)
        elif isinstance(random_init, np.ndarray) and random_init.shape == (3,):
            # Treat as a center for the flock (3D)
            center = random_init
            positions = center + self.init_scatter * np.random.rand(self.n_boids, 3)
            direction = np.array([1.0, 0.0, 0.0])
            velocity = direction * self.max_speed
            velocities = np.tile(velocity, (self.n_boids, 1))
        else:
            assert (
                len(random_init) == 2
            ), "Expected random_init to have length 2 (positions, velocities) or a center (3,)"
            positions, velocities = random_init

        self.goal_positions = self.canonical_goal_positions.copy()
        perturbation = normalize_perturbation_config(perturbation, 3)
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
                positions, velocities, active_mask, removed_total, step,
                perturbation, perturbation_rng,
            )
            history["input_positions"].append(model_positions)
            history["input_velocities"].append(model_velocities)
            history["neighbors"].append(
                self.get_neighbors(model_positions, active_mask=active_mask)
            )
            history["active_masks"].append(active_mask.copy())
            output = self.update_boids(
                model_positions, model_velocities, return_accel=return_accel,
                active_mask=active_mask,
            )
            positions, velocities, _ = output[:3]
            # update_boids() uses the pre-update graph to compute the expert
            # dynamics.  A training sample at the new state, however, must be
            # paired with the graph of that new state.  Storing the graph
            # returned by update_boids() made every sample after t=0 use the
            # previous timestep's adjacency matrix.
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
        goal_bounds=(-4.5, 4.5, -4.5, 4.5, -4.5, 4.5),
        start_bounds=(-5.0, 5.0, -5.0, 5.0, -5.0, 5.0),
        goal_min_distance=2.5,
        goal_arrival_radius=0.5,
        max_steps=10_000,
        perturbation=None,
        perturbation_rng=None,
    ):
        """Generate a reference episode with externally changing 3D goals.

        The optimized cloud path compiles these same dynamics with Numba. This
        method remains the readable reference implementation and supports
        evaluation without changing the fixed-waypoint generator.
        """
        rng = np.random.default_rng() if rng is None else rng
        if random_init is False:
            positions, velocities, _ = self.get_fixed_init(self.n_boids)
        elif random_init is True:
            positions, velocities, _, _ = self.get_random_init(
                self.n_boids, save_config, bounds=tuple(start_bounds)
            )
        elif isinstance(random_init, np.ndarray) and random_init.shape == (3,):
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
        perturbation = normalize_perturbation_config(perturbation, 3)
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
                positions, velocities, active_mask, removed_total, step,
                perturbation, perturbation_rng,
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
                f"Online 3D expert did not reach {n_waypoints} waypoints "
                f"within {max_steps} steps."
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
            selected = self._cap_selected(selected, config.drift_max_per_event, rng)
            if len(selected):
                bounds = np.asarray(config.drift_bounds, dtype=model_positions.dtype)
                model_positions[selected] += rng.uniform(
                    -bounds, bounds, size=(len(selected), 3)
                )
                if config.drift_perturb_velocity:
                    velocity_bounds = np.asarray(
                        config.drift_bounds if config.drift_velocity_bounds is None
                        else config.drift_velocity_bounds,
                        dtype=model_velocities.dtype,
                    )
                    model_velocities[selected] += rng.uniform(
                        -velocity_bounds, velocity_bounds, size=(len(selected), 3)
                    )
        return model_positions, model_velocities, active_mask, removed_total

    @staticmethod
    def _cap_selected(selected, cap, rng):
        if cap is None or len(selected) <= max(0, cap):
            return selected
        return rng.choice(selected, size=max(0, cap), replace=False)

    def avoid_borders(self, positions):
        """Steer boids away from 3D boundaries."""
        in_margin = (
            np.any(positions < self.boundaries[:3], -1) |
            np.any(positions > self.boundaries[3:], -1)
        )
        steering = np.zeros_like(positions)
        steering[in_margin] += self.center - positions[in_margin]
        return steering

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

    def get_random_init(self, n_boids, save_config, bounds=None, exclusion_zone=None, center=None, max_attempts=10000):
        """
        Spawns boids in a tight clump at a random location.
        bounds:         optional (x_min, x_max, y_min, y_max, z_min, z_max)
        exclusion_zone: optional list of (center_3d, radius) spheres to reject
        center:         if provided, skip sampling and use directly
        """
        if center is None:
            if bounds is not None:
                x_min, x_max, y_min, y_max, z_min, z_max = bounds
            else:
                x_min, x_max, y_min, y_max, z_min, z_max = -5, 0, -5, 0, -5, 0
            for _ in range(max_attempts):
                c = np.array([
                    np.random.uniform(x_min, x_max),
                    np.random.uniform(y_min, y_max),
                    np.random.uniform(z_min, z_max),
                ])
                if exclusion_zone is None or not _in_exclusion_zone_3d(c, exclusion_zone):
                    center = c
                    break
            else:
                raise RuntimeError(f"Could not sample valid center after {max_attempts} attempts.")
        if save_config:
            self.rand_configs.append(center)
        positions = center + self.init_scatter * np.random.rand(n_boids, 3)
        direction = np.array([1.0, 0.0, 0.0])
        velocities = np.tile(direction * self.max_speed, (n_boids, 1))
        neighbors = self.get_neighbors(positions)
        return positions, velocities, neighbors, center

    def get_fixed_init(self, n_boids):
        center = np.array([-3.0, -3.0, -3.0])
        self.rand_configs.append(center)
        positions = center + self.init_scatter * np.random.rand(n_boids, 3)
        direction = np.array([1.0, 0.0, 0.0])
        velocity = direction * self.max_speed
        velocities = np.tile(velocity, (n_boids, 1))
        neighbors = self.get_neighbors(positions)
        return positions, velocities, neighbors

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

class BoidsDataset3D(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.graphs = dataset

    def read(self):
        return []


def scale(x, length=1.0):
    return length * x / np.linalg.norm(x, axis=-1, keepdims=True)

def history_to_samples_3d(history, accel=False):
    """Convert 3D trajectory history to graph samples.
    y is [current_state (6D), next_state (6D)] if goal present, else just next_state (6D).
    """
    physical = np.concatenate(
        (history["positions"], history["velocities"]), axis=-1
    )
    inputs = np.concatenate(
        (
            history.get("input_positions", history["positions"][:-1]),
            history.get("input_velocities", history["velocities"][:-1]),
        ),
        axis=-1,
    )
    neighbors = history["neighbors"]
    n_boids = inputs.shape[1]
    if accel and "accelerations" in history:
        targets = history["accelerations"]
        return [(x, a, y) for x, a, y in zip(inputs, neighbors, targets)]
    else:
        # Do not overwrite the previous transition's target with a new impulse.
        base_targets = np.concatenate([inputs, physical[1:]], axis=-1)
        if "goal_positions" in history:
            # goal_positions: [T, 3] -> broadcast to [T-1, n_boids, 3]
            goal_pos = history["goal_positions"][:-1]  # [T-1, 3]
            goal_broadcast = np.broadcast_to(goal_pos[:, None, :], (len(goal_pos), n_boids, 3)).copy()
            targets = np.concatenate([base_targets, goal_broadcast], axis=-1)  # [T-1, n_boids, 15]
        else:
            targets = base_targets  # [T-1, n_boids, 12]
        active_masks = np.asarray(
            history.get(
                "active_masks", np.ones(inputs.shape[:2], dtype=np.float32)
            ),
            dtype=np.float32,
        )
        targets = np.concatenate((targets, active_masks[..., None]), axis=-1)
        return [(x, a, y_) for x, a, y_ in zip(inputs, neighbors, targets)]


def ensure_active_mask_targets_3d(targets):
    """Upgrade cached fixed-waypoint targets that lack the active-agent column by marking every agent active."""
    targets = np.asarray(targets)
    if targets.shape[-1] == BOIDS_3D_MASKED_TARGET_FEATURES:
        return targets
    if targets.shape[-1] != BOIDS_3D_FIXED_TARGET_FEATURES:
        raise ValueError(
            f"Expected {BOIDS_3D_FIXED_TARGET_FEATURES} or "
            f"{BOIDS_3D_MASKED_TARGET_FEATURES} target features, got "
            f"{targets.shape[-1]}."
        )
    active = np.ones(targets.shape[:-1] + (1,), dtype=targets.dtype)
    return np.concatenate((targets, active), axis=-1)


def _load_adaptive_stride_3d(f, start, end, timestep_stride, near_goal_radius):
    """Load timesteps with adaptive stride: dense (stride=1) near any goal, sparse elsewhere.
    y[:, 0, 12:15] = current goal position (same for all boids at each timestep).
    """
    if timestep_stride <= 1 or near_goal_radius <= 0:
        return (
            f['x'][start:end:timestep_stride],
            ensure_active_mask_targets_3d(
                f['y'][start:end:timestep_stride]
            ),
            f['a_row'][start:end:timestep_stride],
            f['a_col'][start:end:timestep_stride],
            f['a_len'][start:end:timestep_stride],
        )

    x_full = f['x'][start:end]   # (T, n_boids, 6)
    y_full = ensure_active_mask_targets_3d(f['y'][start:end])

    active = y_full[..., BOIDS_3D_ACTIVE_MASK_INDEX]
    mean_pos = np.sum(y_full[..., :3] * active[..., None], axis=1)
    mean_pos /= np.maximum(active.sum(axis=1, keepdims=True), 1.0)
    dist = np.min(
        np.linalg.norm(
            mean_pos[:, None, :] - BOIDS_GOAL_POSITIONS_3D[None, :, :],
            axis=-1,
        ),
        axis=1,
    )

    T = end - start
    strided = np.zeros(T, dtype=bool)
    strided[::timestep_stride] = True
    keep = np.where(strided | (dist < near_goal_radius))[0]
    abs_keep = keep + start

    return (
        x_full[keep],
        y_full[keep],
        f['a_row'][abs_keep],
        f['a_col'][abs_keep],
        f['a_len'][abs_keep],
    )


def load_chunk_from_cache_3d(cache_path, flat_indices, boundaries, n_boids_cache,
                              timestep_stride=1, near_goal_radius=1.0):
    """Load a subset of 3D trajectories from HDF5 into a BoidsDataset3D."""
    graphs = []
    with h5py.File(cache_path, 'r') as f:
        for flat_idx in flat_indices:
            start = int(boundaries[flat_idx])
            end   = int(boundaries[flat_idx + 1])
            x_c, y_c, arow_c, acol_c, alen_c = _load_adaptive_stride_3d(
                f, start, end, timestep_stride, near_goal_radius
            )
            for k in range(len(x_c)):
                length = int(alen_c[k])
                row = arow_c[k, :length]
                col = acol_c[k, :length]
                a = sp.coo_matrix(
                    (np.ones(length, dtype=np.float32), (row, col)),
                    shape=(n_boids_cache, n_boids_cache),
                )
                graphs.append(Graph(x=x_c[k], a=a, y=y_c[k]))
    return BoidsDataset3D(graphs)


def make_dataset_3d(unique_reps, repeat_reps, save_config, random_init=True, return_boids=False, accel=False, **kwargs):
    boids_cache_npz = kwargs.pop("boids_cache_npz", None)
    timestep_stride = kwargs.pop("timestep_stride", 1)
    perturbation_configs = kwargs.pop("perturbation_configs", None)

    boids = Boids3D(**kwargs)
    all_graphs = []
    if (
        perturbation_configs is not None
        and len(perturbation_configs) != unique_reps * repeat_reps
    ):
        raise ValueError(
            "perturbation_configs must contain one entry per generated trajectory."
        )

    # Fast path: load precomputed graph tuples from HDF5 cache
    if boids_cache_npz is not None:

        print(f">>> Loading precomputed 3D training data from '{boids_cache_npz}'...")
        with h5py.File(boids_cache_npz, 'r') as f:
            centers       = f['centers'][:]
            cache_unique  = int(f.attrs['unique_reps'])
            cache_repeats = int(f.attrs['repeats'])
            n_boids_cache = int(f.attrs['n_boids'])

            n_train = min(unique_reps, cache_unique)
            train_indices  = np.random.choice(cache_unique, size=n_train, replace=False)
            unseen_indices = np.array([i for i in range(cache_unique) if i not in set(train_indices)])

            boids.rand_configs   = [np.array(centers[i], dtype=np.float32) for i in train_indices]
            boids.unseen_configs = [np.array(centers[i], dtype=np.float32) for i in unseen_indices]
            print(f"Loaded cache: {n_train} training centers, {len(unseen_indices)} unseen centers, timestep_stride={timestep_stride}")

            if cache_repeats < 1:
                raise ValueError("Cache must contain at least one trajectory per center.")
            n_reps = repeat_reps

            if 'traj_lengths' in f:
                traj_lengths = f['traj_lengths'][:]
                boundaries = np.concatenate([[0], np.cumsum(traj_lengths)])
            else:
                total_samples    = f['x'].shape[0]
                samples_per_traj = total_samples // (cache_unique * cache_repeats)
                boundaries = np.arange(cache_unique * cache_repeats + 1) * samples_per_traj

            for traj_i in train_indices:
                for rep_j in range(n_reps):
                    cached_repeat_idx = rep_j % cache_repeats
                    flat_idx = traj_i * cache_repeats + cached_repeat_idx
                    start = int(boundaries[flat_idx])
                    end   = int(boundaries[flat_idx + 1])
                    x_chunk    = f['x'][start:end:timestep_stride]
                    y_chunk    = f['y'][start:end:timestep_stride]
                    arow_chunk = f['a_row'][start:end:timestep_stride]
                    acol_chunk = f['a_col'][start:end:timestep_stride]
                    alen_chunk = f['a_len'][start:end:timestep_stride]
                    for k in range(len(x_chunk)):
                        length = int(alen_chunk[k])
                        row  = arow_chunk[k, :length]
                        col  = acol_chunk[k, :length]
                        data = np.ones(length, dtype=np.float32)
                        a = sp.coo_matrix((data, (row, col)), shape=(n_boids_cache, n_boids_cache))
                        all_graphs.append(Graph(
                            x=x_chunk[k],
                            a=a,
                            y=ensure_active_mask_targets_3d(y_chunk[k]),
                        ))

        dataset = BoidsDataset3D(all_graphs)
        return (dataset, boids) if return_boids else dataset

    # Normal path: simulate trajectories
    print(f">>> Generating {unique_reps} unique 3D trajectories with {repeat_reps} repeats each...")

    use_init_list = isinstance(random_init, list) and len(random_init) == unique_reps

    trajectory_index = 0
    for i in tqdm(range(unique_reps)):
        # Sample one center per unique rep, reuse for all repeats
        if use_init_list:
            center = random_init[i]
        elif random_init is True:
            _, _, _, center = boids.get_random_init(boids.n_boids, save_config=save_config)
        else:
            center = None  # fixed init - handled inside generate_trajectory

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
            samples = history_to_samples_3d(history, accel=accel)
            for x, a, y in samples:
                all_graphs.append(Graph(x=x, a=a, y=y))
            del history, samples

    dataset = BoidsDataset3D(all_graphs)
    return (dataset, boids) if return_boids else dataset
