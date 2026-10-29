"""Import-light, compiled expert rollouts for cloud 3D Boids training.

This module deliberately depends only on NumPy and Numba.  In particular, it
does not import TensorFlow, Spektral, SciPy, h5py, or Matplotlib, so many CPU
workers can use it without each worker initializing the ML stack.

The rollout kernel preserves the equations implemented by
``modules.boids_3d.Boids3D`` and builds the neighborhood graph inside the loop:

* the radius graph is computed once for each state and carried to the next
  update;
* the same graph and degree counts are reused by every force component; and
* row-major adjacency is retained as an eightfold bit-packed mask.

The returned goals and graph are aligned with ``input_states[t]`` and its
expert response ``states[t + 1]``. Both fixed and online waypoints use this
contract, including persistent drift and agent-removal events.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Mapping, Optional, Sequence

import numpy as np
from numba import njit


DEFAULT_FIXED_GOALS = np.asarray(
    [[-4.0, -4.0, -4.0], [4.0, 4.0, 4.0]], dtype=np.float32
)

OCTANT_BOUNDS_3D = (
    (0.0, 5.0, 0.0, 5.0, 0.0, 5.0),
    (-5.0, 0.0, 0.0, 5.0, 0.0, 5.0),
    (-5.0, 0.0, -5.0, 0.0, 0.0, 5.0),
    (0.0, 5.0, -5.0, 0.0, 0.0, 5.0),
    (0.0, 5.0, 0.0, 5.0, -5.0, 0.0),
    (-5.0, 0.0, 0.0, 5.0, -5.0, 0.0),
    (-5.0, 0.0, -5.0, 0.0, -5.0, 0.0),
    (0.0, 5.0, -5.0, 0.0, -5.0, 0.0),
)


@dataclass(frozen=True)
class CompiledTrajectory3D:
    """Compact transition-aligned output of one compiled rollout.

    ``states`` has one more row than the other temporal arrays.  For transition
    ``t``, use ``input_states[t]``, ``states[t + 1]``, ``active_goals[t]``, and
    ``adjacency_bits[t]``. At an event the input includes the impulse, while
    ``states[t]`` retains the preceding transition's output.
    """

    states: np.ndarray
    input_states: np.ndarray
    active_masks: np.ndarray
    active_goals: np.ndarray
    previous_goals: np.ndarray
    max_previous_goal_distances: np.ndarray
    goal_segment_ids: np.ndarray
    adjacency_bits: np.ndarray
    waypoints: np.ndarray
    sampled_center: np.ndarray

    @property
    def steps(self) -> int:
        return int(self.active_goals.shape[0])

    @property
    def n_boids(self) -> int:
        return int(self.states.shape[1])


def _validate_bounds(bounds: Sequence[float], name: str) -> np.ndarray:
    values = np.asarray(bounds, dtype=np.float64)
    if values.shape != (6,):
        raise ValueError(f"{name} must have six values, got shape {values.shape}.")
    if not (
        values[0] < values[1]
        and values[2] < values[3]
        and values[4] < values[5]
    ):
        raise ValueError(f"{name} must contain three increasing min/max pairs.")
    return values


def _validate_goals(goals: Sequence[Sequence[float]]) -> np.ndarray:
    values = np.asarray(goals, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 1:
        raise ValueError(f"goals must have shape (G, 3), got {values.shape}.")
    if not np.all(np.isfinite(values)):
        raise ValueError("goals must contain only finite values.")
    return values


def _inside_cyclic_capsule(center: np.ndarray, goals: np.ndarray, radius: float) -> bool:
    if radius <= 0:
        return False
    for idx in range(len(goals)):
        start = goals[idx].astype(np.float64, copy=False)
        end = goals[(idx + 1) % len(goals)].astype(np.float64, copy=False)
        segment = end - start
        length_squared = float(np.dot(segment, segment))
        if length_squared < 1e-10:
            if np.linalg.norm(center - start) < radius:
                return True
            continue
        fraction = np.clip(
            np.dot(center - start, segment) / length_squared, 0.0, 1.0
        )
        if np.linalg.norm(center - (start + fraction * segment)) < radius:
            return True
    return False


def sample_fixed_initial_state(
    seed: int,
    *,
    octant: int,
    n_boids: int = 100,
    goals: Sequence[Sequence[float]] = DEFAULT_FIXED_GOALS,
    exclusion_radius: float = 0.5,
    init_scatter: float = 0.325,
    max_speed: float = 0.01,
    discard_initial_clump_draw: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample the fixed-task center and initial flock.

    The established fixed-task sampler calls ``get_random_init`` for its center,
    discards the flock sampled by that call, and samples another flock at the
    same center in ``generate_trajectory``. The flag preserves that random-draw
    sequence for exact seeded parity with the local expert.
    """
    if octant < 0 or octant >= len(OCTANT_BOUNDS_3D):
        raise ValueError(f"octant must be in [0, 7], got {octant}.")
    if n_boids < 1:
        raise ValueError("n_boids must be positive.")
    goals_array = _validate_goals(goals)
    bounds = np.asarray(OCTANT_BOUNDS_3D[octant], dtype=np.float64)
    rng = np.random.RandomState(int(seed) % (2**32 - 1))

    for _ in range(10_000):
        center = np.asarray(
            [
                rng.uniform(bounds[0], bounds[1]),
                rng.uniform(bounds[2], bounds[3]),
                rng.uniform(bounds[4], bounds[5]),
            ],
            dtype=np.float64,
        )
        if not _inside_cyclic_capsule(center, goals_array, exclusion_radius):
            break
    else:
        raise RuntimeError("Could not sample a center outside the exclusion capsule.")

    if discard_initial_clump_draw:
        rng.rand(n_boids, 3)
    positions = center + float(init_scatter) * rng.rand(n_boids, 3)
    velocities = np.zeros((n_boids, 3), dtype=np.float64)
    velocities[:, 0] = float(max_speed)
    return center, positions, velocities


def sample_online_initial_state_and_goals(
    seed: int,
    *,
    n_boids: int = 100,
    n_waypoints: int = 2,
    start_bounds: Sequence[float] = (-5.0, 5.0, -5.0, 5.0, -5.0, 5.0),
    goal_bounds: Sequence[float] = (-4.5, 4.5, -4.5, 4.5, -4.5, 4.5),
    goal_min_distance: float = 2.5,
    init_scatter: float = 0.325,
    max_speed: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample a 3D clump and sequential externally supplied random waypoints.

    This mirrors the 2D online experiment: the initial goal is separated from
    the initial flock centroid; each later goal is separated from the goal it
    replaces.  The initial-state RNG and waypoint RNG are independent streams
    initialized with the same seed, matching the 2D online expert convention.
    """
    if n_boids < 1 or n_waypoints < 1:
        raise ValueError("n_boids and n_waypoints must be positive.")
    if goal_min_distance < 0:
        raise ValueError("goal_min_distance cannot be negative.")
    start = _validate_bounds(start_bounds, "start_bounds")
    goal_box = _validate_bounds(goal_bounds, "goal_bounds")
    seed = int(seed) % (2**32 - 1)

    state_rng = np.random.RandomState(seed)
    center = np.asarray(
        [
            state_rng.uniform(start[0], start[1]),
            state_rng.uniform(start[2], start[3]),
            state_rng.uniform(start[4], start[5]),
        ],
        dtype=np.float64,
    )
    positions = center + float(init_scatter) * state_rng.rand(n_boids, 3)
    velocities = np.zeros((n_boids, 3), dtype=np.float64)
    velocities[:, 0] = float(max_speed)

    waypoint_rng = np.random.default_rng(seed)
    waypoints = np.empty((n_waypoints, 3), dtype=np.float32)
    reference = positions.mean(axis=0)
    for waypoint_idx in range(n_waypoints):
        for _ in range(10_000):
            candidate = np.asarray(
                [
                    waypoint_rng.uniform(goal_box[0], goal_box[1]),
                    waypoint_rng.uniform(goal_box[2], goal_box[3]),
                    waypoint_rng.uniform(goal_box[4], goal_box[5]),
                ],
                dtype=np.float32,
            )
            if np.linalg.norm(candidate - reference) >= goal_min_distance:
                waypoints[waypoint_idx] = candidate
                reference = candidate
                break
        else:
            raise RuntimeError(
                "Could not sample separated online waypoint; enlarge goal bounds "
                "or reduce goal_min_distance."
            )
    return center, positions, velocities, waypoints


@njit(cache=True, nogil=True)
def _fill_graph_and_sums(
    positions,
    velocities,
    active_mask,
    perception_squared,
    crowding_squared,
    neighbors,
    degrees,
    position_sums,
    velocity_sums,
    separation,
):
    n_boids = positions.shape[0]
    for idx in range(n_boids):
        degrees[idx] = 0
        for dim in range(3):
            position_sums[idx, dim] = 0.0
            velocity_sums[idx, dim] = 0.0
            separation[idx, dim] = 0.0
        for other in range(n_boids):
            neighbors[idx, other] = 0

    # The graph is undirected, so evaluate each pair once and update both rows.
    for left in range(n_boids):
        if not active_mask[left]:
            continue
        for right in range(left + 1, n_boids):
            if not active_mask[right]:
                continue
            dx = positions[left, 0] - positions[right, 0]
            dy = positions[left, 1] - positions[right, 1]
            dz = positions[left, 2] - positions[right, 2]
            distance_squared = dx * dx + dy * dy + dz * dz
            if distance_squared < perception_squared:
                neighbors[left, right] = 1
                neighbors[right, left] = 1
                degrees[left] += 1
                degrees[right] += 1
                for dim in range(3):
                    position_sums[left, dim] += positions[right, dim]
                    position_sums[right, dim] += positions[left, dim]
                    velocity_sums[left, dim] += velocities[right, dim]
                    velocity_sums[right, dim] += velocities[left, dim]
                if distance_squared < crowding_squared:
                    separation[left, 0] += dx
                    separation[left, 1] += dy
                    separation[left, 2] += dz
                    separation[right, 0] -= dx
                    separation[right, 1] -= dy
                    separation[right, 2] -= dz


@njit(cache=True, nogil=True)
def _clamp_vector_in_place(vector, maximum):
    norm = np.sqrt(
        vector[0] * vector[0]
        + vector[1] * vector[1]
        + vector[2] * vector[2]
    )
    if norm > maximum:
        factor = maximum / norm
        vector[0] *= factor
        vector[1] *= factor
        vector[2] *= factor


@njit(cache=True, nogil=True)
def _store_packed_adjacency(neighbors, destination):
    for byte_idx in range(destination.shape[0]):
        destination[byte_idx] = 0
    n_boids = neighbors.shape[0]
    for row in range(n_boids):
        base = row * n_boids
        for col in range(n_boids):
            if neighbors[row, col] != 0:
                flat_idx = base + col
                destination[flat_idx >> 3] |= np.uint8(1 << (flat_idx & 7))


@njit(cache=True, nogil=True)
def _rollout_kernel_3d(
    initial_positions,
    initial_velocities,
    waypoints,
    arrival_radius,
    max_steps,
    min_speed,
    max_speed,
    max_force,
    max_turn_degrees,
    perception,
    crowding,
    dt,
    borders,
    noise_seed,
    perturbation_mode,
    perturbation_interval,
    perturbation_probability,
    removal_max_per_event,
    removal_max_total,
    drift_max_per_event,
    drift_bounds,
    drift_perturb_velocity,
    drift_velocity_bounds=None,
):
    n_boids = initial_positions.shape[0]
    n_goals = waypoints.shape[0]
    packed_width = (n_boids * n_boids + 7) // 8

    states = np.empty((max_steps + 1, n_boids, 6), dtype=np.float32)
    input_states = np.empty((max_steps, n_boids, 6), dtype=np.float32)
    active_masks = np.ones((max_steps, n_boids), dtype=np.float32)
    active_goals = np.empty((max_steps, 3), dtype=np.float32)
    previous_goals = np.zeros((max_steps, 3), dtype=np.float32)
    previous_max_distances = np.full(max_steps, -1.0, dtype=np.float32)
    goal_segments = np.empty(max_steps, dtype=np.int16)
    adjacency_bits = np.empty((max_steps, packed_width), dtype=np.uint8)

    positions = initial_positions.copy()
    velocities = initial_velocities.copy()
    neighbors = np.zeros((n_boids, n_boids), dtype=np.uint8)
    degrees = np.zeros(n_boids, dtype=np.int32)
    position_sums = np.zeros((n_boids, 3), dtype=np.float64)
    velocity_sums = np.zeros((n_boids, 3), dtype=np.float64)
    separation = np.zeros((n_boids, 3), dtype=np.float64)

    for boid in range(n_boids):
        for dim in range(3):
            states[0, boid, dim] = positions[boid, dim]
            states[0, boid, dim + 3] = velocities[boid, dim]

    current_goal = 0
    running_previous_distance = -1.0
    perception_squared = perception * perception
    crowding_squared = crowding * crowding
    max_turn_radians = max_turn_degrees * np.pi / 180.0
    active_mask = np.ones(n_boids, dtype=np.bool_)
    removed_total = 0
    np.random.seed(noise_seed)

    for step in range(max_steps):
        model_positions = positions.copy()
        model_velocities = velocities.copy()
        if (
            step > 0
            and step % perturbation_interval == 0
            and perturbation_mode != 0
            and perturbation_probability > 0.0
        ):
            selected = np.empty(n_boids, dtype=np.int32)
            selected_count = 0
            for boid in range(n_boids):
                if active_mask[boid] and np.random.random() < perturbation_probability:
                    selected[selected_count] = boid
                    selected_count += 1
            for index in range(selected_count - 1, 0, -1):
                swap = np.random.randint(0, index + 1)
                value = selected[index]
                selected[index] = selected[swap]
                selected[swap] = value
            if perturbation_mode == 1:
                allowed = max(0, int(active_mask.sum()) - 1)
                if removal_max_per_event >= 0:
                    allowed = min(allowed, removal_max_per_event)
                if removal_max_total >= 0:
                    allowed = min(allowed, removal_max_total - removed_total)
                selected_count = min(selected_count, max(0, allowed))
                for index in range(selected_count):
                    active_mask[selected[index]] = False
                removed_total += selected_count
            elif perturbation_mode == 2:
                if drift_max_per_event >= 0:
                    selected_count = min(selected_count, drift_max_per_event)
                for index in range(selected_count):
                    boid = selected[index]
                    for dim in range(3):
                        model_positions[boid, dim] += np.random.uniform(
                            -drift_bounds[dim], drift_bounds[dim]
                        )
                        if drift_perturb_velocity:
                            velocity_bound = (
                                drift_bounds[dim] if drift_velocity_bounds is None
                                else drift_velocity_bounds[dim]
                            )
                            model_velocities[boid, dim] += np.random.uniform(
                                -velocity_bound, velocity_bound
                            )

        for boid in range(n_boids):
            active_masks[step, boid] = 1.0 if active_mask[boid] else 0.0
            for dim in range(3):
                input_states[step, boid, dim] = model_positions[boid, dim]
                input_states[step, boid, dim + 3] = model_velocities[boid, dim]
        # The expert and GNCA see the same disturbed state. Keep its response
        # for subsequent recovery steps instead of reverting to a clean path.
        positions = model_positions
        velocities = model_velocities
        _fill_graph_and_sums(
            positions,
            velocities,
            active_mask,
            perception_squared,
            crowding_squared,
            neighbors,
            degrees,
            position_sums,
            velocity_sums,
            separation,
        )
        _store_packed_adjacency(neighbors, adjacency_bits[step])

        for dim in range(3):
            active_goals[step, dim] = waypoints[current_goal, dim]
        goal_segments[step] = current_goal

        if current_goal > 0:
            centroid0 = 0.0
            centroid1 = 0.0
            centroid2 = 0.0
            active_count = 0
            for boid in range(n_boids):
                if not active_mask[boid]:
                    continue
                centroid0 += positions[boid, 0]
                centroid1 += positions[boid, 1]
                centroid2 += positions[boid, 2]
                active_count += 1
            centroid0 /= active_count
            centroid1 /= active_count
            centroid2 /= active_count
            pdx = centroid0 - waypoints[current_goal - 1, 0]
            pdy = centroid1 - waypoints[current_goal - 1, 1]
            pdz = centroid2 - waypoints[current_goal - 1, 2]
            previous_distance = np.sqrt(pdx * pdx + pdy * pdy + pdz * pdz)
            if previous_distance > running_previous_distance:
                running_previous_distance = previous_distance
            for dim in range(3):
                previous_goals[step, dim] = waypoints[current_goal - 1, dim]
            previous_max_distances[step] = running_previous_distance

        for boid in range(n_boids):
            if not active_mask[boid]:
                velocities[boid, 0] = 0.0
                velocities[boid, 1] = 0.0
                velocities[boid, 2] = 0.0
                continue
            degree = degrees[boid]

            # Separation, alignment, and cohesion are clamped independently,
            # exactly as in Boids3D before their scalar weights are applied.
            sep = separation[boid].copy()
            _clamp_vector_in_place(sep, max_force)

            alignment = np.empty(3, dtype=np.float64)
            cohesion = np.empty(3, dtype=np.float64)
            if degree > 0:
                inverse_degree = 1.0 / degree
                for dim in range(3):
                    alignment[dim] = (
                        velocity_sums[boid, dim] * inverse_degree
                        - velocities[boid, dim]
                    )
                    cohesion[dim] = (
                        position_sums[boid, dim] * inverse_degree
                        - positions[boid, dim]
                    )
            else:
                for dim in range(3):
                    alignment[dim] = 0.0
                    cohesion[dim] = 0.0
            _clamp_vector_in_place(alignment, max_force)
            _clamp_vector_in_place(cohesion, max_force)

            goal_force = np.zeros(3, dtype=np.float64)
            # Including self keeps goal steering well-defined at degree zero.
            denominator = degree + 1.0
            for dim in range(3):
                local_centroid = (
                    position_sums[boid, dim] + positions[boid, dim]
                ) / denominator
                goal_force[dim] = waypoints[current_goal, dim] - local_centroid
            _clamp_vector_in_place(goal_force, max_force)

            proposed0 = velocities[boid, 0] + dt * (
                0.35 * sep[0]
                + 0.35 * alignment[0]
                + 0.001 * cohesion[0]
                + 0.01 * goal_force[0]
            )
            proposed1 = velocities[boid, 1] + dt * (
                0.35 * sep[1]
                + 0.35 * alignment[1]
                + 0.001 * cohesion[1]
                + 0.01 * goal_force[1]
            )
            proposed2 = velocities[boid, 2] + dt * (
                0.35 * sep[2]
                + 0.35 * alignment[2]
                + 0.001 * cohesion[2]
                + 0.01 * goal_force[2]
            )

            old0 = velocities[boid, 0]
            old1 = velocities[boid, 1]
            old2 = velocities[boid, 2]
            old_speed = np.sqrt(old0 * old0 + old1 * old1 + old2 * old2)
            new_speed = np.sqrt(
                proposed0 * proposed0
                + proposed1 * proposed1
                + proposed2 * proposed2
            )

            if old_speed > 1e-8:
                old_dir0 = old0 / old_speed
                old_dir1 = old1 / old_speed
                old_dir2 = old2 / old_speed
            else:
                old_dir0, old_dir1, old_dir2 = 1.0, 0.0, 0.0
            if new_speed > 1e-8:
                new_dir0 = proposed0 / new_speed
                new_dir1 = proposed1 / new_speed
                new_dir2 = proposed2 / new_speed
            else:
                new_dir0, new_dir1, new_dir2 = old_dir0, old_dir1, old_dir2

            cosine = (
                old_dir0 * new_dir0
                + old_dir1 * new_dir1
                + old_dir2 * new_dir2
            )
            cosine = min(1.0, max(-1.0, cosine))
            angle = np.arccos(cosine)
            if angle > max_turn_radians:
                # Rotate the old direction by exactly max_turn toward the new one,
                # within the plane of the two directions.
                perp0 = new_dir0 - cosine * old_dir0
                perp1 = new_dir1 - cosine * old_dir1
                perp2 = new_dir2 - cosine * old_dir2
                perp_norm = np.sqrt(perp0 * perp0 + perp1 * perp1 + perp2 * perp2)
                if perp_norm <= 1e-8:
                    # Opposite directions span no unique plane: turn toward a
                    # fixed perpendicular, as in modules/boids_3d.py.
                    if abs(old_dir0) < 0.9:
                        axis0, axis1, axis2 = 1.0, 0.0, 0.0
                    else:
                        axis0, axis1, axis2 = 0.0, 1.0, 0.0
                    along = axis0 * old_dir0 + axis1 * old_dir1 + axis2 * old_dir2
                    perp0 = axis0 - along * old_dir0
                    perp1 = axis1 - along * old_dir1
                    perp2 = axis2 - along * old_dir2
                    perp_norm = np.sqrt(perp0 * perp0 + perp1 * perp1 + perp2 * perp2)
                cos_turn = np.cos(max_turn_radians)
                sin_turn = np.sin(max_turn_radians)
                proposed0 = (cos_turn * old_dir0 + sin_turn * perp0 / perp_norm) * new_speed
                proposed1 = (cos_turn * old_dir1 + sin_turn * perp1 / perp_norm) * new_speed
                proposed2 = (cos_turn * old_dir2 + sin_turn * perp2 / perp_norm) * new_speed

            limited_speed = np.sqrt(
                proposed0 * proposed0
                + proposed1 * proposed1
                + proposed2 * proposed2
            )
            if limited_speed < min_speed and limited_speed > 0.0:
                factor = min_speed / limited_speed
                proposed0 *= factor
                proposed1 *= factor
                proposed2 *= factor
            elif limited_speed > max_speed:
                factor = max_speed / limited_speed
                proposed0 *= factor
                proposed1 *= factor
                proposed2 *= factor

            velocities[boid, 0] = proposed0
            velocities[boid, 1] = proposed1
            velocities[boid, 2] = proposed2

        for boid in range(n_boids):
            for dim in range(3):
                if active_mask[boid]:
                    positions[boid, dim] += velocities[boid, dim] * dt
                positions[boid, dim] = min(
                    borders[dim + 3], max(borders[dim], positions[boid, dim])
                )
                states[step + 1, boid, dim] = positions[boid, dim]
                states[step + 1, boid, dim + 3] = velocities[boid, dim]

        centroid0 = 0.0
        centroid1 = 0.0
        centroid2 = 0.0
        active_count = 0
        for boid in range(n_boids):
            if not active_mask[boid]:
                continue
            centroid0 += positions[boid, 0]
            centroid1 += positions[boid, 1]
            centroid2 += positions[boid, 2]
            active_count += 1
        if active_count > 0:
            centroid0 /= active_count
            centroid1 /= active_count
            centroid2 /= active_count
        dx = centroid0 - waypoints[current_goal, 0]
        dy = centroid1 - waypoints[current_goal, 1]
        dz = centroid2 - waypoints[current_goal, 2]
        # The next waypoint becomes active when the centroid of the active
        # agents is within the arrival radius; evaluation scores the same way.
        arrival_distance = np.sqrt(dx * dx + dy * dy + dz * dz)
        if active_count > 0 and arrival_distance <= arrival_radius:
            current_goal += 1
            running_previous_distance = 0.0
            if current_goal == n_goals:
                completed_steps = step + 1
                return (
                    states[: completed_steps + 1].copy(),
                    input_states[:completed_steps].copy(),
                    active_masks[:completed_steps].copy(),
                    active_goals[:completed_steps].copy(),
                    previous_goals[:completed_steps].copy(),
                    previous_max_distances[:completed_steps].copy(),
                    goal_segments[:completed_steps].copy(),
                    adjacency_bits[:completed_steps].copy(),
                    current_goal,
                )

    return (
        states.copy(),
        input_states.copy(),
        active_masks.copy(),
        active_goals.copy(),
        previous_goals.copy(),
        previous_max_distances.copy(),
        goal_segments.copy(),
        adjacency_bits.copy(),
        current_goal,
    )


def rollout_waypoints(
    initial_positions: np.ndarray,
    initial_velocities: np.ndarray,
    waypoints: Sequence[Sequence[float]],
    *,
    sampled_center: Optional[Sequence[float]] = None,
    arrival_radius: float = 0.5,
    max_steps: int = 50_000,
    min_speed: float = 0.0001,
    max_speed: float = 0.01,
    max_force: float = 0.1,
    max_turn_degrees: float = 15.0,
    perception: float = 0.1,
    crowding: float = 0.02,
    dt: float = 1.0,
    borders: Sequence[float] = (-5.0, -5.0, -5.0, 5.0, 5.0, 5.0),
    noise_seed: int = 0,
    perturbation: Optional[Mapping[str, object]] = None,
) -> CompiledTrajectory3D:
    """Roll out arbitrary sequential 3D waypoints with unchanged Boids rules."""
    positions = np.ascontiguousarray(initial_positions, dtype=np.float64)
    velocities = np.ascontiguousarray(initial_velocities, dtype=np.float64)
    goals = np.ascontiguousarray(_validate_goals(waypoints), dtype=np.float64)
    border_array = np.ascontiguousarray(borders, dtype=np.float64)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("initial_positions must have shape (N, 3).")
    if velocities.shape != positions.shape:
        raise ValueError("initial_velocities must match initial_positions.")
    if border_array.shape != (6,):
        raise ValueError("borders must have shape (6,).")
    if arrival_radius <= 0 or max_steps < 1:
        raise ValueError("arrival_radius and max_steps must be positive.")

    perturbation = {} if perturbation is None else perturbation
    mode = str(perturbation.get("mode", "normal"))
    mode_code = {"normal": 0, "removal": 1, "drift": 2}[mode]
    result = _rollout_kernel_3d(
        positions,
        velocities,
        goals,
        float(arrival_radius),
        int(max_steps),
        float(min_speed),
        float(max_speed),
        float(max_force),
        float(max_turn_degrees),
        float(perception),
        float(crowding),
        float(dt),
        border_array,
        int(noise_seed) % (2**32 - 1),
        mode_code,
        int(perturbation.get("interval", 1)),
        float(perturbation.get("probability", 0.0)),
        -1 if perturbation.get("removal_max_per_event") is None else int(perturbation["removal_max_per_event"]),
        -1 if perturbation.get("removal_max_total") is None else int(perturbation["removal_max_total"]),
        -1 if perturbation.get("drift_max_per_event") is None else int(perturbation["drift_max_per_event"]),
        np.asarray(perturbation.get("drift_bounds", (0.0, 0.0, 0.0)), dtype=np.float64),
        bool(perturbation.get("drift_perturb_velocity", False)),
        (None if perturbation.get("drift_velocity_bounds") is None else
         np.asarray(perturbation["drift_velocity_bounds"], dtype=np.float64)),
    )
    (
        states,
        input_states,
        active_masks,
        active_goals,
        previous_goals,
        previous_max_distances,
        goal_segments,
        adjacency_bits,
        completed_goals,
    ) = result
    if completed_goals != len(goals):
        raise RuntimeError(
            f"Compiled expert reached {completed_goals}/{len(goals)} waypoints "
            f"within max_steps={max_steps}."
        )
    center = (
        positions.mean(axis=0)
        if sampled_center is None
        else np.asarray(sampled_center, dtype=np.float32)
    )
    return CompiledTrajectory3D(
        states=states,
        input_states=input_states,
        active_masks=active_masks,
        active_goals=active_goals,
        previous_goals=previous_goals,
        max_previous_goal_distances=previous_max_distances,
        goal_segment_ids=goal_segments,
        adjacency_bits=adjacency_bits,
        waypoints=np.asarray(goals, dtype=np.float32),
        sampled_center=np.asarray(center, dtype=np.float32),
    )


def generate_fixed_trajectory(
    config: Mapping[str, object],
) -> CompiledTrajectory3D:
    """Generate one fixed-task trajectory from a cloud-worker-style config."""
    seed = int(config["seed"])
    n_boids = int(config.get("n_boids", 100))
    goals = _validate_goals(config.get("goals", DEFAULT_FIXED_GOALS))
    center, positions, velocities = sample_fixed_initial_state(
        seed,
        octant=int(config["octant"]),
        n_boids=n_boids,
        goals=goals,
        exclusion_radius=float(config.get("goal_exclusion_size", 0.5)),
        init_scatter=float(config.get("init_scatter", 0.325)),
        max_speed=float(config.get("max_speed", 0.01)),
        discard_initial_clump_draw=bool(
            config.get("discard_initial_clump_draw", True)
        ),
    )
    return rollout_waypoints(
        positions,
        velocities,
        goals,
        sampled_center=center,
        arrival_radius=float(config.get("goal_arrival_radius", 0.5)),
        max_steps=int(config.get("expert_max_steps", 50_000)),
        min_speed=float(config.get("min_speed", 0.0001)),
        max_speed=float(config.get("max_speed", 0.01)),
        max_force=float(config.get("max_force", 0.1)),
        max_turn_degrees=float(config.get("max_turn", 15.0)),
        perception=float(config.get("perception", 0.1)),
        crowding=float(config.get("crowding", 0.02)),
        dt=float(config.get("dt", 1.0)),
        noise_seed=seed,
        perturbation=config.get("perturbation"),
    )


def generate_online_trajectory(
    config: Mapping[str, object],
) -> CompiledTrajectory3D:
    """Generate one arbitrary two-online-waypoint 3D expert episode."""
    seed = int(config["seed"])
    center, positions, velocities, waypoints = sample_online_initial_state_and_goals(
        seed,
        n_boids=int(config.get("n_boids", 100)),
        n_waypoints=int(config.get("n_waypoints", 2)),
        start_bounds=config.get(
            "start_bounds", (-5.0, 5.0, -5.0, 5.0, -5.0, 5.0)
        ),
        goal_bounds=config.get(
            "goal_bounds", (-4.5, 4.5, -4.5, 4.5, -4.5, 4.5)
        ),
        goal_min_distance=float(config.get("goal_min_distance", 2.5)),
        init_scatter=float(config.get("init_scatter", 0.325)),
        max_speed=float(config.get("max_speed", 0.01)),
    )
    return rollout_waypoints(
        positions,
        velocities,
        waypoints,
        sampled_center=center,
        arrival_radius=float(config.get("goal_arrival_radius", 0.5)),
        max_steps=int(config.get("expert_max_steps", 10_000)),
        min_speed=float(config.get("min_speed", 0.0001)),
        max_speed=float(config.get("max_speed", 0.01)),
        max_force=float(config.get("max_force", 0.1)),
        max_turn_degrees=float(config.get("max_turn", 15.0)),
        perception=float(config.get("perception", 0.1)),
        crowding=float(config.get("crowding", 0.02)),
        dt=float(config.get("dt", 1.0)),
        noise_seed=seed,
        perturbation=config.get("perturbation"),
    )


def unpack_adjacency(adjacency_bits: np.ndarray, n_boids: int) -> np.ndarray:
    """Decode little-endian packed rows to ``(..., N, N)`` Boolean masks."""
    packed = np.asarray(adjacency_bits, dtype=np.uint8)
    flat = np.unpackbits(packed, axis=-1, bitorder="little")
    flat = flat[..., : n_boids * n_boids]
    return flat.reshape(packed.shape[:-1] + (n_boids, n_boids)).astype(bool)


def materialize_goal_conditioned_arrays(
    trajectory: CompiledTrajectory3D,
) -> tuple[np.ndarray, np.ndarray]:
    """Build online-GNCA inputs and transition-aware targets on demand.

    Inputs are ``[position, velocity, active_goal - position]`` (9 features).
    Targets are ``[current physical state, next physical state, active goal,
    previous goal, max distance from previous goal, active mask]`` (20 features).
    """
    model_input = trajectory.input_states
    following = trajectory.states[1:]
    goals = trajectory.active_goals[:, None, :]
    goal_vectors = goals - model_input[..., :3]
    inputs = np.concatenate((model_input, goal_vectors), axis=-1).astype(
        np.float32, copy=False
    )
    n_boids = trajectory.n_boids
    active = np.broadcast_to(goals, (trajectory.steps, n_boids, 3))
    previous = np.broadcast_to(
        trajectory.previous_goals[:, None, :], (trajectory.steps, n_boids, 3)
    )
    previous_max = np.broadcast_to(
        trajectory.max_previous_goal_distances[:, None, None],
        (trajectory.steps, n_boids, 1),
    )
    targets = np.concatenate(
        (
            model_input,
            following,
            active,
            previous,
            previous_max,
            trajectory.active_masks[..., None],
        ),
        axis=-1,
    ).astype(np.float32, copy=False)
    return np.ascontiguousarray(inputs), np.ascontiguousarray(targets)


def warm_up_compiled_expert_3d() -> float:
    """Compile the zero-noise rollout before starting the measured pipeline.

    Numba specializes on dtype and rank rather than the concrete number of
    boids, so this tiny one-step rollout compiles the same kernel signature used
    by a production 100-boid trajectory without generating a training sample.
    The returned value is compilation wall time in seconds.
    """
    positions = np.asarray(
        [
            [-0.015, 0.000, 0.000],
            [0.015, 0.000, 0.000],
            [0.000, -0.015, 0.000],
            [0.000, 0.015, 0.000],
        ],
        dtype=np.float64,
    )
    velocities = np.zeros((4, 3), dtype=np.float64)
    velocities[:, 0] = 0.01
    started = time.perf_counter()
    rollout_waypoints(
        positions,
        velocities,
        np.zeros((1, 3), dtype=np.float32),
        arrival_radius=100.0,
        max_steps=1,
    )
    return time.perf_counter() - started
