"""Numba-compiled online-goal Boids experts for the 2D cloud backend.

The kernel implements the equations in :mod:`modules.boids` and builds the
neighborhood graph inside the compiled loop.  Adjacency is retained as a
little-endian bit-packed mask and decoded lazily by the TensorFlow dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
from numba import njit


@dataclass(frozen=True)
class CompiledTrajectory2D:
    """Use input_states[t] -> states[t + 1] for each expert transition.

    At an event input_states[t] includes the impulse; states[t] remains the
    preceding transition's output so its supervision is not overwritten.
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
    def steps(self):
        return int(self.active_goals.shape[0])

    @property
    def n_boids(self):
        return int(self.states.shape[1])


def _validate_bounds(bounds: Sequence[float], name: str) -> np.ndarray:
    values = np.asarray(bounds, dtype=np.float64)
    if values.shape != (4,) or not (
        values[0] < values[1] and values[2] < values[3]
    ):
        raise ValueError(f"{name} must contain two increasing min/max pairs.")
    return values


def sample_online_initial_state_and_goals(
    seed,
    *,
    n_boids=100,
    n_waypoints=2,
    start_bounds=(-5.0, 5.0, -5.0, 5.0),
    goal_bounds=(-4.5, 4.5, -4.5, 4.5),
    goal_min_distance=2.5,
    init_scatter=0.325,
    max_speed=0.01,
):
    """Match the seeded initial-state and waypoint streams of ``Boids``."""
    if n_boids < 1 or n_waypoints < 1:
        raise ValueError("n_boids and n_waypoints must be positive.")
    start = _validate_bounds(start_bounds, "start_bounds")
    goal_box = _validate_bounds(goal_bounds, "goal_bounds")
    seed = int(seed) % (2**32 - 1)

    state_rng = np.random.RandomState(seed)
    center = np.asarray(
        [
            state_rng.uniform(start[0], start[1]),
            state_rng.uniform(start[2], start[3]),
        ],
        dtype=np.float64,
    )
    positions = center + float(init_scatter) * state_rng.rand(n_boids, 2)
    velocities = np.zeros((n_boids, 2), dtype=np.float64)
    velocities[:, 0] = float(max_speed)

    waypoint_rng = np.random.default_rng(seed)
    waypoints = np.empty((n_waypoints, 2), dtype=np.float32)
    reference = positions.mean(axis=0)
    for waypoint_idx in range(n_waypoints):
        for _ in range(10_000):
            candidate = np.asarray(
                [
                    waypoint_rng.uniform(goal_box[0], goal_box[1]),
                    waypoint_rng.uniform(goal_box[2], goal_box[3]),
                ],
                dtype=np.float32,
            )
            if np.linalg.norm(candidate - reference) >= goal_min_distance:
                waypoints[waypoint_idx] = candidate
                reference = candidate
                break
        else:
            raise RuntimeError("Could not sample a sufficiently separated waypoint.")
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
    neighbors[:, :] = 0
    degrees[:] = 0
    position_sums[:, :] = 0.0
    velocity_sums[:, :] = 0.0
    separation[:, :] = 0.0
    for left in range(n_boids):
        if not active_mask[left]:
            continue
        for right in range(left + 1, n_boids):
            if not active_mask[right]:
                continue
            dx = positions[left, 0] - positions[right, 0]
            dy = positions[left, 1] - positions[right, 1]
            distance_squared = dx * dx + dy * dy
            if distance_squared < perception_squared:
                neighbors[left, right] = 1
                neighbors[right, left] = 1
                degrees[left] += 1
                degrees[right] += 1
                for dim in range(2):
                    position_sums[left, dim] += positions[right, dim]
                    position_sums[right, dim] += positions[left, dim]
                    velocity_sums[left, dim] += velocities[right, dim]
                    velocity_sums[right, dim] += velocities[left, dim]
                if distance_squared < crowding_squared:
                    separation[left, 0] += dx
                    separation[left, 1] += dy
                    separation[right, 0] -= dx
                    separation[right, 1] -= dy


@njit(cache=True, nogil=True)
def _clamp(vector, maximum):
    norm = np.sqrt(vector[0] * vector[0] + vector[1] * vector[1])
    if norm > maximum:
        vector *= maximum / norm


@njit(cache=True, nogil=True)
def _pack_adjacency(neighbors, destination):
    destination[:] = 0
    n_boids = neighbors.shape[0]
    for row in range(n_boids):
        for col in range(n_boids):
            if neighbors[row, col]:
                flat = row * n_boids + col
                destination[flat >> 3] |= np.uint8(1 << (flat & 7))


@njit(cache=True, nogil=True)
def _rollout_kernel_2d(
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
    states = np.empty((max_steps + 1, n_boids, 4), np.float32)
    input_states = np.empty((max_steps, n_boids, 4), np.float32)
    active_masks = np.ones((max_steps, n_boids), np.float32)
    active_goals = np.empty((max_steps, 2), np.float32)
    previous_goals = np.zeros((max_steps, 2), np.float32)
    previous_max = np.full(max_steps, -1.0, np.float32)
    goal_segments = np.empty(max_steps, np.int16)
    adjacency_bits = np.empty((max_steps, packed_width), np.uint8)

    positions = initial_positions.copy()
    velocities = initial_velocities.copy()
    neighbors = np.zeros((n_boids, n_boids), np.uint8)
    degrees = np.zeros(n_boids, np.int32)
    position_sums = np.zeros((n_boids, 2), np.float64)
    velocity_sums = np.zeros((n_boids, 2), np.float64)
    separation = np.zeros((n_boids, 2), np.float64)
    states[0, :, :2] = positions
    states[0, :, 2:] = velocities
    current_goal = 0
    running_previous_distance = -1.0
    active_mask = np.ones(n_boids, np.bool_)
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
            selected = np.empty(n_boids, np.int32)
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
                    for dim in range(2):
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

        input_states[step, :, :2] = model_positions
        input_states[step, :, 2:] = model_velocities
        active_masks[step] = active_mask
        # Evolve the same disturbed state given to the GNCA, including its
        # velocity and graph. The expert response persists into later steps.
        positions = model_positions
        velocities = model_velocities
        _fill_graph_and_sums(
            positions,
            velocities,
            active_mask,
            perception * perception,
            crowding * crowding,
            neighbors,
            degrees,
            position_sums,
            velocity_sums,
            separation,
        )
        _pack_adjacency(neighbors, adjacency_bits[step])
        active_goals[step] = waypoints[current_goal]
        goal_segments[step] = current_goal
        if current_goal > 0:
            centroid_x = 0.0
            centroid_y = 0.0
            active_count = 0
            for boid in range(n_boids):
                if not active_mask[boid]:
                    continue
                centroid_x += positions[boid, 0]
                centroid_y += positions[boid, 1]
                active_count += 1
            centroid_x /= active_count
            centroid_y /= active_count
            distance = np.sqrt(
                (centroid_x - waypoints[current_goal - 1, 0]) ** 2
                + (centroid_y - waypoints[current_goal - 1, 1]) ** 2
            )
            running_previous_distance = max(running_previous_distance, distance)
            previous_goals[step] = waypoints[current_goal - 1]
            previous_max[step] = running_previous_distance

        for boid in range(n_boids):
            if not active_mask[boid]:
                velocities[boid] = 0.0
                continue
            degree = degrees[boid]
            sep = separation[boid].copy()
            _clamp(sep, max_force)
            if degree > 0:
                alignment = velocity_sums[boid] / degree - velocities[boid]
                cohesion = position_sums[boid] / degree - positions[boid]
                goal_force = (
                    waypoints[current_goal]
                    - (position_sums[boid] + positions[boid]) / (degree + 1.0)
                )
            else:
                # An isolated agent still knows its goal; only the
                # neighbor-dependent alignment/cohesion forces vanish.
                alignment = np.zeros(2, np.float64)
                cohesion = np.zeros(2, np.float64)
                goal_force = waypoints[current_goal] - positions[boid]
            _clamp(goal_force, max_force)
            _clamp(alignment, max_force)
            _clamp(cohesion, max_force)
            proposed = velocities[boid] + dt * (
                0.35 * sep
                + 0.35 * alignment
                + 0.001 * cohesion
                + 0.01 * goal_force
            )

            old_angle = np.arctan2(velocities[boid, 1], velocities[boid, 0])
            new_angle = np.arctan2(proposed[1], proposed[0])
            angle_delta = (new_angle - old_angle + np.pi) % (2.0 * np.pi) - np.pi
            max_turn = max_turn_degrees * np.pi / 180.0
            speed = np.sqrt(proposed[0] ** 2 + proposed[1] ** 2)
            if abs(angle_delta) > max_turn:
                new_angle = old_angle + (max_turn if angle_delta > 0 else -max_turn)
                proposed[0] = speed * np.cos(new_angle)
                proposed[1] = speed * np.sin(new_angle)
            if speed < min_speed and speed > 0.0:
                proposed *= min_speed / speed
            elif speed > max_speed:
                proposed *= max_speed / speed
            velocities[boid] = proposed

        for boid in range(n_boids):
            if active_mask[boid]:
                positions[boid] += velocities[boid] * dt
        for boid in range(n_boids):
            positions[boid, 0] = min(borders[2], max(borders[0], positions[boid, 0]))
            positions[boid, 1] = min(borders[3], max(borders[1], positions[boid, 1]))
        states[step + 1, :, :2] = positions
        states[step + 1, :, 2:] = velocities

        centroid_x = 0.0
        centroid_y = 0.0
        active_count = 0
        for boid in range(n_boids):
            if not active_mask[boid]:
                continue
            centroid_x += positions[boid, 0]
            centroid_y += positions[boid, 1]
            active_count += 1
        if active_count > 0:
            centroid_x /= active_count
            centroid_y /= active_count
        dx = centroid_x - waypoints[current_goal, 0]
        dy = centroid_y - waypoints[current_goal, 1]
        # The next waypoint becomes active when the centroid of the active
        # agents is within the arrival radius; evaluation scores the same way.
        arrival_distance = np.sqrt(dx * dx + dy * dy)
        if active_count > 0 and arrival_distance <= arrival_radius:
            current_goal += 1
            running_previous_distance = 0.0
            if current_goal == n_goals:
                completed = step + 1
                return (
                    states[: completed + 1].copy(),
                    input_states[:completed].copy(),
                    active_masks[:completed].copy(),
                    active_goals[:completed].copy(),
                    previous_goals[:completed].copy(),
                    previous_max[:completed].copy(),
                    goal_segments[:completed].copy(),
                    adjacency_bits[:completed].copy(),
                    current_goal,
                )
    return (
        states,
        input_states,
        active_masks,
        active_goals,
        previous_goals,
        previous_max,
        goal_segments,
        adjacency_bits,
        current_goal,
    )


def rollout_waypoints(config, center, positions, velocities, waypoints):
    """Run either task with the same compiled Boids update equations."""
    seed = int(config["seed"])
    perturbation = config.get("perturbation", {})
    mode = str(perturbation.get("mode", "normal"))
    mode_code = {"normal": 0, "removal": 1, "drift": 2}[mode]
    result = _rollout_kernel_2d(
        np.ascontiguousarray(positions),
        np.ascontiguousarray(velocities),
        np.ascontiguousarray(waypoints, dtype=np.float64),
        float(config.get("goal_arrival_radius", 0.5)),
        int(config.get("expert_max_steps", 10_000)),
        float(config.get("min_speed", 0.0001)),
        float(config.get("max_speed", 0.01)),
        float(config.get("max_force", 0.1)),
        float(config.get("max_turn", 5.0)),
        float(config.get("perception", 0.1)),
        float(config.get("crowding", 0.02)),
        float(config.get("dt", 1.0)),
        np.asarray((-5.0, -5.0, 5.0, 5.0), dtype=np.float64),
        seed % (2**32 - 1),
        mode_code,
        int(perturbation.get("interval", 1)),
        float(perturbation.get("probability", 0.0)),
        -1 if perturbation.get("removal_max_per_event") is None else int(perturbation["removal_max_per_event"]),
        -1 if perturbation.get("removal_max_total") is None else int(perturbation["removal_max_total"]),
        -1 if perturbation.get("drift_max_per_event") is None else int(perturbation["drift_max_per_event"]),
        np.asarray(perturbation.get("drift_bounds", (0.0, 0.0)), dtype=np.float64),
        bool(perturbation.get("drift_perturb_velocity", False)),
        (None if perturbation.get("drift_velocity_bounds") is None else
         np.asarray(perturbation["drift_velocity_bounds"], dtype=np.float64)),
    )
    states, input_states, active_masks, goals, previous, previous_max, segments, bits, completed = result
    if completed != len(waypoints):
        raise RuntimeError(
            f"Compiled 2D expert reached {completed}/{len(waypoints)} waypoints."
        )
    return CompiledTrajectory2D(
        states=states,
        input_states=input_states,
        active_masks=active_masks,
        active_goals=goals,
        previous_goals=previous,
        max_previous_goal_distances=previous_max,
        goal_segment_ids=segments,
        adjacency_bits=bits,
        waypoints=waypoints,
        sampled_center=np.asarray(center, dtype=np.float32),
    )


def generate_online_trajectory(config: Mapping[str, object]) -> CompiledTrajectory2D:
    initial = sample_online_initial_state_and_goals(
        int(config["seed"]),
        n_boids=int(config.get("n_boids", 100)),
        n_waypoints=int(config.get("n_waypoints", 2)),
        start_bounds=config.get("start_bounds", (-5.0, 5.0, -5.0, 5.0)),
        goal_bounds=config.get("goal_bounds", (-4.5, 4.5, -4.5, 4.5)),
        goal_min_distance=float(config.get("goal_min_distance", 2.5)),
        init_scatter=float(config.get("init_scatter", 0.325)),
        max_speed=float(config.get("max_speed", 0.01)),
    )
    return rollout_waypoints(config, *initial)


def sample_fixed_initial_state(config):
    """Reuse centers across repeats, with fresh scatter for every episode."""
    from modules.boids import BOIDS_GOAL_POSITIONS
    from shapely.geometry import Point, Polygon

    bounds = _validate_bounds(config["start_bounds"], "start_bounds")
    center_rng = np.random.RandomState(int(config.get("center_seed", config["seed"])))
    exclusion = float(config.get("goal_exclusion_size", 0.5))
    polygon = Polygon(BOIDS_GOAL_POSITIONS).buffer(exclusion) if exclusion > 0 else None
    for _ in range(10_000):
        center = center_rng.uniform(bounds[::2], bounds[1::2])
        if polygon is None or not polygon.contains(Point(center)):
            break
    else:
        raise ValueError("No valid fixed-2D center in the requested bounds/exclusion zone.")
    rng = np.random.RandomState(int(config["seed"]))
    n_boids = int(config.get("n_boids", 100))
    positions = center + float(config.get("init_scatter", 0.325)) * rng.rand(n_boids, 2)
    velocities = np.zeros((n_boids, 2), dtype=np.float64)
    velocities[:, 0] = float(config.get("max_speed", 0.01))
    return center, positions, velocities, BOIDS_GOAL_POSITIONS.copy()


def generate_fixed_trajectory(config):
    return rollout_waypoints(config, *sample_fixed_initial_state(config))


def warm_up_compiled_expert_2d():
    positions = np.asarray(
        [[-0.015, 0.0], [0.015, 0.0], [0.0, -0.015], [0.0, 0.015]],
        dtype=np.float64,
    )
    velocities = np.zeros((4, 2), dtype=np.float64)
    velocities[:, 0] = 0.01
    started = __import__("time").perf_counter()
    _rollout_kernel_2d(
        positions,
        velocities,
        np.zeros((1, 2), dtype=np.float64),
        100.0,
        1,
        0.0001,
        0.01,
        0.1,
        5.0,
        0.1,
        0.02,
        1.0,
        np.asarray((-5.0, -5.0, 5.0, 5.0)),
        0,
        0,
        1,
        0.0,
        -1,
        -1,
        -1,
        np.zeros(2, dtype=np.float64),
        False,
    )
    return __import__("time").perf_counter() - started
