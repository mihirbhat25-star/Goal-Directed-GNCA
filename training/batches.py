"""Bit-packed, lazy batches of expert data for both training stages.

Every trajectory of a chunk stays in host RAM with its adjacency bit-packed; TensorFlow
gathers one batch at a time and decodes its graphs into one disjoint ``tf.SparseTensor``.
The layout is the same in 2D and 3D, with ``d`` the dimension. Model inputs are
``[position, velocity]``, followed by ``[goal - position]`` for goal-conditioned
models. One-step targets, read by ``training.loss``, are ``[state (2d), next state (2d),
active goal (d), previous goal (d), max previous distance, active mask, graph id]``.

One-step batches hold single transitions, subsampled by a timestep stride far from the
goals. Rollout batches hold windows of consecutive transitions, tiling each trajectory
without overlap (with a seeded random offset per trajectory), so one epoch scores every
expert transition about once. Windows that contain an expert-side disturbance after
their first step are skipped, since the expert's response to an impulse the model never
receives is not a valid target.

When the model receives no goal input (the fixed-waypoint task), the expert's waypoint
switch is likewise an event the model cannot observe: it is triggered by the distance
of the whole flock's centroid to the goal. A window that crossed a switch would make
the model pay for every step after a turn whose moment it has to guess, and the
error-minimizing response to that is to slow down and hedge near every waypoint, which
in closed loop becomes a stall. Such windows therefore end at the switch: the segment
of transitions leading up to each switch is tiled backwards from the switch transition,
so the turn itself is scored as a one-step transition with nothing accumulating behind
it, and the next window starts from the expert's state after the turn. The segment
after the last switch is tiled forward with the random offset. Goal-conditioned models
observe the switch through their goal input, so their windows are not aligned.
"""

import time

import numpy as np
import tensorflow as tf


_BIT_MASKS = tf.constant(
    [1, 2, 4, 8, 16, 32, 64, 128], dtype=tf.uint8
)


def _trajectory_arrays(trajectory):
    """The arrays of one compiled trajectory, with their training dtypes."""
    return (
        np.asarray(trajectory.states, dtype=np.float32),
        np.asarray(trajectory.input_states, dtype=np.float32),
        np.asarray(trajectory.active_masks, dtype=np.float32),
        np.asarray(trajectory.active_goals, dtype=np.float32),
        np.asarray(trajectory.previous_goals, dtype=np.float32),
        np.asarray(trajectory.max_previous_goal_distances, dtype=np.float32),
        np.asarray(trajectory.adjacency_bits, dtype=np.uint8),
    )


def _check_shapes(index, n_boids, dimension, arrays):
    """Every field of a trajectory must fit its number of expert steps."""
    states, input_states, active_masks, active_goals, previous_goals, max_previous, bits = arrays
    steps = len(active_goals)
    physical = 2 * dimension
    expected = (
        ("states", states, (steps + 1, n_boids, physical)),
        ("input_states", input_states, (steps, n_boids, physical)),
        ("active_masks", active_masks, (steps, n_boids)),
        ("active_goals", active_goals, (steps, dimension)),
        ("previous_goals", previous_goals, (steps, dimension)),
        ("max_previous_goal_distances", max_previous, (steps,)),
        ("adjacency_bits", bits, (steps, (n_boids * n_boids + 7) // 8)),
    )
    for name, array, shape in expected:
        if array.shape != shape:
            raise ValueError(f"Trajectory {index} has {name} shape {array.shape}; expected {shape}.")


def _decode_disjoint_indices(adjacency_bits, n_boids):
    """Decode packed masks and offset them into one disjoint sparse graph."""
    batch_size = tf.shape(adjacency_bits, out_type=tf.int64)[0]
    unpacked = tf.not_equal(
        tf.bitwise.bitwise_and(adjacency_bits[..., None], _BIT_MASKS),
        tf.constant(0, dtype=tf.uint8),
    )
    unpacked = tf.reshape(
        unpacked,
        tf.stack((batch_size, tf.constant(-1, dtype=tf.int64))),
    )
    unpacked = unpacked[:, : n_boids * n_boids]
    masks = tf.reshape(
        unpacked,
        tf.stack(
            (
                batch_size,
                tf.cast(n_boids, tf.int64),
                tf.cast(n_boids, tf.int64),
            )
        ),
    )

    # tf.where traverses the dense mask in row-major order.  Adding monotonic
    # graph offsets therefore produces indices already ordered as required by
    # TensorFlow sparse operations; no SparseReorder pass is needed.
    local_indices = tf.where(masks)
    graph_offsets = local_indices[:, 0] * tf.cast(n_boids, tf.int64)
    return tf.stack(
        (
            graph_offsets + local_indices[:, 1],
            graph_offsets + local_indices[:, 2],
        ),
        axis=1,
    )


def _disjoint_graph(adjacency_bits, batch_size, n_boids):
    indices = _decode_disjoint_indices(adjacency_bits, n_boids)
    n_nodes = batch_size * n_boids
    adjacency = tf.SparseTensor(
        indices=indices,
        values=tf.ones([tf.shape(indices)[0]], dtype=tf.float32),
        dense_shape=tf.constant([n_nodes, n_nodes], dtype=tf.int64),
    )
    graph_ids = tf.repeat(tf.range(batch_size, dtype=tf.int64), n_boids)
    return adjacency, tf.ensure_shape(graph_ids, [n_nodes])


def _lazy_dataset(sample_batches, make_batch, shuffle_batches, seed):
    """Repeating dataset that builds each batch from one row of sample indices."""
    n_batches = len(sample_batches)
    with tf.device("/CPU:0"):
        dataset = tf.data.Dataset.from_tensor_slices(sample_batches)
    dataset = dataset.map(make_batch, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    if shuffle_batches and n_batches > 1:
        dataset = dataset.shuffle(min(256, n_batches), seed=seed, reshuffle_each_iteration=True)
    options = tf.data.Options()
    # ``experimental_deterministic`` works on both TensorFlow 2.4 and the newer NVIDIA image.
    options.experimental_deterministic = False
    return dataset.with_options(options).repeat().prefetch(tf.data.AUTOTUNE)


# ------------------------------------------------------------------ one-step

def _selected_steps(
    input_states,
    active_masks,
    active_goals,
    timestep_stride,
    near_goal_radius,
    dimension,
    previous_goals,
    max_previous_distances,
):
    """Every ``timestep_stride``-th step, and all steps near the active or just-left goal."""
    n_steps = len(active_goals)
    if timestep_stride <= 1 or near_goal_radius <= 0:
        return np.arange(0, n_steps, timestep_stride, dtype=np.int32)

    mean_pos = np.sum(
        input_states[..., :dimension] * active_masks[..., None], axis=1
    ) / np.maximum(active_masks.sum(axis=1, keepdims=True), 1.0)
    distance = np.linalg.norm(mean_pos - active_goals, axis=-1)
    keep = np.zeros(n_steps, dtype=bool)
    keep[::timestep_stride] = True
    keep |= distance < near_goal_radius
    previous_distance = np.linalg.norm(mean_pos - previous_goals, axis=-1)
    keep |= ((max_previous_distances >= 0)
             & (max_previous_distances < near_goal_radius)
             & (previous_distance < near_goal_radius))
    return np.flatnonzero(keep).astype(np.int32, copy=False)


def _make_transition_batch(sample_ids, arrays, batch_size, n_boids, dimension, goal_conditioned):
    """Gather x/y lazily and decode one bit-packed disjoint graph batch."""
    sample_ids = tf.ensure_shape(sample_ids, [batch_size])
    next_states = tf.gather(arrays["states"], tf.gather(arrays["state_ids"], sample_ids) + 1)
    # Goal weighting must use the actual (possibly disturbed) input, while
    # the next-state label remains the expert response saved for this step.
    current_states = tf.gather(arrays["input_states"], sample_ids)
    broadcast = lambda values, width: tf.broadcast_to(
        tf.reshape(values, [batch_size, 1, width]), [batch_size, n_boids, width]
    )
    active_goals = broadcast(tf.gather(arrays["active_goals"], sample_ids), dimension)
    previous_goals = broadcast(tf.gather(arrays["previous_goals"], sample_ids), dimension)
    max_previous = broadcast(tf.gather(arrays["max_previous_goal_distances"], sample_ids), 1)
    active_masks = tf.gather(arrays["active_masks"], sample_ids)

    columns = [current_states]
    if goal_conditioned:
        columns.append(active_goals - current_states[..., :dimension])
    input_width = (3 if goal_conditioned else 2) * dimension
    x = tf.reshape(tf.concat(columns, axis=-1), [batch_size * n_boids, input_width])
    y = tf.reshape(
        tf.concat(
            (current_states, next_states, active_goals, previous_goals, max_previous, active_masks[..., None]),
            axis=-1,
        ),
        [batch_size * n_boids, 6 * dimension + 2],
    )
    adjacency, graph_ids = _disjoint_graph(tf.gather(arrays["adjacency_bits"], sample_ids), batch_size, n_boids)
    targets = tf.concat((y, tf.cast(graph_ids[:, None], y.dtype)), axis=-1)
    return (
        (tf.ensure_shape(x, [batch_size * n_boids, input_width]), adjacency, graph_ids),
        tf.ensure_shape(targets, [batch_size * n_boids, 6 * dimension + 3]),
    )


def transition_dataset(trajectories, batch_size, n_boids, dimension, goal_conditioned,
                       timestep_stride, near_goal_radius, seed, *, shuffle_batches=True):
    """One-step dataset of single transitions.

    ``batch_size`` is the per-replica graph count. Samples receive one fixed
    timestep-level permutation when the dataset is built; complete batches are
    reshuffled between epochs. Returns the dataset and its number of batches.
    """
    started = time.time()
    batch_size, n_boids, dimension = int(batch_size), int(n_boids), int(dimension)
    if not trajectories:
        raise ValueError("Training received no expert trajectories.")
    parts = {name: [] for name in ("states", "state_ids", "input_states", "active_masks", "active_goals",
                                   "previous_goals", "max_previous_goal_distances", "adjacency_bits")}
    state_cursor = 0
    for index, trajectory in enumerate(trajectories):
        arrays = _trajectory_arrays(trajectory)
        _check_shapes(index, n_boids, dimension, arrays)
        states, input_states, active_masks, active_goals, previous_goals, max_previous, bits = arrays
        selected = _selected_steps(input_states, active_masks, active_goals, timestep_stride, near_goal_radius,
                                   dimension, previous_goals, max_previous)
        parts["states"].append(states)
        parts["state_ids"].append((state_cursor + selected).astype(np.int32))
        parts["input_states"].append(input_states[selected])
        parts["active_masks"].append(active_masks[selected])
        parts["active_goals"].append(active_goals[selected])
        parts["previous_goals"].append(previous_goals[selected])
        parts["max_previous_goal_distances"].append(max_previous[selected])
        parts["adjacency_bits"].append(bits[selected])
        state_cursor += len(states)
    arrays = {name: np.concatenate(values) for name, values in parts.items()}

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(arrays["state_ids"]))
    usable = (len(order) // batch_size) * batch_size
    if usable < batch_size:
        raise ValueError(f"{len(order)} transitions do not fill one batch of {batch_size}.")
    sample_batches = np.ascontiguousarray(order[:usable].reshape(-1, batch_size), dtype=np.int32)
    n_batches = len(sample_batches)

    with tf.device("/CPU:0"):
        tensors = {name: tf.convert_to_tensor(value) for name, value in arrays.items()}
    dataset = _lazy_dataset(
        sample_batches,
        lambda sample_ids: _make_transition_batch(sample_ids, tensors, batch_size, n_boids, dimension,
                                                  goal_conditioned),
        shuffle_batches,
        seed,
    )
    resident = sum(value.nbytes for value in arrays.values()) + sample_batches.nbytes
    print(
        f">>> Transitions: indexed {usable} samples as {n_batches} lazy batches in "
        f"{time.time() - started:.1f}s ({resident / 1024**3:.1f} GiB resident).",
        flush=True,
    )
    return dataset, n_batches


# ------------------------------------------------------------- rollout windows

def window_starts(n_steps, rollout_steps, disturbed, offset):
    """Start steps of non-overlapping windows with no interior disturbance.

    ``disturbed`` flags steps whose model input differs from the expert state
    (an impulse). A window starting at ``t`` covers transitions ``t`` to
    ``t + rollout_steps - 1``; its first step may be disturbed, later ones may not.
    """
    starts = np.arange(int(offset), n_steps - rollout_steps + 1, rollout_steps, dtype=np.int64)
    return _drop_disturbed(starts, rollout_steps, disturbed)


def _drop_disturbed(starts, rollout_steps, disturbed):
    """Keep the windows whose steps after the first are undisturbed."""
    if len(starts) and np.any(disturbed):
        prefix = np.concatenate(([0], np.cumsum(disturbed, dtype=np.int64)))
        interior = prefix[starts + rollout_steps] - prefix[starts + 1]
        starts = starts[interior == 0]
    return starts


def switch_steps(active_goals):
    """Indices of the first transition under each new goal (the turn steps)."""
    active_goals = np.asarray(active_goals)
    if len(active_goals) < 2:
        return np.zeros(0, dtype=np.int64)
    changed = np.any(active_goals[1:] != active_goals[:-1], axis=-1)
    return np.flatnonzero(changed).astype(np.int64) + 1


def window_starts_ending_at_switches(n_steps, rollout_steps, switches, disturbed, offset):
    """Start steps of non-overlapping windows that never cross a goal switch.

    Every segment that ends with a switch transition is tiled backwards from
    that transition, so the switch is the last step of its window. The segment
    after the last switch is tiled forwards from ``offset``. Windows with an
    interior disturbance are dropped as in :func:`window_starts`.
    """
    rollout_steps = int(rollout_steps)
    starts = []
    segment_start = 0
    for switch in sorted(int(step) for step in switches):
        last_start = switch - rollout_steps + 1
        if last_start >= segment_start:
            count = (last_start - segment_start) // rollout_steps + 1
            starts.append(last_start - rollout_steps * np.arange(count, dtype=np.int64))
        segment_start = switch + 1
    starts.append(np.arange(segment_start + int(offset), n_steps - rollout_steps + 1,
                            rollout_steps, dtype=np.int64))
    starts = np.sort(np.concatenate(starts))
    return _drop_disturbed(starts, rollout_steps, disturbed)


def _flatten_windows(trajectories, n_boids, rollout_steps, dimension, rng, end_windows_at_switches):
    """Concatenate trajectories and index the window starts into them."""
    if not trajectories:
        raise ValueError("Rollout training received no expert trajectories.")
    parts = {name: [] for name in ("states", "active_goals", "previous_goals",
                                   "max_previous_goal_distances", "active_masks",
                                   "start_states", "adjacency_bits", "state_ids", "step_ids")}
    state_cursor = step_cursor = 0
    skipped = 0
    for index, trajectory in enumerate(trajectories):
        arrays = _trajectory_arrays(trajectory)
        _check_shapes(index, n_boids, dimension, arrays)
        states, input_states, active_masks, active_goals, previous_goals, max_previous, bits = arrays
        n_steps = len(active_goals)
        disturbed = np.any(input_states != states[:-1], axis=(1, 2))
        if n_steps < rollout_steps:
            starts = np.zeros(0, dtype=np.int64)
        elif end_windows_at_switches:
            starts = window_starts_ending_at_switches(
                n_steps, rollout_steps, switch_steps(active_goals), disturbed,
                rng.integers(rollout_steps))
        else:
            starts = window_starts(n_steps, rollout_steps, disturbed, rng.integers(rollout_steps))
        if not len(starts):
            skipped += 1
            continue
        parts["states"].append(states)
        parts["active_goals"].append(active_goals)
        parts["previous_goals"].append(previous_goals)
        parts["max_previous_goal_distances"].append(max_previous)
        parts["active_masks"].append(active_masks.astype(np.uint8))
        parts["start_states"].append(input_states[starts])
        parts["adjacency_bits"].append(bits[starts])
        parts["state_ids"].append(state_cursor + starts)
        parts["step_ids"].append(step_cursor + starts)
        state_cursor += n_steps + 1
        step_cursor += n_steps
    if not parts["state_ids"]:
        raise ValueError(
            f"No expert trajectory provides a window of {rollout_steps} undisturbed steps."
        )
    return {name: np.concatenate(values) for name, values in parts.items()}, skipped


def _make_window_batch(sample_ids, arrays, batch_size, n_boids, rollout_steps, dimension, goal_conditioned):
    """Gather one batch of windows and decode the expert graph of the first step."""
    sample_ids = tf.ensure_shape(sample_ids, [batch_size])
    start_states = tf.gather(arrays["start_states"], sample_ids)
    state_ids = tf.gather(arrays["state_ids"], sample_ids)
    step_ids = tf.gather(arrays["step_ids"], sample_ids)
    future = state_ids[:, None] + tf.range(1, rollout_steps + 1, dtype=state_ids.dtype)
    window = step_ids[:, None] + tf.range(rollout_steps, dtype=step_ids.dtype)
    expert_states = tf.gather(arrays["states"], future)
    active_goals = tf.gather(arrays["active_goals"], window)
    previous_goals = tf.gather(arrays["previous_goals"], window)
    max_previous = tf.gather(arrays["max_previous_goal_distances"], window)
    active_masks = tf.cast(tf.gather(arrays["active_masks"], window), tf.float32)

    columns = [start_states]
    if goal_conditioned:
        columns.append(active_goals[:, 0][:, None, :] - start_states[..., :dimension])
    input_width = (3 if goal_conditioned else 2) * dimension
    n_nodes = batch_size * n_boids
    features = tf.reshape(tf.concat(columns, axis=-1), [n_nodes, input_width])
    adjacency, graph_ids = _disjoint_graph(tf.gather(arrays["adjacency_bits"], sample_ids), batch_size, n_boids)
    inputs = (tf.ensure_shape(features, [n_nodes, input_width]), adjacency, graph_ids)
    targets = (
        tf.ensure_shape(expert_states, [batch_size, rollout_steps, n_boids, 2 * dimension]),
        tf.ensure_shape(active_goals, [batch_size, rollout_steps, dimension]),
        tf.ensure_shape(previous_goals, [batch_size, rollout_steps, dimension]),
        tf.ensure_shape(max_previous, [batch_size, rollout_steps]),
        tf.ensure_shape(active_masks, [batch_size, rollout_steps, n_boids]),
    )
    return inputs, targets


def window_dataset(trajectories, batch_size, n_boids, dimension, goal_conditioned, rollout_steps, seed,
                   *, shuffle_batches=True, min_batches=0):
    """Repeating dataset of rollout windows, and its number of batches.

    ``batch_size`` is the per-replica number of windows. Each element is
    ``((features, adjacency, graph_ids), (expert_states, active_goals,
    previous_goals, max_previous_goal_distances, active_masks))``: the first
    tuple is the ordinary one-step model input for the window's first step, the
    second holds the expert's next ``rollout_steps`` states and per-step goal
    and mask information. Models without a goal input get windows that end at the
    waypoint switches; see the module docstring.

    Few validation trajectories can give fewer batches than the replicas; with those as
    ``min_batches``, the same windows are then split into smaller batches, which the
    per-node mean loss allows.
    """
    started = time.time()
    batch_size, n_boids = int(batch_size), int(n_boids)
    rollout_steps, dimension = int(rollout_steps), int(dimension)
    if rollout_steps < 2:
        raise ValueError("Rollout windows need at least two steps.")
    rng = np.random.default_rng(seed)
    arrays, skipped = _flatten_windows(trajectories, n_boids, rollout_steps, dimension, rng,
                                       end_windows_at_switches=not goal_conditioned)
    n_windows = len(arrays["state_ids"])
    order = rng.permutation(n_windows)
    full_batches = n_windows // batch_size
    if full_batches < min_batches:
        batch_size = max(1, (batch_size * full_batches if full_batches else n_windows) // min_batches)
    usable = (n_windows // batch_size) * batch_size
    if usable < batch_size:
        raise ValueError(
            f"{n_windows} rollout windows do not fill one batch of {batch_size}; train on more trajectories."
        )
    sample_batches = np.ascontiguousarray(order[:usable].reshape(-1, batch_size), dtype=np.int32)
    n_batches = len(sample_batches)

    with tf.device("/CPU:0"):
        tensors = {name: tf.convert_to_tensor(value) for name, value in arrays.items()}
    dataset = _lazy_dataset(
        sample_batches,
        lambda sample_ids: _make_window_batch(sample_ids, tensors, batch_size, n_boids, rollout_steps, dimension,
                                              goal_conditioned),
        shuffle_batches,
        seed,
    )
    resident = sum(value.nbytes for value in arrays.values()) + sample_batches.nbytes
    print(
        f">>> Rollout windows: {usable} windows of {rollout_steps} steps as {n_batches} "
        f"batches of {batch_size} in {time.time() - started:.1f}s ({resident / 1024**3:.1f} GiB resident"
        + (f", {skipped} trajectories without a usable window" if skipped else "")
        + (", windows end at waypoint switches" if not goal_conditioned else "") + ").",
        flush=True,
    )
    return dataset, n_batches
