"""The multi-step objective of the BPTT stage.

One-step training scores a single transition: the model reads an expert state and is
compared with the next expert state. Rollout training instead takes a window of
consecutive expert transitions, starts the model from the expert's state at the
beginning of the window, feeds the model its own prediction for the remaining steps,
rebuilds the neighborhood graph from the predicted positions after every step, and
averages the per-step transition loss over the window. Gradients flow through the
whole unrolled sequence. Agents inactive at a step are pinned to the expert's
recorded state and take no part in the graph or the loss at that step.
"""

import tensorflow as tf


def radius_graph(positions, active, perception):
    """Disjoint neighbor graph of a batch of flocks, as the Boids expert defines it.

    Two active agents of the same flock are neighbors when their distance is
    below ``perception``; there are no self edges. ``positions`` has shape
    ``(batch, n_boids, dimension)`` and ``active`` is a boolean
    ``(batch, n_boids)`` mask. Indices come out in row-major order, which is the
    canonical ordering TensorFlow sparse operations expect.
    """
    n_boids = positions.shape[1]
    delta = positions[:, :, None, :] - positions[:, None, :, :]
    within = tf.reduce_sum(tf.square(delta), axis=-1) < tf.cast(perception, positions.dtype) ** 2
    pair_active = active[:, :, None] & active[:, None, :]
    agents = tf.range(n_boids)
    not_self = tf.not_equal(agents[:, None], agents[None, :])[None]
    local = tf.where(within & pair_active & not_self)
    offsets = local[:, 0] * n_boids
    indices = tf.stack((offsets + local[:, 1], offsets + local[:, 2]), axis=1)
    n_nodes = tf.shape(positions, out_type=tf.int64)[0] * n_boids
    return tf.SparseTensor(
        indices=indices,
        values=tf.ones([tf.shape(indices)[0]], dtype=tf.float32),
        dense_shape=tf.stack((n_nodes, n_nodes)),
    )


def make_rollout_forward(model, loss_fn, rollout_steps, perception, goal_conditioned, dimension):
    """Return ``forward(inputs, targets, training)`` for the rollout objective.

    The returned function unrolls ``model`` for ``rollout_steps`` steps from the
    window's first expert state and returns the per-node transition loss averaged
    over the steps, in the layout ``loss_fn`` produces for one-step batches, so
    the surrounding training loop and its normalization are unchanged.
    """
    rollout_steps, dimension = int(rollout_steps), int(dimension)
    physical = 2 * dimension
    perception = float(perception)

    def forward(inputs, targets, training):
        features, adjacency, graph_ids = inputs
        expert_states, active_goals, previous_goals, max_previous, active_masks = targets
        batch = tf.shape(expert_states)[0]
        n_boids = expert_states.shape[2]
        graph_column = tf.cast(graph_ids[:, None], tf.float32)
        state = features[:, :physical]
        step_losses = []
        for step in range(rollout_steps):
            goal = tf.reshape(tf.broadcast_to(active_goals[:, step][:, None, :],
                                              [batch, n_boids, dimension]), [-1, dimension])
            previous = tf.reshape(tf.broadcast_to(previous_goals[:, step][:, None, :],
                                                  [batch, n_boids, dimension]), [-1, dimension])
            max_distance = tf.reshape(tf.broadcast_to(max_previous[:, step][:, None],
                                                      [batch, n_boids]), [-1, 1])
            mask = tf.reshape(active_masks[:, step], [-1, 1])
            target = tf.reshape(expert_states[:, step], [-1, physical])
            model_input = (tf.concat((state, goal - state[:, :dimension]), axis=-1)
                           if goal_conditioned else state)
            prediction = model([model_input, adjacency, graph_ids], training=training)
            y_true = tf.concat(
                (tf.stop_gradient(state), target, goal, previous, max_distance, mask, graph_column),
                axis=-1,
            )
            step_losses.append(loss_fn(y_true, prediction))
            state = mask * prediction + (1.0 - mask) * target
            if step + 1 < rollout_steps:
                adjacency = radius_graph(
                    tf.reshape(state[:, :dimension], [batch, n_boids, dimension]),
                    active_masks[:, step + 1] > 0.5,
                    perception,
                )
        return tf.add_n(step_losses) / float(rollout_steps)

    return forward
