"""The goal-weighted transition loss, for every experiment and both dimensions."""

import tensorflow as tf


def make_transition_loss(dimension, critical_distance, distance_weight):
    """Per-node imitation loss over the target layout of ``training.batches``.

    A graph is upweighted when its flock centroid is within the critical distance of
    the active goal, or of the previous goal while the running maximum distance from it
    since the switch stays below the critical distance.
    """
    d = int(dimension)
    current_state, next_state = slice(0, 2 * d), slice(2 * d, 4 * d)
    active_goal_columns, previous_goal_columns = slice(4 * d, 5 * d), slice(5 * d, 6 * d)
    max_previous_column, mask_column, graph_column = 6 * d, 6 * d + 1, 6 * d + 2
    critical_distance = float(critical_distance)
    distance_weight = float(distance_weight)

    def loss(y_true, y_pred):
        active_mask = tf.cast(y_true[..., mask_column], y_pred.dtype)
        graph_ids = tf.cast(y_true[..., graph_column], tf.int32)
        node_mse = tf.reduce_mean(tf.square(y_true[..., next_state] - y_pred), axis=-1)
        graph_count = tf.reduce_max(graph_ids) + 1
        active_count = tf.math.unsorted_segment_sum(
            active_mask, graph_ids, graph_count
        )
        total_count = tf.math.unsorted_segment_sum(
            tf.ones_like(active_mask), graph_ids, graph_count
        )
        node_mse *= active_mask * tf.gather(
            total_count / tf.maximum(active_count, 1.0), graph_ids
        )

        centroid = tf.math.unsorted_segment_sum(
            y_true[..., current_state][..., :d] * active_mask[:, None], graph_ids, graph_count
        ) / tf.maximum(active_count[:, None], 1.0)
        active_goal = tf.math.unsorted_segment_mean(
            y_true[..., active_goal_columns], graph_ids, graph_count
        )
        previous_goal = tf.math.unsorted_segment_mean(
            y_true[..., previous_goal_columns], graph_ids, graph_count
        )
        max_previous_distance = tf.math.unsorted_segment_max(
            y_true[..., max_previous_column], graph_ids, graph_count
        )
        active_distance = tf.norm(centroid - active_goal, axis=-1)
        previous_distance = tf.norm(centroid - previous_goal, axis=-1)
        near_active = active_distance < critical_distance
        near_eligible_previous = (
            (max_previous_distance >= 0.0)
            & (max_previous_distance < critical_distance)
            & (previous_distance < critical_distance)
        )
        graph_weight = tf.where(
            near_active | near_eligible_previous,
            tf.cast(distance_weight, node_mse.dtype),
            tf.ones_like(active_distance, dtype=node_mse.dtype),
        )
        return node_mse * tf.gather(graph_weight, graph_ids)

    return loss
