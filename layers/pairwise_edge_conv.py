import tensorflow as tf
from spektral.layers import EdgeConv


class PairwiseEdgeConv(EdgeConv):
    """Messages of each neighbor's offset and distance, aggregated by sum or mean."""

    def __init__(self, channels, aggregation="mean", **kwargs):
        if aggregation not in ("sum", "mean"):
            raise ValueError("aggregation must be 'sum' or 'mean'.")
        super().__init__(channels, **kwargs)
        self.aggregation = aggregation

    def propagate(self, x, a, e=None, **kwargs):
        indices = tf.transpose(a.indices)
        i, j = indices[0], indices[1]
        diff = tf.gather(x, j) - tf.gather(x, i)
        # Gradient stays finite when two neighbors share a position, which matters
        # when positions depend on the model, as in rollout training.
        squared = tf.reduce_sum(tf.square(diff), axis=-1, keepdims=True)
        dist = tf.sqrt(tf.maximum(squared, 1e-24)) + 1e-6
        messages = self.mlp(tf.concat([diff, dist], axis=-1))
        n_nodes = tf.shape(x)[0]
        if self.aggregation == "sum":
            return tf.math.unsorted_segment_sum(messages, j, n_nodes)
        return tf.math.unsorted_segment_mean(messages, j, n_nodes)

    def get_config(self):
        return {**super().get_config(), "aggregation": self.aggregation}
