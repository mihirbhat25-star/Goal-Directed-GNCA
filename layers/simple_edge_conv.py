import tensorflow as tf
from spektral.layers import EdgeConv


class SimpleEdgeConv(EdgeConv):
    def propagate(self, x, a, e=None, **kwargs):
        # We bypass Spektral's 'message' and 'get_kwargs' logic entirely.
        # 'a' is the SparseTensor. We get indices directly from it.
        indices = tf.transpose(a.indices)
        i = indices[0] # Source
        j = indices[1] # Target

        x_i = tf.gather(x, i)
        x_j = tf.gather(x, j)

        # Boids logic
        diff = x_j - x_i
        # Euclidean distance, written so its gradient stays finite when two
        # neighbors share a position (the gradient of a plain norm at zero is
        # 0/0). The gradient matters when positions depend on the model, as in
        # rollout training.
        squared = tf.reduce_sum(tf.square(diff), axis=-1, keepdims=True)
        dist = tf.sqrt(tf.maximum(squared, 1e-24)) + 1e-6

        # Calculate messages
        messages = self.mlp(tf.concat([diff, dist], axis=-1))

        # Aggregate (mean messages for each target node j)
        out = tf.math.unsorted_segment_mean(messages, j, tf.shape(x)[0])

        return out
