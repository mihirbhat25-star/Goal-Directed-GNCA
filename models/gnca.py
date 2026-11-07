"""The GNCA: one model for 2D and 3D, fixed and goal-conditioned tasks.

The state branch is a GeneralGNN that averages neighbor features of the full input
state. The relative branch has two message functions over each neighbor's offset
and distance. One is summed over the neighbors, so its total grows with the number
of neighbors that contribute (additive interactions); the other is averaged, so it
depends only on their proportions (normalized interactions). The final MLP maps
``velocity + state term + relative terms`` to the next velocity.

The input holds position and velocity in its first ``2 * dimension`` columns.
Goal-conditioned models receive the displacement to the active goal in
``dimension`` further columns.
"""

import numpy as np
import tensorflow as tf
from spektral.models.general_gnn import MLP, GeneralGNN
from tensorflow.keras.optimizers import Adam

from layers.pairwise_edge_conv import PairwiseEdgeConv


class DualAggregationGNCA(tf.keras.Model):
    """GNCA whose relative branch both sums and averages its messages.

    ``sum_scale`` multiplies the summed term. Set to one over the mean neighbor
    count of the training graphs (``set_sum_scale``), it keeps that term on the scale
    of the averaged one at the training density, while it still grows with the number
    of contributing neighbors. It is a non-trainable weight, so it is saved and
    restored with the checkpoint.
    """

    def __init__(
        self,
        dimension=2,
        message_passing=1,
        batch_norm=False,
        hidden=256,
        hidden_activation="relu",
        connectivity="cat",
        sum_scale=1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dimension = dimension
        self.message_passing = message_passing
        self.batch_norm = batch_norm
        self.hidden = hidden
        self.hidden_activation = hidden_activation
        self.connectivity = connectivity
        self.initial_sum_scale = float(sum_scale)

    def build(self, input_shape):
        d = self.dimension
        self.mp = GeneralGNN(
            d,
            activation="linear",
            message_passing=self.message_passing,
            pool=None,
            batch_norm=self.batch_norm,
            hidden=self.hidden,
            hidden_activation=self.hidden_activation,
            connectivity=self.connectivity,
            aggregate="mean",
        )
        self.mp_diff_sum = PairwiseEdgeConv(d, aggregation="sum", activation="linear", mlp_hidden=[self.hidden])
        self.mp_diff_mean = PairwiseEdgeConv(d, aggregation="mean", activation="linear", mlp_hidden=[self.hidden])
        self.limits_model = MLP(d, batch_norm=self.batch_norm, activation=self.hidden_activation)
        self.sum_scale = self.add_weight(
            name="sum_scale", shape=(), trainable=False,
            initializer=tf.keras.initializers.Constant(self.initial_sum_scale),
        )

    def set_sum_scale(self, mean_degree):
        """Scale the summed term by one over the mean neighbor count of the training graphs."""
        self.sum_scale.assign(1.0 / max(float(mean_degree), 1.0))

    def call(self, inputs, training=False):
        state, adjacency = inputs[0], inputs[1]
        d = self.dimension
        position = state[:, :d]
        velocity = state[:, d : 2 * d]
        summed = self.sum_scale * self.mp_diff_sum([position, adjacency], training=training)
        averaged = self.mp_diff_mean([position, adjacency], training=training)
        state_effect = self.mp([state, adjacency], training=training)
        velocity_next = self.limits_model(velocity + state_effect + summed + averaged, training=training)
        return tf.concat([position + velocity_next, velocity_next], axis=-1)


def input_features(dimension, goal_conditioned):
    """Input columns: position and velocity, plus the goal displacement when conditioned."""
    return (3 if goal_conditioned else 2) * int(dimension)


def build_gnca(dimension, goal_conditioned, n_boids=100, learning_rate=None):
    """Create the model and its variables; with ``learning_rate``, also its Adam optimizer."""
    model = DualAggregationGNCA(dimension=int(dimension))
    if learning_rate is not None:
        model.compile(optimizer=Adam(learning_rate=learning_rate))
    adjacency = tf.SparseTensor(
        indices=np.zeros((0, 2), dtype=np.int64),
        values=np.zeros((0,), dtype=np.float32),
        dense_shape=(n_boids, n_boids),
    )
    state = tf.zeros((n_boids, input_features(dimension, goal_conditioned)), dtype=tf.float32)
    model([state, tf.sparse.reorder(adjacency), tf.constant(0)], training=False)
    return model


def load_gnca(weights, dimension, goal_conditioned, n_boids=100):
    """Build the model and restore a TensorFlow checkpoint prefix into it."""
    model = build_gnca(dimension, goal_conditioned, n_boids)
    # The checkpoint also holds the optimizer of the run that saved it, which evaluation does not use.
    model.load_weights(weights).assert_existing_objects_matched().expect_partial()
    return model
