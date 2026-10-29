"""Goal-conditioned 2D GNCA with the original Boids GNCA update structure."""

import tensorflow as tf
from spektral.models.general_gnn import GeneralGNN, MLP

from layers.simple_edge_conv import SimpleEdgeConv


class GoalConditionedGNNCABoids(tf.keras.Model):
    """Evolve physical state from ``[position, velocity, relative_goal]``."""

    def __init__(
        self,
        activation=None,
        message_passing=1,
        batch_norm=False,
        hidden=256,
        hidden_activation="relu",
        connectivity="cat",
        aggregate="mean",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.boids_activation = activation
        self.message_passing = message_passing
        self.batch_norm = batch_norm
        self.hidden = hidden
        self.hidden_activation = hidden_activation
        self.connectivity = connectivity
        self.aggregate = aggregate

    def build(self, input_shape):
        self.mp = GeneralGNN(
            2,
            activation="linear",
            message_passing=self.message_passing,
            pool=None,
            batch_norm=self.batch_norm,
            hidden=self.hidden,
            hidden_activation=self.hidden_activation,
            connectivity=self.connectivity,
            aggregate=self.aggregate,
        )
        self.mp_diff = SimpleEdgeConv(
            2, activation="linear", mlp_hidden=[self.hidden]
        )
        self.limits_model = MLP(
            2,
            batch_norm=self.batch_norm,
            activation=self.hidden_activation,
        )

    def call(self, inputs, training=False):
        conditioned_state, adjacency = inputs[:2]
        position = conditioned_state[:, :2]
        velocity = conditioned_state[:, 2:4]

        relative_effect = self.mp_diff(
            [position, adjacency], training=training
        )
        absolute_effect = self.mp(
            [conditioned_state, adjacency], training=training
        )
        velocity_next = self.limits_model(
            velocity + absolute_effect + relative_effect,
            training=training,
        )
        position_next = position + velocity_next
        return tf.concat((position_next, velocity_next), axis=-1)
