import tensorflow as tf
from spektral.models.general_gnn import MLP, GeneralGNN
from layers.simple_edge_conv import SimpleEdgeConv


class GNNCASimpleBoids3D(tf.keras.Model):
    def __init__(
        self,
        activation=None,
        message_passing=1,
        batch_norm=False,
        hidden=256,
        hidden_activation="relu",
        connectivity="cat",
        aggregate="mean",
        **kwargs
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
        # MP for full 3D state (pos_xyz + vel_xyz = 6 features)
        self.mp = GeneralGNN(
            3,  # output 3D velocity
            activation="linear",
            message_passing=self.message_passing,
            pool=None,
            batch_norm=self.batch_norm,
            hidden=self.hidden,
            hidden_activation=self.hidden_activation,
            connectivity=self.connectivity,
            aggregate=self.aggregate,
        )

        # MP for relative 3D position differences
        self.mp_diff = SimpleEdgeConv(3, activation="linear", mlp_hidden=[self.hidden])

        # Final velocity model
        self.limits_model = MLP(
            3, batch_norm=self.batch_norm, activation=self.hidden_activation
        )

    def call(self, inputs, training=False):
        x_v = inputs[0]
        adj = inputs[1]

        pos = x_v[:, :3]   # 3D position
        vel = x_v[:, 3:]   # 3D velocity

        diff_effect = self.mp_diff([pos, adj], training=training)
        mp_effect = self.mp([x_v, adj], training=training)

        # Spektral's two message-passing implementations do not always return
        # the same compute dtype under a mixed-precision policy. Normalize both
        # residual branches to the state dtype before combining them. These are
        # no-op casts in the existing/default float32 execution path.
        diff_effect = tf.cast(diff_effect, vel.dtype)
        mp_effect = tf.cast(mp_effect, vel.dtype)

        v_next = vel + mp_effect + diff_effect
        v_next = self.limits_model(v_next, training=training)

        x_next = pos + v_next
        return tf.concat([x_next, v_next], axis=-1)

    @tf.function
    def steps(self, inputs, steps):
        state, adj = inputs
        steps_int = tf.cast(steps, tf.int32)
        for _ in tf.range(steps_int):
            state = self([state, adj], training=False)
            state = tf.ensure_shape(state, [None, 6])
        return state
