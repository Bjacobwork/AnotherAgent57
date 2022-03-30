import tensorflow as tf


def h(x):
    eps = 1e-3
    sign = tf.math.sign(x)
    a = tf.math.abs(x)
    sqr = tf.math.sqrt(a + 1.)
    sqr = sign * (sqr - 1)
    return sqr + eps * x


def h_inverse(x):
    eps = 1e-3
    sign = tf.math.sign(x)
    a = tf.abs(x)
    return sign * (tf.square((tf.math.sqrt(1. + 4. * eps * (a + 1 + eps)) - 1.) / (2 * eps)) - 1.)


class BetterFlatten(tf.keras.layers.Layer):

    def __init__(self):
        super(BetterFlatten, self).__init__()

    def call(self, inputs):
        shape = inputs.shape.as_list()
        shape = shape[:-3] + [shape[-3] * shape[-2] * shape[-1]]
        shape[0] = -1
        return tf.reshape(inputs, shape)


def get_convolutional_torso(params):
    layers = []
    layer_params = params["conv_layers"]
    for key in layer_params.keys():
        filters = layer_params[key]['filters']
        kernel_size = layer_params[key]['kernel_size']
        strides = layer_params[key]['strides']
        activation = layer_params[key]['activation']
        layers.append(tf.keras.layers.Conv2D(
            filters, kernel_size, strides, activation=activation
        ))
    layers.append(tf.keras.layers.Flatten())
    layer_params = params['dense_layers']
    for key in layer_params.keys():
        units = layer_params[key]['units']
        activation = layer_params[key]['activation']
        layers.append(tf.keras.layers.Dense(units, activation=activation))
    return tf.keras.Sequential(layers)


class DualingHeads(tf.keras.Model):

    def __init__(self, params):
        super(DualingHeads, self).__init__()
        self.advantage_0 = tf.keras.layers.Dense(params['hidden_units'], activation=params['hidden_activation'])
        self.advantage_1 = tf.keras.layers.Dense(1)
        self.state_value_0 = tf.keras.layers.Dense(params['hidden_units'], activation=params['hidden_activation'])
        self.state_value_1 = tf.keras.layers.Dense(params['num_actions'])

    def call(self, x):
        adv = self.advantage_0(x)
        adv = self.advantage_1(adv)
        x = self.state_value_0(x)
        x = self.state_value_1(x)
        m = tf.reduce_mean(x, axis=-1, keepdims=True)
        x = tf.add(x, adv)
        return tf.subtract(x, m)


class R2D2(tf.keras.Model):

    def __init__(self, params):
        super(R2D2, self).__init__()
        self.conv_torso = get_convolutional_torso(params['torso'])
        self.lstm = tf.keras.layers.LSTM(params['lstm']['units'], return_state=True)
        self.dual_heads = DualingHeads(params['dual_heads'])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, epsilon=0.0001, clipnorm=40.)

    def call(self, x, prev_a, prev_r_e, prev_r_i, one_hot_j, state_h, state_c):
        x = self.conv_torso(x)
        x = tf.concat([x, prev_a, prev_r_e, prev_r_i, one_hot_j], axis=-1)
        x = tf.expand_dims(x, 1)
        x, state_h, state_c = self.lstm(x, initial_state=[state_h, state_c])
        return self.dual_heads(x), state_h, state_c


class Agent57(tf.keras.Model):

    def __init__(self, params):
        super(Agent57, self).__init__()
        self.extrinsic_model = R2D2(params)
        self.intrinsic_model = R2D2(params)

    def separate_models(self, x, prev_a, prev_r_e, prev_r_i, one_hot_j, eh, ec, ih, ic):
        q_e, eh, ec = self.extrinsic_model(x, prev_a, prev_r_e, prev_r_i, one_hot_j, eh, ec)
        q_i, ih, ic = self.intrinsic_model(x, prev_a, prev_r_e, prev_r_i, one_hot_j, ih, ic)
        return q_e, q_i, eh, ec, ih, ic

    def separate_hidden(self, x, prev_a, prev_r_e, prev_r_i, one_hot_j, beta, eh, ec, ih, ic):
        q_e, q_i, eh, ec, ih, ic = self.separate_models(x, prev_a, prev_r_e, prev_r_i, one_hot_j, eh, ec, ih, ic)
        q_e = h_inverse(q_e)
        q_i = h_inverse(q_i)
        q_i = beta * q_i
        return h(q_e + q_i), eh, ec, ih, ic

    def call(self, x, prev_a, prev_r_e, prev_r_i, one_hot_j, beta, hidden_state):
        eh, ec, ih, ic = tf.split(hidden_state, num_or_size_splits=4, axis=-1)
        q, eh, ec, ih, ic = self.separate_hidden(x, prev_a, prev_r_e, prev_r_i, one_hot_j, beta, eh, ec, ih, ic)
        return q, tf.concat([eh, ec, ih, ic], axis=-1)


def get_agent57_model(params, weight_path=None, with_sequence=False):
    dtype = params['Misc']['dtype']
    if dtype == 'float16':
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
    model = Agent57(params['Agent57'])
    if weight_path:
        x = tf.ones((1, 210, 160, 1), dtype=dtype)
        a = tf.zeros((1, params['Agent57']['dual_heads']['num_actions']), dtype=dtype)
        r = tf.zeros((1, 1), dtype=dtype)
        hot = tf.zeros((1, params['Misc']['N']), dtype=dtype)
        if with_sequence:
            x = tf.expand_dims(x, 1)
            a = tf.expand_dims(a, 1)
            r = tf.expand_dims(r, 1)
            hot = tf.expand_dims(hot, 1)
        h = tf.zeros((1, params['Agent57']['lstm']['units'] * 4), dtype=dtype)
        model(x, a, r, r, hot, 0., h)
        model.summary()
        import os
        if os.path.exists(weight_path):
            model.load_weights(weight_path)
        else:
            model.save_weights(weight_path)
    return model


if __name__ == "__main__":
    import yaml
    import tensorflow as tf

    with open('../params.yml', 'r') as file:
        params = yaml.full_load(file)
    get_agent57_model(params, "../weights/checkpoints/agent57_0_dqn.h5")
