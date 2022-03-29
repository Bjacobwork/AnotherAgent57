import tensorflow as tf
from models.dqn import get_convolutional_torso
from models.running_vars import RunningVars

class RND(tf.keras.Model):

    def __init__(self, params):
        super(RND, self).__init__()
        self.prediction = get_convolutional_torso(params)
        self.random = get_convolutional_torso(params)
        self.vars = RunningVars()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005,epsilon=0.0001, clipnorm=40.)

    def reset(self):
        self.vars.reset([True])

    def call(self, x):
        r = self.random(x)
        x = self.prediction(x)
        return tf.sqrt(tf.reduce_sum((r-x)**2, axis=-1, keepdims=True), + 1.0e-12)

    def get_novelty(self, x):
        x = self(x)
        m = tf.reduce_mean(x, axis=0)
        m = tf.where(tf.math.is_nan(m), tf.zeros_like(m), m)
        self.vars.append(m)
        m = self.vars.mean()
        v = self.vars.variance()
        a = 1.+((x-m)/v)
        return tf.where(self.vars.count <= 1, tf.zeros_like(a), a)