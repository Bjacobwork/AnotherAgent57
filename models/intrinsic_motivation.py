import tensorflow as tf
from models.rnd import RND
from models.embedding_network import EmbeddingNetwork, EpisodicMemory


class IntrinsicMotivation(tf.keras.Model):

    def __init__(self, params, batch_sizes):
        super(IntrinsicMotivation, self).__init__()
        self.rnd = RND(params['RND'])
        self.embedding_network = EmbeddingNetwork(params['EmbeddingNetwork'])
        self.memory = [EpisodicMemory(params['EpisodicMemory'], b, params['Misc']['dtype']) for b in batch_sizes]
        self.batch_sizes = batch_sizes

    def reset(self, batch_index, resets):
        self.memory[batch_index].reset(resets)

    def get_both_rewards(self, observations, batch_index):
        L = 5.
        novelty = self.rnd.get_novelty(observations)
        memories = self.embedding_network.embedding_fn(observations)
        episodic_rewards = self.memory[batch_index].get_reward(memories)
        reward = tf.multiply(episodic_rewards, tf.math.minimum(tf.math.maximum(novelty, 1.), L))
        reward = tf.where(tf.math.is_nan(reward), tf.zeros_like(reward), reward)
        return reward, episodic_rewards

    def call(self, observations, batch_index):
        L = 5.
        novelty = self.rnd.get_novelty(observations)
        memories = self.embedding_network.embedding_fn(observations)
        episodic_rewards = self.memory[batch_index].get_reward(memories)
        reward = tf.multiply(episodic_rewards, tf.math.minimum(tf.math.maximum(novelty, 1.), L))
        reward = tf.where(tf.math.is_nan(reward), tf.zeros_like(reward), reward)
        return reward


def get_intrinsic_motivation_model(params, batch_sizes, weight_path=None):
    dtype = params['Misc']['dtype']
    if dtype == 'float16':
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
    model = IntrinsicMotivation(params, batch_sizes)
    if weight_path:
        x = tf.ones((batch_sizes[0], 210, 160, 1), dtype=dtype)
        model(x, 0)
        model.reset(0, [True for _ in range(batch_sizes[0])])
        model.rnd.reset()
        model.embedding_network(x, x)
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
    get_intrinsic_motivation_model(params, [1], "../weights/checkpoints/agent57_0_im.h5")
