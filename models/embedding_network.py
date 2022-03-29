import tensorflow as tf
from models.dqn import get_convolutional_torso
from models.running_vars import RunningVars

class EmbeddingNetwork(tf.keras.Model):

    def __init__(self, params):
        super(EmbeddingNetwork, self).__init__()
        self.embedding_fn = get_convolutional_torso(params['torso'])
        layers = []
        params = params['predictor']
        for key in params.keys():
            units = params[key]['units']
            activation = params[key]['activation']
            layers.append(tf.keras.layers.Dense(units, activation=activation))
        self.action_predictor = tf.keras.Sequential(layers)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005,epsilon=0.0001, clipnorm=40.)
        self.loss_fn = tf.keras.losses.CategoricalCrossentropy()

    def call(self, x, x1):
        x = self.embedding_fn(x)
        x1 = self.embedding_fn(x1)
        x = tf.concat([x, x1], axis=-1)
        return self.action_predictor(x)

class EpisodicMemory:

    def __init__(self, params, batch_size, dtype='float32'):
        self.dtype = dtype
        self.vars = RunningVars()
        self.buffer = None
        self.k = params['k']
        self.max_size = params['max_size']
        self.depth = params['depth']
        self.memory_bank = tf.zeros((batch_size, self.max_size, self.depth), dtype)
        self.in_use = tf.zeros((batch_size, self.max_size), dtype)
        self.current_index = 0
        self.epsilon = params['epsilon']
        self.c = params['c']
        self.max_sim = params['maximum_similarity']

    def reset(self, resets):
        self.vars.reset(resets)
        resets = tf.convert_to_tensor(resets)
        resets = tf.expand_dims(resets, -1)
        self.in_use = tf.where(resets, tf.zeros_like(self.in_use), self.in_use)
        resets = tf.expand_dims(resets, -1)
        self.memory_bank = tf.where(resets, tf.zeros_like(self.memory_bank), self.memory_bank)

    def get_reward(self, memory):
        memory = tf.expand_dims(memory, 1)

        distances = tf.subtract(self.memory_bank, memory)
        distances = tf.square(distances)
        distances = tf.reduce_sum(distances, axis=-1)
        #distances = tf.sqrt(distances)
        distances = 1.+(1./distances)

        working_index = tf.one_hot(self.current_index, self.max_size, dtype=self.dtype)
        working_index = tf.expand_dims(working_index, 0)
        clear = tf.ones_like(working_index)-working_index

        self.in_use = self.in_use*clear
        distances = distances*self.in_use
        distances = tf.where(tf.math.is_nan(distances), tf.zeros_like(distances), distances)
        distances = tf.sort(distances, axis=1, direction='DESCENDING')[:,:self.k]
        distances = 1./(distances-1.)
        count = tf.where(tf.equal(distances, -1.), tf.zeros_like(distances), tf.ones_like(distances))
        distances = tf.where(tf.equal(distances, -1.), tf.zeros_like(distances), distances)
        #distances = tf.square(distances)
        batch_mean = tf.reduce_sum(distances, axis=-1,keepdims=True)/tf.reduce_sum(count, axis=-1, keepdims=True)
        batch_mean = tf.where(tf.math.is_nan(batch_mean), tf.zeros_like(batch_mean), batch_mean)
        self.vars.append(batch_mean)
        norm_dist = distances/self.vars.mean()
        distances = tf.where(tf.math.is_nan(norm_dist), distances, norm_dist)
        distances = (self.epsilon/(distances+self.epsilon))
        distances = distances*count
        s = tf.reduce_sum(distances, axis=-1, keepdims=True)
        s = tf.sqrt(s)
        s = s+self.c
        r = tf.where(tf.logical_or(tf.greater(s, self.max_sim), tf.equal(s, self.c)), tf.zeros_like(s), 1/s)

        self.in_use += working_index
        memory = tf.tile(memory, [1,self.max_size, 1])
        self.memory_bank = tf.where(tf.equal(tf.expand_dims(working_index, -1), 1.), memory, self.memory_bank)
        self.current_index += 1
        self.current_index %= self.max_size
        return r





if __name__ == "__main__":
    import yaml
    with open('../params.yml', 'r') as file:
        params = yaml.full_load(file)
    params = params['EpisodicMemory']
    memory = EpisodicMemory(params, 3)
    depth = params['depth']
    dtype = params['Misc']['dtype']
    for i in range(3000):
        mems = tf.ones((3,depth), dtype=dtype)
        r = memory.get_reward(mems)
    print(r)

    memory.reset([True, False, False])

    mems = tf.ones((3,depth), dtype=dtype)
    r = memory.get_reward(mems)
    print(r)
