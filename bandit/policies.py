import math
import numpy as np
import random


def python_sigmoid(number):
    return 1 / (1 + math.exp(-number))


def beta_scale(j, N):
    foo = N - 2
    return python_sigmoid(10 * ((2 * j - foo) / foo))


def tf_beta_scale(j, N):
    import tensorflow as tf
    foo = N - 2
    return tf.nn.sigmoid(10 * ((2 * j - foo) / foo))


def gamma_sigmoid(j, k, gamma_0, gamma_1):
    return gamma_1 + (gamma_0 - gamma_1) * (1 - python_sigmoid(10 * ((2 * j - k) / k)))


def tf_gamma_sigmoid(j, k, gamma_0, gamma_1):
    import tensorflow as tf
    return gamma_1 + (gamma_0 - gamma_1) * (1 - tf.nn.sigmoid(10 * ((2 * j - k) / k)))


def gamma_decay(j, N, gamma_1, gamma_2):
    scale = (j - 8) / (N - 9)
    return 1 - math.exp((1 - scale) * math.log(1 - gamma_1) + scale * math.log(1 - gamma_2))


def tf_gamma_decay(j, N, gamma_1, gamma_2):
    import tensorflow as tf
    scale = (j - 8) / (N - 9)
    return 1 - tf.math.exp((1 - scale) * tf.math.log(1 - gamma_1) + scale * tf.math.log(1 - gamma_2))


def get_policy(j, N, max_beta=.3, gamma_0=.9999, gamma_1=0.997, gamma_2=.99):
    if j == 0:
        beta = 0.
        gamma = gamma_0
    elif j >= N - 1:
        beta = max_beta
        gamma = gamma_decay(j, N, gamma_1, gamma_2)
    else:
        beta = max_beta * beta_scale(j, N)
        if j < 7:
            gamma = gamma_sigmoid(j, 6, gamma_0, gamma_1)
        elif j == 7:
            gamma = gamma_1
        else:
            gamma = gamma_decay(j, N, gamma_1, gamma_2)
    return beta, gamma


def tf_get_policy(j, N, dtype, max_beta=.3, gamma_0=.9999, gamma_1=0.997, gamma_2=.99):
    import tensorflow as tf
    max_beta = tf.convert_to_tensor(max_beta, dtype)
    gamma_0 = tf.convert_to_tensor(gamma_0, dtype)
    gamma_1 = tf.convert_to_tensor(gamma_1, dtype)
    gamma_2 = tf.convert_to_tensor(gamma_2, dtype)
    beta = max_beta * tf_beta_scale(j, N)
    beta = tf.where(j >= N - 1., tf.cast(tf.fill(j.shape, max_beta), dtype), beta)
    beta = tf.where(j == 0., tf.zeros_like(beta), beta)
    gamma = tf_gamma_decay(j, N, gamma_1, gamma_2)
    gamma = tf.where(j == 7., tf.cast(tf.fill(j.shape, gamma_1), dtype), gamma)
    gamma = tf.where(j < 7., tf_gamma_sigmoid(j, 6., gamma_0, gamma_1), gamma)
    gamma = tf.where(j == 0., tf.cast(tf.fill(j.shape, gamma_0), dtype), gamma)
    return beta, gamma


class MAB:

    def __init__(self, N, epsilon, beta, window_size):
        self.N = N
        self.epsilon = epsilon
        self.beta = beta
        self.window_size = window_size
        self.rewards = np.zeros(N)
        self.counts = np.zeros(N)
        self.window = []
        self.k = 0

    def save(self, root, print_exception=False):
        print("Saving")
        try:
            import json
            import os
            os.makedirs(root, exist_ok=True)
            with open(root + "mab.json", 'w') as file:
                json.dump({"mab": self.window, "k": self.k}, file)
            return True
        except Exception as e:
            if print_exception:
                print(e)
            return False

    def load(self, root):
        import json
        import os
        if os.path.isfile(root + "mab.json"):
            with open(root + "mab.json", 'r') as file:
                data = json.load(file)
                self.window = data['mab']
                self.k = data['k']
                for elm in self.window:
                    arm, reward = elm
                    self.counts[arm] += 1
                    self.rewards[arm] += reward
            return True
        return False

    def greed(self):
        den = self.counts + 1e-10
        ranks = np.divide(self.rewards, den)
        self.k -= 1
        return int(np.argmax(ranks))

    def ucb(self):
        num = math.log(min(self.k - 1, self.window_size))
        den = self.counts + 1e-10
        ucb = self.beta * np.sqrt(np.divide(num, den))
        ranks = ucb + np.divide(self.rewards, den)
        return int(np.argmax(ranks))

    def get_j(self):
        if self.k < self.N:
            return self.k
        if random.random() < self.epsilon:
            return random.randint(0, self.N - 1)
        return self.ucb()

    def update_reward(self, arm, reward):
        self.counts[arm] += 1
        self.rewards[arm] += reward
        self.window.append((arm, reward))
        self.k += 1
        if len(self.window) > self.window_size:
            arm, reward = self.window.pop(0)
            self.counts[arm] -= 1
            self.rewards[arm] -= reward


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    N = 32
    betas = []
    gammas = []
    indicies = []
    for j in range(N):
        indicies.append(j)
        b, g = get_policy(j, N)
        betas.append(b)
        gammas.append(g)
    plt.plot(indicies, betas, 'bs')
    plt.show()
    plt.plot(indicies, gammas, 'bs')
    plt.show()
    print(gammas)
