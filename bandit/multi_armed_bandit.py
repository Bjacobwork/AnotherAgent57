import numpy as np
from multiprocessing import shared_memory
import random
import math


class MultiArmedBandit:

    def __init__(self, params, lock, address=None, save_root=""):
        self.lock = lock
        self.N = params['Misc']['N']
        self.epsilon = params['Misc']['bandit_e']
        self.beta = params['Misc']['bandit_beta']
        self.window_size = params['Misc']['bandit_window_size']
        self.save_every = params['Misc']['bandit_save_period']
        self.root = save_root
        dtype = params['Misc']['dtype']
        element_size = 4
        dtype_size = {"float16": 2, "float32": 4, "float64": 8}[dtype]
        mem_size = dtype_size * (self.window_size + 2 * self.N + 1) + element_size * (2 + self.window_size)
        if not address:
            self.mem = shared_memory.SharedMemory(create=True, size=mem_size)
        else:
            self.mem = shared_memory.SharedMemory(name=address)
        start = 0
        end = self.N * dtype_size
        self.rewards = np.ndarray(self.N, dtype=dtype, buffer=self.mem.buf[start:end])
        start = end
        end += self.N * dtype_size
        self.counts = np.ndarray(self.N, dtype=dtype, buffer=self.mem.buf[start:end])
        start = end
        end += dtype_size
        self.high_score = np.ndarray(1, dtype=dtype, buffer=self.mem.buf[start:end])
        start = end
        end += element_size
        self.k = np.ndarray(1, dtype=np.int32, buffer=self.mem.buf[start:end])
        start = end
        end += element_size
        self.window_indices = np.ndarray(1, dtype=np.int32, buffer=self.mem.buf[start:end])
        self.window_indices[0] = -1
        start = end
        end += self.window_size * element_size
        self.window_arms = np.ndarray(self.window_size, dtype=np.int32, buffer=self.mem.buf[start:end])
        self.window_arms[:] = -1
        start = end
        end += self.window_size * dtype_size
        self.window_rewards = np.ndarray(self.window_size, dtype=dtype, buffer=self.mem.buf[start:end])

    def append_window(self, arm, reward):
        self.window_indices[0] = (self.window_indices[0] + 1) % self.window_size
        wi = self.window_indices[0]
        self.counts[arm] += 1
        self.rewards[arm] += reward
        a = self.window_arms[wi]
        if a >= 0:
            self.counts[a] -= 1
            self.rewards[a] -= self.window_rewards[wi]
        self.window_arms[wi] = arm
        self.window_rewards[wi] = reward

    def update_reward(self, arm, reward, score):
        with self.lock:
            self.append_window(arm, reward)
            self.high_score[0] = max(self.high_score[0], score)
            self.k[0] += 1
            saving = self.k[0] % self.save_every == 0
        if saving:
            self.save(self.root)

    def greed(self):
        with self.lock:
            ranks = np.divide(self.rewards, self.counts + 1e-10)
        return int(np.argmax(ranks))

    def ucb(self):
        with self.lock:
            num = math.log(min(self.k[0] - 1, self.window_size))
            den = self.counts + 1e-10
            ucb = self.beta * np.sqrt(np.divide(num, den))
            ranks = ucb + np.divide(self.rewards, den)
        return int(np.argmax(ranks))

    def get_j(self):
        if self.k[0] < self.N:
            return int(self.k[0])
        if random.random() < self.epsilon:
            return random.randint(0, self.N - 1)
        return self.ucb()

    def grab_mab_data(self):
        with self.lock:
            data = {
                "mab": [],
                "k": int(self.k[0]),
                "high_score": float(self.high_score[0])
            }
            if self.window_indices[0] >= 0:
                for i in range(self.window_indices[0] + 1):
                    data['mab'].append((int(self.window_arms[i]), float(self.window_rewards[i])))
        return data

    def save(self, root, print_exception=False):
        print("Saving")
        try:
            import json
            import os
            os.makedirs(root, exist_ok=True)
            with open(root + "/mab.json", 'w') as file:
                data = self.grab_mab_data()
                json.dump(data, file)
            return True
        except Exception as e:
            if print_exception:
                print(e)
            return False

    def load(self, root):
        import json
        import os
        if os.path.isfile(root + "/mab.json"):
            with open(root + "/mab.json", 'r') as file:
                data = json.load(file)
                with self.lock:
                    self.k[0] = data['k']
                    self.high_score[0] = data['high_score']
                    for elm in data['mab']:
                        arm, reward = elm
                        self.append_window(arm, reward)
            return True
        return False
