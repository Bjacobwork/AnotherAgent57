from multiprocessing import shared_memory, Lock
import numpy as np
import functools

class ActorData:

    def __init__(self, params, batch_size, address=None):
        dtype = params['Misc']['dtype']
        element_size = 4
        dtype_size = {"float16":2,"float32":4,"float64":8}[dtype]
        hidden_size = params['Agent57']['lstm']['units']*4
        obs_shape = params['Misc']['obs_shape']
        obs_size = functools.reduce(lambda a, b: a*b, obs_shape)
        memory_size = batch_size*(3+5*dtype_size+dtype_size*hidden_size+obs_size+4*element_size)+2*element_size+1
        if address:
            self.shared_mem = shared_memory.SharedMemory(name=address)
        else:
            self.shared_mem = shared_memory.SharedMemory(create=True, size=memory_size)
        self.lock = Lock()
        start = 0
        end = 1
        self.status = np.ndarray(1, dtype=np.uint8, buffer=self.shared_mem.buf[start:end])
        start = 1
        end += 2*element_size
        self.timer = np.ndarray(1, dtype=np.float64, buffer=self.shared_mem.buf[start:end])
        start = end
        end += batch_size*element_size
        self.episode_ids = np.ndarray(batch_size, dtype=np.uint32, buffer=self.shared_mem.buf[start:end])
        start = end
        end += batch_size*element_size
        self.steps = np.ndarray(batch_size, dtype=np.uint32, buffer=self.shared_mem.buf[start:end])
        start = end
        end += batch_size
        self.j = np.ndarray(batch_size, dtype=np.uint8, buffer=self.shared_mem.buf[start:end])
        start = end
        end += dtype_size*batch_size
        self.extrinsic_rewards = np.ndarray((batch_size,1), dtype=dtype, buffer=self.shared_mem.buf[start:end])
        start = end
        end += dtype_size*batch_size
        self.intrinsic_rewards = np.ndarray((batch_size,1), dtype=dtype, buffer=self.shared_mem.buf[start:end])
        start = end
        end += element_size*batch_size
        self.actions = np.ndarray(batch_size, dtype=np.int32, buffer=self.shared_mem.buf[start:end])
        start = end
        end += element_size*batch_size
        self.prev_actions = np.ndarray(batch_size, dtype=np.int32, buffer=self.shared_mem.buf[start:end])
        start = end
        end += obs_size*batch_size
        self.observations = np.ndarray((batch_size, obs_shape[1], obs_shape[2], obs_shape[3]), dtype=np.uint8, buffer=self.shared_mem.buf[start:end])
        start = end
        end += dtype_size*hidden_size*batch_size
        self.hidden = np.ndarray((batch_size, hidden_size), dtype=dtype, buffer=self.shared_mem.buf[start:end])
        start = end
        end += dtype_size*batch_size
        self.mu = np.ndarray(batch_size, dtype=dtype, buffer=self.shared_mem.buf[start:end])
        start = end
        end += dtype_size*batch_size
        self.q_value = np.ndarray(batch_size, dtype=dtype, buffer=self.shared_mem.buf[start:end])
        start = end
        end += dtype_size*batch_size
        self.discounted_q = np.ndarray(batch_size, dtype=dtype, buffer=self.shared_mem.buf[start:end])
        start = end
        end += batch_size
        self.resets = np.ndarray(batch_size, dtype='bool', buffer=self.shared_mem.buf[start:end])
        start = end
        end += batch_size
        self.loss_of_life = np.ndarray(batch_size, dtype='bool', buffer=self.shared_mem.buf[start:end])


if __name__ == "__main__":
    import yaml
    with open('../actors/params.yml', 'r') as file:
        params = yaml.full_load(file)
    foo = ActorData(params, 6)
    bar = ActorData(params, 6, address=foo.shared_mem.name)

    with bar.lock:
        bar.resets[-1] = True

    print(foo.resets)