from multiprocessing import shared_memory, Lock
import numpy as np
import functools

class LearnerData:

    def __init__(self, params, address=None):
        dtype = params['Misc']['dtype']
        element_size = 4
        dtype_size = {"float16":2,"float32":4,"float64":8}[dtype]
        batch_size = params['Misc']['training_batch_size']
        trace_length = params['Misc']['trace_length']+1
        hidden_size = params['Agent57']['lstm']['units']*4
        obs_shape = params['Misc']['obs_shape']
        obs_size = functools.reduce(lambda a, b: a*b, obs_shape)
        memory_size = batch_size*(dtype_size*(4*trace_length+hidden_size+1+obs_size*trace_length)+2*element_size+1)+dtype_size+2
        if address:
            self.shared_mem = shared_memory.SharedMemory(name=address)
        else:
            self.shared_mem = shared_memory.SharedMemory(create=True, size=memory_size)
        self.lock = Lock()
        start = 0
        end = 1
        self.status = np.ndarray(1, dtype=np.uint8, buffer=self.shared_mem.buf[start:end])
        start = end
        end += 1
        self.init_step_count = np.ndarray(1, dtype=np.uint8, buffer=self.shared_mem.buf[start:end])
        start = end
        end += dtype_size
        self.timer = np.ndarray(1, dtype=dtype, buffer=self.shared_mem.buf[start:end])
        start = end
        end += element_size*batch_size
        self.trace_ids = np.ndarray(batch_size, dtype=np.int32, buffer=self.shared_mem.buf[start:end])
        start = end
        end += element_size*batch_size
        self.episode_ids = np.ndarray(batch_size, dtype=np.int32, buffer=self.shared_mem.buf[start:end])
        start = end
        end += dtype_size*batch_size
        self.priority = np.ndarray(batch_size, dtype=dtype, buffer=self.shared_mem.buf[start:end])
        start = end
        end += batch_size
        self.j = np.ndarray(batch_size, dtype=np.uint8, buffer=self.shared_mem.buf[start:end])
        start = end
        end += obs_size*batch_size*trace_length*dtype_size
        self.observations = np.ndarray((trace_length, batch_size, obs_shape[1], obs_shape[2], obs_shape[3]), dtype=dtype, buffer=self.shared_mem.buf[start:end])
        start = end
        end += dtype_size*hidden_size*batch_size
        self.hidden = np.ndarray((batch_size, hidden_size), dtype=dtype, buffer=self.shared_mem.buf[start:end])
        start = end
        end += dtype_size*batch_size*trace_length
        self.prev_extrinsic_rewards = np.ndarray((trace_length, batch_size,1), dtype=dtype, buffer=self.shared_mem.buf[start:end])
        start = end
        end += dtype_size*batch_size*trace_length
        self.prev_intrinsic_rewards = np.ndarray((trace_length, batch_size,1), dtype=dtype, buffer=self.shared_mem.buf[start:end])
        start = end
        end += dtype_size*batch_size*trace_length
        self.actions = np.ndarray((trace_length, batch_size), dtype=np.int32, buffer=self.shared_mem.buf[start:end])
        start = end
        end += dtype_size*batch_size*trace_length
        self.mu = np.ndarray((batch_size, trace_length), dtype=dtype, buffer=self.shared_mem.buf[start:end])

if __name__ == "__main__":
    import yaml
    with open('../actors/params.yml', 'r') as file:
        params = yaml.full_load(file)
    foo = LearnerData(params)
    bar = LearnerData(params, foo.shared_mem.name)

    with bar.lock:
        bar.mu[-1][-1] = 69.420

    print(foo.mu)