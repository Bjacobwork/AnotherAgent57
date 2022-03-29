import socket
import time
from multiprocessing import Process, shared_memory, Value, Lock, Queue
import database
import numpy as np
import traceback
import datetime

def request_episode_id(params, cm, transition_space_available, space_lock, in_progress, progress_lock, batches, batch_lock, next_episode_id, episode_queue, args):
    try:
        from bandit import bandit_client
        j = bandit_client.BanditClient(params).get_j()
        in_db = False
        with next_episode_id.get_lock():
            episode_id = next_episode_id.value
            in_db = cm.init_episode(episode_id, j)
            if in_db:
                next_episode_id.value += 1
        if in_db:
            with progress_lock:
                check = np.sum(np.where(in_progress == -1, np.ones_like(in_progress), np.zeros_like(in_progress)))
                print(f"\n{check} spaces open for episodes.\n {datetime.datetime.now()}")
                if check > 0:
                    for i in range(len(in_progress)):
                        if in_progress[i] < 0:
                            in_progress[i] = episode_id
                            return f"{episode_id}_{j}"
    except Exception as e:
        print(e)
        print(traceback.print_exc())
    return "-1_-1"

def request_trace_batch(params, cm, transition_space_available, space_lock, in_progress, progress_lock, batches, batch_lock, next_episode_id, episode_queue, args):
    batch_index = int(args[0])
    with progress_lock:
        check = np.sum(np.where(in_progress == -1, np.ones_like(in_progress), np.zeros_like(in_progress)))
    if check == 0:
        return "-1"
    has_batch = cm.get_trace_batch_ids(params['Misc']['min_required_sequences'], batches, batch_lock, batch_index)
    if not has_batch:
        return "-1"
    else:
        return "_".join(str(v) for v in batches[batch_index])

def request_trace_update(params, cm, transition_space_available, space_lock, in_progress, progress_lock, batches, batch_lock, next_episode_id, episode_queue, args):
    batch_index = int(args[0])
    return "_".join(str(v) for v in batches[batch_index]), True

def request_transition_upload(params, cm, transition_space_available, space_lock, in_progress, progress_lock, batches, batch_lock, next_episode_id, episode_queue, args):
    with space_lock:
        can_upload = min(transition_space_available[0], int(args[0]))
        transition_space_available[0] -= can_upload
    return str(can_upload)

def priority_calculator(params, config, episode_queue):
    trace_length = params['Misc']['trace_length']
    replay_period = params['Misc']['replay_period']
    N = params['Misc']['N']
    training_splits = params['Misc']['training_splits'] if params['Misc']['break_training_loop_early'] else 1
    cm = database.ConnectionManager(config)
    while True:
        episode_id = episode_queue.get()
        cm.calculate_priorities(trace_length, replay_period, N, episode_id, training_splits)

def signal_episode_end(params, cm, transition_space_available, space_lock, in_progress, progress_lock, batches, batch_lock, next_episode_id, episode_queue, args):
    episode_id = int(args[0])
    print(f"\nEnd of episode {episode_id}\n {datetime.datetime.now()}")
    with progress_lock:
        in_progress[:] = np.where(in_progress == episode_id, np.full(in_progress.shape, fill_value=-1, dtype='i'), in_progress)
    episode_queue.put(episode_id)
    return "1"

def signal_trace_update(params, cm, transition_space_available, space_lock, in_progress, progress_lock, batches, batch_lock, next_episode_id, episode_queue, args):
    batch_index = int(args[0])
    with batch_lock:
        batches[batch_index, :] = -1
    return "1"

def worker(socket, params, config, shared_mem, space_lock, progress_lock, batch_lock, next_episode_id, episode_queue):
    shared_mem = shared_memory.SharedMemory(name=shared_mem)
    L = params['Misc']['L']*params['Misc']['consecutive_batches']*2
    b = params['Misc']['training_batch_size']
    c = params['Misc']['consecutive_training_batches']
    in_progress = np.ndarray(L+(b*c), dtype='i', buffer=shared_mem.buf[:-8])
    in_progress, batches = np.split(in_progress, [L])
    batches = batches.reshape((c, b))
    transition_space_available = np.ndarray(1, dtype=np.int64, buffer=shared_mem.buf[-8:])
    lookup = {
        "request": {
            "episode": {
                "id": (0,request_episode_id)
            },
            "trace": {
                "batch": (1,request_trace_batch),
                "update": (1,request_trace_update)
            },
            "transition": {
                "upload": (1,request_transition_upload)
            }

        },
        "signal": {
            "episode": {
                "end": (1,signal_episode_end)
            },
            "trace":{
                "update": (1,signal_trace_update)
            }
        }
    }
    cm = database.ConnectionManager(config)
    while True:
        try:
            client, address = socket.accept()
            data = client.recv(1024)
            data = iter(data.decode('utf-8').split("_"))
            additional_arg_count, function = lookup[next(data)][next(data)][next(data)]
            message = function(params, cm, transition_space_available, space_lock, in_progress, progress_lock, batches, batch_lock, next_episode_id, episode_queue, [next(data) for i in range(additional_arg_count)])
            client.send(bytes(message, 'utf-8'))
            client.close()
        except Exception as e:
            print(e)
            print(traceback.print_exc())
            print(datetime.datetime.now())

def space_allocator(params, config, shared_mem, space_lock, progress_lock, batch_lock):
    shared_mem = shared_memory.SharedMemory(name=shared_mem)
    L = params['Misc']['L']*params['Misc']['consecutive_batches']*2
    b = params['Misc']['training_batch_size']
    c = params['Misc']['consecutive_training_batches']
    in_progress = np.ndarray(L+(b*c),dtype='i', buffer=shared_mem.buf[:-8])
    in_progress, batches = np.split(in_progress, [L])
    batches = batches.reshape((c, b))
    transition_space_available = np.ndarray(1, dtype=np.int64, buffer=shared_mem.buf[-8:])
    cm = database.ConnectionManager(config)
    target_free_space = params['Misc']['target_free_space']*(1024**3)
    byte_limit = params['Misc']['database_size_limit']*(1024**3)
    bytes_per_transition = 41840
    num_episodes_allowed = len(in_progress)*2

    while True:
        db_size = cm.get_database_size()
        if db_size > byte_limit-target_free_space and cm.get_episode_count() > num_episodes_allowed:
            print("\nALLOCATING SPACE")
            print(datetime.datetime.now())
            cm.remove_all_but(num_episodes_allowed, in_progress, progress_lock, batches, batch_lock)
            db_size = cm.get_database_size()
        free_space = byte_limit-db_size
        with space_lock:
            transition_space_available[0] = free_space//bytes_per_transition
        time.sleep(10)

def server(params, config):
    import mariadb
    from replay_server import database
    sockets = []
    ip = params['Misc']['replay_ip']
    port_range = params['Misc']['replay_port_range']
    for port in range(min(port_range), max(port_range)+1):
        serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        serversocket.bind((ip, port))
        serversocket.listen(5)
        sockets.append(serversocket)
    L = params['Misc']['L']*params['Misc']['consecutive_batches']*2
    b = params['Misc']['training_batch_size']
    c = params['Misc']['consecutive_training_batches']
    shared_mem = shared_memory.SharedMemory(create=True, size=(L+(b*c))*4+8)
    init_array = np.ndarray((L+(b*c)), dtype='i', buffer=shared_mem.buf[:-8])
    transition_space_available = np.ndarray(1, dtype=np.int64, buffer=shared_mem.buf[-8:])
    init_array[:] = -1
    progress_lock = Lock()
    batch_lock = Lock()
    space_lock = Lock()
    next_episode_id = Value('i', 0)
    cm = database.ConnectionManager(config)
    cm.cur.execute("SELECT episode_id FROM episode ORDER BY episode_id DESC")
    for r in cm.cur:
        with next_episode_id.get_lock():
            next_episode_id.value = r[0]+1
        break
    del cm
    episode_queue = Queue()
    Process(target=space_allocator, args=(params, config, shared_mem.name, space_lock, progress_lock, batch_lock)).start()
    Process(target=priority_calculator, args=(params, config, episode_queue)).start()
    workers = [Process(target=worker, args=(serversocket,params, config, shared_mem.name, space_lock, progress_lock, batch_lock, next_episode_id, episode_queue)) for i in range(params['Misc']['replay_workers']) for serversocket in sockets]
    for p in workers:
        p.daemon = True
        p.start()
    while True:
        pass

if __name__ == "__main__":
    import yaml
    with open('../params.yml', 'r') as file:
        params = yaml.full_load(file)
    server(params, database.DEFAULT_CONFIG)