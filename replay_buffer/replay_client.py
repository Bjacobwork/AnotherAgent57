import socket
import traceback
import time


def request_episode_id(params):
    time.sleep(.001)
    ip = params['Misc']['replay_ip']
    port_range = params['Misc']['replay_port_range']
    for port in range(min(port_range), max(port_range) + 1):
        try:
            s = socket.socket()
            s.connect((params['Misc']['replay_ip'], port))
            s.send(bytes("request_episode_id", 'utf-8'))
            message = s.recv(1024).decode('utf-8').split("_")
            s.close()
            episode_id = int(message[0])
            j = int(message[1])
            return episode_id, j
        except Exception as e:
            print(e)
    print(traceback.print_exc())
    return -2, -2


def request_trace_batch(params, batch_index):
    time.sleep(.001)
    ip = params['Misc']['replay_ip']
    port_range = params['Misc']['replay_port_range']
    for port in range(min(port_range), max(port_range) + 1):
        try:
            s = socket.socket()
            s.connect((params['Misc']['replay_ip'], port))
            s.send(bytes(f"request_trace_batch_{batch_index}", 'utf-8'))
            message = s.recv(1024).decode('utf-8').split("_")
            s.close()
            trace_ids = []
            for m in message:
                tid = int(m)
                if tid >= 0:
                    trace_ids.append(tid)
            return trace_ids
        except Exception as e:
            print(e)
    print(traceback.print_exc())
    return []


def request_trace_update(params, batch_index):
    time.sleep(.001)
    ip = params['Misc']['replay_ip']
    port_range = params['Misc']['replay_port_range']
    for port in range(min(port_range), max(port_range) + 1):
        try:
            s = socket.socket()
            s.connect((params['Misc']['replay_ip'], port))
            s.send(bytes(f"request_trace_update_{batch_index}", 'utf-8'))
            message = s.recv(1024).decode('utf-8').split("_")
            s.close()
            trace_ids = []
            for m in message:
                tid = int(m)
                if tid >= 0:
                    trace_ids.append(tid)
            return trace_ids
        except Exception as e:
            print(e)
    print(traceback.print_exc())
    return []


def request_transition_upload(params, transitions_requested):
    time.sleep(.001)
    ip = params['Misc']['replay_ip']
    port_range = params['Misc']['replay_port_range']
    for port in range(min(port_range), max(port_range) + 1):
        try:
            s = socket.socket()
            s.connect((params['Misc']['replay_ip'], port))
            s.send(bytes(f"request_transition_upload_{transitions_requested}", 'utf-8'))
            space_for = int(s.recv(1024).decode('utf-8'))
            s.close()
            return space_for
        except Exception as e:
            print(e)
    print(traceback.print_exc())
    return 0


def signal_episode_end(params, episode_id):
    time.sleep(.001)
    ip = params['Misc']['replay_ip']
    port_range = params['Misc']['replay_port_range']
    for port in range(min(port_range), max(port_range) + 1):
        try:
            s = socket.socket()
            s.connect((params['Misc']['replay_ip'], port))
            s.send(bytes(f"signal_episode_end_{episode_id}", 'utf-8'))
            recieved = bool(int(s.recv(1024).decode('utf-8')))
            s.close()
            return recieved
        except Exception as e:
            print(e)
    print(traceback.print_exc())
    return False


def signal_trace_update(params, batch_index):
    time.sleep(.001)
    ip = params['Misc']['replay_ip']
    port_range = params['Misc']['replay_port_range']
    for port in range(min(port_range), max(port_range) + 1):
        try:
            s = socket.socket()
            s.connect((params['Misc']['replay_ip'], port))
            s.send(bytes(f"signal_trace_update_{batch_index}", 'utf-8'))
            recieved = bool(int(s.recv(1024).decode('utf-8')))
            s.close()
            return recieved
        except Exception as e:
            print(e)
    print(traceback.print_exc())
    return False


if __name__ == "__main__":
    import yaml
    import database

    with open('../params.yml', 'r') as file:
        params = yaml.full_load(file)
    dtype = params['Misc']['dtype']
    eid, j = request_episode_id(params)
    print(eid, j)
    import numpy as np
    import random

    obs = np.zeros((1, 210, 160, 1), dtype=np.uint8)
    h = np.zeros((1, 512 * 4), dtype=dtype)
    offset = 0
    for i in range(250):
        transitions = []
        for i in range(100):
            transitions.append(
                [eid, i + offset, random.randrange(100), random.randrange(100), 0, obs.tobytes(), h.tobytes(),
                 random.random(), random.randrange(100), random.randrange(100)])
        offset += 100
        can_upload = request_transition_upload(params, 100)
        if can_upload:
            database.upload_transitions(database.DEFAULT_CONFIG, transitions[:can_upload])
    signal_episode_end(params, eid)

    print(request_trace_batch(params, 0))
