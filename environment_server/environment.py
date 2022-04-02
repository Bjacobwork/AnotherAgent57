import time
from environment_server.actor_data import ActorData
import traceback


def replay_buffer_process(params, batch_sizes, batch_addresses, transition_queue, replay_lock):
    try:
        from replay_buffer import replay_client
        import random
        batch_data = [ActorData(params, b, a) for b, a in zip(batch_sizes, batch_addresses)]
        consecutive_batches = params['Misc']['consecutive_batches']
        num_actions = params['Agent57']['dual_heads']['num_actions'] - 1
        for b, data in zip(batch_sizes, batch_data):
            with data.lock:
                for i in range(b):
                    with replay_lock:
                        episode_id, j = replay_client.request_episode_id(params)
                    while episode_id < 0 or j < 0:
                        time.sleep(1)
                        with replay_lock:
                            episode_id, j = replay_client.request_episode_id(params)
                    data.episode_ids[i] = episode_id
                    data.j[i] = j
                    data.actions[i] = random.randint(0, num_actions)
                data.resets[:] = True
                data.timer[:] = time.time()
                data.status[:] = 1
        while True:
            for b, data in zip(batch_sizes, batch_data):
                while data.status[0] != 0:
                    pass
                with data.lock:
                    # print("Buffer")
                    total_time = time.time() - data.timer[0]
                    data.timer[0] = time.time()
                    print(f"\rActions per second {consecutive_batches / (total_time * len(batch_sizes))} ", end="")
                    transition_queue.put([
                        data.episode_ids.copy(),
                        data.steps.copy(),
                        data.extrinsic_rewards.copy(),
                        data.intrinsic_rewards.copy(),
                        data.prev_actions.copy(),
                        data.observations.copy(),
                        data.hidden.copy(),
                        data.mu.copy(),
                        data.q_value.copy(),
                        data.discounted_q.copy(),
                        data.resets.copy()
                    ])
                    for i, reset in enumerate(data.resets):
                        if reset:
                            with replay_lock:
                                episode_id, j = replay_client.request_episode_id(params)
                            while episode_id < 0 or j < 0:
                                print(f"\rPausing while eps_id or j < 0: {episode_id} {j}", end="")
                                time.sleep(120)
                                with replay_lock:
                                    episode_id, j = replay_client.request_episode_id(params)
                            data.episode_ids[i] = episode_id
                            data.j[i] = j
                    data.status[0] = 1
    except Exception as e:
        print(e)
        print(traceback.print_exc(4))

def transition_upload_process(params, batch_sizes, batch_addresses, replay_lock, config=None):
    try:
        from replay_buffer import replay_client, database
        import random
        cm = database.ConnectionManager(config)
        batch_data = [ActorData(params, b, a) for b, a in zip(batch_sizes, batch_addresses)]
        consecutive_batches = params['Misc']['consecutive_batches']
        num_actions = params['Agent57']['dual_heads']['num_actions'] - 1
        for b, data in zip(batch_sizes, batch_data):
            with data.lock:
                for i in range(b):
                    with replay_lock:
                        episode_id, j = replay_client.request_episode_id(params)
                    while episode_id < 0 or j < 0:
                        time.sleep(1)
                        with replay_lock:
                            episode_id, j = replay_client.request_episode_id(params)
                    data.episode_ids[i] = episode_id
                    data.j[i] = j
                    data.actions[i] = random.randint(0, num_actions)
                data.resets[:] = True
                data.timer[:] = time.time()
                data.status[:] = 1
        while True:
            for b, data in zip(batch_sizes, batch_data):
                while data.status[0] != 0:
                    pass
                with data.lock:
                    # print("Buffer")
                    total_time = time.time() - data.timer[0]
                    data.timer[0] = time.time()
                    print(f"\rActions per second {consecutive_batches / (total_time * len(batch_sizes))} ", end="")

                    transitions = []
                    resets = []
                    for episode_id, step, extrinsic_reward, intrinsic_reward, action, observation, hidden, mu, value, discounted, reset in zip(
                            data.episode_ids.copy(),
                            data.steps.copy(),
                            data.extrinsic_rewards.copy(),
                            data.intrinsic_rewards.copy(),
                            data.prev_actions.copy(),
                            data.observations.copy(),
                            data.hidden.copy(),
                            data.mu.copy(),
                            data.q_value.copy(),
                            data.discounted_q.copy(),
                            data.resets.copy()):
                        transitions.append([int(episode_id),
                                    int(step),
                                    float(extrinsic_reward),
                                    float(intrinsic_reward),
                                    int(action),
                                    observation.tobytes(),
                                    hidden.tobytes(),
                                    float(mu),
                                    float(value),
                                    float(discounted)])
                        resets.append([episode_id, reset])

                    while len(transitions) > 0:
                        with replay_lock:
                            allowed_to_upload = replay_client.request_transition_upload(params, len(transitions))
                        if allowed_to_upload > 0:
                            packets_1024 = allowed_to_upload // 1024
                            final_packet = allowed_to_upload - (packets_1024 * 1024)
                            packets = [1024 for _ in range(packets_1024)]
                            if final_packet:
                                packets.append(final_packet)
                            for upload in packets:
                                uploading = transitions[:upload]
                                resetting = resets[:upload]
                                transitions = transitions[upload:]
                                resets = resets[upload:]
                                success = cm.upload_transitions(uploading)
                                if success:
                                    for r in resetting:
                                        if r[1]:
                                            with replay_lock:
                                                replay_client.signal_episode_end(params, int(r[0]))
                                else:
                                    transitions += uploading
                                    resets += resetting

                    for i, reset in enumerate(data.resets):
                        if reset:
                            with replay_lock:
                                episode_id, j = replay_client.request_episode_id(params)
                            while episode_id < 0 or j < 0:
                                print(f"\rPausing while eps_id or j < 0: {episode_id} {j}", end="")
                                time.sleep(120)
                                with replay_lock:
                                    episode_id, j = replay_client.request_episode_id(params)
                            data.episode_ids[i] = episode_id
                            data.j[i] = j
                    data.status[0] = 1
    except Exception as e:
        print(e)
        print(traceback.print_exc(4))

def environment_process(params, batch_sizes, batch_addresses):
    try:
        from environment_server.environment_wrapper import Environment
        batch_data = [ActorData(params, b, a) for b, a in zip(batch_sizes, batch_addresses)]
        frameskip = params['Misc']['frameskip'] if type(params['Misc']['frameskip']) == int else False
        multi_envs = [[Environment(params['Misc']['environment'], params['Misc']['obs_type'], frameskip,
                                   params['Misc']['max_episode_length'], i, data.actions, data.observations,
                                   data.extrinsic_rewards, data.steps, data.resets, data.loss_of_life,
                                   params['Misc']['reward_scale']) for i in range(b)] for b, data in
                      zip(batch_sizes, batch_data)]
        while True:
            for b, multi_env, data in zip(batch_sizes, multi_envs, batch_data):
                while data.status[0] != 1:
                    pass
                with data.lock:
                    # print("Env")
                    for env in multi_env:
                        env()
                    data.prev_actions[:] = data.actions
                    data.status[0] = 2
    except Exception as e:
        print(e)
        print(traceback.print_exc(4))
        # environment_process(params, batch_sizes, batch_addresses)


def split_environment_process(params, batch_sizes, batch_addresses, num_splits):
    from multiprocessing import Process, Queue
    from environment_server.environment_wrapper import multi_env
    envs = []
    starts = []
    ends = []
    batch_data = []
    for b, address in zip(batch_sizes, batch_addresses):
        batch_data.append(ActorData(params, b, address))
        env = []
        start = []
        end = []
        for i in range(num_splits):
            s = Queue()
            e = Queue()
            env.append(Process(target=multi_env, args=(params, b, address, s, e, num_splits, i)))
            start.append(s)
            end.append(e)
        envs.append(env)
        starts.append(start)
        ends.append(end)
    for env in envs:
        for e in env:
            e.start()

    while True:
        for data, start, end in zip(batch_data, starts, ends):
            while data.status[0] != 1:
                pass
            with data.lock:
                for s in start:
                    s.put(True)
                for e in end:
                    e.get()
                data.prev_actions[:] = data.actions
                data.status[0] = 2


def intrinsic_motivation_process(params, batch_sizes, path_index, path_template, batch_addresses, device):
    import tensorflow as tf
    dtype = params['Misc']['dtype']
    if dtype == 'float16':
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
    from models.intrinsic_motivation import IntrinsicMotivation, get_intrinsic_motivation_model
    batch_data = [ActorData(params, b, a) for b, a in zip(batch_sizes, batch_addresses)]
    with tf.device(device):
        with path_index.get_lock():
            if path_index.value >= 0:
                path = path_template.format("im", path_index.value)
            else:
                path = None
        print(f"Loading model from {path}\n", end="")
        model = get_intrinsic_motivation_model(params, batch_sizes, path)

        update_every = params['Misc']['actor_weight_update']
        next_update = update_every
        while True:
            for b, batch_size, data in zip(range(len(batch_sizes)), batch_sizes, batch_data):
                while data.status[0] != 2:
                    pass
                with data.lock:
                    observations = tf.cast(data.observations, dtype) / 255
                    intrinsic_rewards = model(observations, b)
                    # transition = [resets, x, prev_reward_e, j, beta, prev_reward_i]
                    data.intrinsic_rewards[:] = intrinsic_rewards.numpy()
                    resets = data.resets.copy()
                    data.status[0] = 3
                # resetting after intrinsic calculation for final intrinsic reward
                model.reset(b, resets)
            next_update -= 1
            if next_update <= 0:
                next_update = update_every
                with path_index.get_lock():
                    if path_index.value >= 0:
                        path = path_template.format("im", path_index.value)
                        print(f"Loading model from {path}\n", end="")
                        model.load_weights(path)


def dqn_process(params, batch_sizes, path_index, path_template, batch_addresses, device):
    import tensorflow as tf
    dtype = params['Misc']['dtype']
    if dtype == 'float16':
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
    from models.dqn import Agent57, get_agent57_model
    from models import dqn
    from bandit import policies
    batch_data = [ActorData(params, b, a) for b, a in zip(batch_sizes, batch_addresses)]
    with tf.device(device):
        with path_index.get_lock():
            if path_index.value >= 0:
                path = path_template.format("dqn", path_index.value)
            else:
                path = None
        print(f"Loading model from {path}\n", end="")
        model = get_agent57_model(params, path)
        num_actions = params['Agent57']['dual_heads']['num_actions']
        hidden_units = 4 * params['Agent57']['lstm']['units']
        N = params['Misc']['N']
        L = params['Misc']['L']
        consecutive_batches = params['Misc']['consecutive_batches']
        update_every = params['Misc']['actor_weight_update']
        next_update = update_every
        zero_discount_on_life_loss = params['Misc']['zero_discount_on_life_loss']

        hiddens = [tf.zeros((b, hidden_units), dtype) for b in batch_sizes]
        greeds = tf.pow(params['Misc']['greed_e'],
                        1. + (params['Misc']['greed_a'] * (tf.range(L, dtype=dtype) / (L - 1))))
        greeds = tf.tile(greeds, [consecutive_batches])
        mu_random = greeds / num_actions
        mu_q = 1. - (greeds * ((num_actions - 1) / num_actions))
        greeds = tf.split(greeds, num_or_size_splits=batch_sizes)
        mu_random = tf.split(mu_random, num_or_size_splits=batch_sizes)
        mu_q = tf.split(mu_q, num_or_size_splits=batch_sizes)

        while True:
            next_hiddens = []
            for b, hidden, greed, m_r, m_q, data in zip(batch_sizes, hiddens, greeds, mu_random, mu_q, batch_data):
                while data.status[0] != 3:
                    pass
                with data.lock:
                    data.hidden[:] = hidden.numpy()
                    resets = tf.convert_to_tensor(data.resets)
                    one_hot_js = tf.one_hot(data.j, depth=N, dtype=dtype)
                    prev_a = tf.one_hot(data.prev_actions, depth=num_actions, dtype=dtype)
                    beta, gamma = policies.tf_get_policy(tf.cast(data.j, dtype), N, dtype)
                    beta = tf.expand_dims(beta, axis=-1)
                    prev_reward_e = dqn.h(tf.convert_to_tensor(data.extrinsic_rewards, dtype))
                    prev_r_i = dqn.h(tf.convert_to_tensor(data.intrinsic_rewards, dtype=dtype))
                    observations = tf.cast(data.observations, dtype) / 255

                    q_values, hidden = model(observations, prev_a, prev_reward_e, prev_r_i,
                                             one_hot_js, beta, hidden)

                    action = tf.argmax(q_values, axis=-1, output_type=tf.int32)
                    q_values = dqn.h_inverse(q_values)
                    random_action = tf.random.uniform(action.shape, 0, num_actions - 1, tf.int32)
                    random_decision = tf.random.uniform(action.shape, 0., 1., dtype) < greed
                    random_decision = random_decision | resets
                    action = tf.where(random_decision, random_action, action)
                    mu = tf.where(random_decision, m_r, m_q)
                    selected_q_values = tf.reduce_sum(
                        tf.one_hot(action, depth=num_actions, dtype=dtype) * q_values, axis=-1)
                    discounted_q = tf.reduce_sum(tf.multiply(tf.nn.softmax(q_values, axis=-1),
                                                             q_values),
                                                 axis=-1)
                    if zero_discount_on_life_loss:
                        discounted_q = tf.multiply(discounted_q, tf.where(tf.convert_to_tensor(data.loss_of_life),
                                                                          tf.zeros_like(gamma), gamma))
                    else:
                        discounted_q = tf.multiply(discounted_q, gamma)

                    data.actions[:] = action.numpy()
                    data.q_value[:] = selected_q_values.numpy()
                    data.discounted_q[:] = discounted_q.numpy()
                    data.mu[:] = mu.numpy()
                    data.status[0] = 0
                if tf.reduce_any(resets):
                    resets = tf.tile(tf.expand_dims(resets, -1), [1, hidden_units])
                    hidden = tf.where(resets, tf.zeros_like(hidden), hidden)
                next_hiddens.append(hidden)
            hiddens = next_hiddens

            next_update -= 1
            if next_update <= 0:
                next_update = update_every
                with path_index.get_lock():
                    if path_index.value >= 0:
                        path = path_template.format("dqn", path_index.value)
                        print(f"Loading model from {path}\n", end="")
                        model.load_weights(path)


def Agent57_process(params, batch_sizes, path_index, path_template, batch_addresses, device, splits, split_position):
    try:
        import tensorflow as tf
        dtype = params['Misc']['dtype']
        if dtype == 'float16':
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
        from models.intrinsic_motivation import IntrinsicMotivation, get_intrinsic_motivation_model
        from models.dqn import Agent57, get_agent57_model
        from models import dqn
        from bandit import policies
        batch_data = [ActorData(params, b, a) for b, a in zip(batch_sizes, batch_addresses)]
        with tf.device(device):
            while True:
                with path_index.get_lock():
                    if path_index.value >= 0:
                        path = path_template.format("{}", path_index.value)
                        break
                    else:
                        time.sleep(3)
            loading = path.format("im")
            print(f"Loading model from {loading}\n", end="")
            im = get_intrinsic_motivation_model(params, batch_sizes, loading)
            loading = path.format("dqn")
            print(f"Loading model from {loading}\n", end="")
            agent57 = get_agent57_model(params, loading)

            num_actions = params['Agent57']['dual_heads']['num_actions']
            hidden_units = 4 * params['Agent57']['lstm']['units']
            N = params['Misc']['N']
            L = params['Misc']['L']
            consecutive_batches = params['Misc']['consecutive_batches']
            zero_discount_on_life_loss = params['Misc']['zero_discount_on_life_loss']

            hiddens = [tf.zeros((b, hidden_units), dtype) for b in batch_sizes]
            greeds = tf.pow(params['Misc']['greed_e'],
                            1. + (params['Misc']['greed_a'] * (tf.cast(tf.range(L), dtype) / (L - 1))))
            greeds = tf.tile(greeds, [consecutive_batches])
            mu_random = greeds / num_actions
            mu_q = 1. - (greeds * ((num_actions - 1) / num_actions))
            greeds = tf.split(greeds, num_or_size_splits=batch_sizes)
            mu_random = tf.split(mu_random, num_or_size_splits=batch_sizes)
            mu_q = tf.split(mu_q, num_or_size_splits=batch_sizes)

            update_every = params['Misc']['actor_weight_update'] * (len(batch_sizes) / consecutive_batches)
            next_update = update_every
            for i in range(len(batch_sizes) - 1, -1, -1):
                if i % splits != split_position:
                    batch_sizes.pop(i)
                    batch_data.pop(i)
                    greeds.pop(i)
                    mu_random.pop(i)
                    mu_q.pop(i)
                    hiddens.pop(i)
            while True:
                next_hiddens = []
                for b, batch_size, data, greed, m_r, m_q, hidden in zip(range(len(batch_sizes)), batch_sizes,
                                                                        batch_data, greeds, mu_random, mu_q, hiddens):
                    while data.status[0] != 2:
                        pass
                    with data.lock:
                        # print("Agent")
                        data.hidden[:] = hidden.numpy()
                        observations = tf.cast(data.observations, dtype) / 255
                        prev_r_i = im(observations, b)
                        data.intrinsic_rewards[:] = prev_r_i.numpy()
                        prev_r_i = dqn.h(prev_r_i)

                        resets = tf.convert_to_tensor(data.resets)
                        one_hot_js = tf.one_hot(data.j, depth=N, dtype=dtype)
                        prev_a = tf.one_hot(data.actions, depth=num_actions, dtype=dtype)
                        beta, gamma = policies.tf_get_policy(tf.cast(data.j, dtype), N, dtype)
                        beta = tf.expand_dims(beta, axis=-1)
                        prev_reward_e = dqn.h(tf.convert_to_tensor(data.extrinsic_rewards, dtype))

                        q_values, hidden = agent57(observations, prev_a, prev_reward_e, prev_r_i,
                                                   one_hot_js, beta, hidden)

                        action = tf.argmax(q_values, axis=-1, output_type=tf.int32)
                        q_values = dqn.h_inverse(q_values)
                        random_action = tf.random.uniform(action.shape, 0, num_actions - 1, tf.int32)
                        random_decision = tf.random.uniform(action.shape, 0., 1., dtype) < greed
                        random_decision = random_decision | resets
                        action = tf.where(random_decision, random_action, action)
                        mu = tf.where(random_decision, m_r, m_q)
                        selected_q_values = tf.reduce_sum(
                            tf.one_hot(action, depth=num_actions, dtype=dtype) * q_values, axis=-1)
                        discounted_q = tf.reduce_sum(tf.multiply(tf.nn.softmax(q_values, axis=-1),
                                                                 q_values),
                                                     axis=-1)
                        if zero_discount_on_life_loss:
                            discounted_q = tf.multiply(discounted_q, tf.where(tf.convert_to_tensor(data.loss_of_life),
                                                                              tf.zeros_like(gamma), gamma))
                        else:
                            discounted_q = tf.multiply(discounted_q, gamma)
                        data.actions[:] = action.numpy()
                        data.q_value[:] = selected_q_values.numpy()
                        data.discounted_q[:] = discounted_q.numpy()
                        data.mu[:] = mu.numpy()
                        resets = data.resets.copy()
                        data.status[0] = 0
                    im.reset(b, resets)
                    if tf.reduce_any(resets):
                        resets = tf.tile(tf.expand_dims(resets, -1), [1, hidden_units])
                        hidden = tf.where(resets, tf.zeros_like(hidden), hidden)
                    next_hiddens.append(hidden)
                hiddens = next_hiddens
                next_update -= 1
                if next_update <= 0:
                    next_update = update_every
                    with path_index.get_lock():
                        if path_index.value >= 0:
                            path = path_template.format("{}", path_index.value)
                        else:
                            path = None
                        loading = path.format("im")
                        print(f"\nLoading model from {loading}\n", end="")
                        im.load_weights(loading)
                        loading = path.format("dqn")
                        print(f"Loading model from {loading}\n", end="")
                        agent57.load_weights(loading)
    except Exception as e:
        print(e)
        print(traceback.print_exc(4))
        # Agent57_process(params, batch_sizes, path_index, path_template, batch_addresses, device, splits, split_position)


def weight_downloading_process(params, path_index, path_template, path_limit, download_period):
    from learning_server.weights_client import download_files
    import os
    import re
    import time
    root = "/".join(re.split(r'/|\\', path_template)[:-1])
    os.makedirs(root, exist_ok=True)
    del root

    while True:
        with path_index.get_lock():
            path = path_template.format("{}", (path_index.value + 1) % path_limit)
            try:
                download_files(params, path)
                path_index.value = (path_index.value + 1) % path_limit
            except Exception as e:
                print(traceback.print_exc())
                print(e)
        time.sleep(download_period)


if __name__ == "__main__":
    import yaml
    import tensorflow as tf
    import numpy as np
    from multiprocessing import Queue, Value, Process, Lock
    from environment_server import local_replay_buffer

    with open('../params.yml', 'r') as file:
        params = yaml.full_load(file)
    queue = Queue()
    path_index = Value('i', -1)
    path_template = '../weights/agent57_{}_{}.h5'
    L = params['Misc']['L']
    path_limit = 3
    download_period = params['Misc']['download_period']

    preferred_device_types = ['TPU', 'GPU']
    secondary_device_types = ['CPU']
    devices = []
    for device_type in preferred_device_types:
        for device in tf.config.get_visible_devices(device_type):
            devices.append(":".join(device.name.split(":")[-2:]))
    if len(devices) == 0:
        for device_type in secondary_device_types:
            for device in tf.config.get_visible_devices(device_type):
                devices.append(":".join(device.name.split(":")[-2:]))
    if len(devices) == 0:
        raise Exception
    elif len(devices) < 2:
        devices.append(devices[0])

    batches = params['Misc']['consecutive_batches'] * params['Misc']['batch_splits']
    num_envs = L * params['Misc']['consecutive_batches']
    batch_size = num_envs // batches
    last_batch_size = num_envs - (batch_size * (batches - 1))
    batch_sizes = [batch_size for _ in range(batches - 1)]
    batch_sizes.append(last_batch_size)
    print(f"Working with batch sizes {batch_sizes}\n", end="")
    print(f"on devices {devices}\n", end="")

    batch_memory = [ActorData(params, b) for b in batch_sizes]
    batch_addresses = [bm.shared_mem.name for bm in batch_memory]
    transition_queue = Queue()
    replay_lock = Lock()

    processes = [
        #Process(target=local_replay_buffer.transition_upload_process, args=(params, transition_queue, replay_lock)),
        Process(target=weight_downloading_process,
                args=(params, path_index, path_template, path_limit, download_period)),
        #Process(target=replay_buffer_process,
        #        args=(params, batch_sizes, batch_addresses, transition_queue, replay_lock)),
        Process(target=transition_upload_process,
                args=(params, batch_sizes, batch_addresses, replay_lock)),
        Process(target=split_environment_process,
                args=(params, batch_sizes, batch_addresses, 2))]
    # Process(target=environment_process,
    #        args=(params, batch_sizes, batch_addresses))]

    if params['Misc']['split_stream']:
        splits = len(devices)
        for i, device in enumerate(devices):
            processes.append(Process(target=Agent57_process,
                                     args=(
                                     params, batch_sizes, path_index, path_template, batch_addresses, device, splits,
                                     i)))
    else:
        processes.append(Process(target=intrinsic_motivation_process,
                                 args=(params, batch_sizes, path_index, path_template, batch_addresses,
                                       devices[0])))
        processes.append(Process(target=dqn_process,
                                 args=(params, batch_sizes, path_index, path_template, batch_addresses,
                                       devices[1] if len(devices) > 1 else devices[0])))

    for p in processes:
        p.start()

    if params['Misc']['render_actor']:
        import cv2

        while True:
            cv2.imshow("Actor", np.concatenate([batch_memory[0].observations[0], batch_memory[-1].observations[-1]], 1))
            cv2.waitKey(1)
