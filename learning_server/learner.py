from learning_server.learner_data import LearnerData
from multiprocessing import Process
import numpy as np
import tensorflow as tf

def load_batch(params, mem_addresses, config=None):
    from replay_buffer import replay_client, database
    import time
    batch_data = [LearnerData(params, address) for address in mem_addresses]
    connection = database.ConnectionManager(config)
    trace_length = params['Misc']['trace_length']
    while True:
        for i, data in enumerate(batch_data):
            while data.status[0] != 0:
                pass
            with data.lock:
                trace_ids = replay_client.request_trace_batch(params, i)
                while len(trace_ids) == 0:
                    time.sleep(60)
                    trace_ids = replay_client.request_trace_batch(params, i)
                data.trace_ids[:] = -1
                index = 0
                for tid in trace_ids:
                    if tid >= 0:
                        data.trace_ids[index] = tid
                        index += 1
                connection.get_batch_by_trace_ids(data, trace_length)
                data.status[0] = 1


def reduced_product(c, dtype):
    """
    Returns the reduced product along each step in the sequence.
    :param c: shape = (batch_size, sequence_length)
    :return: shape = (batch_size, sequence_length+1)
    """
    c = tf.concat([tf.ones((c.shape[0],1), dtype), c], -1)
    c = tf.tile(tf.expand_dims(c, 1), [1,c.shape[-1],1])
    c = tf.linalg.LinearOperatorLowerTriangular(c).to_dense()
    c = tf.where(c == 0., tf.ones_like(c), c)
    c = tf.reduce_prod(c, axis=-1)
    return c


def process_batch(data, dtype, num_actions, N, trace_length, replay_period, retrace_lambda, target, online, im_model, training_splits, break_early, eta):
    from bandit import policies
    from models import dqn
    batch_size = int(np.sum(np.where(data.trace_ids >= 0, np.ones_like(data.trace_ids), np.zeros_like(data.trace_ids))))
    observations = tf.cast(data.observations[:,:batch_size], dtype)
    hidden = tf.cast(data.hidden[:batch_size], dtype)
    teh, tec, tih, tic = tf.split(hidden, num_or_size_splits=4, axis=-1)
    oeh, oec, oih, oic = tf.split(hidden, num_or_size_splits=4, axis=-1)
    one_hot_actions = tf.one_hot(data.actions[:,:batch_size], depth=num_actions, dtype=dtype)
    one_hot_j = tf.one_hot(data.j[:batch_size], depth=N, dtype=dtype)
    beta, gamma = policies.tf_get_policy(tf.cast(data.j[:batch_size], dtype), N, dtype)
    beta = tf.expand_dims(beta, 1)
    prev_extrinsic_rewards = dqn.h(tf.cast(data.prev_extrinsic_rewards[:,:batch_size], dtype))
    prev_intrinsic_rewards = dqn.h(tf.cast(data.prev_intrinsic_rewards[:,:batch_size], dtype))
    mus = tf.cast(data.mu[:batch_size], dtype)

    q_values = []
    for i, x, a, re, ri in zip(range(trace_length+1), observations, one_hot_actions, prev_extrinsic_rewards, prev_intrinsic_rewards):
        tq, teh, tec, tih, tic = target.separate_hidden(x, a, re, ri, one_hot_j, beta, teh, tec, tih, tic)
        q_values.append(tf.expand_dims(tq, 1))
        if i < replay_period:
            oq, oeh, oec, oih, oic = online.separate_hidden(x, a, re, ri, one_hot_j, beta, oeh, oec, oih, oic)


    gamma = tf.expand_dims(gamma, axis=1)
    qt = tf.concat(q_values, 1)
    q_probs = tf.nn.softmax(qt, axis=-1)
    q_values = dqn.h_inverse(qt)
    discounted_q = gamma*tf.reduce_sum(q_probs*q_values, axis=-1)
    discounted_q = tf.where(tf.convert_to_tensor(data.lost_life), tf.zeros_like(discounted_q), discounted_q)
    q_values = tf.reduce_sum(tf.transpose(one_hot_actions, [1,0,2])[:,1:]*q_values[:,:-1], axis=-1)
    temporal_difference = tf.squeeze(tf.transpose(data.prev_extrinsic_rewards[1:,:batch_size] + tf.expand_dims(beta, 0)*data.prev_intrinsic_rewards[1:,:batch_size], [1,0,2]))+discounted_q[:,1:]-q_values

    q_probs = tf.reduce_sum(tf.transpose(one_hot_actions, [1,0,2])[:,replay_period+1:]*q_probs[:,replay_period:-1], axis=-1)
    c = retrace_lambda*tf.minimum(1., q_probs/mus[:,replay_period:-1])
    gamma = tf.pow(gamma, tf.expand_dims(tf.range(trace_length-replay_period, dtype=dtype), 0))
    retrace_targets = [tf.expand_dims(dqn.h(q_values[:,replay_period]+tf.reduce_sum(gamma*reduced_product(c[:,1:], dtype)*temporal_difference[:,replay_period:],axis=-1)),0)]
    for i, q in zip(range(1,trace_length-replay_period), tf.transpose(q_values[:,replay_period+1:], [1,0])):
        retrace_targets.append(tf.expand_dims(dqn.h(q+tf.reduce_sum(gamma[:,:-i]*reduced_product(c[:,i+1:], dtype)*temporal_difference[:,replay_period+i:], axis=-1)),0))

    retrace_targets = tf.concat(retrace_targets, axis=0)
    splits = training_splits

    for x_, a_, re_, ri_, action_taken, retrace_target in zip(tf.split(observations[replay_period:-1], num_or_size_splits=splits, axis=0),
                                                              tf.split(one_hot_actions[replay_period:-1], num_or_size_splits=splits, axis=0),
                                                              tf.split(prev_extrinsic_rewards[replay_period:-1], num_or_size_splits=splits, axis=0),
                                                              tf.split(prev_intrinsic_rewards[replay_period:-1], num_or_size_splits=splits, axis=0),
                                                              tf.split(one_hot_actions[1+replay_period:], num_or_size_splits=splits, axis=0),
                                                              tf.split(retrace_targets, num_or_size_splits=splits, axis=0)):
        with tf.GradientTape() as tape:
            q = []
            for x, a, re, ri in zip(x_, a_, re_, ri_):
                oq, oeh, oec, oih, oic = online.separate_hidden(x, a, re, ri, one_hot_j, beta, oeh, oec, oih, oic)
                q.append(tf.expand_dims(oq, 0))
            q = tf.concat(q, axis=0)
            loss = tf.reduce_sum(tf.square(tf.reduce_sum(action_taken*q, axis=-1)-retrace_target))
        trainable_variables = online.extrinsic_model.trainable_variables
        extrinsic_variable_len = len(trainable_variables)
        trainable_variables += online.intrinsic_model.trainable_variables
        grads = tape.gradient(loss, trainable_variables)
        online.extrinsic_model.optimizer.apply_gradients(zip(grads[:extrinsic_variable_len], trainable_variables[:extrinsic_variable_len]))
        online.intrinsic_model.optimizer.apply_gradients(zip(grads[extrinsic_variable_len:], trainable_variables[extrinsic_variable_len:]))
        if break_early:
            break


    if data.init_step_count[0] > 0:
        one_hot_j = one_hot_j[:data.init_step_count[0]]
        beta = beta[:data.init_step_count[0]]
        oeh, oec, oih, oic = tf.split(tf.zeros((data.init_step_count[0], oeh.shape[1]*4), dtype=dtype), num_or_size_splits=4, axis=-1)
        q_probs = tf.nn.softmax(qt, axis=-1)
        q_probs = tf.reduce_sum(tf.transpose(one_hot_actions, [1,0,2])[:data.init_step_count[0],1:replay_period+1]*q_probs[:data.init_step_count[0],:replay_period], axis=-1)
        c = retrace_lambda*tf.minimum(1., q_probs/mus[:data.init_step_count[0],:replay_period])
        gamma = tf.pow(gamma[:data.init_step_count[0]], tf.expand_dims(tf.range(replay_period, dtype=dtype), 0))
        retrace_targets = [tf.expand_dims(dqn.h(q_values[:data.init_step_count[0],0]+tf.reduce_sum(gamma*reduced_product(c[:,1:], dtype)*temporal_difference[:data.init_step_count[0],:replay_period],axis=-1)),0)]
        for i, q in zip(range(1,replay_period), tf.transpose(q_values[:data.init_step_count[0],1:], [1,0])):
            retrace_targets.append(tf.expand_dims(dqn.h(q+tf.reduce_sum(gamma[:,:replay_period-i]*reduced_product(c[:,i+1:replay_period], dtype)*temporal_difference[:data.init_step_count[0],i:replay_period], axis=-1)),0))

        retrace_targets = tf.concat(retrace_targets, axis=0)
        splits = training_splits*2
        s = 0
        for x_, a_, re_, ri_, action_taken, retrace_target in zip(tf.split(observations[:replay_period, :data.init_step_count[0]], num_or_size_splits=splits, axis=0),
                                                                  tf.split(one_hot_actions[:replay_period,:data.init_step_count[0]], num_or_size_splits=splits, axis=0),
                                                                  tf.split(prev_extrinsic_rewards[:replay_period,:data.init_step_count[0]], num_or_size_splits=splits, axis=0),
                                                                  tf.split(prev_intrinsic_rewards[:replay_period,:data.init_step_count[0]], num_or_size_splits=splits, axis=0),
                                                                  tf.split(one_hot_actions[1:1+replay_period,:data.init_step_count[0]], num_or_size_splits=splits, axis=0),
                                                                  tf.split(retrace_targets, num_or_size_splits=splits, axis=0)):
            with tf.GradientTape() as tape:
                q = []
                for x, a, re, ri in zip(x_, a_, re_, ri_):
                    oq, oeh, oec, oih, oic = online.separate_hidden(x, a, re, ri, one_hot_j, beta, oeh, oec, oih, oic)
                    q.append(tf.expand_dims(oq, 0))
                q = tf.concat(q, axis=0)
                loss = tf.reduce_sum(tf.square(tf.reduce_sum(action_taken*q, axis=-1)-retrace_target))
            trainable_variables = online.extrinsic_model.trainable_variables
            extrinsic_variable_len = len(trainable_variables)
            trainable_variables += online.intrinsic_model.trainable_variables
            grads = tape.gradient(loss, trainable_variables)
            online.extrinsic_model.optimizer.apply_gradients(zip(grads[:extrinsic_variable_len], trainable_variables[:extrinsic_variable_len]))
            online.intrinsic_model.optimizer.apply_gradients(zip(grads[extrinsic_variable_len:], trainable_variables[extrinsic_variable_len:]))
            if break_early:
                s += 1
                if s == 2:
                    break


    splits = training_splits*2
    obs_new_shape = [observations.shape[0]*observations.shape[1]]+[s for s in observations.shape[2:]]
    observations = tf.reshape(observations, obs_new_shape)
    next_observations = tf.split(observations[batch_size:], num_or_size_splits=splits, axis=0)
    observations = tf.split(observations[:-batch_size], num_or_size_splits=splits, axis=0)
    one_hot_actions = tf.split(tf.reshape(one_hot_actions, (one_hot_actions.shape[0]*one_hot_actions.shape[1], one_hot_actions.shape[2]))[batch_size:],
                               num_or_size_splits=splits, axis=0)

    for w, x, a in zip(observations, next_observations, one_hot_actions):
        with tf.GradientTape() as tape:
            y = im_model.embedding_network.call(w,x)
            loss = im_model.embedding_network.loss_fn(a, y)
        grads = tape.gradient(loss, im_model.embedding_network.trainable_variables)
        im_model.embedding_network.optimizer.apply_gradients(zip(grads, im_model.embedding_network.trainable_variables))
        with tf.GradientTape() as tape:
            loss = tf.reduce_mean(im_model.rnd.call(w), 0)
        grads = tape.gradient(loss, im_model.rnd.prediction.trainable_weights)
        im_model.rnd.optimizer.apply_gradients(zip(grads, im_model.rnd.prediction.trainable_weights))

    temporal_difference = tf.math.abs(temporal_difference[:,replay_period:])
    data.priority[:] = (eta*tf.reduce_max(temporal_difference, axis=-1)+(1-eta)*tf.reduce_mean(temporal_difference, axis=-1)).numpy()
    data.status[0] = 2
    tf.keras.backend.clear_session()

def learn_batch(params, mem_addresses, checkpoint_dir, device, eta=.9):
    import traceback
    try:
        import tensorflow as tf
        dtype = params['Misc']['dtype']
        checkpoint_period = params['Misc']['checkpoint_period']
        if dtype == 'float16':
            tf.keras.mixed_precision.set_global_policy('mixed_float16')
        from models import intrinsic_motivation, dqn
        from bandit import policies
        import os
        with tf.device(device):
            batch_data = [LearnerData(params, address) for address in mem_addresses]
            checkpoint_im = 0
            checkpoint_dqn = 0
            im_checkpoints = []
            dqn_checkpoints = []
            for element in os.listdir(checkpoint_dir):
                if os.path.isfile(checkpoint_dir+"/"+element):
                    if element.endswith(".h5") and element.startswith("agent57_"):
                        tokens = element.split("_")
                        if tokens[-1] == "dqn.h5":
                            checkpoint_dqn = max(checkpoint_dqn, int(tokens[1]))
                        if tokens[-1] == "im.h5":
                            checkpoint_im = max(checkpoint_im, int(tokens[1]))
            dqn_path_template = checkpoint_dir+"/agent57_{}_dqn.h5"
            im_path_template = checkpoint_dir+"/agent57_{}_im.h5"

            num_actions = params['Agent57']['dual_heads']['num_actions']
            N = params['Misc']['N']
            replay_period = params['Misc']['replay_period']
            trace_length = params['Misc']['trace_length']
            retrace_lambda = params['Misc']['retrace_lambda']
            update_every = params['Misc']['target_weight_update']//(trace_length-replay_period)
            next_update = update_every

            im_model = intrinsic_motivation.get_intrinsic_motivation_model(params,[1],im_path_template.format(checkpoint_im))
            target = dqn.get_agent57_model(params, dqn_path_template.format(checkpoint_dqn))
            online = dqn.get_agent57_model(params, dqn_path_template.format(checkpoint_dqn))
            import time
            while True:
                for data in batch_data:
                    while data.status[0] != 1:
                        pass
                    with data.lock:
                        timer = time.time()
                        process_batch(data, dtype, num_actions, N, trace_length, replay_period, retrace_lambda, target, online, im_model, params['Misc']['training_splits'], params['Misc']['break_training_loop_early'], eta)
                    next_update -= 1

                    if next_update <= 0:
                        print("Updating weights.")
                        checkpoint_im += 1
                        checkpoint_dqn += 1
                        im_model.save_weights(im_path_template.format(checkpoint_im))
                        online.save_weights(dqn_path_template.format(checkpoint_dqn))
                        target.load_weights(dqn_path_template.format(checkpoint_dqn))
                        im_checkpoints.append(checkpoint_im)
                        dqn_checkpoints.append(checkpoint_dqn)
                        if len(im_checkpoints) > 3:
                            removing = im_checkpoints.pop(0)
                            if removing % checkpoint_period != 0:
                                os.remove(im_path_template.format(removing))
                        if len(dqn_checkpoints) > 3:
                            removing = dqn_checkpoints.pop(0)
                            if removing % checkpoint_period != 0:
                                os.remove(dqn_path_template.format(removing))
                        print("Weights updated.")
                        next_update = update_every
                    print(f"Total time: {(time.time()-timer)}")
    except Exception as e:
        print(traceback.print_exc())
        print(e)

def update_batch(params, mem_addresses):
    from replay_buffer import replay_client, database
    batch_data = [LearnerData(params, address) for address in mem_addresses]
    cm = database.ConnectionManager()
    while True:
        for i, data in enumerate(batch_data):
            while data.status[0] != 2:
                pass
            with data.lock:
                cm.update_priorities(data.episode_ids, data.trace_ids, data.priority)
                replay_client.signal_trace_update(params, i)
                data.status[0] = 0

if __name__ == "__main__":
    import yaml
    import tensorflow as tf
    import os
    with open('../params.yml', 'r') as file:
        params = yaml.full_load(file)
    batch_data = [LearnerData(params) for b in range(params['Misc']['consecutive_training_batches'])]
    mem_addresses = [ld.shared_mem.name for ld in batch_data]
    checkpoint_dir = "../weights/checkpoints"
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    loading_batches = Process(target=load_batch, args=(params, mem_addresses))
    learning_from_data = Process(target=learn_batch, args=(params, mem_addresses, checkpoint_dir, tf.config.list_logical_devices("GPU")[0].name))
    updating_data = Process(target=update_batch, args=(params, mem_addresses))

    loading_batches.start()
    learning_from_data.start()
    updating_data.start()
