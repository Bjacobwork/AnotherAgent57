from models import dqn, intrinsic_motivation
import datetime
from bandit import bandit_client, policies
import os
import tensorflow as tf
import gym
import random
import cv2

def load_models(params, checkpoint_dir, motivation=None, agent=None):
    dqn_p = 0
    im_p = 0
    path_template = checkpoint_dir+"/agent57_{}_{}.h5"
    for obj in os.listdir(checkpoint_dir):
        if os.path.isfile(checkpoint_dir+"/"+obj):
            if obj.endswith(".h5"):
                tokens = obj.split("_")
                if tokens[0] == 'agent57':
                    if tokens[-1] == "dqn.h5":
                        dqn_p = max(dqn_p, int(tokens[1]))
                    elif tokens[-1] == "im.h5":
                        im_p = max(im_p, int(tokens[1]))
    if type(motivation) == type(None):
        motivation = intrinsic_motivation.get_intrinsic_motivation_model(params, [1], path_template.format(im_p, "im"))
        agent = dqn.get_agent57_model(params, path_template.format(dqn_p, "dqn"))
    else:
        motivation.load_weights(path_template.format(im_p, "im"))
        agent.load_weights(path_template.format(dqn_p, "dqn"))
    return motivation, agent

def episode(dtype, env, env_name, obs_shape, num_actions, hidden_depth, max_episode_length, N, j, beta, motivation, agent, reward_scale, reward_cutout_length=500):
    from models import dqn
    motivation.reset(0, [True])
    total_reward = tf.zeros((1,1), dtype=dtype)
    score = tf.zeros((1,1), dtype=dtype)
    obs = tf.reshape(tf.cast(env.reset(), dtype)/255, obs_shape)
    reward_i = motivation(obs, 0)
    reward_e = tf.zeros((1,1), dtype=dtype)
    one_hot_a = tf.one_hot([random.randint(0, num_actions-1)], depth=num_actions, dtype=dtype)
    one_hot_j = tf.one_hot([j], depth=N, dtype=dtype)
    eh, ec, ih, ic = tf.split(tf.zeros((1,hidden_depth), dtype=dtype), num_or_size_splits=4, axis=-1)
    timed_out = True
    rewards = []
    for s in range(max_episode_length):
        reward_e = dqn.h(reward_e)
        reward_i = dqn.h(reward_i)
        q, eh, ec, ih, ic = agent.separate_hidden(obs, one_hot_a, reward_e, reward_i, one_hot_j,beta, eh, ec, ih, ic)
        action = tf.argmax(q, axis=-1)
        obs, reward_e, done, info = env.step(int(action[0]))
        reward_e = reward_e*reward_scale
        rewards.append(reward_e)
        if len(rewards) > reward_cutout_length:
            rewards.pop(0)
        arr = env.render(mode="rgb_array")
        cv2.imshow(env_name, cv2.cvtColor(arr, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
        obs = tf.reshape(tf.cast(obs, dtype)/255, obs_shape)
        reward_e = tf.convert_to_tensor([[reward_e]], dtype=dtype)
        reward_i, episodic_reward = motivation.get_both_rewards(obs, 0)
        total_reward += reward_e+beta*reward_i
        score += reward_e
        if sum(rewards) == 0 and len(rewards) >= reward_cutout_length:
            print(f"\nNo extrinsic rewards found in {len(rewards)} steps.")
            timed_out = True
            break
        if done:
            print("\nThe gym environment ended the episode.")
            timed_out = False
            break
    if timed_out:
        print("\nThe actor ran out of time.")
    return float(total_reward[0,0]), float(score[0,0])

def evaluate(params, device, checkpoint_dir):
    dtype = params['Misc']['dtype']
    if dtype == 'float16':
        tf.keras.mixed_precision.set_global_policy('mixed_float16')
    with tf.device(device):
        motivation, agent = load_models(params, checkpoint_dir)
        client = bandit_client.BanditClient(params)
        evaluation_iterations = params['Misc']['evaluation_iterations']
        env_name = params['Misc']['environment']
        N = params['Misc']['N']
        K = params['EpisodicMemory']['k']
        num_actions = params['Agent57']['dual_heads']['num_actions']
        hidden_depth = params['Agent57']['lstm']['units']*4
        max_episode_length = params['Misc']['max_episode_length']
        obs_shape = params['Misc']['obs_shape']
        reward_scale = params['Misc']['reward_scale']
        env = gym.make(env_name,
                       obs_type=params['Misc']['obs_type'],
                       full_action_space=True)
        high_score = 0
        while True:
            # Training bandit algorithm:
            for i in range(evaluation_iterations):
                j = client.get_non_greedy_j()
                beta, gamma = policies.get_policy(j, N)
                print(f"\nEvaluating policy {j}\n  Beta  = {'%.5f'%beta}\n  Gamma = {'%.5f'%gamma}\n {datetime.datetime.now()}")
                total_reward, score = episode(dtype, env, env_name, obs_shape, num_actions, hidden_depth, max_episode_length,N, j, beta, motivation, agent, reward_scale)
                high_score = max(score, high_score)
                print(f"Reward: {total_reward}\nScore: {score}\nHigh Score: {high_score}\n{datetime.datetime.now()}")
                client.update_reward(j, score, score)
            motivation, agent = load_models(params, checkpoint_dir, motivation, agent)
            # Evaluation:
            evaluated_rewards = []
            evaluated_scores = []
            j = client.get_greedy_j()
            beta, gamma = policies.get_policy(j, N)
            print(f"\nEvaluating greedy policy {j}\n  Beta  = {'%.5f'%beta}\n  Gamma = {'%.5f'%gamma}\n {datetime.datetime.now()}")
            for i in range(evaluation_iterations):
                total_reward, score = episode(dtype, env, env_name, obs_shape, num_actions, hidden_depth, max_episode_length,N, j, beta, motivation, agent, reward_scale)
                high_score = max(score, high_score)
                print(f"Reward: {total_reward}\nScore: {score}\nHigh Score: {high_score}\n{datetime.datetime.now()}")
                evaluated_rewards.append(total_reward)
                evaluated_scores.append(score)
            client.update_reward(j,sum(evaluated_scores)/evaluation_iterations, max(evaluated_scores))
            motivation, agent = load_models(params, checkpoint_dir, motivation, agent)

if __name__ == "__main__":
    import yaml
    import traceback
    with open('../params.yml', 'r') as file:
        params = yaml.full_load(file)
    checkpoint_dir = "../weights/checkpoints"
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
    while True:
        try:
            evaluate(params, tf.config.list_logical_devices("CPU")[0].name, checkpoint_dir)
        except Exception as e:
            traceback.print_exc()
            print(e)
