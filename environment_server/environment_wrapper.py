import gym
import numpy as np
import cv2


class Environment:

    def __init__(self, environment_name, obs_type, frameskip, max_episode_length, index, actions, observations, rewards, steps, resets, loss_of_life, reward_scale):
        if type(frameskip) == int:
            self.env = gym.make(environment_name, obs_type=obs_type, frameskip=frameskip, full_action_space=True)
        else:
            self.env = gym.make(environment_name, obs_type=obs_type,  full_action_space=True)
        self.max_episode_length = max_episode_length
        self.timeout = 1
        self.index = index
        self.actions = actions
        self.observations = observations
        self.rewards = rewards
        self.steps = steps
        self.resets = resets
        self.loss_of_life = loss_of_life
        self.reward_scale = reward_scale
        self.life_count = -1

    def __call__(self, *args, **kwargs):
        if self.resets[self.index]:
            obs = self.env.reset()
            done = False
            reward = 0.
            self.life_count = -1
            self.loss_of_life[self.index] = False
            self.steps[self.index] = 0
        else:
            obs, reward, done, info = self.env.step(self.actions[self.index])
            self.steps[self.index] += 1
            if 'lives' in info:
                lives = info['lives']
                if self.life_count >= 0:
                    if lives < self.life_count:
                        self.loss_of_life[self.index] = True
                    else:
                        self.loss_of_life[self.index] = False
                self.life_count = lives
        if self.steps[self.index] == self.max_episode_length or done:
            done = True
        self.observations[self.index] = np.reshape(obs,  (210, 160, -1))
        self.rewards[self.index] = reward*self.reward_scale
        self.resets[self.index] = done

def multi_env(params, batch_size, mem_address, start, end, env_splits, split_index):
    from environment_server.actor_data import ActorData
    data = ActorData(params, batch_size, mem_address)
    envs = []
    frameskip = params['Misc']['frameskip'] if type(params['Misc']['frameskip']) == int else False
    reward_scale = params['Misc']['reward_scale']
    for i in range(batch_size):
        if i%env_splits == split_index:
            envs.append(Environment(params['Misc']['environment'],
                                    params['Misc']['obs_type'],
                                    frameskip,
                                    params['Misc']['max_episode_length'],
                                    i,
                                    data.actions,
                                    data.observations,
                                    data.extrinsic_rewards,
                                    data.steps,
                                    data.resets,
                                    data.loss_of_life,
                                    reward_scale))
    while True:
        start.get()
        for env in envs:
            env()
        end.put(True)

if __name__ == "__main__":
    import random
    import time
    import sys
    from environment_server.actor_data import ActorData
    import yaml
    with open('../actors/params.yml', 'r') as file:
        params = yaml.full_load(file)
    foo = ActorData(params, 1)
    foo.resets[0] = True
    environment_name = 'MontezumaRevenge-v0'
    obs_type = 'grayscale'
    max_episode_length = 468000

    steps_in_test = 12800

    env = Environment(environment_name, obs_type, max_episode_length,0, foo.actions, foo.observations, foo.extrinsic_rewards, foo.resets)
    times = []
    for i in range(steps_in_test):
        timer = time.time()
        env()
        timer = time.time()-timer
        times.append(timer)
        cv2.imshow("Actor", foo.observations[0][:])
        cv2.waitKey(1)
    print(f"Total time spent acting {sum(times)}")
    print(f"Average time {sum(times)/len(times)}")
    print(f"Steps per second {steps_in_test/sum(times)}")
    print(foo.observations)