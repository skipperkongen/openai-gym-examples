"""
Reproduction of http://kvfrans.com/simple-algoritms-for-solving-cartpole/
"""
import gym
import numpy as np

env = gym.make('CartPole-v0')

def run_episode(env, params):
    observation = env.reset()
    total_reward = 0
    for _ in range(200):
        env.render()
        action = 0 if np.matmul(params, observation) < 0 else 1
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward

best_params = None
best_reward = 0
for i_episode in range(10000):
    params = np.random.rand(4) * 2 - 1
    reward = run_episode(env, params)
    print('Reward:', reward)
    if reward > best_reward:
        best_reward = reward
        best_params = params
    if reward >= 200:
        break
