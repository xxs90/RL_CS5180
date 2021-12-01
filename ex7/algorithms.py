"""
    CS 4180/5180 RL and SDM
    Exercise 7: Function Approximation
    Prof: Robert Platt
    Date: November 30th, 2021
    Author: Guanang Su
"""

import gym
import numpy as np
import random


def create_epsilon_policy(env:gym.Env, state, weights, epsilon):
    num_actions = env.action_space.n
    next_action_q = np.array([])

    for i in range(num_actions):
        next_action_q = np.append(next_action_q, np.dot(env.aggregation(state, i), weights))

    if np.random.random() < epsilon:
        action = random.randint(0, num_actions-1)
    else:
        action = np.random.choice(np.where(next_action_q == next_action_q.max())[0])

    return action


def semi_grad_sarsa(env: gym.Env, num_episode: int, gamma: float, epsilon: float, step_size: float):
    data = np.array([])
    n = 0

    aggregation = env.aggregation
    state = env.observation_space.sample()
    action = env.action_space.sample()
    w = np.zeros(len(aggregation(state, action)))

    while n < num_episode:
        state = env.reset()
        count = 0
        action = create_epsilon_policy(env, state, w, epsilon)

        while True:
            count += 1
            next_state, reward, done, _ = env.step(action)

            if done:
                w += step_size * (reward - np.dot(aggregation(state, action), w)) * aggregation(state, action)
                break

            next_action = create_epsilon_policy(env, state, w, epsilon)
            w += step_size * (reward + gamma * np.dot(aggregation(next_state, next_action), w) - np.dot(aggregation(state, action), w)) * aggregation(state, action)

            state = next_state
            action = next_action

        n += 1
        data = np.append(data, count)

    return w, data


