"""
    CS 4180/5180 RL and SDM
    Exercise 5: Temporal_Difference Learning
    Prof: Robert Platt
    Date: October 23rd, 2021
    Author: Guanang Su
"""

import gym
from typing import Optional
from collections import defaultdict
import numpy as np
import random
from typing import Callable, Tuple
from tqdm import trange
import env as env0


def generate_episode(env: gym.Env, policy: Callable, es: bool = False):
    """A function to generate one episode and collect the sequence of (s, a, r) tuples
    This function will be useful for implementing the MC methods
    Args:
        env (gym.Env): a Gym API compatible environment
        policy (Callable): A function that represents the policy.
        es (bool): Whether to use exploring starts or not
    """
    episode = []
    state = env.reset()
    while True:
        if es and len(episode) == 0:
            action = env.action_space.sample()
        else:
            action = policy(state)

        next_state, reward, done, _ = env.step(action)
        episode.append((state, action, reward))
        if done:
            break
        state = next_state

    return episode


def create_epsilon_policy(Q: defaultdict, epsilon: float) -> Callable:
    """Creates an epsilon soft policy from Q values.

    A policy is represented as a function here because the policies are simple. More complex policies can be represented using classes.

    Args:
        Q (defaultdict): current Q-values
        epsilon (float): softness parameter
    Returns:
        get_action (Callable): Takes a state as input and outputs an action
    """
    # Get number of actions
    num_actions = len(Q[0])

    def get_action(state: Tuple) -> int:
        if np.random.random() < epsilon:
            action = random.choice(range(0, num_actions))
        else:
            action = np.random.choice(np.where(Q[state] == Q[state].max())[0])

        return action

    print(get_action)
    return get_action


def on_policy_mc_control_epsilon_soft(
    env: gym.Env, num_steps: int, gamma: float, epsilon: float,
):
    """On-policy Monte Carlo policy control for epsilon soft policies.

    Args:
        env (gym.Env): a Gym API compatible environment
        num_episodes (int): Number of episodes
        gamma (float): Discount factor of MDP
        epsilon (float): Parameter for epsilon soft policy (0 <= epsilon <= 1)
    Returns:

    """
    # Q = defaultdict(lambda: np.zeros(env.action_space.n))
    # N = defaultdict(lambda: np.zeros(env.action_space.n))
    #
    # policy = create_epsilon_policy(Q, epsilon)
    # returns = np.zeros(num_steps)
    #
    # for i in trange(num_steps, desc="Episode", leave=False):
    #     episode = generate_episode(env, policy)
    #     G = 0
    #     episode_list = []
    #
    #     for t in range(len(episode) - 1, -1, -1):
    #         state, action, reward = episode[t]
    #         G = gamma * G + reward
    #
    #         if state not in episode_list:
    #             episode_list.append(state)
    #             N[state][action] += 1
    #             Q[state][action] += (G - Q[state][action]) / N[state][action]
    #
    #     returns[i] = G
    #
    # return returns


def sarsa(env: gym.Env, num_steps: int, gamma: float, epsilon: float, step_size: float):
    """SARSA algorithm.

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    # TODO
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    episode_list = np.zeros(num_steps)
    count = 0
    n = 0

    while n < num_steps:
        state = env.reset()
        policy = create_epsilon_policy(Q, epsilon)
        action = policy(state)

        while n < num_steps:
            next_state, reward, done, _ = env.step(action)
            next_action = policy(next_state)
            Q[state][action] += step_size * (reward + gamma * Q[next_state][next_action] - Q[state][action])

            if done:
                count += 1
                episode_list[n] = count
                break

            episode_list[n] = count

            action = next_action
            state = next_state
            n += 1

    #print(episode_list)
    return episode_list


def nstep_sarsa(
    env: gym.Env,
    num_steps: int,
    gamma: float,
    epsilon: float,
    step_size: float,
):
    """N-step SARSA

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    # TODO
    pass


def exp_sarsa(
    env: gym.Env,
    num_steps: int,
    gamma: float,
    epsilon: float,
    step_size: float,
):
    """Expected SARSA

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    # TODO
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    episode_list = np.zeros(num_steps)
    count = 0
    n = 0

    while n < num_steps:
        state = env.reset()
        policy = create_epsilon_policy(Q, epsilon)
        action = policy(state)

        while n < num_steps:
            next_state, reward, done, _ = env.step(action)
            next_action = policy(next_state)
            Q[state][action] += step_size * (reward + gamma * Q[next_state][next_action] - Q[state][action])

            if done:
                count += 1
                episode_list[n] = count
                break

            episode_list[n] = count

            action = next_action
            state = next_state
            n += 1

    # print(episode_list)
    return episode_list


def q_learning(
    env: gym.Env,
    num_steps: int,
    gamma: float,
    epsilon: float,
    step_size: float,
):
    """Q-learning

    Args:
        env (gym.Env): a Gym API compatible environment
        num_steps (int): Number of steps
        gamma (float): Discount factor of MDP
        epsilon (float): epsilon for epsilon greedy
        step_size (float): step size
    """
    # TODO
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    episode_list = np.zeros(num_steps)
    count = 0
    n = 0

    while n < num_steps:
        state = env.reset()

        while n < num_steps:
            policy = create_epsilon_policy(Q, epsilon)
            action = policy(state)
            next_state, reward, done, _ = env.step(action)
            # next_action = policy(next_state)
            value = []
            #print(action)
            for a in env0.Action:
                value.append(Q[next_state][a])
            Q[state][action] += step_size * (reward + gamma * np.max(value) - Q[state][action])

            if done:
                count += 1
                episode_list[n] = count
                break

            episode_list[n] = count

            state = next_state
            n += 1

    # print(episode_list)
    return episode_list


def td_prediction(env: gym.Env, gamma: float, episodes, n=1) -> defaultdict:
    """TD Prediction

    This generic function performs TD prediction for any n >= 1. TD(0) corresponds to n=1.

    Args:
        env (gym.Env): a Gym API compatible environment
        gamma (float): Discount factor of MDP
        episodes : the evaluation episodes. Should be a sequence of (s, a, r) tuples or a dict.
        n (int): The number of steps to use for TD update. Use n=1 for TD(0).
    """
    # TODO
    # Q = defaultdict(lambda: np.zeros(env.action_space.n))
    # episode_list = np.zeros(num_steps)
    # count = 0
    # n = 0
    #
    # while n < num_steps:
    #     state = env.reset()
    #     policy = create_epsilon_policy(Q, epsilon)
    #     action = policy(state)
    #
    #     while n < num_steps:
    #         next_state, reward, done, _ = env.step(action)
    #         next_action = policy(next_state)
    #         Q[state][action] += step_size * (reward + gamma * Q[next_state][next_action] - Q[state][action])
    #
    #         if done:
    #             count += 1
    #             episode_list[n] = count
    #             break
    #
    #         episode_list[n] = count
    #
    #         action = next_action
    #         state = next_state
    #         n += 1
    #
    # # print(episode_list)
    # return episode_list


def learning_targets(
    V: defaultdict, gamma: float, episodes, n: Optional[int] = None
) -> np.ndarray:
    """Compute the learning targets for the given evaluation episodes.

    This generic function computes the learning targets for Monte Carlo (n=None), TD(0) (n=1), or TD(n) (n=n).

    Args:
        V (defaultdict) : A dict of state values
        gamma (float): Discount factor of MDP
        episodes : the evaluation episodes. Should be a sequence of (s, a, r) tuples or a dict.
        n (int or None): The number of steps for the learning targets. Use n=1 for TD(0), n=None for MC.
    """
    # TODO
    targets = np.zeros(len(episodes))

    pass
