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


def generate_episode(env: gym.Env, policy: Callable, es: bool = False, limit: int = 8000):
    """A function to generate one episode and collect the sequence of (s, a, r) tuples
    This function will be useful for implementing the MC methods
    Args:
        limit:
        env (gym.Env): a Gym API compatible environment
        policy (Callable): A function that represents the policy.
        es (bool): Whether to use exploring starts or not
    """
    episode = []
    state = env.reset()
    i = 0
    while i < limit:
        if es and len(episode) == 0:
            action = env.action_space.sample()
        else:
            action = policy(state)

        next_state, reward, done, _ = env.step(action)
        episode.append((state, action, reward))
        if done:
            break
        state = next_state
        i += 1

    return episode, done


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
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    N = defaultdict(lambda: np.zeros(env.action_space.n))

    policy = create_epsilon_policy(Q, epsilon)
    state_list = []
    episode_list = np.zeros(num_steps)

    count = 0.0
    n = 0

    while n < num_steps:
        episode, done = generate_episode(env, policy)
        G = 0
        i = n
        if done:
            count += 1.0
        n += len(episode)
        episode_list[i:n] = count

        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = gamma * G + reward

            if state not in state_list:
                state_list.append(state)
                N[state][action] += 1
                Q[state][action] += (G - Q[state][action]) / N[state][action]

    #print(episode_list)
    return episode_list


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

        while True:
            next_state, reward, done, _ = env.step(action)
            policy = create_epsilon_policy(Q, epsilon)
            next_action = policy(next_state)
            Q[state][action] += step_size * (reward + gamma * Q[next_state][next_action] - Q[state][action])

            if done:
                count += 1
                break

            if n < num_steps:
                episode_list[n] = count

            action = next_action
            state = next_state
            n += 1

    # print(episode_list)
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
    n_step = 4

    Q = defaultdict(lambda: np.zeros(env.action_space.n))
    S = [() for _ in range(n_step)]
    A = np.zeros(n_step, dtype=int)
    R = np.zeros(n_step, dtype=float)

    policy = create_epsilon_policy(Q, epsilon)
    episode_list = np.zeros(num_steps)
    episode_count = 0
    step_count = 0

    while step_count < num_steps:
        t = 0
        S[t] = env.reset()
        T = np.infty

        while t < T:
            t_index = t % (n_step - 1)
            action = policy(S[t_index])
            A[t_index] = action
            S[t_index + 1], R[t_index + 1], done, _ = env.step(A[t_index])

            if step_count >= num_steps:
                break
            if done:
                T = t + 1
                episode_count += 1
                episode_list[step_count] = episode_count
            else:
                policy = create_epsilon_policy(Q, epsilon)
                A[t_index + 1] = policy(S[t_index + 1])
                episode_list[step_count] = episode_count

            tau = t - n_step + 2
            if tau >= 0:
                G = sum([(gamma ** (i - tau - 1)) * R[i % n_step] for i in range(tau + 1, min(tau + n_step - 1, T) + 1)])
                if tau >= 0:
                    G += (gamma ** n_step) * Q[S[t_index+1]][A[t_index+1]]
                Q[S[t_index]][A[t_index]] += step_size * (G - Q[S[t_index]][A[t_index]])
            t += 1
            step_count += 1

    # print(episode_list)
    return episode_list


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

        while True:

            next_state, reward, done, _ = env.step(action)
            next_action = policy(next_state)
            value = 0

            prob = np.ones(env.action_space.n) * (epsilon / env.action_space.n)
            prob[np.argmax(Q[next_state])] += (1 - epsilon)
            for i in range(env.action_space.n):
                value += (prob[i] * Q[next_state][i])
            Q[state][action] += step_size * (reward + gamma * value - Q[state][action])

            if done:
                count += 1
                break

            if n < num_steps:
                episode_list[n] = count

            action = next_action
            state = next_state
            n += 1

    #print(episode_list)
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
            Q[state][action] += step_size * (reward + gamma * np.max(Q[next_state]) - Q[state][action])

            if done:
                count += 1
                break

            if n < num_steps:
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
    pass


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
