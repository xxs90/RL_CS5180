"""
    CS 4180/5180 RL and SDM
    Exercise 3: Dynamic Programming
    Prof: Robert Platt
    Date: October 9th, 2021
    Author: Guanang Su
"""

import numpy as np
import env
import random


def iterativePolicyEvaluation(V, state, theta, gamma):
    grid = env.Gridworld5x5()
    while True:
        delta = 0
        for s in grid.state_space:
            v = V[s]

            state_value = 0
            for a in env.Action:
                next_s, reward = grid.transitions(s, a)
                state_value += 0.25 * (reward + gamma * V[next_s])

            v[s] = state_value
            delta = max(delta, abs(v - V[s]))

        if delta < theta:
            break

    return V


def valueIteration(theta, state, gamma):
    grid = env.Gridworld5x5()
    V = np.empty(shape=(5, 5), dtype=float)

    while True:
        delta = 0
        for s in grid.state_space:
            v = V[s]

            value_list = []
            for a in env.Action:
                next_s, reward = grid.transitions(s, a)
                value = reward + gamma * V[next_s]
                value_list.append(value)

            v[s] = max(value_list)
            delta = max(delta, abs(v - V[s]))

        if delta < theta:
            break

    return V


def policy(V, s, gamma):
    action = ''
    max_value = -999

    for a in env.Action:
        next_s, reward = env.Gridworld5x5.transitions(s, a)
        value = reward + gamma * V[next_s]

        if value > max_value:
            action = a
            max_value = value
        elif value == max_value:
            action += '/' + a

    return action


def pick(pi, s):
    action = pi[s].decode('utf-8')
    if '/' in action:
        tokens = action.split('/')
        action = random.choice(tokens)
    return action


def policyIteration(S, V, pi, theta, gamma):
    policy_stable = False
    while not policy_stable:
        while True:
            delta = 0
            for s in S:
                v = V[s]
                state_value = 0
                next_state, reward = env.Gridworld5x5.transitions(s, pick(pi, s))
                state_value += reward + gamma * V[next_state]
                V[s] = state_value
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break

        policy_stable = True
        for s in S:
            next_state, reward = env.Gridworld5x5.transitions(s, pick(pi, s))
            old_value = reward + gamma * V[next_state]
            pi[s] = policy(V, s, gamma)
            s_prime, reward = env.Gridworld5x5.transitions(s, pick(pi, s))
            new_value = reward + gamma * V[s_prime]
            if old_value != new_value:
                policy_stable = False
    return V, pi
