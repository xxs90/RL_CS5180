"""
    CS 4180/5180 RL and SDM
    Exercise 3: Dynamic Programming
    Prof: Robert Platt
    Date: October 8th, 2021
    Author: Guanang Su
"""

import env as env
import numpy as np


def iterative_policy_evaluation(environment, V, theta, gamma):
    while True:
        delta = 0

        for s in environment.state_space:
            v = V[s]
            state_value = 0

            for a in env.Action:
                s_prime, reward = environment.transitions(s, a)
                state_value += 0.25 * (reward + gamma * V[s_prime])

            V[s] = state_value
            delta = max(delta, abs(v - V[s]))

        if delta < theta:
            break
    return V


def value_iteration(environment, V, theta, gamma):
    while True:
        delta = 0
        for state in environment.state_space:
            v = V[state]
            value_list = []

            for action in env.Action:
                s_prime, reward = environment.transitions(state, action)
                value_list.append(reward + gamma * V[s_prime])

            V[state] = max(value_list)
            delta = max(delta, abs(v - V[state]))

        if delta < theta:
            break

    def policy(state):
        a = ''
        max_value = -999

        for action in env.Action:
            s_prime, reward = environment.transitions(state, action)
            value = reward + gamma * V[s_prime]

            if value > max_value:
                max_value = value
                if action == env.Action.UP:  # right
                    a = '\u2192'
                elif action == env.Action.DOWN:  # left
                    a = '\u2190'
                elif action == env.Action.LEFT:  # down
                    a = '\u2191'
                elif action == env.Action.RIGHT:  # up
                    a = '\u2193'

            elif value == max_value:
                if action == env.Action.UP:
                    a += '/\u2192'
                elif action == env.Action.DOWN:
                    a += '/\u2190'
                elif action == env.Action.LEFT:
                    a += '/\u2191'
                elif action == env.Action.RIGHT:
                    a += '/\u2193'
        return a

    return V, policy


def policy_evaluation(environment, V, pi, gamma, theta):

    while True:
        delta = 0

        for state in environment.state_space:
            v = V[state]

            s_prime, reward = environment.transitions(state, pi[state])
            state_value = reward + gamma * V[s_prime]

            V[state] = state_value
            delta = max(delta, abs(v - V[state]))

        if delta < theta:
            break
    return V


def policy_iteration(environment, V, theta, gamma):
    pi = np.zeros(shape=(5, 5), dtype=float)
    policy_evaluation(environment, V, pi, gamma, theta)
    policy_stable = True

    while policy_stable:
        for state in environment.state_space:
            old_action = pi[state]
            state_value = []

            for action in env.Action:
                s_prime, reward = environment.transitions(state, action)
                value = reward + gamma * V[s_prime]
                state_value.append(value)
            pi[state] = np.argmax(state_value)

            if old_action != pi[state]:
                policy_stable = False

        if policy_stable:
            break
        else:
            policy_evaluation(environment, V, pi, gamma, theta)

    return V, pi
