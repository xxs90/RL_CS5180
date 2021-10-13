"""
    CS 4180/5180 RL and SDM
    Exercise 3: Dynamic Programming
    Prof: Robert Platt
    Date: October 9th, 2021
    Author: Guanang Su
"""

import problem5algorithm as ag
import numpy as np
import problem5env as env


def grid_world(theta, gamma):
    print("Which dynamic programming algorithm do you want to choose?")
    str1 = input("(a: iterative policy evaluation, b: value iteration, c: policy iteration): ")
    grid = env.Gridworld5x5()
    V = np.zeros((5, 5), dtype=float)

    if str1 == 'a':
        V = ag.iterative_policy_evaluation(grid, V, theta, gamma)
        print(np.round(V[::-1], 1))

    elif str1 == 'b':
        optimal_value, optimal_policy = ag.value_iteration(grid, V, theta, gamma)
        optimal_action = [[''] * 5] * 5

        for i in range(5):
            for j in range(5):
                optimal_action[i][j] = optimal_policy((i, j))

        print(np.around(optimal_value[::-1], 1))
        print(*optimal_action[::-1], sep='\n')

    elif str1 == 'c':
        V, pi = ag.policy_iteration(grid, V, theta, gamma)
        pi = pi[::-1]
        policy_list = []

        for i in range(5):
            for j in range(5):
                if pi[i][j] == 0:
                    optimal_policy = '\u2193'
                elif pi[i][j] == 1:
                    optimal_policy = '\u2190'
                elif pi[i][j] == 2:
                    optimal_policy = '\u2191'
                else:
                    optimal_policy = '\u2192'
                policy_list.append(optimal_policy)
        print(np.round(V[::-1], 1))
        print(np.array(policy_list).reshape(5, 5), sep='\n')


def carRental():
    print("AAAAAA")


def main():
    threshold = 10**(-3)
    gamma = 0.9

    str0 = input("Which problem do you choose? ('g': gridWorld, 'c': jackCarRental): ")
    if str0 == 'g':
        grid_world(threshold, gamma)
    elif str0 == 'c':
        carRental()
    else:
        print("Invalid selection. Do it again.")
        main()


if __name__ == "__main__":
    main()
