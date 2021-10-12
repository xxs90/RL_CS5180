"""
    CS 4180/5180 RL and SDM
    Exercise 3: Dynamic Programming
    Prof: Robert Platt
    Date: October 9th, 2021
    Author: Guanang Su
"""

import algorithms as ag
import numpy as np
import env


def grid_world(theta, gamma):
    print("Which dynamic programming algorithm do you want to choose?")
    str1 = input("(a: iterative policy evaluation, b: value iteration, c: policy iteration): ")
    grid = env.Gridworld5x5()
    V = np.zeros((5, 5), dtype=float)

    if str1 == 'a':
        V = ag.iterative_policy_evaluation(grid, V, theta, gamma)
        print(np.round(V, 1))

    elif str1 == 'b':
        optimal_value, optimal_policy = ag.value_iteration(grid, V, theta, gamma)
        optimal_action = [[''] * 5] * 5

        for i in range(5):
            for j in range(5):
                optimal_action[i][j] = optimal_policy((i, j))

        print(np.around(optimal_value, 1))
        print(*optimal_action, sep='\n')

    elif str1 == 'c':
        Q = np.zeros(shape=(5, 5), dtype=str)
        optimal_value, optimal_policy = ag.policy_iteration(grid, V, Q, theta, gamma)

        print(np.around(optimal_value, 1))
        print(*optimal_policy, sep='\n')


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
