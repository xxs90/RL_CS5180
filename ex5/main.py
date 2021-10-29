"""
    CS 4180/5180 RL and SDM
    Exercise 5: Temporal_Difference Learning
    Prof: Robert Platt
    Date: October 23rd, 2021
    Author: Guanang Su
"""
import numpy as np

import env
import gym
import algorithms as ag
from matplotlib import pyplot as plt
def q4b():
    #print('4b')
    #n = 2000
    step_num = 8000
    env = gym.make('WindyGridWorld-v0')

    data_1 = []
    data_2 = []
    for i in range(1):
        #returns = ag.on_policy_mc_control_epsilon_soft(env=env, num_steps=step_num, gamma=1, epsilon=0.1)
        returns_1 = ag.sarsa(env=env, num_steps=step_num, gamma=1, epsilon=0.1, step_size=0.5)
        returns_2 = ag.q_learning(env=env, num_steps=step_num, gamma=1, epsilon=0.1, step_size=0.5)
        data_1.append(returns_1)
        data_2.append(returns_2)

    #print(data)
    #t = range(len(data))
    #steps = np.average(data_2, axis=0)
    plt.xlabel('Time steps')
    plt.ylabel('Episodes')
    # plt.title('SARSA (on-policy TD control)')
    #plt.title('Q-learning (off-policy TD control)')
    plt.plot(np.average(data_1, axis=0), 'r', label='sarsa')
    plt.plot(np.average(data_2, axis=0), 'b', label='q_learning')
    plt.legend()
    plt.show()

    #t = range(len(steps))


def q4c():
    print('4c')


def q4d():
    print('4d')


def q5():
    print('5')


def main():
    print('\nWhich question do you want to run? (4: windy gridworld, 5: bias-variance trade-off)')
    str0 = input('(4: "b": method compare, "c": king\'s move, "d": stochastic wind and "5" ): ')
    env.register_env()
    if str0 == 'b':
        q4b()
    elif str0 == 'c':
        q4c()
    elif str0 == 'd':
        q4d()
    elif str0 == '5':
        q5()
    else:
        main()


if __name__ == "__main__":
    main()
