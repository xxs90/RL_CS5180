"""
    CS 4180/5180 RL and SDM
    Exercise 4: Monte-Carlo Methods
    Prof: Robert Platt
    Date: October 16th, 2021
    Author: Guanang Su
"""

import env
import gym
from matplotlib import pyplot as plt
import algorithms as ag
import policy
import numpy as np
from tqdm import trange
import q2


def q2c():
    q2.infinite_variance()


def q3a():
    env = gym.make('Blackjack-v1')
    for num in [10000, 500000]:
        value = ag.on_policy_mc_evaluation(env, policy.default_blackjack_policy, num, 1)
        #print(value)
        v_usable = np.ones(shape=[21, 10])
        v_no_usable = np.ones(shape=[21, 10])

        key = list(value.keys())
        #print(key)
        for k in key:
            if k[2] == True:
                v_usable[k[0] - 1, k[1] - 1] = value[k]
            else:
                v_no_usable[k[0] - 1, k[1] - 1] = value[k]

        #print(v_usable)
        #print(v_no_usable)
        X = np.arange(1, 11)
        Y = np.arange(12, 21)
        x, y = np.meshgrid(X, Y)
        fig1, ax1 = plt.subplots(subplot_kw={'projection': '3d'})
        fig2, ax2 = plt.subplots(subplot_kw={'projection': '3d'})
        plt.figure(1)
        surface1 = ax1.plot_surface(x, y, v_usable[12:], cmap='turbo')
        fig1.colorbar(surface1)
        plt.title('MC prediction (Usable Ace, ' + str(num) + ' Episodes)')
        plt.xlabel('Dealer showing')
        plt.ylabel('Player sum')
        plt.figure(2)
        surface2 = ax2.plot_surface(x, y, v_no_usable[12:], cmap='turbo')
        fig2.colorbar(surface2)
        plt.title('MC prediction (No Usable Ace, ' + str(num) + ' Episodes)')
        plt.xlabel('Dealer showing')
        plt.ylabel('Player sum')
        plt.show()


def q3b():
    env = gym.make('Blackjack-v1')
    num = 3000000
    Q_value, policy_value = ag.on_policy_mc_control_es(env, num, 1)
    #print(Q_value)
    #print(policy_value)
    v_usable = np.ones(shape=[21, 10])
    v_no_usable = np.ones(shape=[21, 10])
    pi_usable = np.ones(shape=[21, 10])
    pi_no_usable = np.ones(shape=[21, 10])

    key = list(Q_value.keys())
    for k in key:
        if k[2] == True:
            v_usable[k[0] - 1, k[1] - 1] = max(Q_value[k])
            pi_usable[k[0] - 1, k[1] - 1] = policy_value((k[0], k[1], True))
        else:
            v_no_usable[k[0] - 1, k[1] - 1] = max(Q_value[k])
            pi_no_usable[k[0] - 1, k[1] - 1] = policy_value((k[0], k[1], False))


    #print(v_usable)
    #print(v_no_usable)
    #print(p_usable)
    #print(p_no_usable)
    # X = np.arange(1, 11)
    # Y = np.arange(12, 21)
    # x, y = np.meshgrid(X, Y)
    # fig1, ax1 = plt.subplots(subplot_kw={'projection': '3d'})
    # fig3, ax3 = plt.subplots(subplot_kw={'projection': '3d'})
    # plt.figure(1)
    # surface1 = ax1.plot_surface(x, y, v_usable[12:], cmap='turbo')
    # fig1.colorbar(surface1)
    # plt.title('MC prediction (Usable Ace, ' + str(num) + ' Episodes)')
    # plt.xlabel('Dealer showing')
    # plt.ylabel('Player sum')
    plt.figure(0)
    ax2 = plt.axes()
    #print(pi_usable)
    p1 = ax2.imshow(np.fliplr(np.flipud(pi_usable[11:])), cmap='bwr', extent=[1, 10, 11, 21])
    ax2.colorbar(p1)
    plt.title('MC ES (Usable Ace, ' + str(num) + ' Episodes)')
    plt.xlabel('Dealer showing')
    plt.ylabel('Player sum')
    # plt.figure(3)
    # surface2 = ax3.plot_surface(x, y, v_no_usable[12:], cmap='turbo')
    # fig3.colorbar(surface2)
    # plt.title('MC prediction (No Usable Ace, ' + str(num) + ' Episodes)')
    # plt.xlabel('Dealer showing')
    # plt.ylabel('Player sum')
    plt.figure(2)
    ax4 = plt.axes()
    #print(pi_no_usable)
    p2 = ax4.imshow(np.fliplr(np.flipud(pi_usable[10:])), cmap='bwr', extent=[1, 10, 11, 21])
    ax4.colorbar(p2)
    plt.title('MC ES (No Usable Ace, ' + str(num) + ' Episodes)')
    plt.xlabel('Dealer showing')
    plt.ylabel('Player sum')
    plt.show()


def q4a():
    env_room = gym.make('FourRooms-v0', goal_pos=(2, 8))
    num_episodes = 10000
    Q_1, return_1 = ag.on_policy_mc_control_epsilon_soft(env_room, num_episodes=num_episodes, gamma=0.99, epsilon=0.1)
    X = np.arange(1, 11)
    Y = np.arange(12, 21)
    x, y = np.meshgrid(X, Y)
    fig1, ax1 = plt.subplots(subplot_kw={'projection': '3d'})
    plt.figure(1)
    surface1 = ax1.plot_surface(x, y, Q_1, cmap='turbo')
    fig1.colorbar(surface1)
    plt.title('MC prediction (Usable Ace, ' + str(num_episodes) + ' Episodes)')
    plt.xlabel('Dealer showing')
    plt.ylabel('Player sum')

def q4b():
    env_room = gym.make('FourRooms-v0')
    steps = 100

    return_average = []
    trails = 10

    for t in trange(trails, desc="Trails"):
        return_agent = []

        return_agent.append(ag.on_policy_mc_control_epsilon_soft(env_room, num_episodes=steps, gamma=0.99, epsilon=0))
        return_agent.append(ag.on_policy_mc_control_epsilon_soft(env_room, num_episodes=steps, gamma=0.99, epsilon=0.01))
        return_agent.append(ag.on_policy_mc_control_epsilon_soft(env_room, num_episodes=steps, gamma=0.99, epsilon=0.1))

        return_average.append(return_agent)

    average_return = np.average(return_average, axis=0)
    #print(np.shape(average_return))
    standard_error = np.std(return_average, axis=0)
    error = 1.96 * standard_error / np.sqrt(trails)
    upper_bound = [np.amax(average_return)] * steps

    plt.xlabel('Steps')
    plt.ylabel('Average return')
    plt.plot(upper_bound, '--', linewidth=1.0)
    x = np.arange(steps)
    plt.plot(average_return[0], 'C2', linewidth=0.6, label='ε=0 (greedy)')
    plt.fill_between(x, (average_return[0] - error[0]), (average_return[0] + error[0]), color='C2', alpha=0.3)
    plt.plot(average_return[1], 'C1', linewidth=0.6, label='ε=0.01')
    plt.fill_between(x, (average_return[1] - error[1]), (average_return[1] + error[1]), color='C1', alpha=0.3)
    plt.plot(average_return[2], 'C0', linewidth=0.6, label='ε=0.1')
    plt.fill_between(x, (average_return[2] - error[2]), (average_return[2] + error[2]), color='C0', alpha=0.3)
    plt.legend()
    plt.show()

def q6a():
    print('6a')


def q6b():
    print('6b')


def main():
    env.register_env()
    print('What question do you want to do?')
    print('("2": question2, "3": blackjack, "4": four rooms, "6": racetrack)')
    str0 = input('("3a": MC evaluation, "3b": Monte-Carlo ES, "4a": test, "4b": real, "6a": on-policy, "6b": off policy): ')
    if str0 == '2':
        q2c()
    elif str0 == '3a':
        q3a()
    elif str0 == '3b':
        q3b()
    elif str0 == '4a':
        q4a()
    elif str0 == '4b':
        q4b()
    elif str0 == '6a':
        q6a()
    elif str0 == '6b':
        q6b()
    else:
        main()


if __name__ == '__main__':
    main()
