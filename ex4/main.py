import env
import gym
from matplotlib import pyplot as plt
import algorithms as ag
import policy
import numpy as np


def q2c():
    print('2c')


def q3a():
    env = gym.make('Blackjack-v1')
    for num in [10000, 500000]:
        value = ag.on_policy_mc_evaluation(env, policy.default_blackjack_policy, num, 1)
        #print(value)
        v_usable = np.ones(shape=[21, 10])
        v_no_usable = np.ones(shape=[21, 10])

        key = np.array(list(value.keys()))
        #print(key)
        for k in key:
            if k[2] == True:
                v_usable[k[0] - 1, k[1] - 1] = value[tuple(k)]
            else:
                v_no_usable[k[0] - 1, k[1] - 1] = value[tuple(k)]

        print(v_usable)
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
    num = 300000
    Q_value, policy = ag.on_policy_mc_control_es(env, num, 1)
    print(Q_value)
    print(policy)
    q_usable = np.ones(shape=[21, 10])
    q_no_usable = np.ones(shape=[21, 10])

    key = np.array(list(Q_value.keys()))
    for k in key:
        if k[2] == True:
            #print(Q_value[tuple(k)])
            a, b = Q_value[tuple(k)]
            if a >= b:
                q_usable[k[0] - 1, k[1] - 1] = a
            else:
                q_usable[k[0] - 1, k[1] - 1] = b
        else:
            #print(Q_value[tuple(k)])
            a, b = Q_value[tuple(k)]
            if a >= b:
                q_no_usable[k[0] - 1, k[1] - 1] = a
            else:
                q_no_usable[k[0] - 1, k[1] - 1] = b

    #print(v_usable)
    #print(v_no_usable)
    #print(p_usable)
    #print(p_no_usable)
    X = np.arange(1, 11)
    Y = np.arange(12, 21)
    x, y = np.meshgrid(X, Y)
    fig1, ax1 = plt.subplots(subplot_kw={'projection': '3d'})
    fig3, ax3 = plt.subplots(subplot_kw={'projection': '3d'})
    plt.figure(1)
    surface1 = ax1.plot_surface(x, y, q_usable[12:], cmap='turbo')
    fig1.colorbar(surface1)
    plt.title('MC prediction (Usable Ace, ' + str(num) + ' Episodes)')
    plt.xlabel('Dealer showing')
    plt.ylabel('Player sum')
    fig2 = plt.figure(2)
    p_1 = plt.imshow(p_usable[:, :, 1])
    fig2.colorbar(p_1)
    plt.title('MC prediction (Usable Ace, ' + str(num) + ' Episodes)')
    plt.xlabel('Dealer showing')
    plt.ylabel('Player sum')
    plt.figure(3)
    surface2 = ax3.plot_surface(x, y, q_no_usable[12:], cmap='turbo')
    fig3.colorbar(surface2)
    plt.title('MC prediction (No Usable Ace, ' + str(num) + ' Episodes)')
    plt.xlabel('Dealer showing')
    plt.ylabel('Player sum')
    plt.show()


def q4():
    print('4')


def q6a():
    print('6a')


def q6b():
    print('6b')


def main():
    env.register_env()
    print('What question do you want to do?')
    print('("2": question2, "3": blackjack, "4": four rooms, "6": racetrack)')
    str0 = input('("3a": MC evaluation, "3b": Monte-Carlo ES, "6a": on-policy, "6b": off policy): ')
    if str0 == '2':
        q2c()
    elif str0 == '3a':
        q3a()
    elif str0 == '3b':
        q3b()
    elif str0 == '4':
        q4()
    elif str0 == '6a':
        q6a()
    elif str0 == '6b':
        q6b()
    else:
        main()


if __name__ == '__main__':
    main()
