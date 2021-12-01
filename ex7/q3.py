"""
    CS 4180/5180 RL and SDM
    Exercise 7: Function Approximation
    Prof: Robert Platt
    Date: November 30th, 2021
    Author: Guanang Su
"""

import env
import gym
import numpy as np
from tqdm import trange
import algorithms as ag
import matplotlib.pyplot as plt

gamma = 0.99
epsilon = 0.1
step_size = 0.1
trials = 100
num_episode = 100


def main():
    env.register_env()
    env0 = gym.make('FourRooms-v0')  # tabular equivalent

    envs = {'Tabular': env0,
            'Row Aggregation': env.FourRoomsWrapperRow(env0),
            'Room Aggregation': env.FourRoomsWrapperRow(env0),
            '3x3 Aggregation': env.FourRoomsWrapper3x3(env0),
            '2x2 Aggregation': env.FourRoomsWrapper2x2(env0),
            'Feature x,y,1': env.FourRoomsWrapper_d(env0),
            'Feature Room': env.FourRoomsWrapperFeatureRoom(env0),
            'Feature Room Number': env.FourRoomsWrapperFeatureRoomNum(env0),
            'Feature Goal Distance': env.FourRoomsWrapperFeatureGoal(env0),
            }

    feature = 'Row Aggregation'
    environment = envs[feature]

    data = np.zeros([trials, num_episode])
    for trial in trange(trials, desc="Trial"):
        _, plot_data = ag.semi_grad_sarsa(environment, num_episode, gamma, epsilon, step_size)
        data[trial] = plot_data[0:num_episode]

    average_data = np.average(data, axis=0)
    plt.plot(average_data, label="Semi_gradient_SARSA")
    x = np.linspace(0, num_episode, num_episode)
    error = np.std(data, axis=0) / trials ** 0.5 * 1.96
    plt.fill_between(x, average_data - error, average_data + error, alpha=0.2)

    plt.title(feature + ' (Num trails = ' + str(trials) + ' )')
    plt.xlabel('Episodes')
    plt.ylabel('Steps per episode')
    plt.legend()
    plt.savefig(feature + '.png')
    plt.clf()


if __name__ == "__main__":
    main()
