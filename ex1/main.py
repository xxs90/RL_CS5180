"""
    CS 4180/5180 RL and SDM
    Exercise 1: Multi-armed Bandits
    Prof: Robert Platt
    Date: September 22nd, 2021
    Author: Guanang Su
"""

from env import BanditEnv
from tqdm import trange
import agent as ag
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def q4(k: int, num_samples: int):
    """Q4

    Structure:
        1. Create multi-armed bandit env
        2. Pull each arm `num_samples` times and record the rewards
        3. Plot the rewards (e.g. violinplot, stripplot)

    Args:
        k (int): Number of arms in bandit environment
        num_samples (int): number of samples to take for each arm
    """

    env = BanditEnv(k=k)
    env.reset()

    # TODO
    reward_list = []

    for i in range(k):
        rewards = []
        for n in range(num_samples):
            reward = env.step(i)
            rewards.append(reward)

        reward_list.append(rewards)

    #print(reward_list)
    plt.xlabel("Action")
    plt.ylabel("Reward distribution")
    plt.violinplot(dataset=reward_list)

    plt.show()


def q6(k: int, trials: int, steps: int):
    """Q6

    Implement epsilon greedy bandit agents with an initial estimate of 0

    Args:
        k (int): number of arms in bandit environment
        trials (int): number of trials
        steps (int): total number of steps for each trial
    """
    # TODO initialize env and agents here
    env = BanditEnv(k=k)
    agents = [ag.EpsilonGreedy(k=k, init=0, epsilon=0),
              ag.EpsilonGreedy(k=k, init=0, epsilon=0.01),
              ag.EpsilonGreedy(k=k, init=0, epsilon=0.1)]

    reward_list = [[0]*steps, [0]*steps, [0]*steps]
    #optimal_list = []
    #reward_exp = []

    # Loop over trials
    for t in trange(trials, desc="Trials"):
        # Reset environment and agents after every trial
        env.reset()
        reward_agent = []
        #optimal_agent = []

        for agent in agents:
            agent.reset()
            # TODO For each trial, perform specified number of steps for each type of agent
            reward_step = []
            #optimal_step = []

            for i in range(steps):
                action = agent.choose_action()
                reward = env.step(action=action)
                agent.update(action=action, reward=reward)
                reward_step.append(reward)

                #if action ==
            reward_agent.append(reward_step)
            #print(reward_agent)

        #print("reward_list: " + str(reward_list))
        #print("reward_agent: " + str(reward_agent))
        reward_list = np.add(reward_agent, reward_list)

    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.plot((reward_list[0] / trials), '-g', label='ε=0')
    plt.plot((reward_list[1] / trials), '-r', label='ε=0.01')
    plt.plot((reward_list[2] / trials), '-b', label='ε=0.1')
    plt.legend()
    plt.show()


def q7(k: int, trials: int, steps: int):
    """Q7

    Compare epsilon greedy bandit agents and UCB agents

    Args:
        k (int): number of arms in bandit environment
        trials (int): number of trials
        steps (int): total number of steps for each trial
    """
    # TODO initialize env and agents here
    env = None
    agents = []

    # Loop over trials
    for t in trange(trials, desc="Trials"):
        # Reset environment and agents after every trial
        env.reset()
        for agent in agents:
            agent.reset()

        # TODO For each trial, perform specified number of steps for each type of agent

    pass


def main():
    # TODO run code for all questions
    #q4(10, 2000)
    q6(10, 2000, 1000)


if __name__ == "__main__":
    main()
