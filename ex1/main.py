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
    plt.xlabel('Action')
    plt.ylabel('Reward distribution')
    plt.violinplot(dataset=reward_list, showmeans=True, showmedians=True)
    plt.axhline(y=0, linestyle='dashed')

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

    # Loop over trials
    reward_average = []
    action_optimal = []
    for t in trange(trials, desc="Trials"):
        # Reset environment and agents after every trial
        env.reset()
        optimal = np.argmax(env.means)
        reward_agent = []
        action_agent = []

        for agent in agents:
            agent.reset()
            # TODO For each trial, perform specified number of steps for each type of agent
            reward_step = []
            action_step = []

            for i in range(steps):
                action = agent.choose_action()
                reward = env.step(action=action)
                agent.update(action=action, reward=reward)
                reward_step.append(reward)

                if action == optimal:
                    action_step.append(1)
                else:
                    action_step.append(0)

            reward_agent.append(reward_step)
            action_agent.append(action_step)

        reward_average.append(reward_agent)
        action_optimal.append(action_agent)

    average_reward = np.average(reward_average, axis=0)
    average_action = np.average(action_optimal, axis=0)
    standard_error = np.std(reward_average, axis=0)
    error = 1.96 * standard_error / np.sqrt(trials)
    upper_bound = [np.amax(average_reward)] * steps

    plt.figure(0)
    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.plot(upper_bound, '--', linewidth=1.0)
    x = np.arange(steps)
    plt.plot(average_reward[0], 'C2', linewidth=0.6, label='ε=0 (greedy)')
    plt.fill_between(x, (average_reward[0]-error[0]), (average_reward[0]+error[0]), color='C2', alpha=0.3)
    plt.plot(average_reward[1], 'C1', linewidth=0.6, label='ε=0.01')
    plt.fill_between(x, (average_reward[1] - error[1]), (average_reward[1] + error[1]), color='C1', alpha=0.3)
    plt.plot(average_reward[2], 'C0', linewidth=0.6, label='ε=0.1')
    plt.fill_between(x, (average_reward[2] - error[2]), (average_reward[2] + error[2]), color='C0', alpha=0.3)
    plt.legend()

    plt.figure(1)
    plt.xlabel('Steps')
    plt.ylabel('Optimal Action %')
    plt.plot(average_action[0], 'C2', linewidth=0.6, label='ε=0 (greedy)')
    plt.plot(average_action[1], 'C1', linewidth=0.6, label='ε=0.01')
    plt.plot(average_action[2], 'C0', linewidth=0.6, label='ε=0.1')
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
    env = BanditEnv(k=k)
    agents = [ag.EpsilonGreedy(k=k, init=0, epsilon=0, step_size=0.1),
              ag.EpsilonGreedy(k=k, init=5, epsilon=0, step_size=0.1),
              ag.EpsilonGreedy(k=k, init=0, epsilon=0.1, step_size=0.1),
              ag.EpsilonGreedy(k=k, init=5, epsilon=0.1, step_size=0.1),
              ag.UCB(k=k, init=0, c=2, step_size=0.1)]

    # Loop over trials
    reward_average = []
    action_optimal = []
    for t in trange(trials, desc="Trials"):
        # Reset environment and agents after every trial
        env.reset()
        optimal = np.argmax(env.means)
        reward_agent = []
        action_agent = []
        for agent in agents:
            agent.reset()
            # TODO For each trial, perform specified number of steps for each type of agent
            reward_step = []
            action_step = []

            for i in range(steps):
                action = agent.choose_action()
                reward = env.step(action=action)
                agent.update(action=action, reward=reward)
                reward_step.append(reward)

                if action == optimal:
                    action_step.append(1)
                else:
                    action_step.append(0)

            reward_agent.append(reward_step)
            action_agent.append(action_step)

        reward_average.append(reward_agent)
        action_optimal.append(action_agent)

    average_reward = np.average(reward_average, axis=0)
    average_action = np.average(action_optimal, axis=0)
    standard_error = np.std(reward_average, axis=0)
    error = 1.96 * standard_error / np.sqrt(trials)
    upper_bound = [np.amax(average_reward)] * steps

    plt.figure(0)
    plt.xlabel('Steps')
    plt.ylabel('Average reward')
    plt.plot(upper_bound, 'k--', linewidth=1.0)
    x = np.arange(steps)
    plt.plot(average_reward[0], linewidth=0.6, label='ε-greedy(Q1=0, ε=0)')
    plt.fill_between(x, (average_reward[0] - error[0]), (average_reward[0] + error[0]), alpha=0.3)
    plt.plot(average_reward[1], linewidth=0.6, label='ε-greedy(Q1=5, ε=0)')
    plt.fill_between(x, (average_reward[1] - error[1]), (average_reward[1] + error[1]), alpha=0.3)
    plt.plot(average_reward[2], linewidth=0.6, label='ε-greedy(Q1=0, ε=0.1)')
    plt.fill_between(x, (average_reward[2] - error[2]), (average_reward[2] + error[2]), alpha=0.3)
    plt.plot(average_reward[3], linewidth=0.6, label='ε-greedy(Q1=5, ε=0.1)')
    plt.fill_between(x, (average_reward[3] - error[3]), (average_reward[3] + error[3]), alpha=0.3)
    plt.plot(average_reward[4], linewidth=0.6, label='UCB (c=2)')
    plt.fill_between(x, (average_reward[4] - error[4]), (average_reward[4] + error[4]), alpha=0.3)
    plt.legend()

    plt.figure(1)
    plt.xlabel('Steps')
    plt.ylabel('Optimal Action %')
    plt.plot(average_action[0], linewidth=0.6, label='ε-greedy(Q1=0, ε=0)')
    plt.plot(average_action[1], linewidth=0.6, label='ε-greedy(Q1=5, ε=0)')
    plt.plot(average_action[2], linewidth=0.6, label='ε-greedy(Q1=0, ε=0.1)')
    plt.plot(average_action[3], linewidth=0.6, label='ε-greedy(Q1=5, ε=0.1)')
    plt.plot(average_action[4], linewidth=0.6, label='UCB (c=2)')
    plt.legend()

    plt.show()


def main():
    # TODO run code for all questions
    q4(10, 2000)
    #q6(10, 2000, 1000)
    #q7(10, 2000, 1000)


if __name__ == "__main__":
    main()
