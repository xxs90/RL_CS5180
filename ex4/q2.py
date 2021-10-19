import numpy as np
from matplotlib import pyplot as plt

ACTION_LEFT = 0
ACTION_RIGHT = 1


def behavior_policy():
    return np.random.binomial(1, 0.5)


def target_policy():
    return ACTION_LEFT


def play():
    action = []
    while True:
        a = behavior_policy()
        action.append(a)
        if a == ACTION_RIGHT:
            return 0, action
        if np.random.binomial(1, 0.9) == 0:
            return 1, action


def infinite_variance():
    runs = 10
    episodes = 100000
    for run in range(runs):
        rewards = []
        for episode in range(0, episodes):
            reward, action = play()
            if action[-1] == ACTION_RIGHT:
                rho = 0
            else:
                rho = 1.0 / pow(0.5, len(action))
            rewards.append(rho * reward)
        rewards = np.add.accumulate(rewards)
        estimations = np.asarray(rewards) / np.arange(1, episodes + 1)
        plt.plot(estimations)
    plt.ylim(-0.05, 3.2)
    plt.title('Example 5.5: Infinite Variance')
    plt.xlabel('Episodes (log scale)')
    plt.ylabel('Ordinary Importance Sampling')
    plt.xscale('log')

    plt.show()


if __name__ == '__main__':
    infinite_variance()
