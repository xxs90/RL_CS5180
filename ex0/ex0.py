import numpy as np
import matplotlib.pyplot as plt
import random
import math
from typing import Tuple, Callable
from enum import IntEnum


class Action(IntEnum):
    """Action"""

    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


def actions_to_dxdy(action: Action):
    """
    Helper function to map action to changes in x and y coordinates

    Args:
        action (Action): taken action

    Returns:
        dxdy (Tuple[int, int]): Change in x and y coordinates
    """
    mapping = {
        Action.LEFT: (-1, 0),
        Action.DOWN: (0, -1),
        Action.RIGHT: (1, 0),
        Action.UP: (0, 1),
    }
    return mapping[action]


def reset():
    """Return agent to start state"""
    return (0, 0)


# Q1
def simulate(state: Tuple[int, int], action: Action):
    """Simulate function for Four Rooms environment

    Implements the transition function p(next_state, reward | state, action).
    The general structure of this function is:
        1. If goal was reached, reset agent to start state
        2. Calculate the action taken from selected action (stochastic transition)
        3. Calculate the next state from the action taken (accounting for boundaries/walls)
        4. Calculate the reward

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))
        action (Action): selected action from current agent position (must be of type Action defined above)

    Returns:
        next_state (Tuple[int, int]): next agent position
        reward (float): reward for taking action in state
    """
    # Walls are listed for you
    # Coordinate system is (x, y) where x is the horizontal and y is the vertical direction
    walls = [
        (0, 5), (2, 5), (3, 5), (4, 5),
        (5, 0), (5, 2), (5, 3), (5, 4),
        (5, 5), (5, 6), (5, 7), (5, 9), (5, 10),
        (6, 4), (7, 4), (9, 4), (10, 4),
    ]

    # TODO check if goal was reached
    goal_state = (10, 10)
    if state == goal_state:
        state = reset()
        return state, 0

    # TODO modify action_taken so that 10% of the time, the action_taken is perpendicular to action (there are 2 perpendicular actions for each action)
    random_number = random.randint(1, 10)
    #print("There will be noise that affect your step, noise num is: " + str(random_number))
    if random_number == 10:
        action_taken = Action((action + 1) % 4)
    elif random_number == 9:
        action_taken = Action((action + 3) % 4)
    else:
        action_taken = action

    # TODO calculate the next state and reward given state and action_taken
    # You can use actions_to_dxdy() to calculate the next state
    # Check that the next state is within boundaries and is not a wall
    # One possible way to work with boundaries is to add a boundary wall around environment and
    # simply check whether the next state is a wall
    reward = 0

    boundaries = [
        (-1, -1), (-1, 0), (-1, 1), (-1, 2), (-1,  3), (-1,  4),
        (-1,  5), (-1, 6), (-1, 7), (-1, 8), (-1,  9), (-1, 10),
        (11,  0), (11, 1), (11, 2), (11, 3), (11,  4), (11,  5),
        (11,  6), (11, 7), (11, 8), (11, 9), (11, 10), (11, 11),
        (0,  -1), (1, -1), (2, -1), (3, -1), (4,  -1), (5,  -1),
        (6,  -1), (7, -1), (8, -1), (9, -1), (10, -1), (11, -1),
        (-1, 11), (0, 11), (1, 11), (2, 11), (3,  11), (4,  11),
        (5,  11), (6, 11), (7, 11), (8, 11), (9,  11), (10, 11),
    ]
    no_way = walls + boundaries

    next_state = tuple(map(sum, zip(state, actions_to_dxdy(Action(action_taken)))))
    if next_state in no_way:
        next_state = state

    if next_state == goal_state:
        reward = 1
        #print("You did it! You arrive at (10, 10) and your reward is 1.")
    #else:
        #print("You arrive at " + str(next_state) + " with " + str(action_taken) + ". The reward is " + str(reward) + ".")

    return next_state, reward


# Q2
def manual_policy(state: Tuple[int, int]):
    """A manual policy that queries user for action and returns that action

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))

    Returns:
        action (Action)
    """
    # TODO

    print("\nThis is a Four Rooms Environment, you are currently at " + str(state) + ".")
    print("What is the action you want to do?")
    action_input = input("(l: \"left\", d: \"down\", r: \"right\", u: \"up\"): ")

    if action_input is not None:
        if action_input == "l":
            action = Action(0)
        elif action_input == "d":
            action = Action(1)
        elif action_input == "r":
            action = Action(2)
        elif action_input == "u":
            action = Action(3)
        else:
            print("Invalid action input, please try again.")

    return action


# Q2
def agent(
    steps: int = 1000,
    trials: int = 1,
    policy=Callable[[Tuple[int, int]], Action],
):
    """
    An agent that provides actions to the environment (actions are determined by policy), and receives
    next_state and reward from the environment

    The general structure of this function is:
        1. Loop over the number of trials
        2. Loop over total number of steps
        3. While t < steps
            - Get action from policy
            - Take a step in the environment using simulate()
            - Keep track of the reward
        4. Compute cumulative reward of trial

    Args:
        steps (int): steps
        trials (int): trials
        policy: a function that represents the current policy. Agent follows policy for interacting with environment.
            (e.g. policy=manual_policy, policy=random_policy)

    """
    # TODO you can use the following structure and add to it as needed
    rewards = []

    for t in range(trials):
        state = reset()
        i = 0
        reward_list = []
        cumulative_reward = 0

        while i < steps:
            # TODO select action to take
            action = policy(state)

            # TODO take step in environment using simulate()
            state, reward = simulate(state, action)

            # TODO record the reward
            cumulative_reward += reward
            reward_list.append(cumulative_reward)
            i += 1

        rewards.append(reward_list)
    return rewards

# Q3
def random_policy(state: Tuple[int, int]):
    """A random policy that returns an action uniformly at random

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))

    Returns:
        action (Action)
    """
    # TODO
    random_number = random.uniform(0, 4)
    action = Action(math.trunc(random_number))

    #print(random_number)


    return action


# Q4
def worse_policy(state: Tuple[int, int]):
    """A policy that is worse than the random_policy

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))

    Returns:
        action (Action)
    """
    # TODO
    pass


# Q4
def better_policy(state: Tuple[int, int]):
    """A policy that is better than the random_policy

    Args:
        state (Tuple[int, int]): current agent position (e.g. (1, 3))

    Returns:
        action (Action)
    """
    # TODO
    pass


def main():
    # TODO run code for Q2~Q4 and plot results
    # You may be able to reuse the agent() function for each question
    print("This is a Four Room Problem, the initial state is (0, 0) and the final goal is (10, 10)")
    policy_select = input("Do you want to choose manual policy? (Y for yes, N for no)")

    # manual policy
    if policy_select == "Y":
        agent(100, 1, manual_policy)

    else:
        steps = 10000
        trails = 10

        # random policy
        #plt.title("Random Policy")
        plt.xlabel("Steps")
        plt.ylabel("Cumulative reward")
        rewards = agent(steps, trails, random_policy)
        total_list = [0] * steps
        step = np.arange(steps)
        #print(rewards)

        for i in range(trails):
            rewards_list = rewards[i]
            total_list = np.add(total_list, rewards_list)
            plt.plot(step, rewards_list, ':')

        print(total_list)
        mean_list = (np.array(total_list)) / trails
        plt.plot(step, mean_list, 'k', )

        plt.grid(color="gray", linestyle="--", linewidth=0.3)
        plt.figure(figsize=(20, 16))
        plt.show()


    # three policy comparison


if __name__ == "__main__":
    main()
