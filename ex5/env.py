"""
    CS 4180/5180 RL and SDM
    Exercise 5: Temporal_Difference Learning
    Prof: Robert Platt
    Date: October 23rd, 2021
    Author: Guanang Su
"""

from enum import IntEnum
from typing import Tuple, Optional, List
from gym import Env, spaces
from gym.utils import seeding
from gym.envs.registration import register
from collections import defaultdict
import numpy as np
import random


def register_env() -> None:
    """Register custom gym environment so that we can use `gym.make()`

    In your main file, call this function before using `gym.make()` to use the Four Rooms environment.
        register_env()
        env = gym.make('WindyGridWorld-v0')

    There are a couple of ways to create Gym environments of the different variants of Windy Grid World.
    1. Create separate classes for each env and register each env separately.
    2. Create one class that has flags for each variant and register each env separately.

        Example:
        (Original)     register(id="WindyGridWorld-v0", entry_point="env:WindyGridWorldEnv")
        (King's moves) register(id="WindyGridWorldKings-v0", entry_point="env:WindyGridWorldEnv", **kwargs)

        The kwargs will be passed to the entry_point class.

    3. Create one class that has flags for each variant and register env once. You can then call gym.make using kwargs.

        Example:
        (Original)     gym.make("WindyGridWorld-v0")
        (King's moves) gym.make("WindyGridWorld-v0", **kwargs)

        The kwargs will be passed to the __init__() function.

    Choose whichever method you like.
    """
    # TODO
    register(id="WindyGridWorld-v0", entry_point="env:WindyGridWorldEnv")#, isKing=False, isStop=False, isStochastic=False)
    # register(id="WindyGridWorldKings-v0", entry_point="env:WindyGridWorldEnv", **kwargs)


class Action(IntEnum):
    """Action"""

    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3


def actions_to_dxdy(action: Action) -> Tuple[int, int]:
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


class WindyGridWorldEnv(Env):
    def __init__(self): #, isKing=False, isStop=False, isStochastic=False):
        """Windy grid world gym environment
        This is the template for Q4a. You can use this class or modify it to create the variants for parts c and d.
        """

        # Grid dimensions (x, y)
        self.rows = 10
        self.cols = 7

        # Wind
        # TODO define self.wind as either a dict (keys would be states) or multidimensional array (states correspond to indices)
        self.wind = np.zeros(shape=(self.cols, self.rows))

        for i in range(len(self.wind)):
            self.wind[i][3:6] = 1
            self.wind[i][8:9] = 1
            self.wind[i][6:8] = 2

        self.action_space = spaces.Discrete(len(Action))
        #print(self.action_space)
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(self.rows), spaces.Discrete(self.cols))
        )

        # Set start_pos and goal_pos
        self.start_pos = (0, 3)
        self.goal_pos = (7, 3)
        self.agent_pos = None

    def seed(self, seed: Optional[int] = None) -> List[int]:
        """Fix seed of environment

        In order to make the environment completely reproducible, call this function and seed the action space as well.
            env = gym.make(...)
            env.seed(seed)
            env.action_space.seed(seed)

        This function does not need to be used for this assignment, it is given only for reference.
        """

        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.agent_pos = self.start_pos
        return self.agent_pos

    def step(self, action: Action) -> Tuple[Tuple[int, int], float, bool, dict]:
        """Take one step in the environment.

        Takes in an action and returns the (next state, reward, done, info).
        See https://github.com/openai/gym/blob/master/gym/core.py#L42-L58 found r more info.

        Args:
            action (Action): an action provided by the agent

        Returns:
            observation (object): agent's observation after taking one step in environment (this would be the next state s')
            reward (float) : reward for this transition
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning). Not used in this assignment.
        """

        # TODO
        #1. check if goal state is reached
        if self.agent_pos == self.goal_pos:
            done = True
            reward = 0.0
        else:
            done = False
            reward = -1.0

        #
        # # 2. calculate next position
        # observation = tuple(map(sum, zip(self.agent_pos, actions_to_dxdy(Action(action_taken)))))
        #
        # # 3. check if position is feasible
        # next_pos = self.agent_pos
        # if self.cols > observation[0] > -1:
        #     next_pos = observation
        # elif observation[0] > self.cols:
        #     pos_x, pos_y = observation
        #     next_pos = pos_x, self.cols - 1
        # elif observation[1] > self.rows:
        #     pos_x, pos_y = observation
        #     next_pos = self.cols - 1, pos_y
        #
        # print(next_pos)
        # self.agent_pos = next_pos

        # next_pos_no = self.agent_pos[0] + actions_to_dxdy(action)[0], self.agent_pos[1] + actions_to_dxdy(action)[1]
        # in_boundary = 0 <= next_pos_no[0] <= self.rows and 0 <= next_pos_no[1] <= self.cols
        # # print('next', next_pos_no)
        # if in_boundary:
        #     #print()
        #     next_pos_no = next_pos_no
        # else:
        #     next_pos_no = self.agent_pos
        #
        # next_row = next_pos_no[0]
        # next_with_wind = self.agent_pos[0] + self.wind[str(next_row)][0], self.agent_pos[1] + self.wind[str(next_row)][1]
        #
        # in_boundary_wind = 0 <= next_with_wind[0] <= self.rows and 0 <= next_with_wind[1] <= self.cols
        # if in_boundary_wind:
        #     self.agent_pos = next_with_wind
        # else:
        #     self.agent_pos = next_pos_no
        # #print(self.agent_pos)

        walls = []
        for i in range(7):
            w = [(-1, i), (10, i)]
            walls += w

        for i in range(10):
            w = [(i, -1), (i, 7)]
            walls += w

        next_pos = tuple(map(lambda i, j: i + j, actions_to_dxdy(action), self.agent_pos))

        if (next_pos in walls):
            next_pos = self.agent_pos
            # print(self.wind)

        # print(self.agent_pos[0], self.agent_pos[1])
        wind1 = (0, int(self.wind[self.agent_pos[1] - 1][self.agent_pos[0] - 1]))
        t_next_pos = tuple(map(lambda i, j: i + j, next_pos, wind1))

        if (t_next_pos[1] > 6):
            t_next_pos = (t_next_pos[0], 6)
            next_pos = t_next_pos
        else:
            next_pos = t_next_pos

        self.agent_pos = next_pos

        return self.agent_pos, reward, done, {}
