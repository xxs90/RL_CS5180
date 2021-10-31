"""
    CS 4180/5180 RL and SDM
    Exercise 5: Temporal_Difference Learning
    Prof: Robert Platt
    Date: October 28th, 2021
    Author: Guanang Su
"""

from enum import IntEnum
from typing import Tuple, Optional, List
from gym import Env, spaces
from gym.utils import seeding
from gym.envs.registration import register
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
    register(id="WindyGridWorldSto-v0", entry_point="envStochastic:WindyGridWorldEnv")


class Action(IntEnum):
    """Action"""

    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3
    LEFT_UP = 4
    RIGHT_UP = 5
    LEFT_DOWN = 6
    RIGHT_DOWN = 7


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
        Action.LEFT_UP: (-1, 1),
        Action.RIGHT_UP: (1, 1),
        Action.LEFT_DOWN: (-1, -1),
        Action.RIGHT_DOWN: (1, -1),
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

        random_number = random.randint(1, 100)
        if random_number <= 33:
            action_taken = Action((action + 1) % 4)
        elif random_number <= 66:
            action_taken = Action((action + 3) % 4)
        else:
            action_taken = action

        next_pos = tuple(map(sum, zip(self.agent_pos, actions_to_dxdy(action_taken))))

        if not (9 >= next_pos[0] >= 0 and 6 >= next_pos[1] >= 0):
            next_pos = self.agent_pos
            # print(self.wind)

        # print(self.agent_pos[0], self.agent_pos[1])
        a = int(self.wind[self.agent_pos[1] - 1][self.agent_pos[0] - 1])
        print('pos', self.agent_pos[0], self.agent_pos[1])
        #print(self.wind)
        print('wind', a)
        if self.wind[self.agent_pos[1] - 1][self.agent_pos[0] - 1] != 0:
            wind_random = [a-1, a, a+1]
        else:
            wind_random = [a]
        print(wind_random)
        print(random.choice(wind_random))

        wind1 = (0, int(random.choice(wind_random)))

        next_row, next_col = tuple(map(sum, zip(next_pos, wind1)))

        if next_col > 6:
            next_col = 6
        elif next_col < 0:
            next_col = 0

        if next_row > 9:
            next_row = 9
        elif next_row < 0:
            next_row = 0

        next_pos = next_row, next_col

        self.agent_pos = next_pos
        #print(self.agent_pos)

        return self.agent_pos, reward, done, {}
