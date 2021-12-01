"""
    CS 4180/5180 RL and SDM
    Exercise 7: Function Approximation
    Prof: Robert Platt
    Date: November 30th, 2021
    Author: Guanang Su
    Reference: Some structure of the code are discussed with Mingxi
"""
from enum import IntEnum
from typing import Tuple, Optional, List
from gym import Env, spaces, ObservationWrapper
from gym.utils import seeding
from gym.envs.registration import register
import random
import numpy as np


def register_env() -> None:
    """Register custom gym environment so that we can use `gym.make()`

    In your main file, call this function before using `gym.make()` to use the Four Rooms environment.
        register_env()
        env = gym.make('FourRooms-v0')
    """
    # TODO
    register(id="FourRooms-v0", entry_point="env:FourRoomsEnv", max_episode_steps=10000)


def min_max_scaling(value):
    scale = np.max(value) - np.min(value)
    return (value - np.min(value)) / scale


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


class FourRoomsEnv(Env):
    """Four Rooms gym environment."""

    def __init__(self, goal_pos=(10, 10)) -> None:
        self.rows = 11
        self.cols = 11

        # Coordinate system is (x, y) where x is the horizontal and y is the vertical direction
        self.walls = [
            (0, 5),
            (2, 5),
            (3, 5),
            (4, 5),
            (5, 0),
            (5, 2),
            (5, 3),
            (5, 4),
            (5, 5),
            (5, 6),
            (5, 7),
            (5, 9),
            (5, 10),
            (6, 4),
            (7, 4),
            (9, 4),
            (10, 4),
        ]

        self.start_pos = (0, 0)
        self.goal_pos = goal_pos
        self.agent_pos = None

        self.action_space = spaces.Discrete(len(Action))
        self.observation_space = spaces.Tuple(
            (spaces.Discrete(self.rows), spaces.Discrete(self.cols))
        )

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

    def reset(self) -> Tuple[int, int]:
        """Reset agent to the starting position.

        Returns:
            observation (Tuple[int,int]): returns the initial observation
        """
        self.agent_pos = self.start_pos

        return self.agent_pos

    def step(self, action: Action) -> Tuple[Tuple[int, int], float, bool, dict]:
        """Take one step in the environment.

        Takes in an action and returns the (next state, reward, done, info).
        See https://github.com/openai/gym/blob/master/gym/core.py#L42-L58 foand r more info.

        Args:
            action (Action): an action provided by the agent

        Returns:
            observation (object): agent's observation after taking one step in environment (this would be the next state s')
            reward (float) : reward for this transition
            done (bool): whether the episode has ended, in which case further step() calls will return undefined results
            info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning). Not used in this assignment.
        """

        # Check if goal was reached
        if self.agent_pos == self.goal_pos:
            done = True
            reward = 1.0
        else:
            done = False
            reward = 0.0

        # TODO modify action_taken so that 10% of the time, the action_taken is perpendicular to action (there are 2 perpendicular actions for each action).
        # You can reuse your code from ex0
        change_p = random.random()
        if change_p < 0.2:

            if change_p < 0.1:
                action_taken = (action + 1) % 4
            else:
                action_taken = (action - 1) % 4
        else:
            action_taken = action

        # TODO calculate the next position using actions_to_dxdy()
        # You can reuse your code from ex0
        random_number = random.randint(1, 10)
        if random_number == 10:
            action_taken = Action((action + 1) % 4)
        elif random_number == 9:
            action_taken = Action((action + 3) % 4)
        else:
            action_taken = action

        # TODO calculate the next position using actions_to_dxdy()
        # You can reuse your code from ex0
        next_pos = tuple(map(sum, zip(self.agent_pos, actions_to_dxdy(Action(action_taken)))))

        boundaries = [
            (-1, -1), (-1, 0), (-1, 1), (-1, 2), (-1, 3), (-1, 4),
            (-1, 5), (-1, 6), (-1, 7), (-1, 8), (-1, 9), (-1, 10),
            (11, 0), (11, 1), (11, 2), (11, 3), (11, 4), (11, 5),
            (11, 6), (11, 7), (11, 8), (11, 9), (11, 10), (11, 11),
            (0, -1), (1, -1), (2, -1), (3, -1), (4, -1), (5, -1),
            (6, -1), (7, -1), (8, -1), (9, -1), (10, -1), (11, -1),
            (-1, 11), (0, 11), (1, 11), (2, 11), (3, 11), (4, 11),
            (5, 11), (6, 11), (7, 11), (8, 11), (9, 11), (10, 11),
        ]
        no_way = self.walls + boundaries

        # TODO check if next position is feasible
        # If the next position is a wall or out of bounds, stay at current position
        # Set self.agent_pos
        if next_pos in no_way:
            next_pos = self.agent_pos

        self.agent_pos = next_pos

        return self.agent_pos, reward, done, {}

    # tabular aggregation
    def aggregation(self, state, action):
        state_one_hot = np.zeros([self.cols, self.rows])
        state_one_hot[state[0], state[1]] = 1
        state_one_hot = state_one_hot.flatten()
        action_one_hot = np.zeros(self.action_space.n)
        action_one_hot[action] = 1
        return np.append(state_one_hot, action_one_hot)


class FourRoomsWrapperRow(ObservationWrapper):
    """Wrapper for FourRoomsEnv to aggregate rows.

    This wrapper aggregates rows, i.e. the original observation of the position (x,y) becomes just y

    To use this wrapper, create the environment using gym.make() and wrap the environment with this class.

    Ex) env = gym.make('FourRooms-v0')
        env = FourRoomsWrapper1(env)

    To use different function approximation schemes, create more wrappers, following this code.
    """

    def __init__(self, env):
        super().__init__(env)
        # Since this wrapper changes the observation space from a tuple of scalars to a single scalar,
        # need to change the observation_space
        self.observation_space = spaces.Discrete(self.rows)

    def observation(self, observation):
        """Return the y-value of position"""
        return observation[1]

    def aggregation(self, state, action):
        state_one_hot = np.zeros(self.observation_space.n)
        state_one_hot[state] = 1
        action_one_hot = np.zeros(self.action_space.n)
        action_one_hot[action] = 1
        return np.append(state_one_hot, action_one_hot)


class FourRoomsWrapperRoom(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Discrete(4)

    def observation(self, observation):
        x = observation[0]
        y = observation[1]
        if 0 <= x < 5 and 0 <= y <= 5:
            obs = 0
        elif 0 <= x <= 5 and y > 5:
            obs = 1
        elif x > 5 and y >= 4:
            obs = 2
        elif x >= 5 and 0 <= y < 4:
            obs = 3
        return obs

    def aggregation(self, state, action):
        state_one_hot = np.zeros(self.observation_space.n)
        state_one_hot[state] = 1
        action_one_hot = np.zeros(self.action_space.n)
        action_one_hot[action] = 1
        return np.append(state_one_hot, action_one_hot)


class FourRoomsWrapper3x3(ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Tuple((spaces.Discrete(4), spaces.Discrete(4)))

    def observation(self, observation):
        x = observation[0]
        y = observation[1]

        arr = np.array([0, 1, 2, 3])
        obs = (int(np.where(arr == x // 3)[0]), int(np.where(arr == y // 3)[0]))

        return obs

    def aggregation(self, state, action):
        state_one_hot = np.zeros([4, 4])
        state_one_hot[state[0], state[1]] = 1
        state_one_hot = state_one_hot.flatten()
        action_one_hot = np.zeros(self.action_space.n)
        action_one_hot[action] = 1
        return np.append(state_one_hot, action_one_hot)


class FourRoomsWrapper2x2(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Tuple((spaces.Discrete(6), spaces.Discrete(6)))

    def observation(self, observation):
        """Return the aggregation-value of position"""
        x = observation[0]
        y = observation[1]

        arr = np.array([0, 1, 2, 3, 4, 5])
        obs = (int(np.where(arr == x // 2)[0]), int(np.where(arr == y // 2)[0]))

        return obs

    def aggregation(self, state, action):
        state_one_hot = np.zeros([6, 6])
        state_one_hot[state[0], state[1]] = 1
        state_one_hot = state_one_hot.flatten()
        action_one_hot = np.zeros(self.action_space.n)
        action_one_hot[action] = 1
        return np.append(state_one_hot, action_one_hot)


class FourRoomsWrapper_d(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Tuple((spaces.Discrete(11), spaces.Discrete(11)))

    def observation(self, observation):
        x = observation[0]
        y = observation[1]

        return x, y

    def aggregation(self, state, action):
        x = state[0]
        y = state[1]
        action_one_hot = np.zeros(self.action_space.n)
        action_one_hot[action] = 1
        return np.append(min_max_scaling(np.array([1, x, y])), action_one_hot)


class FourRoomsWrapperFeatureRoom(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Tuple((spaces.Discrete(11), spaces.Discrete(11)))

    def observation(self, observation):
        x = observation[0]
        y = observation[1]

        return x, y

    def aggregation(self, state, action):
        x = state[0]
        y = state[1]
        if 0 <= x < 5 and 0 <= y <= 5:
            obs = 0
        elif 0 <= x <= 5 and y > 5:
            obs = 1
        elif x > 5 and y >= 4:
            obs = 2
        elif x >= 5 and 0 <= y < 4:
            obs = 3
        else:
            obs = 3
        state_one_hot = np.zeros(23)
        state_one_hot[state[0]] = 1
        state_one_hot[state[1] + 11] = 1
        state_one_hot[-1] = 1
        obs_one_hot = np.zeros(4)
        obs_one_hot[obs] = 1
        state_room_one_hot = np.append(state_one_hot, obs_one_hot)
        action_one_hot = np.zeros(self.action_space.n)
        action_one_hot[action] = 1
        return np.append(state_room_one_hot, action_one_hot)


class FourRoomsWrapperFeatureRoomNum(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Tuple((spaces.Discrete(11), spaces.Discrete(11)))

    def observation(self, observation):
        x = observation[0]
        y = observation[1]

        return x, y

    def aggregation(self, state, action):
        x = state[0]
        y = state[1]
        if 0 <= x < 5 and 0 <= y <= 5:
            obs = 0
        elif 0 <= x <= 5 and y > 5:
            obs = 1
        elif x > 5 and y >= 4:
            obs = 2
        elif x >= 5 and 0 <= y < 4:
            obs = 3
        else:
            obs = 3
            print(state)
        state_one_hot = np.zeros(23)
        state_one_hot[state[0]] = 1
        state_one_hot[state[1] + 11] = 1
        state_one_hot[-1] = 1
        obs_one_hot = np.zeros(4)
        obs_one_hot[obs] = 1
        action_one_hot = np.zeros(self.action_space.n)
        action_one_hot[action] = 1
        return np.append(obs, action_one_hot)


class FourRoomsWrapperFeatureGoal(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = spaces.Tuple((spaces.Discrete(11), spaces.Discrete(11)))

    def observation(self, observation):
        x = observation[0]
        y = observation[1]

        return x, y

    def aggregation(self, state, action):
        x = state[0]
        y = state[1]
        if 0 <= x < 5 and 0 <= y <= 5:
            obs = 0
        elif 0 <= x <= 5 and y > 5:
            obs = 1
        elif x > 5 and y >= 4:
            obs = 2
        elif x >= 5 and 0 <= y < 4:
            obs = 3
        else:
            obs = 3
        state_one_hot = np.zeros(23)
        state_one_hot[state[0]] = 1
        state_one_hot[state[1] + 11] = 1
        state_one_hot[-1] = 1
        manhattan_distance_to_goal = abs(self.goal_pos[0] - x) + abs(self.goal_pos[1] - y)
        action_one_hot = np.zeros(self.action_space.n)
        action_one_hot[action] = 1

        return np.append(
            min_max_scaling(np.array([1, x, y, x * y, x ** 2, y ** 2, x * y ** 2, x ** 2 * y, x ** 2 * y ** 2,
                                      manhattan_distance_to_goal, obs])), action_one_hot)

