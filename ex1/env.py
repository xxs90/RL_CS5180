"""
    CS 4180/5180 RL and SDM
    Exercise 1: Multi-armed Bandits
    Prof: Robert Platt
    Date: September 22nd, 2021
    Author: Guanang Su
"""

import numpy as np
from typing import Tuple


class BanditEnv:
    """Multi-armed bandit environment"""

    def __init__(self, k: int) -> None:
        """__init__.

        Args:
            k (int): number of arms/bandits
        """
        self.k = k

    def reset(self) -> None:
        """Resets the mean payout/reward of each arm.
        This function should be called at least once after __init__()
        """
        # Initialize means of each arm distributed according to standard normal
        self.means = np.random.normal(size=self.k)
        """ random.normal(loc=0.0, scale=1.0, size=None)
            -parameter
            loc: float or array_like of floats
            Mean (“centre”) of the distribution.
            
            scale: float or array_like of floats
            Standard deviation (spread or “width”) of the distribution. Must be non-negative.
            
            size: int or tuple of ints, optional
        """

    def step(self, action: int) -> Tuple[float, int]:
        """Take one step in env (pull one arm) and observe reward

        Args:
            action (int): index of arm to pull
        """
        # TODO calculate reward of arm given by action
        """page 29 from book
        Then, when a learning method applied to that problem selected 
        action At at time step t, the actual reward, Rt, was selected 
        from a normal distribution with mean q*(At) and variance 1."""
        reward = np.random.normal(loc=self.means[action], scale=1.0, size=1)
        return reward[0]
