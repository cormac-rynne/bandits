"""
The `bandits.py` module forms a core component of a multi-armed bandit simulation framework, offering
a structured approach to defining and interacting with various types of bandit problems. By abstracting
the common attributes and behaviors of bandit problems into a hierarchy of classes, this module facilitates
the exploration of different bandit environments, ranging from stationary to dynamically changing (non-stationary)
reward distributions.

Classes:
    BanditBase (ABC): Serves as an abstract base class for defining the interface and shared behavior of all
                      bandit problems. It specifies the fundamental operations such as pulling an arm and
                      resetting the bandit's state, while leaving the implementation of specific behaviors
                      to its subclasses.

    StationaryBandit: Implements a bandit problem where the reward probabilities of each arm are fixed
                      and do not change over time. This class is ideal for studying algorithms in a
                      stable environment where the optimal strategy remains constant.

    NonStationaryBandit: Extends `BanditBase` to model environments where the reward probabilities
                         of each arm can change over time, simulating real-world scenarios where
                         conditions evolve. This class challenges algorithms to adapt their strategies
                         dynamically to continue maximizing rewards.

Purpose:
    This module is designed to support the empirical analysis of multi-armed bandit algorithms, providing
    the necessary infrastructure to simulate interactions with bandit environments under controlled conditions.
    It enables researchers and practitioners to investigate the effectiveness of different exploration and
    exploitation strategies across a spectrum of bandit problem complexities.

Usage:
    The classes within this module are intended to be instantiated as part of an experimental setup involving
    multi-armed bandit simulations. Users can define their own agents (strategies) and utilize instances of
    `StationaryBandit` or `NonStationaryBandit` to assess their performance across numerous trials, thereby
    gaining insights into the behavior and efficacy of the algorithms under study.

Example:
    # Example instantiation of a stationary bandit with 10 arms
    stationary_bandit = StationaryBandit(num_arms=10)

    # Example instantiation of a non-stationary bandit with 5 arms, a change probability of 0.01, and a change scale of 0.2
    non_stationary_bandit = NonStationaryBandit(num_arms=5, change_prob=0.01, change_scale=0.2)

    # These instances can then be used in conjunction with agent classes and simulation frameworks to conduct experiments.
"""


from abc import ABC, abstractmethod
import numpy as np


class BanditBase(ABC):
    """
    An abstract base class representing a generic multi-armed bandit problem.

    This class provides the basic functionality for simulating a bandit with a specified number of arms,
    where each arm has a reward distribution defined by a normal distribution with unknown means.

    Attributes:
        num_arms (int): The number of arms in the bandit.
        means (np.ndarray): An array of the true means of the reward distributions for each arm.
        best_arm (int): The index of the arm with the highest mean reward.
        best_value (float): The mean reward of the best arm.

    Args:
        num_arms (int): The number of arms for the bandit.
    """

    def __init__(self, num_arms):
        """
        Initializes the BanditBase with a specified number of arms.
        """
        self.num_arms = num_arms
        self.means = np.random.normal(0, 1, self.num_arms)
        self.best_arm = np.argmax(self.means)
        self.best_value = np.max(self.means)

    def pull(self, arm_index):
        """
        Simulates pulling an arm of the bandit.

        Args:
            arm_index (int): The index of the arm to pull.

        Returns:
            float: The reward obtained from pulling the arm, sampled from a normal distribution
                   centered around the true mean of the arm with a standard deviation of 1.
        """
        return np.round(np.random.normal(self.means[arm_index], 1), 3)

    def reset(self):
        """
        Resets the bandit by randomly reinitializing the means of all arms.
        """
        self.means = np.random.normal(0, 1, self.num_arms)
        self.best_arm = np.argmax(self.means)
        self.best_value = np.max(self.means)

    @abstractmethod
    def __repr__(self):
        """
        Abstract method for returning a string representation of the bandit instance.
        Must be implemented by subclasses.
        """
        pass

    @property
    def name(self):
        return self.__repr__()


class StationaryBandit(BanditBase):
    """
    A concrete implementation of the BanditBase class for a stationary bandit problem.

    In a stationary bandit problem, the reward distributions of the arms do not change over time.
    This class inherits from BanditBase and does not modify the base functionality, assuming
    that the reward distributions are stationary.

    Args:
        num_arms (int): The number of arms for the bandit.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self, num_arms, **kwargs):
        """
        Initializes the StationaryBandit with a specified number of arms.
        """
        super().__init__(num_arms=num_arms)

    def __repr__(self):
        """
        Returns a string representation of the StationaryBandit instance.

        Returns:
            str: A string that includes the number of arms in the bandit.
        """
        return f'StationaryBandit(num_arms={self.num_arms})'


class NonStationaryBandit(BanditBase):
    """
    A subclass of BanditBase for simulating non-stationary multi-armed bandit problems.

    In this class, the means of the reward distributions for the arms have a probability of changing
    after each arm pull. This change simulates a non-stationary environment where the optimal action
    can vary over time.

    Attributes:
        change_prob (float): The probability that the mean reward of any given arm will change after a pull.
        change_scale (float): The maximum magnitude of change in the mean reward of an arm.

    Args:
        num_arms (int): The number of arms in the bandit problem.
        change_prob (float, optional): The probability of change in the mean reward for each arm. Defaults to 0.01.
        change_scale (float, optional): The scale of change for each arm's reward distribution. Defaults to 0.3.
    """

    def __init__(self, num_arms, change_prob=0.01, change_scale=0.3):
        """
        Initializes the NonStationaryBandit with the specified number of arms and change parameters.
        """
        super().__init__(num_arms=num_arms)
        self.change_prob = change_prob
        self.change_scale = change_scale

    def pull(self, arm_index):
        """
        Pulls an arm of the bandit, returns the reward, and potentially updates the mean rewards.

        Args:
            arm_index (int): The index of the arm to pull.

        Returns:
            float: The reward obtained from pulling the arm.
        """
        # Call the pull method from BanditBase to get the reward
        reward = super().pull(arm_index)
        # Update the means of the reward distributions
        self.update()
        return reward

    def update(self):
        """
        Randomly updates the means of the reward distributions for each arm based on the change probability and scale.
        """
        # Determine which arms will change
        arms_to_change = np.random.choice(
            [0, 1],
            size=self.num_arms,
            p=[1 - self.change_prob, self.change_prob]
        )
        # Calculate the change size for each arm
        change_size = np.random.uniform(
            -self.change_scale,
            self.change_scale,
            self.num_arms
        )
        # Apply changes to the means
        self.means += change_size * arms_to_change
        # Update the best arm and its value
        self.best_arm = np.argmax(self.means)
        self.best_value = np.max(self.means)

    def __repr__(self):
        """
        Returns a string representation of the NonStationaryBandit instance.

        Returns:
            str: A string that includes the number of arms and the change parameters of the bandit.
        """
        return f'NonStationaryBandit(num_arms={self.num_arms}, change_prob={self.change_prob}, change_scale={self.change_scale})'

