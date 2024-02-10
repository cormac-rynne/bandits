"""
The `agents.py` module is a central component of the multi-armed bandit simulation framework, encapsulating
a diverse array of agent strategies designed to navigate the exploration-exploitation trade-off inherent
in bandit problems. By offering a broad selection of agents, from basic to more sophisticated approaches,
this module enables an in-depth comparative analysis of strategies under a variety of bandit environments.

Classes:
    AgentBase (ABC): An abstract base class that defines the common interface and foundational behaviors
                     for all agent strategies. It ensures consistency in how agents interact with bandit
                     environments, facilitating ease of extension and experimentation with new strategies.

    EGreedyAgent: Implements the ε-greedy strategy, balancing exploration and exploitation by selecting
                  actions either randomly or by choosing the current best estimate with probability ε.

    WeightedAgent: A variant of the ε-greedy strategy that uses a weighted approach for updating estimated
                   rewards, potentially favoring recent observations over older ones.

    OptimisticAgent: An adaptation of the ε-greedy strategy that initializes action value estimates with
                     optimistic values to encourage initial exploration.

    UCBAgent: Utilizes the Upper Confidence Bound (UCB) algorithm to select actions based on a trade-off
              between the estimated reward and the uncertainty or variance in the estimate.

    GradientAgent: Employs a policy gradient method to select actions based on preferences that are
                   adjusted in proportion to received rewards, promoting actions that have resulted in
                   higher rewards.

    ThompsonSamplingAgent: Based on Bayesian principles, this strategy selects actions by sampling from
                           probability distributions that model the uncertainty in the estimated rewards
                           of actions.

    SlidingWindowUCBAgent: A variant of the UCBAgent that incorporates a sliding window mechanism to focus
                           on recent observations, particularly useful in non-stationary environments.

    GPKernelAgent: Uses Gaussian Process Regression with a Radial Basis Function (RBF) kernel to model
                   the reward function across actions, incorporating uncertainty in the selection process.

    EXP3Agent: Designed for adversarial bandit problems, the EXP3 (Exponential-weight algorithm for
               Exploration and Exploitation) algorithm selects actions based on weights that are
               exponentially adjusted in response to received rewards.

Purpose:
    This module serves as a comprehensive toolkit for exploring various agent strategies in the context
    of multi-armed bandit problems. From simple, heuristic-based methods to more complex, statistically
    grounded approaches, the agents provided here cover a broad spectrum of strategies for balancing
    exploration and exploitation.

Usage:
    The agents in this module are designed to be easily integrated into simulation frameworks alongside
    different bandit environments, enabling users to conduct experiments and analyze the performance
    and characteristics of each strategy under controlled conditions.

Example:
    # Example instantiation of an ε-greedy agent with ε value of 0.1
    epsilon_greedy_agent = EGreedyAgent(num_arms=10, epsilon=0.1)

    # These agent instances can then be utilized in conjunction with bandit environments
    # and simulation classes to run experiments and evaluate their effectiveness.
"""


import numpy as np
import math
from abc import ABC, abstractmethod
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import scipy.special


class Agent(ABC):
    """
    An abstract base class representing an agent for a multi-armed bandit problem.

    This class provides a foundational structure for implementing various strategies
    for action selection and value updating in a multi-armed bandit setting.

    Attributes:
        num_actions (int): The number of actions available to the agent.
        epsilon (float): The probability of choosing a random action in the ε-greedy strategy.
        alpha (float): The step size for updating the estimated action values in the weighted average method.
        update_method (str): The method used for updating action values ('average' or 'weighted_average').
        optimism (float): The initial value for optimism toward action values, influencing initial exploration.
        actions (np.ndarray): An array of action indices available to the agent.
        action_values (np.ndarray): An array holding the estimated values of each action.
        action_counts (np.ndarray): An array tracking the number of times each action has been selected.
    """
    def __init__(self, num_actions, epsilon=0.1, update_method='average', alpha=0.1, optimism=0):
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.alpha = alpha
        self.update_method = update_method
        self.optimism = optimism

        self.actions = np.arange(0, self.num_actions)
        self.action_values = np.ones(self.num_actions) * self.optimism
        self.action_counts = np.zeros(self.num_actions)
        self.timestep = 0

    def reset(self):
        """
        Resets the agent's internal state, re-initializing action values and action counts.

        Args:
            bandit (bool, optional): Placeholder, not used in the base class.
                                     May have different implications in subclass implementations.
        """
        self.actions = np.arange(0, self.num_actions)
        self.action_values = np.ones(self.num_actions) * self.optimism
        self.action_counts = np.zeros(self.num_actions)
        self.timestep = 0

    def choose_action(self):
        """
        Selects an action using an epsilon-greedy strategy.

        Returns:
            int: The index of the chosen action.
        """
        if np.random.random() < self.epsilon:
            return np.random.choice(self.actions)
        else:
            return np.argmax(self.action_values)

    def update(self, action, reward):
        """
        Updates the action-value estimates based on the selected action and received reward.
        This method must be implemented by concrete subclasses.

        Args:
            action (int): The index of the action that was taken.
            reward (float): The reward received from the environment.
        """
        self.action_counts[action] += 1

        if self.update_method == 'average':
            step = 1/self.action_counts[action]
        if self.update_method == 'weighted_average':
            step = self.alpha

        reward_increment = step * (reward - self.action_values[action])
        self.action_values[action] += reward_increment

    @property
    def name(self):
        return self.__repr__()

    @abstractmethod
    def __repr__(self):
        pass


class EGreedyAgent(Agent):
    """
    Represents an ε-greedy agent for the multi-armed bandit problem.

    This agent extends the basic Agent class, specializing it to use an ε-greedy strategy
    for action selection. In the ε-greedy strategy, with probability ε, a random action is chosen
    (exploration), and with probability 1-ε, the action with the highest estimated value is chosen
    (exploitation).

    Attributes:
        Inherits all attributes from the Agent class.

    Args:
        num_actions (int): The number of actions available to the agent.
        epsilon (float, optional): The probability of selecting a random action to encourage exploration.
                                   Defaults to 0.1.
        **kwargs: Additional keyword arguments not specifically handled by EGreedyAgent, which are
                  passed to the Agent base class.

    Note:
        The `update_method`, `alpha`, and `optimism` parameters are set to their default values
        for the ε-greedy agent upon initialization, which are 'average', 0, and 0, respectively.
        These can be adjusted in the Agent class if a different behavior is desired.
    """

    def __init__(self, num_actions, epsilon=0.1, **kwargs):
        """
        Initializes an EGreedyAgent instance with specified parameters.

        Args:
            num_actions (int): Number of actions available to the agent.
            epsilon (float, optional): Epsilon value for ε-greedy action selection. Defaults to 0.1.
            **kwargs: Arbitrary keyword arguments, allowing for additional parameters that might be
                      required by the base class or for future extensions.
        """
        super().__init__(num_actions=num_actions, epsilon=epsilon, update_method='average',
                         alpha=0, optimism=0)

    def __repr__(self):
        """
        Returns a string representation of the EGreedyAgent instance, including its main parameters.

        Returns:
            str: A string that represents the EGreedyAgent object, detailing its ε-greedy configuration.
        """
        return f'EGreedyAgent(epsilon={self.epsilon})'


class WeightedAgent(Agent):
    """
    An agent that uses a weighted average strategy for updating action value estimates.

    The `WeightedAgent` extends the basic `Agent` class by specifically utilizing a weighted average
    method to update the estimated values of actions based on received rewards. This approach gives
    more weight to recent rewards, which can be particularly useful in non-stationary environments
    where the probabilities of rewards may change over time.

    Attributes:
        Inherits all attributes from the `Agent` class.

    Args:
        num_actions (int): The number of actions available to the agent.
        epsilon (float, optional): The probability of selecting a random action to encourage exploration.
                                   Defaults to 0.1.
        alpha (float, optional): The step size parameter used in the weighted average update method.
                                 It determines the weight given to the most recent reward. Defaults to 0.1.
        **kwargs: Additional keyword arguments not specifically handled by `WeightedAgent`, which are
                  passed to the `Agent` base class.

    Note:
        The `update_method` parameter is set to 'weighted_average' to indicate the use of the weighted
        average strategy for updating action values. The `optimism` parameter is set to 0 by default,
        assuming no initial optimism towards the actions' values. These settings are designed to tailor
        the `Agent` class's behavior for scenarios best suited to the weighted average update method.
    """

    def __init__(self, num_actions, epsilon=0.1, alpha=0.1, **kwargs):
        """
        Initializes a `WeightedAgent` instance with specified parameters.

        Args:
            num_actions (int): Number of actions available to the agent.
            epsilon (float, optional): Epsilon value for ε-greedy action selection. Defaults to 0.1.
            alpha (float, optional): Step size for the weighted average update. Defaults to 0.1.
            **kwargs: Arbitrary keyword arguments for additional flexibility and future proofing.
        """
        super().__init__(num_actions=num_actions, epsilon=epsilon, update_method='weighted_average',
                         alpha=alpha, optimism=0)

    def __repr__(self):
        """
        Returns a concise string representation of the `WeightedAgent` instance, including its key configuration.

        Returns:
            str: A string that succinctly describes the `WeightedAgent` object, focusing on its core parameters.
        """
        return f'WeightedAgent(epsilon={self.epsilon}, alpha={self.alpha})'


class OptimisticAgent(Agent):
    """
    An agent that employs optimistic initialization of action values for the multi-armed bandit problem.

    The `OptimisticAgent` extends the `Agent` class by initializing the action values with optimistic
    values. This encourages exploration early in the learning process by making the agent more likely
    to try out all actions to discover their true values. Optimistic initialization is a simple yet
    effective technique for encouraging exploration, especially in environments where exploration
    is crucial in the early stages.

    Attributes:
        Inherits all attributes from the `Agent` class.

    Args:
        num_actions (int): The number of actions available to the agent.
        epsilon (float, optional): The probability of selecting a random action to encourage exploration.
                                   Defaults to 0.1.
        update_method (str, optional): The method used for updating action values ('average' or 'weighted_average').
                                       Defaults to 'average'.
        alpha (float, optional): The step size parameter used in the weighted average update method.
                                 It determines the weight given to the most recent reward. Defaults to 0.1.
        optimism (float, optional): The initial optimistic value assigned to all actions. This value is
                                    used to encourage initial exploration. Defaults to 5.

    Note:
        The `optimism` parameter significantly influences the agent's initial behavior by encouraging
        exploration of actions that have not yet been tried. This approach can lead to faster convergence
        on the optimal action in certain environments, especially those with sparse or delayed rewards.
    """

    def __init__(self, num_actions, epsilon=0.1, optimism=5):
        """
        Initializes an `OptimisticAgent` instance with specified parameters.

        Args:
            num_actions (int): Number of actions available to the agent.
            epsilon (float, optional): Epsilon value for ε-greedy action selection. Defaults to 0.1.
            update_method (str, optional): Method for updating action values. Defaults to 'average'.
            alpha (float, optional): Step size for the weighted average update. Defaults to 0.1.
            optimism (float, optional): Initial optimism level for all action values. Defaults to 5.
        """
        super().__init__(num_actions=num_actions, epsilon=epsilon, update_method='average',
                         alpha=0, optimism=optimism)

    def __repr__(self):
        """
        Returns a detailed string representation of the `OptimisticAgent` instance, highlighting its configuration.

        Returns:
            str: A string that provides a detailed description of the `OptimisticAgent` object,
                 including its key parameters such as epsilon, update method, alpha, and optimism.
        """
        return f'OptimisticAgent(epsilon={self.epsilon}, optimism={self.optimism})'


class UCBAgent(Agent):
    """
    An agent that implements the Upper Confidence Bound (UCB) strategy for the multi-armed bandit problem.

    The `UCBAgent` extends the `Agent` class, utilizing the UCB algorithm for action selection. This strategy
    selects actions based on their estimated values and the uncertainty around those estimates, effectively
    balancing the exploration of actions with uncertain values and the exploitation of actions known to yield
    high rewards. The UCB algorithm is particularly effective in environments where a balance between exploration
    and exploitation is crucial for optimizing performance over time.

    Attributes:
        Inherits all attributes from the `Agent` class.
        c (float): The exploration parameter that controls the degree of exploration. Higher values
                   indicate a greater emphasis on exploration.
        timestep (int): The current timestep of the agent, used in the UCB calculation to adjust the
                        confidence interval based on the number of actions taken.

    Args:
        num_actions (int): The number of actions available to the agent.
        c (float, optional): The exploration parameter for the UCB algorithm. Defaults to 2.
        **kwargs: Additional keyword arguments passed to the `Agent` base class.
    """

    def __init__(self, num_actions, c=2, **kwargs):
        """
        Initializes a `UCBAgent` instance with the specified parameters.

        Args:
            num_actions (int): Number of actions available to the agent.
            c (float, optional): Exploration parameter for the UCB algorithm. Defaults to 2.
            **kwargs: Arbitrary keyword arguments providing flexibility for future extensions and
                      passing additional parameters to the base class.
        """
        super().__init__(num_actions=num_actions, update_method='average', **kwargs)
        self.c = c  # Exploration parameter
        self.timestep = 0  # Initialize the timestep at which the agent starts

    def choose_action(self):
        """
        Selects an action using the UCB algorithm.

        This method calculates the Upper Confidence Bound for each action, considering both the
        estimated value of the action and the uncertainty of that estimate. It then selects the
        action with the highest UCB value.

        Returns:
            int: The index of the selected action.
        """
        if np.all(self.action_counts == 0):
            # Initially explore actions randomly if none have been selected yet
            return np.random.choice(self.actions)

        log_term = math.log(self.timestep + 1)
        exploration_param = self.c * np.sqrt(log_term / self.action_counts)
        ucb_values = self.action_values + exploration_param

        self.timestep += 1  # Increment the timestep after selecting an action
        return np.argmax(ucb_values)  # Select the action with the highest UCB value

    def __repr__(self):
        """
        Returns a string representation of the `UCBAgent` instance, highlighting its configuration.

        Returns:
            str: A string that describes the `UCBAgent` object, particularly its exploration parameter (c).
        """
        return f'UCBAgent(c={self.c})'


class GradientAgent(Agent):
    """
    Implements a gradient-based learning strategy for a multi-armed bandit problem.

    The GradientAgent class extends the Agent class, using a policy gradient method where
    the action preferences are updated based on the received rewards, and actions are selected
    according to a softmax distribution over these preferences. This approach can effectively
    balance exploration and exploitation by adjusting the preferences (and thus the action
    probabilities) in the direction that increases the expected reward.

    Attributes:
        preferences (np.ndarray): A vector of preferences for each action, which determine the
                                  action selection probabilities through a softmax function.
        baseline (bool): Indicates whether to use a baseline to reduce variance in the update step.
                         The baseline used is the average reward.
        average_reward (float): The running average of the received rewards, serving as the baseline
                                if enabled.
        alpha (float): The step size parameter for updating action preferences.
        beta (float): The step size parameter for updating the average reward baseline.

    Args:
        num_actions (int): The number of actions available to the agent.
        alpha (float, optional): The step size for updating preferences. Defaults to 0.1.
        beta (float, optional): The step size for updating the average reward baseline. Defaults to 0.2.
        baseline (bool, optional): Whether to use an average reward baseline. Defaults to True.
    """

    def __init__(self, num_actions, alpha=0.1, beta=0.2, baseline=True):
        """
        Initializes a GradientAgent instance with the specified parameters.
        """
        super().__init__(num_actions, epsilon=0, update_method='gradient', alpha=alpha, optimism=0)
        self.preferences = np.zeros(num_actions)  # Initialize preferences to zero
        self.baseline = baseline
        self.average_reward = 0  # Initial average reward for baseline
        self.beta = beta

    def choose_action(self):
        """
        Selects an action based on the current preferences, using a softmax distribution.

        Returns:
            int: The index of the selected action.
        """
        action_probs = scipy.special.softmax(self.preferences)
        action = np.random.choice(self.actions, p=action_probs)
        return action

    def update(self, action, reward):
        """
        Updates the action preferences and the average reward baseline based on the received reward.

        Args:
            action (int): The index of the action taken.
            reward (float): The reward received from taking the action.
        """
        action_probs = scipy.special.softmax(self.preferences)
        chosen_action_prob = action_probs[action]

        if self.baseline:
            # Update the average reward baseline
            self.average_reward += self.beta * (reward - self.average_reward)

        # Update preferences for all actions
        for a in range(self.num_actions):
            if a == action:
                self.preferences[a] += self.alpha * (reward - self.average_reward) * (1 - chosen_action_prob)
            else:
                self.preferences[a] -= self.alpha * (reward - self.average_reward) * action_probs[a]

    def reset(self):
        """
        Resets the agent's state, including preferences and average reward baseline.
        """
        super().reset()
        self.preferences = np.zeros(self.num_actions)  # Reset preferences to zero
        self.average_reward = 0  # Reset average reward for baseline

    def __repr__(self):
        """
        Returns a string representation of the GradientAgent instance.

        Returns:
            str: A string that includes the agent's parameters.
        """
        return f'GradientAgent(alpha={self.alpha}, beta={self.beta}, baseline={self.baseline})'


class ThompsonSamplingAgent(Agent):
    def __init__(self, num_actions, prior_a=1, prior_b=1):
        super().__init__(num_actions=num_actions)
        self.prior_a = prior_a  # Parameter of the Beta prior distribution
        self.prior_b = prior_b  # Parameter of the Beta prior distribution
        self.successes = np.full(num_actions, prior_a).astype(float)  # Initialize successes
        self.failures = np.full(num_actions, prior_b).astype(float)  # Initialize failures

    def choose_action(self):
        samples = np.random.beta(self.successes, self.failures)
        return np.argmax(samples)

    def update(self, action, reward):
        epsilon = 0.001
        self.successes[action] = np.maximum(self.successes[action] + reward, epsilon)
        self.failures[action] = np.maximum(self.failures[action] + (1 - reward), epsilon)

    def reset(self):
        self.successes = np.full(self.num_actions, self.prior_a).astype(float)  # Initialize successes
        self.failures = np.full(self.num_actions, self.prior_b).astype(float)  # Initialize failures

    def __repr__(self):
        return f'ThompsonSamplingAgent(prior_a={self.prior_a}, prior_b={self.prior_b})'


class SlidingWindowUCBAgent(Agent):
    def __init__(self, num_actions, window_size=50, c=1):
        super().__init__(num_actions=num_actions)
        self.window_size = window_size
        self.c = c
        self.action_counts = [np.array([]) for _ in range(self.num_actions)]  # Store timestamps of rewards
        self.action_values = [np.array([]) for _ in range(self.num_actions)]  # Store rewards
        self.history = []  # Store (action, reward) tuples

    def choose_action(self):
        ucb_values = np.zeros(self.num_actions)

        for a in range(self.num_actions):
            if len(self.action_values[a]) > 0:
                # Calculate the average reward for the action within the window
                average_reward = np.mean(self.action_values[a][-self.window_size:])
                # Calculate the confidence interval
                confidence = np.sqrt(self.c * np.log(min(self.timestep, self.window_size)) / len(self.action_values[a]))
                ucb_values[a] = average_reward + confidence
            else:
                # If an action has never been taken, it is considered infinitely interesting
                ucb_values[a] = np.inf

        # Select the action with the highest UCB value
        return np.argmax(ucb_values)

    def update(self, action, reward):
        """
                Updates the agent's knowledge about the action.

                Parameters:
                - action (int): The action taken.
                - reward (float): The reward received.
                """
        self.timestep += 1  # Increment time step

        # Update actions and rewards, ensuring the window does not exceed its size
        if len(self.action_counts[action]) >= self.window_size:
            self.action_counts[action] = self.action_counts[action][1:]
            self.action_values[action] = self.action_values[action][1:]

        self.action_counts[action] = np.append(self.action_counts[action], self.timestep)
        self.action_values[action] = np.append(self.action_values[action], reward)

    def reset(self):
        super().reset()
        self.action_counts = [np.array([]) for _ in range(self.num_actions)]  # Store timestamps of rewards
        self.action_values = [np.array([]) for _ in range(self.num_actions)]  # Store rewards
        self.history = []  # Store (action, reward) tuples

    def __repr__(self):
        return f'SlidingWindowUCBAgent(window_size={self.window_size}, c={self.c})'


class GPKernelAgent(Agent):
    """
    An agent that uses Gaussian Process Regression to model the reward function of actions in a multi-armed bandit problem.

    The GPKernelAgent extends a base Agent class, utilizing a Gaussian Process (GP) model with a Radial Basis Function (RBF)
    kernel to estimate the expected rewards of actions. The agent selects actions based on an ε-greedy policy, where it
    either explores randomly with probability ε or exploits by choosing the action with the highest predicted reward based
    on the GP model.

    Attributes:
        gp (GaussianProcessRegressor): The Gaussian Process Regressor model used to predict action rewards.
        X_train (np.ndarray): The feature matrix storing the actions that have been taken.
        y_train (np.ndarray): The target vector storing the observed rewards for the taken actions.
        window_size (int): The size of the sliding window for training data to limit the memory and focus on recent observations.

    Args:
        num_actions (int): The number of available actions.
        epsilon (float, optional): The exploration probability for the ε-greedy strategy. Defaults to 0.1.
        window_size (int, optional): The maximum number of recent observations to keep for training the GP model. Defaults to 50.
        **kwargs: Additional keyword arguments for the Agent base class.
    """

    def __init__(self, num_actions, epsilon=0.1, window_size=50, **kwargs):
        """
        Initializes a GPKernelAgent instance with specified parameters, including the GP model and training data.
        """
        super().__init__(num_actions, epsilon=epsilon, **kwargs)
        # Initialize Gaussian Process Regressor with RBF kernel and White kernel for noise
        kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-10, 1e5))
        self.gp = GaussianProcessRegressor(kernel=kernel + WhiteKernel(noise_level=1.0))
        self.X_train = np.empty((0, 1))  # Initialize feature matrix for actions
        self.y_train = np.empty((0, 1))  # Initialize target vector for rewards
        self.window_size = window_size  # Set the sliding window size for training data

    def choose_action(self):
        """
        Selects an action based on the ε-greedy policy, using the GP model to predict the reward of each action.

        Returns:
            int: The index of the selected action.
        """
        if np.random.random() < self.epsilon:
            # Exploration: randomly select an action
            return np.random.choice(self.actions)
        else:
            # Exploitation: predict rewards for each action and select the one with the highest predicted reward
            if len(self.X_train) > 0:  # Ensure there is data to make a prediction
                mu, sigma = self.gp.predict(self.actions[:, None], return_std=True)
                return np.argmax(mu)
            else:
                # If no data is available, choose randomly
                return np.random.choice(self.actions)

    def update(self, action, reward):
        """
        Updates the training data with the new action and reward, and retrains the GP model.

        Args:
            action (int): The action that was taken.
            reward (float): The reward received from taking the action.
        """
        # Update training data with the new action and reward
        self.X_train = np.vstack([self.X_train, [[action]]])
        self.y_train = np.vstack([self.y_train, [[reward]]])

        # Keep only the most recent observations within the window size
        if len(self.X_train) >= self.window_size:
            self.X_train = self.X_train[-self.window_size:]
            self.y_train = self.y_train[-self.window_size:]

        # Retrain the GP model with updated data
        self.gp.fit(self.X_train, self.y_train)

    def reset(self):
        """
        Resets the agent's state, including the GP model and training data, for a fresh start.
        """
        # Reinitialize the GP model and training data
        kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-10, 1e5))
        self.gp = GaussianProcessRegressor(kernel=kernel + WhiteKernel(noise_level=1.0))
        self.X_train = np.empty((0, 1))
        self.y_train = np.empty((0, 1))

    def __repr__(self):
        """
        Returns a string representation of the GPKernelAgent instance, highlighting its key configurations.

        Returns:
            str: A string that includes the agent's epsilon value and kernel configuration.
        """
        return 'GPKernelAgent(epsilon={}, kernel={})'.format(self.epsilon, self.gp.kernel)


class EXP3Agent(Agent):
    """
    An agent implementing the EXP3 algorithm for adversarial bandit problems.

    The EXP3 (Exponential-weight algorithm for Exploration and Exploitation) algorithm is designed for
    multi-armed bandit problems where the reward distribution can change over time or might be generated
    by an adversarial process. The algorithm maintains a set of weights for each action, which are updated
    based on the received rewards, and selects actions based on a probability distribution derived from these
    weights, incorporating exploration through a parameter gamma.

    Attributes:
        gamma (float): The exploration parameter that controls the trade-off between exploration and exploitation.
                       A higher gamma value encourages more exploration.
        weights (np.ndarray): A numpy array holding the weights for each action.
        probabilities (np.ndarray): A numpy array of the probabilities for selecting each action, derived from the weights.

    Args:
        num_actions (int): The number of actions available to the agent.
        gamma (float, optional): The exploration parameter. Defaults to 0.1.
    """

    def __init__(self, num_actions, gamma=0.1):
        """
        Initializes the EXP3Agent with the specified number of actions and exploration parameter.
        """
        super().__init__(num_actions)
        self.gamma = gamma
        self.weights = np.ones(self.num_actions)  # Initialize weights for each action
        self.probabilities = np.ones(self.num_actions) / self.num_actions  # Initialize with uniform distribution

    def choose_action(self):
        """
        Selects an action based on the current probability distribution.

        Returns:
            int: The index of the selected action.
        """
        self.probabilities = (1 - self.gamma) * (self.weights / np.sum(self.weights)) + (self.gamma / self.num_actions)
        action = np.random.choice(range(self.num_actions), p=self.probabilities)
        return action

    def update(self, action, reward):
        """
        Updates the weights and probabilities based on the received reward for the chosen action.

        Args:
            action (int): The index of the action that was taken.
            reward (float): The reward received from taking the action.
        """
        estimated_reward = reward / self.probabilities[action]

        # Update weights
        self.weights[action] *= np.exp((self.gamma / self.num_actions) * estimated_reward)

    def reset(self):
        """
        Resets the agent's weights and probabilities to their initial state.
        """
        self.weights = np.ones(self.num_actions)  # Initialize weights for each action
        self.probabilities = np.ones(self.num_actions) / self.num_actions  # Initialize with uniform distribution

    def __repr__(self):
        """
        Returns a string representation of the EXP3Agent.

        Returns:
            str: A string that includes the agent's exploration parameter (gamma).
        """
        return f'EXP3Agent(gamma={self.gamma})'
