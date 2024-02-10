"""
The `simulation.py` module is part of a larger project aimed at exploring and analyzing the performance
of various strategies (agents) in solving multi-armed bandit problems. This module provides the foundational
classes necessary for setting up and running simulations, allowing for an empirical comparison of different
agents' abilities to maximize rewards in both stationary and non-stationary bandit environments.

Classes:
    Experiment: Facilitates the execution of a single experiment, tracking the interactions between
                an agent and a bandit problem over a series of steps. It captures key metrics such
                as average rewards, optimal action selections, and cumulative rewards, enabling a detailed
                analysis of an agent's performance.

    Simulations: Manages the execution of multiple experiments across different agents and potentially
                 varying bandit problem configurations. This class automates the process of running
                 numerous simulations, collecting results, and providing tools for visualizing comparative
                 performance metrics such as average reward, optimal action frequency, and various forms
                 of regret analysis.

This module is designed to be flexible, supporting experiments with a wide range of bandit problem types
and agent strategies. By providing a structured approach to experimentation and simulation, it serves as a
valuable tool for researchers and practitioners in the field of reinforcement learning, particularly those
focused on the exploration-exploitation trade-off inherent in multi-armed bandit problems.

Usage:
    The classes within this module can be used as part of a larger framework for conducting empirical
    research on multi-armed bandit algorithms. Users can define their own agents and bandit problems,
    then utilize the `Experiment` and `Simulations` classes to evaluate and compare their performance
    under controlled conditions.

Example:
    See the documentation for `Experiment` and `Simulations` classes for example usage.
"""

import numpy as np
import matplotlib.pyplot as plt


class Experiment:
    """
    A class to conduct experiments with multi-armed bandit problems.

    This class encapsulates the setup required to run simulations where an agent interacts with a bandit environment.
    It tracks the actions taken by the agent, the rewards received, and how often the agent selects the optimal action.
    This facilitates analysis of the agent's performance and learning behavior over time.

    Attributes:
        bandit (Bandit): An instance of a Bandit class, representing the multi-armed bandit problem.
        agent (Agent): An instance of an Agent class, representing the strategy or algorithm used to select actions.
        steps (int): The number of steps (iterations) the experiment will run.
        optimal_actions (np.ndarray): An array tracking whether the agent selected the optimal action at each step.
        optimal_values (np.ndarray): An array storing the optimal value (best possible reward) at each step.
        reward_hist (np.ndarray): An array recording the reward received by the agent at each step.

    Args:
        bandit (Bandit): The bandit problem instance the agent will interact with.
        agent (Agent): The agent that will be making decisions on which arm to pull.
        steps (int): The total number of steps the experiment will execute.
    """

    def __init__(self, bandit, agent, steps):
        """
        Initializes the Experiment with a bandit, an agent, and a specified number of steps.
        """
        self.bandit = bandit
        self.agent = agent
        self.steps = steps
        self.optimal_actions = np.empty(steps)
        self.optimal_values = np.empty(steps)
        self.reward_hist = np.empty(steps)

    def run(self):
        """
        Executes the experiment over the specified number of steps.

        At each step, the agent selects an action (arm to pull) based on its strategy,
        receives a reward from the bandit, and updates its knowledge based on the outcome.
        The experiment tracks the reward received, whether the chosen action was optimal,
        and the optimal value at each step.
        """
        for step in range(self.steps):
            action = self.agent.choose_action()
            reward = self.bandit.pull(action)
            self.agent.update(action=action, reward=reward)

            # Record whether the selected action was the optimal one
            self.optimal_actions[step] = int(action == self.bandit.best_arm)
            # Record the reward received
            self.reward_hist[step] = reward
            # Record the optimal value for reference
            self.optimal_values[step] = self.bandit.best_value

    @property
    def name(self):
        """
        A property to get the name of the agent being used in the experiment.

        Returns:
            str: The name of the agent.
        """
        return self.agent.name


class Simulations:
    """
    A class for conducting multiple simulation runs of bandit problems with various agents.

    This class orchestrates the setup, execution, and visualization of experiments comparing
    different strategies (agents) in interacting with a specified bandit environment over a series
    of runs and steps.

    Attributes:
        runs (int): The number of simulation runs for each agent.
        steps (int): The number of steps (actions taken) in each run.
        num_arms (int): The number of arms in the bandit problem.
        bandit (Bandit): The class of the bandit problem to be used in the simulations.
        bandit_args (dict): Arguments required to initialize instances of the bandit class.
        agents_lst (list): A list of dictionaries, each specifying an agent class and its arguments.
        experiments_lst (list): A list of Experiment instances, each corresponding to an agent.
        results_dct (dict): A dictionary storing the results of the simulations for each agent.
        results_template (NoneType): Placeholder for future use.

    Args:
        runs (int): Number of runs to simulate.
        steps (int): Number of steps per run.
        arms (int): Number of arms in the bandit problem.
        bandit (Bandit): The bandit class to be used in the experiments.
        bandit_args (dict): Initialization arguments for the bandit class.
        agents (list): List of dictionaries specifying agents and their initialization arguments.
    """

    def __init__(self, runs, steps, arms, bandit, bandit_args, agents):
        self.runs = runs
        self.steps = steps
        self.num_arms = arms
        self.bandit = bandit
        self.bandit_args = bandit_args
        self.agents_lst = agents
        self.experiments_lst = []
        self.results_dct = {}
        self.results_template = None
        self.bandit_name = None
        self.build_experiments()

    def build_experiments(self):
        """
        Constructs Experiment instances for each specified agent and bandit configuration.
        """
        for agent_dct in self.agents_lst:
            bandit_instance = self.bandit(num_arms=self.num_arms, **self.bandit_args)
            agent_instance = agent_dct['agent'](self.num_arms, **agent_dct['args'])

            exp = Experiment(agent=agent_instance, bandit=bandit_instance, steps=self.steps)
            self.experiments_lst.append(exp)
            self.results_dct[exp.name] = {
                'optimal_action_count': np.zeros((self.runs, self.steps)),
                'avg_rewards': np.zeros((self.runs, self.steps)),
                'optimal_value': np.zeros((self.runs, self.steps)),
            }
        self.bandit_name = bandit_instance.name

    def run(self):
        """
        Executes each experiment across the specified number of runs, collecting performance metrics.
        """
        for exp in self.experiments_lst:
            for r in range(self.runs):
                exp.run()
                self.results_dct[exp.name]['avg_rewards'][r] = exp.reward_hist
                self.results_dct[exp.name]['optimal_action_count'][r] = exp.optimal_actions
                self.results_dct[exp.name]['optimal_value'][r] = exp.optimal_values
                exp.agent.reset()
                exp.bandit.reset()

    def visualise(self, window=10):
        """
        Visualizes the results of the simulations, including average rewards, optimal action counts,
        cumulative rewards, reward variability, cumulative regret, and instantaneous regret.

        Args:
            window (int): The size of the moving average window for smoothing variability and instantaneous regret plots.
        """
        # Plot setup
        fig, axes = plt.subplots(6, 1, figsize=(10, 24))

        steps = np.arange(0, self.steps)

        linewidth = 1

        conv_kernel = np.ones(window)/window
        for name, dct in self.results_dct.items():
            axes[0].plot(steps, dct['avg_rewards'].mean(0), label=name, linewidth=linewidth)
            axes[1].plot(steps, dct['optimal_action_count'].mean(0), label=name, linewidth=linewidth)
            axes[2].plot(steps, np.cumsum(dct['avg_rewards'].mean(0), axis=0), label=name, linewidth=linewidth)

            reward_variability = np.std(dct['avg_rewards'], axis=0)
            reward_variability = np.convolve(reward_variability, conv_kernel, mode='valid')
            axes[3].plot(steps[:len(reward_variability)], reward_variability[:self.steps], label=name, linewidth=linewidth)

            optimal_reward_cum = np.cumsum(dct['optimal_value'].mean(0), axis=0)
            actual_rewards_cum = np.cumsum(dct['avg_rewards'].mean(0), axis=0)
            regret_cum = optimal_reward_cum - actual_rewards_cum
            axes[4].plot(regret_cum, label=name, linewidth=linewidth)

            regret_inst = np.convolve(np.diff(regret_cum), conv_kernel, mode='valid')
            axes[5].plot(steps[:len(regret_inst)], regret_inst[:self.steps], label=name, linewidth=linewidth)

        # Plot average rewards
        axes[0].set_title('Average reward')
        axes[0].set_xlabel('Steps')
        axes[0].set_ylabel('Average reward')

        # Plot percentage of optimal actions
        axes[1].set_title('Percentage of optimal action')
        axes[1].set_xlabel('Steps')
        axes[1].set_ylabel('% Optimal action')

        axes[2].set_title('Cumsum Reward')
        axes[2].set_xlabel('Steps')
        axes[2].set_ylabel('Reward')

        # Plot percentage of optimal actions
        axes[3].set_title(f'Reward Variability (window={window})')
        axes[3].set_xlabel('Steps')
        axes[3].set_ylabel('Stddev')

        axes[4].set_title('Cumulative Regret')
        axes[4].set_xlabel('Steps')
        axes[4].set_ylabel('Cumulative Regret')

        axes[5].set_title(f'Instantaneous Regret (window={window})')
        axes[5].set_xlabel('Steps')
        axes[5].set_ylabel('Instantaneous Regret')

        for ax in axes:
            ax.legend()
            ax.grid()

        fig.suptitle(f'{self.bandit_name}', fontsize=16)
        plt.tight_layout()
        fig.subplots_adjust(top=.96)
        plt.show()
