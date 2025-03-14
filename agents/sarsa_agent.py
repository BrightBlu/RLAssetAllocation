from typing import Dict, Any, List, Tuple
import numpy as np
from environments.market_environment import MarketEnvironment

class Agent:
    """SARSA agent for asset allocation.
    
    This agent implements SARSA (State-Action-Reward-State-Action) method for policy evaluation
    and improvement. It uses epsilon-greedy exploration and maintains state-action value
    estimates based on temporal difference learning.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the SARSA agent.
        
        Args:
            config: Configuration dictionary containing:
                - epsilon: Exploration rate for epsilon-greedy policy
                - gamma: Discount factor
                - learning_rate: Learning rate for TD update
                - n_actions: Number of discrete actions
                - n_wealth_bins: Number of bins for wealth discretization
        """
        agent_config = config.get('agent', {})
        env_config = config.get('environment', {})
        self.epsilon = agent_config.get('epsilon', 0.1)
        self.gamma = agent_config.get('gamma', 0.99)
        self.learning_rate = agent_config.get('learning_rate', 0.01)
        self.n_actions = agent_config.get('n_actions', 10)  # Discretize action space
        self.n_wealth_bins = agent_config.get('n_wealth_bins', 100)  # Number of wealth bins
        self.initial_wealth = env_config.get('initial_wealth', 1)  # Initial wealth from environment config
        self.action_space: np.ndarray = np.linspace(0, 1, self.n_actions)  # Discrete action space
        self.q_table: Dict[Tuple[int, int], np.ndarray] = {}  # State-action value table
        self.policy: Dict[Tuple[int, int], np.ndarray] = {}  # Policy table
        
    def _discretize_state(self, state: np.ndarray) -> Tuple[int, int]:
        """Discretize the continuous state space.
        
        Args:
            state: Current state observation [current wealth, remaining time steps]
            
        Returns:
            Tuple of (discretized_wealth_index, time_step)
        """
        wealth, time = state
        # Discretize wealth using logarithmic bins
        min_wealth = 0.1  # Lower bound for wealth
        max_wealth = self.initial_wealth * 5  # Assume max wealth is 5x initial
        log_wealth = np.log(max(wealth, min_wealth))
        log_bins = np.linspace(np.log(min_wealth), np.log(max_wealth), self.n_wealth_bins)
        wealth_idx = np.digitize(log_wealth, log_bins) - 1
        return (int(wealth_idx), int(time))

    def select_action(self, state: np.ndarray) -> float:
        """Choose an action based on the current state and policy.

        Args:
            state: Current state observation [current wealth, remaining time steps]

        Returns:
            Action to take
        """
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.action_space)  # Explore
        else:
            return self.get_best_action(state)  # Exploit

    def get_best_action(self, state: np.ndarray) -> float:
        """Get the best action for a given state based on the Q-table.
        Args:
            state: Current state observation [current wealth, remaining time steps]
        Returns:
            Best action for the given state
        """
        state_key = self._discretize_state(state)
        if state_key in self.q_table:
            return self.action_space[np.argmax(self.q_table[state_key])]
        else:
            # Initialize state entry in Q-table and policy
            self.q_table[state_key] = np.zeros(self.n_actions)
            self.policy[state_key] = np.ones(self.n_actions) / self.n_actions
            return np.random.choice(self.action_space)

    def train(self, env: MarketEnvironment, n_episodes: int) -> Dict[str, List]:
        """Train the agent using SARSA method.
        
        Args:
            env: Market environment instance
            n_episodes: Number of episodes to train
            
        Returns:
            Dictionary containing training metrics
        """
        returns = []
        epsilons = []
        wealth_history = []
        action_history = []
        
        for episode in range(n_episodes):
            state = env.reset()
            done = False
            episode_rewards = []
            episode_wealth = [state[0]]
            episode_actions = []
            
            # Select initial action
            action = self.select_action(state)
            
            while not done:
                # Take action and observe next state and reward
                next_state, reward, done, info = env.step(action)
                
                # Select next action using the same policy
                next_action = self.select_action(next_state)
                
                # SARSA update
                state_key = self._discretize_state(state)
                next_state_key = self._discretize_state(next_state)
                action_idx = np.argmax(self.action_space == action)
                next_action_idx = np.argmax(self.action_space == next_action)
                
                # Initialize state-action values if not exists
                if state_key not in self.q_table:
                    self.q_table[state_key] = np.zeros(self.n_actions)
                    self.policy[state_key] = np.ones(self.n_actions) / self.n_actions
                if next_state_key not in self.q_table:
                    self.q_table[next_state_key] = np.zeros(self.n_actions)
                    self.policy[next_state_key] = np.ones(self.n_actions) / self.n_actions
                
                # Q-value update using SARSA
                target = reward
                if not done:
                    target += self.gamma * self.q_table[next_state_key][next_action_idx]
                current = self.q_table[state_key][action_idx]
                self.q_table[state_key][action_idx] = current + self.learning_rate * (target - current)
                
                # Policy improvement
                best_action_idx = np.argmax(self.q_table[state_key])
                for a_idx in range(self.n_actions):
                    if a_idx == best_action_idx:
                        self.policy[state_key][a_idx] = 1 - (self.n_actions - 1) / self.n_actions * self.epsilon
                    else:
                        self.policy[state_key][a_idx] = self.epsilon / self.n_actions
                
                # Record step information
                episode_rewards.append(reward)
                episode_wealth.append(next_state[0])
                episode_actions.append(action)
                
                # Move to next state and action
                state = next_state
                action = next_action
            
            # Record episode metrics
            returns.append(sum(episode_rewards))
            epsilons.append(self.epsilon)
            wealth_history.append(episode_wealth)
            action_history.append(episode_actions)
        
        return {
            'returns': returns,
            'epsilons': epsilons,
            'wealth_history': wealth_history,
            'action_history': action_history
        }
    
    def save(self, path: str):
        """Save the agent's Q-table and policy to a file.
        
        Args:
            path: Path to save the agent's data
        """
        np.save(path, {
            'q_table': self.q_table,
            'policy': self.policy
        })
    
    def load(self, path: str):
        """Load the agent's Q-table and policy from a file.
        
        Args:
            path: Path to load the agent's data from
        """
        data = np.load(path, allow_pickle=True).item()
        self.q_table = data['q_table']
        self.policy = data['policy']