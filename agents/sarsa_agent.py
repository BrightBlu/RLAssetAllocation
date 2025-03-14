from typing import Dict, Any, List, Tuple
import numpy as np
from environments.market_environment import MarketEnvironment
from tqdm import tqdm

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
        """
        agent_config = config.get('agent', {})
        env_config = config.get('environment', {})
        self.epsilon = agent_config.get('epsilon', 0.1)
        self.gamma = agent_config.get('gamma', 0.99)
        self.learning_rate = agent_config.get('learning_rate', 0.01)
        self.initial_wealth = env_config.get('initial_wealth', 1)  # Initial wealth from environment config
        self.q_table: Dict[Tuple[Tuple[int, int], int], Dict[Tuple[int, int], float]] = {}  # State-action value table
        self.policy: Dict[Tuple[Tuple[int, int], int], Tuple[int, int]] = {}  # Policy table
        # Training metrics
        self.returns = []
        self.epsilons = []
        self.wealth_history = []
        self.action_history = []
        
    def _discretize_state(self, state: np.ndarray) -> Tuple[Tuple[int, int], int]:
        """Discretize the continuous state space.
        
        Args:
            state: Current state observation [current wealth, remaining time steps]
            
        Returns:
            Tuple of ((sign, log_value), time_step)
        """
        state_value = state[0]  # Use wealth as state value
        time_step = int(state[1])  # Get time step
        
        if state_value >= 0:
            sign = 1
            log_value = np.clip(round(np.log(state_value)), -8, 8)
        else:
            sign = 0
            log_value = np.clip(round(np.log(abs(state_value))), -8, 8)

        return ((sign, int(log_value)), time_step)

    def _action_to_xt(self, action: Tuple[int, int]) -> float:
        """Convert discrete action to continuous portfolio weight.
        
        Args:
            action: Tuple of (sign, log_value) where sign is 0 or 1 and log_value is in [-8, 8]
            
        Returns:
            Portfolio weight as a float value
        """
        sign, log_value = action
        if sign == 1:
            return np.exp(log_value)
        else:
            return -np.exp(log_value)
    
    def _xt_to_action(self, xt: float) -> Tuple[int, int]:
        """Convert continuous portfolio weight to discrete action.
        
        Args:
            xt: Portfolio weight as a float value
            
        Returns:
            Tuple of (sign, log_value)
        """
        if xt >= 0:
            sign = 1
            log_value = np.clip(round(np.log(xt)), -8, 8)
        else:
            sign = 0
            log_value = np.clip(round(np.log(abs(xt))), -8, 8)
        return (sign, int(log_value))
    
    def select_action(self, state: np.ndarray) -> float:
        """Choose an action based on the current state and policy.

        Args:
            state: Current state observation [current wealth, remaining time steps]

        Returns:
            Action to take
        """
        if np.random.uniform(0, 1) < self.epsilon:
            # Random discrete action
            sign = np.random.choice([0, 1])
            log_value = np.random.randint(-8, 9)
            return self._action_to_xt((sign, log_value))  # Convert to continuous action
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
        if state_key in self.policy:
            return self._action_to_xt(self.policy[state_key])
        else:
            # Initialize state entry in Q-table and policy
            sign = np.random.choice([0, 1])
            log_value = np.random.randint(-8, 9)
            self.policy[state_key] = (sign, log_value)
            self.q_table[state_key] = {}
            return self._action_to_xt(self.policy[state_key])

    def train(self, env: MarketEnvironment, n_episodes: int) -> Dict[str, List]:
        """Train the agent using SARSA method.
        
        Args:
            env: Market environment instance
            n_episodes: Number of episodes to train
            
        Returns:
            Dictionary containing training metrics
        """
        for episode in tqdm(range(n_episodes)):
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
                discrete_action = self._xt_to_action(action)
                discrete_next_action = self._xt_to_action(next_action)
                
                # Initialize state-action values if not exists
                if state_key not in self.q_table:
                    self.q_table[state_key] = {}
                    self.policy[state_key] = discrete_action
                if next_state_key not in self.q_table:
                    self.q_table[next_state_key] = {}
                    self.policy[next_state_key] = discrete_next_action
                if discrete_action not in self.q_table[state_key]:
                    self.q_table[state_key][discrete_action] = 0.0
                if discrete_next_action not in self.q_table[next_state_key]:
                    self.q_table[next_state_key][discrete_next_action] = 0.0
                
                # Q-value update using SARSA
                target = reward
                if not done:
                    target += self.gamma * self.q_table[next_state_key][discrete_next_action]
                current = self.q_table[state_key][discrete_action]
                self.q_table[state_key][discrete_action] = current + self.learning_rate * (target - current)
                
                # Policy improvement
                if np.random.uniform(0, 1) < self.epsilon:
                    # Random discrete action
                    sign = np.random.choice([0, 1])
                    log_value = np.random.randint(-8, 9)
                    self.policy[state_key] = (sign, log_value)
                else:
                    # Choose action with highest Q-value
                    best_action = None
                    best_value = float('-inf')
                    for act, q_value in self.q_table[state_key].items():
                        if q_value > best_value:
                            best_value = q_value
                            best_action = act
                    if best_action is not None:
                        self.policy[state_key] = best_action
                
                # Record step information
                episode_rewards.append(reward)
                episode_wealth.append(next_state[0])
                episode_actions.append(action)
                
                # Move to next state and action
                state = next_state
                action = next_action
            
            # Record episode metrics
            self.returns.append(sum(episode_rewards))
            self.epsilons.append(self.epsilon)
            self.wealth_history.append(episode_wealth)
            self.action_history.append(episode_actions)
        
        return {
            'returns': self.returns,
            'epsilons': self.epsilons,
            'wealth_history': self.wealth_history,
            'action_history': self.action_history
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