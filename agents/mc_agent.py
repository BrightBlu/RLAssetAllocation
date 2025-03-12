from typing import Dict, Any, List, Tuple
import numpy as np
from environments.market_environment import MarketEnvironment

class MCAgent:
    """Monte Carlo agent for asset allocation.
    
    This agent implements Monte Carlo methods for policy evaluation and improvement.
    It uses epsilon-greedy exploration and maintains state-action value estimates.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the MC agent.
        
        Args:
            config: Configuration dictionary containing:
                - epsilon: Exploration rate for epsilon-greedy policy
                - gamma: Discount factor
                - n_actions: Number of discrete actions
                - learning_rate: Learning rate for value function updates
        """
        agent_config = config.get('agent', {})
        self.epsilon = agent_config.get('epsilon', 0.1)
        self.gamma = agent_config.get('gamma', 0.99)
        self.n_actions = agent_config.get('n_actions', 10)  # Discretize action space
        self.learning_rate = agent_config.get('learning_rate', 0.01)
        
        # Initialize state-action value table
        self.q_table = {}
        self.returns = {}
        self.episode_history: List[Tuple] = []
    
    def _discretize_state(self, state: np.ndarray) -> Tuple[int, int]:
        """Convert continuous state to discrete state for table lookup.
        
        Args:
            state: Continuous state array [wealth, remaining_time]
            
        Returns:
            Tuple of discretized (wealth_level, time_step)
        """
        wealth, time = state
        # Discretize wealth into 100 levels
        wealth_level = int(np.clip(wealth / 100, 0, 99))
        return (wealth_level, int(time))
    
    def _get_action_from_value(self, state: Tuple[int, int], explore: bool = True) -> float:
        """Select action using epsilon-greedy policy.
        
        Args:
            state: Discretized state tuple
            explore: Whether to use exploration
            
        Returns:
            Selected action (proportion of wealth in risky asset)
        """
        if explore and np.random.random() < self.epsilon:
            # Random exploration
            return np.random.random()
        
        # Get Q-values for this state
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions)
        
        # Select best action
        action_idx = np.argmax(self.q_table[state])
        return action_idx / (self.n_actions - 1)  # Convert to [0,1] range
    
    def select_action(self, state: np.ndarray) -> float:
        """Select action for given state using current policy.
        
        Args:
            state: Environment state observation
            
        Returns:
            Selected action value
        """
        discrete_state = self._discretize_state(state)
        return self._get_action_from_value(discrete_state)
    
    def store_transition(self, state: np.ndarray, action: float, reward: float,
                        next_state: np.ndarray, done: bool) -> None:
        """Store transition for episode history.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode is done
        """
        discrete_state = self._discretize_state(state)
        action_idx = int(action * (self.n_actions - 1))
        self.episode_history.append((discrete_state, action_idx, reward))
    
    def update(self) -> Dict[str, float]:
        """Update value functions using episode history.
        
        Returns:
            Dictionary with training metrics
        """
        if not self.episode_history:
            return {}
        
        # Calculate returns for each step
        G = 0
        for t in reversed(range(len(self.episode_history))):
            state, action, reward = self.episode_history[t]
            G = reward + self.gamma * G
            
            # Store state-action return
            if (state, action) not in self.returns:
                self.returns[(state, action)] = []
            self.returns[(state, action)].append(G)
            
            # Update Q-value
            if state not in self.q_table:
                self.q_table[state] = np.zeros(self.n_actions)
            
            # Incremental update
            old_q = self.q_table[state][action]
            self.q_table[state][action] = old_q + self.learning_rate * (G - old_q)
        
        # Clear episode history
        episode_return = sum(x[2] for x in self.episode_history)
        self.episode_history = []
        
        return {
            'episode_return': episode_return,
            'epsilon': self.epsilon
        }
    
    def save(self, path: str) -> None:
        """Save agent state to file.
        
        Args:
            path: Path to save file
        """
        np.save(path, {
            'q_table': self.q_table,
            'returns': self.returns,
            'epsilon': self.epsilon
        })
    
    def load(self, path: str) -> None:
        """Load agent state from file.
        
        Args:
            path: Path to load file
        """
        data = np.load(path, allow_pickle=True).item()
        self.q_table = data['q_table']
        self.returns = data['returns']
        self.epsilon = data['epsilon']