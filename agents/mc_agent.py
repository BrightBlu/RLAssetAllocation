from typing import Dict, Any, List, Tuple
import numpy as np
from environments.market_environment import MarketEnvironment
from tqdm import tqdm

class Agent:
    """Monte Carlo agent for asset allocation.
    
    This agent implements Monte Carlo methods for policy evaluation and improvement.
    It uses epsilon-greedy exploration and maintains state-action value estimates based on sampling.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the MC agent.
        
        Args:
            config: Configuration dictionary containing:
                - epsilon: Exploration rate for epsilon-greedy policy
                - gamma: Discount factor
        """
        agent_config = config.get('agent', {})
        env_config = config.get('environment', {})
        self.epsilon = agent_config.get('epsilon', 0.1)
        self.gamma = agent_config.get('gamma', 0.99)
        self.initial_wealth = env_config.get('initial_wealth', 1)  # Initial wealth from environment config
        
        # Determine action sign based on market parameters
        risky_return_a = env_config.get('risky_return_a', 0.05)
        risky_return_b = env_config.get('risky_return_b', 0.01)
        risk_free_rate = env_config.get('risk_free_rate', 0.02)
        self.action_sign = 1 if (risky_return_b - risk_free_rate) * (risky_return_b - risky_return_a) > 0 else 0
        
        self.q_table: Dict[Tuple[int, int], Dict[int, List[float]]] = {}  # State-action value table
        self.policy: Dict[Tuple[int, int], int] = {}  # Policy table: state -> action (log_value)
        # Training metrics
        self.returns = []
        self.epsilons = []
        self.wealth_history = []
        self.action_history = []
        
    def _discretize_state(self, state: np.ndarray) -> Tuple[Tuple[int, int], int]:
        """
        Args:
            state: Current state observation [current wealth, remaining time steps]
            
        Returns:
            Tuple of ((sign, log_value), time_step) representing the state
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

    def _action_to_xt(self, log_value: int) -> float:
        """Convert discrete action to continuous portfolio weight.
        
        Args:
            log_value: Integer in range [-8, 8] representing the log of portfolio weight
            
        Returns:
            Portfolio weight as a float value
        """
        if self.action_sign == 1:
            return np.exp(log_value)
        else:
            return -np.exp(log_value)
    
    def _xt_to_action(self, xt: float) -> int:
        """Convert continuous portfolio weight to discrete action.
        
        Args:
            xt: Portfolio weight as a float value
            
        Returns:
            Integer in range [-8, 8] representing the log of portfolio weight
        """
        return int(np.clip(round(np.log(abs(xt))), -8, 8))
    
    def select_action(self, state: np.ndarray) -> float:
        """Choose an action based on the current state and policy.

        Args:
            state: Current state observation [current wealth, remaining time steps]

        Returns:
            Action to take
        """
        if np.random.uniform(0, 1) < self.epsilon:
            # Random discrete action
            log_value = np.random.randint(-8, 9)
            return self._action_to_xt(log_value)  # Convert to continuous action
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
            log_value = np.random.randint(-8, 9)
            self.policy[state_key] = log_value
            self.q_table[state_key] = {}
            return self._action_to_xt(self.policy[state_key])

    def train(self, env: MarketEnvironment, n_episodes: int) -> Dict[str, List]:
        """Train the agent using Monte Carlo every-visit method.
        
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
        
        for episode in tqdm(range(n_episodes)):
            # Episode generation
            state = env.reset()
            done = False
            episode_states = []
            episode_actions = []
            episode_rewards = []
            episode_wealth = [state[0]]
            
            while not done:
                action = self.select_action(state)
                next_state, reward, done, info = env.step(action)
                
                episode_states.append(state)
                episode_actions.append(action)
                episode_rewards.append(reward)
                episode_wealth.append(next_state[0])
                
                state = next_state
            
            # Policy evaluation and improvement using every-visit MC
            G = 0
            for t in range(len(episode_states) - 1, -1, -1):
                G = self.gamma * G + episode_rewards[t]
                state = episode_states[t]
                action = episode_actions[t]
                state_key = self._discretize_state(state)
                discrete_action = self._xt_to_action(action)
                
                # Initialize state-action value if not exists
                if state_key not in self.q_table:
                    self.q_table[state_key] = {}
                    self.policy[state_key] = discrete_action
                if discrete_action not in self.q_table[state_key]:
                    self.q_table[state_key][discrete_action] = []
                
                # Update Q-value for state-action pair
                self.q_table[state_key][discrete_action].append(G)
                
                # Policy improvement
                if np.random.uniform(0, 1) < self.epsilon:
                    # Random discrete action
                    log_value = np.random.randint(-8, 9)
                    self.policy[state_key] = log_value
                else:
                    # Choose action with highest average return
                    best_action = None
                    best_value = float('-inf')
                    for act, returns in self.q_table[state_key].items():
                        avg_return = np.mean(returns)
                        if avg_return > best_value:
                            best_value = avg_return
                            best_action = act
                    if best_action is not None:
                        self.policy[state_key] = best_action
            
            # Record metrics
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
            'policy': self.policy,
            'returns': self.returns
        })
    
    def load(self, path: str):
        """Load the agent's Q-table and policy from a file.
        
        Args:
            path: Path to load the agent's data from
        """
        data = np.load(path, allow_pickle=True).item()
        self.q_table = data['q_table']
        self.policy = data['policy']
        self.returns = data['returns']