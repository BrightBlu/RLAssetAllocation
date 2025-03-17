from typing import Dict, Any, List, Tuple
import numpy as np
from environments.market_environment import MarketEnvironment
from tqdm import tqdm

class Agent:
    """TD(0) agent for asset allocation."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the TD(0) agent."""
        agent_config = config.get('agent', {})
        env_config = config.get('environment', {})
        
        # Initial values for epsilon and learning rate
        self.initial_epsilon = agent_config.get('epsilon', 0.9)
        self.min_epsilon = agent_config.get('min_epsilon', 0.01)
        self.epsilon_decay = agent_config.get('epsilon_decay', 0.99995)
        self.epsilon = self.initial_epsilon
        
        self.initial_lr = agent_config.get('learning_rate', 0.1)
        self.min_lr = agent_config.get('min_learning_rate', 0.001)
        self.lr_decay = agent_config.get('learning_rate_decay', 0.99995)
        self.learning_rate = self.initial_lr
        
        self.gamma = agent_config.get('gamma', 0.99)
        self.initial_wealth = env_config.get('initial_wealth', 1)
        
        # Determine action sign based on market parameters
        self.risky_return_a = env_config.get('risky_return_a', 0.05)
        self.risky_return_b = env_config.get('risky_return_b', 0.01)
        self.risk_free_rate = env_config.get('risk_free_rate', 0.02)
        self.risky_return_p = env_config.get('risky_return_p', 0.05)
        self.T = env_config.get('T', 10)
        self.alpha = env_config.get('alpha', 0.1)
        self.action_sign = 1 if (self.risk_free_rate - self.risky_return_b)*(1 - self.risky_return_p) < self.risky_return_p * (self.risky_return_a - self.risk_free_rate) else 0

        
        # State -> value function
        self.value_table: Dict[Tuple[int, int], float] = {}
        # Greedy policy: state -> best discrete action
        self.policy: Dict[Tuple[int, int], int] = {}
        
        # Track some training metrics
        self.returns: List[float] = []
        self.epsilons: List[float] = []
        self.wealth_history: List[List[float]] = []
        self.action_history: List[List[float]] = []
        self.td_errors: List[float] = []  # Track TD errors for each episode
        
        # Predefine the full action space (discrete)
        self.action_upper_bound = env_config.get('action_upper_bound', 3)
        self.action_lower_bound = env_config.get('action_lower_bound', -3)
        self.action_space_shift = env_config.get('action_space_shift', False)

        self.state_upper_bound = env_config.get('state_upper_bound', 2)
        self.state_lower_bound = env_config.get('state_lower_bound', -2)

        self.all_actions = list(range(self.action_lower_bound, self.action_upper_bound + 1))

        if self.action_space_shift:
            self.action_shift_k = np.log(1+self.risk_free_rate)
            self.action_shift_b = - (self.T - 1) * np.log(1+self.risk_free_rate) + np.log(np.log(((self.risk_free_rate - self.risky_return_b)*(1-self.risky_return_p)/(self.risky_return_p*(self.risky_return_a-self.risk_free_rate))))/(self.alpha * (self.risky_return_b - self.risky_return_a)))

    def _discretize_state(self, state: np.ndarray) -> Tuple[int, int]:
        """Discretize the continuous state space."""
        state_value = state[0]
        time_step = int(state[1])

        if state_value >= 0:
            sign = 1
            log_value = np.clip(round(np.log(state_value)), self.state_lower_bound, self.state_upper_bound)
        else:
            sign = 0
            log_value = np.clip(round(np.log(abs(state_value))), self.state_lower_bound, self.state_upper_bound)
        
        return (sign, int(log_value))

    def _action_to_xt(self, log_value: int) -> float:
        """Convert discrete action (log_value) to continuous portfolio weight."""
        if self.action_space_shift:
            if self.action_sign == 1:
                return np.exp(self.action_shift_k * log_value + self.action_shift_b)
            else:
                return -np.exp(self.action_shift_k * log_value + self.action_shift_b)

        else:    
            if self.action_sign == 1:
                return np.exp(log_value)
            else:
                return -np.exp(log_value)

    def _xt_to_action(self, xt: float) -> int:
        """Convert continuous portfolio weight to discrete action (log_value)."""
        if self.action_space_shift:
            return int(np.clip(round(np.log(abs(xt)) / self.action_shift_k - self.action_shift_b / self.action_shift_k), self.action_lower_bound, self.action_upper_bound))
        else:
            return int(np.clip(round(np.log(abs(xt))), self.action_lower_bound, self.action_upper_bound))

    def _init_state_if_needed(self, state_key: Tuple[int, int]):
        """Initialize value function for new state."""
        if state_key not in self.value_table:
            self.value_table[state_key] = 0.0
            random_act = np.random.choice(self.all_actions)
            self.policy[state_key] = random_act

    def select_action(self, state: np.ndarray) -> float:
        """Select action using epsilon-greedy policy."""
        state_key = self._discretize_state(state)
        
        # Ensure value exists for this state
        self._init_state_if_needed(state_key)
        
        if np.random.rand() < self.epsilon:
            # Explore: pick a random discrete action
            discrete_action = np.random.choice(self.all_actions)
        else:
            # Exploit: use current policy
            discrete_action = self.policy[state_key]

        return self._action_to_xt(discrete_action)

    def get_best_action(self, state: np.ndarray) -> float:
        """Return the greedy action from current policy."""
        state_key = self._discretize_state(state)
        self._init_state_if_needed(state_key)
        
        discrete_action = self.policy[state_key]
        return self._action_to_xt(discrete_action)

    def train(self, env: MarketEnvironment, n_episodes: int) -> Dict[str, List]:
        for episode in tqdm(range(n_episodes)):
            state = env.reset()
            done = False

            episode_rewards = []
            episode_wealth = [state[0]]
            episode_actions = []
            episode_td_errors = []

            while not done:
                # --- 1) Select action via epsilon-greedy
                action = self.select_action(state)
                discrete_action = self._xt_to_action(action)

                # --- 2) Step environment
                next_state, reward, done, info = env.step(action)
                episode_rewards.append(reward)
                episode_wealth.append(next_state[0])
                episode_actions.append(action)

                # --- 3) TD(0) update
                state_key = self._discretize_state(state)
                next_state_key = self._discretize_state(next_state)

                # Ensure both states are initialized
                self._init_state_if_needed(state_key)
                self._init_state_if_needed(next_state_key)

                current_value = self.value_table[state_key]
                next_value = self.value_table[next_state_key]

                # Calculate TD target and error
                if done:
                    target = reward
                else:
                    target = reward + self.gamma * next_value

                td_error = target - current_value
                log_td_error = np.log(abs(td_error) + 1)  # Add 1 to avoid log(0)
                episode_td_errors.append(log_td_error)

                # Update value function
                self.value_table[state_key] = current_value + self.learning_rate * td_error

                # Update policy for the state
                best_action = max(self.all_actions, 
                                key=lambda a: reward + self.gamma * self.value_table[next_state_key])
                self.policy[state_key] = best_action

                # Move to next state
                state = next_state

            # Decay epsilon and learning rate
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            self.learning_rate = max(self.min_lr, self.learning_rate * self.lr_decay)
            
            # Collect episode stats
            self.returns.append(sum(episode_rewards))
            self.epsilons.append(self.epsilon)
            self.wealth_history.append(episode_wealth)
            self.action_history.append(episode_actions)
            self.td_errors.append(np.mean(episode_td_errors))

        return {
            'returns': self.returns,
            'epsilons': self.epsilons,
            'wealth_history': self.wealth_history,
            'action_history': self.action_history,
            'td_errors': self.td_errors
        }

    def save(self, path: str):
        """Save the agent's value table and policy to a file."""
        np.save(path, {
            'value_table': self.value_table,
            'policy': self.policy
        }, allow_pickle=True)

    
    def load(self, path: str):
        """Load the agent's value table and policy from a file."""
        data = np.load(path, allow_pickle=True).item()
        self.value_table = data['value_table']
        self.policy = data['policy']