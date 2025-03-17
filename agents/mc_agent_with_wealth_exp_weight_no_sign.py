from typing import Dict, Any, List, Tuple
import numpy as np
from environments.market_environment import MarketEnvironment
from tqdm import tqdm

class Agent:
    """Monte Carlo agent for asset allocation."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the Monte Carlo agent."""
        agent_config = config.get('agent', {})
        env_config = config.get('environment', {})
        
        # Initial values for epsilon (no decay needed for MC)
        self.epsilon = agent_config.get('epsilon', 0.1)
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

        # State -> (action -> value)
        self.value_table: Dict[Tuple[int, int], Dict[int, float]] = {}
        # State -> (action -> returns)
        self.returns_table: Dict[Tuple[int, int], Dict[int, List[float]]] = {}
        # Greedy policy: state -> best discrete action
        self.policy: Dict[Tuple[int, int], int] = {}
        
        # Track some training metrics
        self.returns: List[float] = []
        self.epsilons: List[float] = []
        self.wealth_history: List[List[float]] = []
        self.action_history: List[List[float]] = []
        
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

    def _discretize_state(self, state: np.ndarray) -> Tuple[Tuple[int, int], int]:
        """
        Discretize the continuous state space.
        state[0]: wealth
        state[1]: time_step
        """
        state_value = state[0]
        time_step = int(state[1])

        if state_value >= 0:
            sign = 1
            log_value = np.clip(round(np.log(state_value)), self.state_lower_bound, self.state_upper_bound)
        else:
            sign = 0
            log_value = np.clip(round(np.log(abs(state_value))), self.state_lower_bound, self.state_upper_bound)
        
        return (sign, int(log_value)), time_step

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

    def _init_state_if_needed(self, state_key: Tuple[Tuple[int,int],int]):
        """
        If we haven't seen this state yet, initialize values and returns for all possible actions.
        And set the policy to a random action for now (optional).
        """
        if state_key not in self.value_table:
            # Initialize dictionaries for values and returns
            self.value_table[state_key] = {a: 0.0 for a in self.all_actions}
            self.returns_table[state_key] = {a: [] for a in self.all_actions}
            # Pick any random action as a default
            random_act = np.random.choice(self.all_actions)
            self.policy[state_key] = random_act

    def select_action(self, state: np.ndarray) -> float:
        """Select action using epsilon-greedy from value table."""
        state_key = self._discretize_state(state)
        
        # Ensure values exist for this state
        self._init_state_if_needed(state_key)
        
        if np.random.rand() < self.epsilon:
            # Explore: pick a random discrete action
            discrete_action = np.random.choice(self.all_actions)
        else:
            # Exploit: pick action with highest value
            values = self.value_table[state_key]
            discrete_action = max(values, key=values.get)

        return self._action_to_xt(discrete_action)

    def get_best_action(self, state: np.ndarray) -> float:
        """Return the greedy action (argmax over values) – no exploration."""
        state_key = self._discretize_state(state)
        self._init_state_if_needed(state_key)
        
        values = self.value_table[state_key]
        best_action = max(values, key=values.get)
        return self._action_to_xt(best_action)

    def train(self, env: MarketEnvironment, n_episodes: int) -> Dict[str, List]:
        for episode in tqdm(range(n_episodes)):
            # Generate episode
            episode_states = []
            episode_actions = []
            episode_rewards = []
            episode_wealth = []
            
            state = env.reset()
            episode_wealth.append(state[0])
            done = False

            while not done:
                # Select and take action
                action = self.select_action(state)
                discrete_action = self._xt_to_action(action)
                
                next_state, reward, done, info = env.step(action)
                
                # Store state, action, reward
                episode_states.append(state)
                episode_actions.append(discrete_action)
                episode_rewards.append(reward)
                episode_wealth.append(next_state[0])
                
                state = next_state

            # Calculate returns for each step
            G = 0
            returns = []
            for r in reversed(episode_rewards):
                G = r + self.gamma * G
                returns.insert(0, G)

            # Update value estimates for each state-action pair
            for t in range(len(episode_states)):
                state_key = self._discretize_state(episode_states[t])
                action = episode_actions[t]
                G = returns[t]
                
                # Ensure state is initialized
                self._init_state_if_needed(state_key)
                
                # Add return to returns list for this state-action pair
                self.returns_table[state_key][action].append(G)
                
                # Update value to be mean of all returns
                self.value_table[state_key][action] = np.mean(self.returns_table[state_key][action])
                
                # Update policy for this state
                best_action = max(self.value_table[state_key], key=self.value_table[state_key].get)
                self.policy[state_key] = best_action
            
            # Collect some stats
            self.returns.append(sum(episode_rewards))
            self.epsilons.append(self.epsilon)
            self.wealth_history.append(episode_wealth)
            self.action_history.append([self._action_to_xt(a) for a in episode_actions])

        return {
            'returns': self.returns,
            'epsilons': self.epsilons,
            'wealth_history': self.wealth_history,
            'action_history': self.action_history,
            "td_errors": []
        }

    def save(self, path: str):
        """Save the agent's value table and policy to a file."""
        np.save(path, {
            'value_table': self.value_table,
            'returns_table': self.returns_table,
            'policy': self.policy
        }, allow_pickle=True)

    def load(self, path: str):
        """Load the agent's value table and policy from a file."""
        data = np.load(path, allow_pickle=True).item()
        self.value_table = data['value_table']
        self.returns_table = data['returns_table']
        self.policy = data['policy']