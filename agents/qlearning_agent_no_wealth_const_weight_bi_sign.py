from typing import Dict, Any, List, Tuple
import numpy as np
from environments.market_environment import MarketEnvironment
from tqdm import tqdm

class Agent:
    """Q-learning agent for asset allocation."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the Q-learning agent."""
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
        self.action_value = env_config.get('action_value', 300)

        
        # State -> (action -> Q-value)
        self.q_table: Dict[Tuple[int, int], Dict[int, float]] = {}
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

        self.all_actions = [0,1]

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
            # log_value = np.clip(round(np.log(state_value)), -2, 2)
            log_value = 1
        else:
            # sign = 0
            # log_value = np.clip(round(np.log(abs(state_value))), -2, 2)
            sign = 1
            log_value = 1
        
        return (sign, int(log_value)), time_step

    def _action_to_xt(self, action_value: int) -> float:
        """Convert discrete action (log_value) to continuous portfolio weight."""
        if action_value > 0:
            return self.action_value
        else:
            return -self.action_value


    def _xt_to_action(self, xt: float) -> int:
        """Convert continuous portfolio weight to discrete action (log_value)."""
        if xt > 0:
            return 1
        else:
            return 0

    def _init_state_if_needed(self, state_key: Tuple[Tuple[int,int],int]):
        """
        If we haven't seen this state yet, initialize Q-values for all possible actions to 0.
        And set the policy to a random action for now (optional).
        """
        if state_key not in self.q_table:
            # Initialize a dictionary: action -> 0.0
            self.q_table[state_key] = {a: 0.0 for a in self.all_actions}
            # Pick any random action as a default
            random_act = np.random.choice(self.all_actions)
            self.policy[state_key] = random_act

    def select_action(self, state: np.ndarray) -> float:
        """Select action using epsilon-greedy from Q-table."""
        state_key = self._discretize_state(state)
        
        # Ensure Q-values exist for this state
        self._init_state_if_needed(state_key)
        
        if np.random.rand() < self.epsilon:
            # Explore: pick a random discrete action
            discrete_action = np.random.choice(self.all_actions)
            # print(f"Exploring: Random action {discrete_action}")
        else:
            # Exploit: pick argmax Q(s, a)
            q_values = self.q_table[state_key]
            discrete_action = max(q_values, key=q_values.get)  # returns the action with highest Q-value
            # print(f"Exploiting: Chose action {discrete_action}")

        return self._action_to_xt(discrete_action)

    def get_best_action(self, state: np.ndarray) -> float:
        """Return the greedy action (argmax over Q) – no exploration."""
        state_key = self._discretize_state(state)
        self._init_state_if_needed(state_key)
        
        q_values = self.q_table[state_key]
        best_action = max(q_values, key=q_values.get)
        return self._action_to_xt(best_action)

    def train(self, env: MarketEnvironment, n_episodes: int) -> Dict[str, List]:
        for episode in tqdm(range(n_episodes)):
            state = env.reset()
            done = False

            episode_rewards = []
            episode_wealth = [state[0]]
            episode_actions = []
            episode_td_errors = []  # Track TD errors for this episode

            while not done:
                # --- 1) Select action via epsilon-greedy
                action = self.select_action(state)
                discrete_action = self._xt_to_action(action)

                # --- 2) Step environment
                next_state, reward, done, info = env.step(action)
                episode_rewards.append(reward)
                episode_wealth.append(next_state[0])
                episode_actions.append(action)

                # --- 3) Q-learning update
                state_key = self._discretize_state(state)
                next_state_key = self._discretize_state(next_state)
                # print(f"Current state: {state_key}, next state: {next_state_key}")

                # Ensure both states are in Q-table
                self._init_state_if_needed(state_key)
                self._init_state_if_needed(next_state_key)

                current_q = self.q_table[state_key][discrete_action]

                # Off-policy target: r + gamma * max(Q(next_state, a'))
                if done:
                    target = reward
                else:
                    next_q_values = self.q_table[next_state_key]
                    target = reward + self.gamma * max(next_q_values.values())

                # Calculate TD error
                td_error = target - current_q
                # Convert TD error to log scale to handle extreme values
                log_td_error = np.log(abs(td_error) + 1)  # Add 1 to avoid log(0)
                episode_td_errors.append(log_td_error)  # Record log-scaled TD error
                
                # TD update
                self.q_table[state_key][discrete_action] = current_q + \
                    self.learning_rate * td_error

                # --- 4) (Optional) Update greedy policy for this state to reflect new Q
                best_action = max(self.q_table[state_key], key=self.q_table[state_key].get)
                self.policy[state_key] = best_action

                # --- 5) Move on
                state = next_state
                # print(f"Episode {episode}, Step {env.current_step}, State: {state_key}, Action: {discrete_action}, Reward: {reward}")



            # Decay epsilon and learning rate
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            self.learning_rate = max(self.min_lr, self.learning_rate * self.lr_decay)
            
            # Collect some stats
            self.returns.append(sum(episode_rewards))
            self.epsilons.append(self.epsilon)
            self.wealth_history.append(episode_wealth)
            self.action_history.append(episode_actions)
            self.td_errors.append(np.mean(episode_td_errors))  # Average TD error for this episode


        return {
            'returns': self.returns,
            'epsilons': self.epsilons,
            'wealth_history': self.wealth_history,
            'action_history': self.action_history,
            'td_errors': self.td_errors
        }

    def save(self, path: str):
        """Save the agent's Q-table and policy to a file."""
        np.save(path, {
            'q_table': self.q_table,
            'policy': self.policy
        }, allow_pickle=True)

    
    def load(self, path: str):
        """Load the agent's Q-table and policy from a file."""
        data = np.load(path, allow_pickle=True).item()
        self.q_table = data['q_table']
        self.policy = data['policy']
