from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from environments.market_environment import MarketEnvironment
from tqdm import tqdm

class DQN(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        return self.network(x)

class Agent:
    """DQN agent for asset allocation."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the DQN agent."""
        agent_config = config.get('agent', {})
        env_config = config.get('environment', {})
        
        # Initial values for epsilon and learning rate
        self.initial_epsilon = agent_config.get('epsilon', 0.9)
        self.min_epsilon = agent_config.get('min_epsilon', 0.01)
        self.epsilon_decay = agent_config.get('epsilon_decay', 0.99995)
        self.epsilon = self.initial_epsilon
        
        self.learning_rate = agent_config.get('learning_rate', 0.001)
        self.gamma = agent_config.get('gamma', 0.99)
        self.initial_wealth = env_config.get('initial_wealth', 1)
        
        # DQN specific parameters
        self.batch_size = agent_config.get('batch_size', 64)
        self.target_update = agent_config.get('target_update', 10)
        self.memory_size = agent_config.get('memory_size', 10000)
        self.memory = deque(maxlen=self.memory_size)
        
        # Determine action sign based on market parameters
        self.risky_return_a = env_config.get('risky_return_a', 0.05)
        self.risky_return_b = env_config.get('risky_return_b', 0.01)
        self.risk_free_rate = env_config.get('risk_free_rate', 0.02)
        self.risky_return_p = env_config.get('risky_return_p', 0.05)
        self.T = env_config.get('T', 10)
        self.alpha = env_config.get('alpha', 0.1)
        self.action_sign = 1 if (self.risk_free_rate - self.risky_return_b)*(1 - self.risky_return_p) < self.risky_return_p * (self.risky_return_a - self.risk_free_rate) else 0

        # Predefine the full action space (discrete)
        self.action_upper_bound = env_config.get('action_upper_bound', 3)
        self.action_lower_bound = env_config.get('action_lower_bound', -3)
        self.action_space_shift = env_config.get('action_space_shift', False)

        self.state_upper_bound = env_config.get('state_upper_bound', 2)
        self.state_lower_bound = env_config.get('state_lower_bound', -2)

        self.all_actions = list(range(self.action_lower_bound, self.action_upper_bound + 1))
        
        # Initialize networks
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(3, len(self.all_actions)).to(self.device)  # 3 = [wealth_sign, log_wealth, time_step]
        self.target_net = DQN(3, len(self.all_actions)).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.criterion = nn.MSELoss()

        if self.action_space_shift:
            self.action_shift_k = np.log(1+self.risk_free_rate)
            self.action_shift_b = - (self.T - 1) * np.log(1+self.risk_free_rate) + np.log(np.log((self.risk_free_rate - self.risky_return_b)*(1-self.risky_return_p)/(self.risky_return_p*(self.risky_return_a-self.risk_free_rate)))/(self.alpha * (self.risky_return_b - self.risky_return_a)))
        
        # Track training metrics
        self.returns: List[float] = []
        self.epsilons: List[float] = []
        self.wealth_history: List[List[float]] = []
        self.action_history: List[List[float]] = []
        self.td_errors: List[float] = []

    def _state_to_tensor(self, state: np.ndarray) -> torch.Tensor:
        """Convert state to tensor format for neural network."""
        wealth = state[0]
        time_step = state[1]
        
        if wealth >= 0:
            sign = 1
            log_value = np.clip(np.log(wealth), self.state_lower_bound, self.state_upper_bound)
        else:
            sign = 0
            log_value = np.clip(np.log(abs(wealth)), self.state_lower_bound, self.state_upper_bound)
        
        return torch.FloatTensor([sign, log_value, time_step]).to(self.device)

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

    def select_action(self, state: np.ndarray) -> float:
        """Select action using epsilon-greedy from DQN."""
        if np.random.rand() < self.epsilon:
            # Explore: pick a random discrete action
            discrete_action = np.random.choice(self.all_actions)
        else:
            # Exploit: pick argmax Q(s, a)
            with torch.no_grad():
                state_tensor = self._state_to_tensor(state)
                q_values = self.policy_net(state_tensor)
                discrete_action = self.all_actions[q_values.argmax().item()]

        return self._action_to_xt(discrete_action)

    def get_best_action(self, state: np.ndarray) -> float:
        """Return the greedy action (argmax over Q) – no exploration."""
        with torch.no_grad():
            state_tensor = self._state_to_tensor(state)
            q_values = self.policy_net(state_tensor)
            discrete_action = self.all_actions[q_values.argmax().item()]
        return self._action_to_xt(discrete_action)

    def _optimize_model(self):
        """Perform one step of optimization on the DQN."""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample from memory
        transitions = random.sample(self.memory, self.batch_size)
        batch = list(zip(*transitions))
        
        # Convert to tensors
        state_batch = torch.stack([self._state_to_tensor(s) for s in batch[0]])
        action_batch = torch.LongTensor([[self.all_actions.index(self._xt_to_action(a))] for a in batch[1]]).to(self.device)
        reward_batch = torch.FloatTensor(batch[2]).to(self.device)
        next_state_batch = torch.stack([self._state_to_tensor(s) for s in batch[3]])
        done_batch = torch.BoolTensor(batch[4]).to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute next Q values
        next_q_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            next_q_values[~done_batch] = self.target_net(next_state_batch[~done_batch]).max(1)[0]
        
        # Compute expected Q values
        expected_q_values = reward_batch + (self.gamma * next_q_values).unsqueeze(1)
        
        # Compute loss and optimize
        loss = self.criterion(current_q_values, expected_q_values)
        
        # Calculate TD error for logging
        td_error = (expected_q_values - current_q_values).abs().mean().item()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return td_error

    def train(self, env: MarketEnvironment, n_episodes: int) -> Dict[str, List]:
        for episode in tqdm(range(n_episodes)):
            state = env.reset()
            done = False

            episode_rewards = []
            episode_wealth = [state[0]]
            episode_actions = []
            episode_td_errors = []

            while not done:
                # Select and perform action
                action = self.select_action(state)
                next_state, reward, done, info = env.step(action)
                
                # Store transition in memory
                self.memory.append((state, action, reward, next_state, done))
                
                # Optimize model
                if td_error := self._optimize_model():
                    episode_td_errors.append(td_error)
                
                # Update target network
                if env.current_step % self.target_update == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                
                # Record metrics
                episode_rewards.append(reward)
                episode_wealth.append(next_state[0])
                episode_actions.append(action)
                
                state = next_state

            # Decay epsilon
            self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
            
            # Collect episode stats
            self.returns.append(sum(episode_rewards))
            self.epsilons.append(self.epsilon)
            self.wealth_history.append(episode_wealth)
            self.action_history.append(episode_actions)
            self.td_errors.append(np.mean(episode_td_errors) if episode_td_errors else 0)

        return {
            'returns': self.returns,
            'epsilons': self.epsilons,
            'wealth_history': self.wealth_history,
            'action_history': self.action_history,
            'td_errors': self.td_errors
        }

    def save(self, path: str):
        """Save the DQN model state."""
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        """Load the DQN model state."""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])