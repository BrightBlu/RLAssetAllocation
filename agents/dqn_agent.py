from typing import Dict, Any, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
from environments.market_environment import MarketEnvironment

class DQNetwork(nn.Module):
    """Deep Q-Network for continuous state space.
    
    This network maps state observations to Q-values for each action.
    The network architecture consists of fully connected layers with ReLU activations.
    """
    
    def __init__(self, state_dim: int, n_actions: int):
        """Initialize the Q-Network.
        
        Args:
            state_dim: Dimension of the state space
            n_actions: Number of discrete actions
        """
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, n_actions)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input state tensor
            
        Returns:
            Q-values for each action
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class Agent:
    """Deep Q-Learning agent for asset allocation.
    
    This agent implements DQN with experience replay and target network.
    It uses a neural network to approximate Q-values for continuous state spaces.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the DQN agent.
        
        Args:
            config: Configuration dictionary containing:
                - epsilon: Exploration rate for epsilon-greedy policy
                - gamma: Discount factor
                - learning_rate: Learning rate for optimizer
                - batch_size: Size of training batch
                - memory_size: Size of replay memory
                - target_update: Frequency of target network update
                - n_actions: Number of discrete actions
        """
        agent_config = config.get('agent', {})
        env_config = config.get('environment', {})
        
        # DQN parameters
        self.epsilon = agent_config.get('epsilon', 0.1)
        self.gamma = agent_config.get('gamma', 0.99)
        self.learning_rate = agent_config.get('learning_rate', 0.001)
        self.batch_size = agent_config.get('batch_size', 32)
        self.memory_size = agent_config.get('memory_size', 10000)
        self.target_update = agent_config.get('target_update', 10)
        self.n_actions = agent_config.get('n_actions', 10)
        
        # Initialize action space
        self.action_space = np.linspace(0, 1, self.n_actions)
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.policy_net = DQNetwork(state_dim=2, n_actions=self.n_actions).to(self.device)
        self.target_net = DQNetwork(state_dim=2, n_actions=self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Setup optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        
        # Initialize replay memory
        self.memory = deque(maxlen=self.memory_size)
        
    def select_action(self, state: np.ndarray) -> float:
        """Choose an action using epsilon-greedy policy.
        
        Args:
            state: Current state observation [current wealth, remaining time steps]
            
        Returns:
            Selected action
        """
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.action_space)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            action_idx = q_values.max(1)[1].item()
            return self.action_space[action_idx]
    
    def _optimize_model(self):
        """Perform one step of optimization on the Q-network."""
        if len(self.memory) < self.batch_size:
            return
        
        # Sample batch from memory
        transitions = random.sample(self.memory, self.batch_size)
        batch = list(zip(*transitions))
        
        # Prepare batch tensors
        state_batch = torch.FloatTensor(batch[0]).to(self.device)
        action_batch = torch.LongTensor([[self.action_space.tolist().index(a)] for a in batch[1]]).to(self.device)
        reward_batch = torch.FloatTensor(batch[2]).to(self.device)
        next_state_batch = torch.FloatTensor(batch[3]).to(self.device)
        done_batch = torch.FloatTensor(batch[4]).to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)
        
        # Compute target Q values
        with torch.no_grad():
            max_next_q_values = self.target_net(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * max_next_q_values
        
        # Compute loss and optimize
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def train(self, env: MarketEnvironment, n_episodes: int) -> Dict[str, List]:
        """Train the agent using DQN.
        
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
        losses = []
        
        for episode in range(n_episodes):
            state = env.reset()
            done = False
            episode_rewards = []
            episode_wealth = [state[0]]
            episode_actions = []
            
            while not done:
                # Select and take action
                action = self.select_action(state)
                next_state, reward, done, info = env.step(action)
                
                # Store transition in memory
                self.memory.append((state, action, reward, next_state, float(done)))
                
                # Optimize model
                self._optimize_model()
                
                # Update target network
                if episode % self.target_update == 0:
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                
                # Record step information
                episode_rewards.append(reward)
                episode_wealth.append(next_state[0])
                episode_actions.append(action)
                
                # Move to next state
                state = next_state
            
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
        """Save the agent's networks to files.
        
        Args:
            path: Path to save the agent's data
        """
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    def load(self, path: str):
        """Load the agent's networks from files.
        
        Args:
            path: Path to load the agent's data from
        """
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])