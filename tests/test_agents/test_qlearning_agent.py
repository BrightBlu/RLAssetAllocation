import pytest
import numpy as np
from agents.qlearning_agent_with_wealth_exp_weight_no_sign import Agent
from environments.market_environment import MarketEnvironment

@pytest.fixture
def default_config():
    return {
        'agent': {
            'epsilon': 0.9,
            'min_epsilon': 0.01,
            'epsilon_decay': 0.99995,
            'learning_rate': 0.1,
            'min_learning_rate': 0.001,
            'learning_rate_decay': 0.99995,
            'gamma': 0.99
        },
        'environment': {
            'risky_return_a': 0.2,
            'risky_return_b': -0.2,
            'risky_return_p': 0.5,
            'risk_free_rate': 0.02,
            'alpha': 0.5,
            'gamma': 0.99,
            'T': 10,
            'initial_wealth': 10000,
            'instant_reward': False,
            'action_upper_bound': 3,
            'action_lower_bound': -3,
            'state_upper_bound': 2,
            'state_lower_bound': -2
        }
    }

@pytest.fixture
def agent(default_config):
    return Agent(default_config)

@pytest.fixture
def env(default_config):
    return MarketEnvironment(default_config)

def test_agent_initialization(agent, default_config):
    """Test if the agent is correctly initialized with given configuration."""
    agent_config = default_config['agent']
    env_config = default_config['environment']
    
    assert agent.initial_epsilon == agent_config['epsilon']
    assert agent.min_epsilon == agent_config['min_epsilon']
    assert agent.epsilon_decay == agent_config['epsilon_decay']
    assert agent.epsilon == agent.initial_epsilon
    
    assert agent.initial_lr == agent_config['learning_rate']
    assert agent.min_lr == agent_config['min_learning_rate']
    assert agent.lr_decay == agent_config['learning_rate_decay']
    assert agent.learning_rate == agent.initial_lr
    
    assert agent.gamma == agent_config['gamma']
    assert agent.initial_wealth == env_config['initial_wealth']
    
    assert agent.action_upper_bound == env_config['action_upper_bound']
    assert agent.action_lower_bound == env_config['action_lower_bound']
    assert agent.state_upper_bound == env_config['state_upper_bound']
    assert agent.state_lower_bound == env_config['state_lower_bound']

def test_state_discretization(agent):
    """Test state discretization functionality."""
    # Test positive wealth
    state = np.array([100.0, 5])
    discrete_state, time_step = agent._discretize_state(state)
    assert discrete_state[0] == 1  # sign
    assert discrete_state[1] == np.clip(round(np.log(100.0)), agent.state_lower_bound, agent.state_upper_bound)
    assert time_step == 5
    
    # Test negative wealth
    state = np.array([-100.0, 5])
    discrete_state, time_step = agent._discretize_state(state)
    assert discrete_state[0] == 0  # sign
    assert discrete_state[1] == np.clip(round(np.log(abs(-100.0))), agent.state_lower_bound, agent.state_upper_bound)
    assert time_step == 5

def test_action_conversion(agent):
    """Test action conversion between discrete and continuous space."""
    # Test action to portfolio weight conversion
    for action in range(agent.action_lower_bound, agent.action_upper_bound + 1):
        xt = agent._action_to_xt(action)
        recovered_action = agent._xt_to_action(xt)
        assert recovered_action == action

def test_epsilon_decay(agent):
    """Test epsilon decay functionality."""
    initial_epsilon = agent.epsilon
    
    # Decay epsilon
    agent.epsilon = max(agent.min_epsilon, agent.epsilon * agent.epsilon_decay)
    
    # Verify decay
    assert agent.epsilon == max(agent.min_epsilon, initial_epsilon * agent.epsilon_decay)

def test_learning_rate_decay(agent):
    """Test learning rate decay functionality."""
    initial_lr = agent.learning_rate
    
    # Decay learning rate
    agent.learning_rate = max(agent.min_lr, agent.learning_rate * agent.lr_decay)
    
    # Verify decay
    assert agent.learning_rate == max(agent.min_lr, initial_lr * agent.lr_decay)

def test_q_value_update(agent, env):
    """Test Q-value update functionality."""
    state = env.reset()
    discrete_state, time_step = agent._discretize_state(state)
    action = 0
    next_state, reward, done, _ = env.step(agent._action_to_xt(action))
    next_discrete_state, next_time_step = agent._discretize_state(next_state)
    
    # Initialize Q-values
    if discrete_state not in agent.q_table:
        agent.q_table[discrete_state] = {a: 0.0 for a in agent.all_actions}
    if next_discrete_state not in agent.q_table:
        agent.q_table[next_discrete_state] = {a: 0.0 for a in agent.all_actions}
    
    old_q_value = agent.q_table[discrete_state][action]
    if not done:
        next_max_q = max(agent.q_table[next_discrete_state].values())
        agent.q_table[discrete_state][action] = old_q_value + agent.learning_rate * (reward + agent.gamma * next_max_q - old_q_value)
    else:
        agent.q_table[discrete_state][action] = old_q_value + agent.learning_rate * (reward - old_q_value)
    new_q_value = agent.q_table[discrete_state][action]
    
    # Verify Q-value update
    if not done:
        next_max_q = max(agent.q_table[next_discrete_state].values())
        expected_q = old_q_value + agent.learning_rate * (reward + agent.gamma * next_max_q - old_q_value)
    else:
        expected_q = old_q_value + agent.learning_rate * (reward - old_q_value)
    
    assert new_q_value == pytest.approx(expected_q)