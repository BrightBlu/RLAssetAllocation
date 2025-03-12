import pytest
import numpy as np
from environments.market_environment import MarketEnvironment

@pytest.fixture
def default_config():
    return {
        'environment': {
            'risky_return_a': 0.2,
            'risky_return_b': -0.2,
            'risky_return_p': 0.5,
            'risk_free_rate': 0.02,
            'alpha': 0.5,
            'gamma': 0.99,
            'T': 10,
            'initial_wealth': 10000
        }
    }

@pytest.fixture
def env(default_config):
    return MarketEnvironment(default_config)

def test_environment_initialization(env, default_config):
    """Test if the environment is correctly initialized with given configuration."""
    # Test environment parameters initialization
    assert env.risky_return_a == default_config['environment']['risky_return_a']
    assert env.risky_return_b == default_config['environment']['risky_return_b']
    assert env.risky_return_p == default_config['environment']['risky_return_p']
    assert env.risk_free_rate == default_config['environment']['risk_free_rate']
    assert env.alpha == default_config['environment']['alpha']
    assert env.gamma == default_config['environment']['gamma']
    assert env.T == default_config['environment']['T']
    assert env.initial_wealth == default_config['environment']['initial_wealth']
    
    # Test initial state
    assert env.current_step == 0
    assert env.wealth == env.initial_wealth

def test_reset(env):
    """Test environment reset functionality."""
    # Change environment state
    env.current_step = 5
    env.wealth = 12000

    # Reset environment
    observation = env.reset()
    
    # Verify reset state
    assert env.current_step == 0
    assert env.wealth == env.initial_wealth
    np.testing.assert_array_equal(observation, [env.initial_wealth, env.T])

def test_step_action_clipping(env):
    """Test action clipping functionality."""
    # Test action below lower bound (0)
    observation, reward, done, info = env.step(-0.5)
    assert info['portfolio_return'] == (0 * (1 + info['risky_return']) + 1 * (1 + env.risk_free_rate))

    env.reset()
    # Test action above upper bound (1)
    observation, reward, done, info = env.step(1.5)
    assert info['portfolio_return'] == (1 * (1 + info['risky_return']) + 0 * (1 + env.risk_free_rate))

def test_step_wealth_update(env):
    """Test wealth update calculations."""
    np.random.seed(42)  # Set random seed for reproducibility
    
    initial_wealth = env.wealth
    action = 0.5
    observation, reward, done, info = env.step(action)
    
    # Verify wealth update calculations
    expected_wealth = initial_wealth * info['portfolio_return']
    assert env.wealth == pytest.approx(expected_wealth)
    assert info['wealth_change'] == pytest.approx(env.wealth - initial_wealth)

def test_step_reward_calculation(env):
    """Test reward calculation."""
    # Test intermediate step reward
    observation, reward, done, info = env.step(0.5)
    assert reward == 0

    # Test final step reward
    env.current_step = env.T - 1
    observation, reward, done, info = env.step(0.5)
    expected_reward = (1 - np.exp(-env.alpha * env.wealth)) / env.alpha
    assert reward == pytest.approx(expected_reward)
    assert done == True

def test_state_space_and_action_space(env):
    """Test state space and action space definitions."""
    assert env.state_space == 2  # Current wealth and remaining time steps
    assert env.action_space == (0.0, 1.0)  # Continuous action space in range [0,1]

def test_render(env, capsys):
    """Test environment rendering functionality."""
    env.render()
    captured = capsys.readouterr()
    expected_output = f'Step: {env.current_step}/{env.T}\nCurrent Wealth: {env.wealth:.2f}\n'
    assert captured.out == expected_output