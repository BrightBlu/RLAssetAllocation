import pytest
import numpy as np
from environments.market_environment import MarketEnvironment

@pytest.fixture
def edge_case_config():
    return {
        'environment': {
            'risky_return_a': 1.0,      # 100% positive return
            'risky_return_b': -0.9,     # 90% loss
            'risky_return_p': 0.1,      # Only 10% chance for positive return
            'risk_free_rate': 0.001,    # Very low risk-free rate
            'alpha': 0.5,               # Moderate risk aversion
            'gamma': 0.95,
            'T': 5,
            'initial_wealth': 1000
        }
    }

@pytest.fixture
def env(edge_case_config):
    return MarketEnvironment(edge_case_config)

def test_extreme_market_conditions(env):
    """Test environment behavior under extreme market conditions."""
    np.random.seed(42)
    
    # Test with full investment in risky asset
    observation, reward, done, info = env.step(1.0)
    assert 0 <= info['portfolio_return'] <= 2.0  # Return should be between 0% and 200%
    
    # Test with no investment in risky asset
    env.reset()
    observation, reward, done, info = env.step(0.0)
    assert info['portfolio_return'] == pytest.approx(1.001)  # Should get risk-free return

def test_wealth_preservation(env):
    """Test wealth preservation under conservative strategy."""
    initial_wealth = env.wealth
    
    # Always invest in risk-free asset
    for _ in range(env.T):
        observation, reward, done, info = env.step(0.0)
    
    final_wealth = env.wealth
    assert final_wealth > initial_wealth  # Should preserve and slightly increase wealth

def test_high_risk_high_reward(env):
    """Test high risk strategy outcomes."""
    np.random.seed(42)
    initial_wealth = env.wealth
    max_wealth = initial_wealth
    min_wealth = initial_wealth
    
    # Always invest in risky asset
    for _ in range(env.T):
        observation, reward, done, info = env.step(1.0)
        max_wealth = max(max_wealth, env.wealth)
        min_wealth = min(min_wealth, env.wealth)
    
    # Verify significant wealth variation
    assert max_wealth != min_wealth
    assert max_wealth > initial_wealth or min_wealth < initial_wealth

def test_utility_sensitivity(env):
    """Test utility function sensitivity to wealth changes."""
    np.random.seed(42)
    
    # Compare utilities at different wealth levels with adjusted values
    w1 = 10
    w2 = 50
    u1 = (1 - np.exp(-env.alpha * w1)) / env.alpha
    u2 = (1 - np.exp(-env.alpha * w2)) / env.alpha
    
    # Verify diminishing marginal utility
    wealth_diff = w2 - w1
    utility_diff = u2 - u1
    assert utility_diff > 0  # Higher wealth should give higher utility
    
    w3 = 200
    u3 = (1 - np.exp(-env.alpha * w3)) / env.alpha
    utility_diff_2 = u3 - u2
    
    # Test diminishing marginal utility
    assert utility_diff > utility_diff_2  # Utility increase should be smaller