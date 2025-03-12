import pytest
import numpy as np
from environments.market_environment import MarketEnvironment

@pytest.fixture
def balanced_config():
    return {
        'environment': {
            'risky_return_a': 0.2,      # 20% positive return
            'risky_return_b': -0.2,     # 20% negative return
            'risky_return_p': 0.5,      # 50% chance for each outcome
            'risk_free_rate': 0.02,     # 2% risk-free rate
            'alpha': 0.5,               # Moderate risk aversion
            'gamma': 0.99,
            'T': 10,
            'initial_wealth': 10000
        }
    }

def test_balanced_strategy(balanced_config):
    """Test the performance of a balanced strategy (50% risky, 50% risk-free)."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Initialize environment
    env = MarketEnvironment(balanced_config)
    observation = env.reset()
    
    # Record data for analysis
    wealth_history = [env.wealth]
    portfolio_returns = []
    risky_returns = []
    
    # Run simulation with balanced strategy
    done = False
    while not done:
        # Always invest 50% in each asset
        action = 0.5
        observation, reward, done, info = env.step(action)
        
        # Record results
        wealth_history.append(env.wealth)
        portfolio_returns.append(info['portfolio_return'])
        risky_returns.append(info['risky_return'])
        
        # Print step-by-step information
        print(f'\nStep {env.current_step}:')
        print(f'Asset Allocation: {action*100:.1f}% Risky, {(1-action)*100:.1f}% Risk-free')
        print(f'Risky Asset Return: {info["risky_return"]*100:.2f}%')
        print(f'Portfolio Return: {(info["portfolio_return"]-1)*100:.2f}%')
        print(f'Current Wealth: {env.wealth:.2f}')
    
    # Verify strategy execution
    assert len(wealth_history) == env.T + 1  # Initial wealth plus T steps
    assert len(portfolio_returns) == env.T
    
    # Calculate and verify final results
    final_wealth = wealth_history[-1]
    final_utility = (1 - np.exp(-env.alpha * final_wealth)) / env.alpha
    
    # Verify wealth is positive
    assert final_wealth > 0
    
    # Verify utility is reasonable
    assert final_utility ==(1 - np.exp(-env.alpha * final_wealth)) / env.alpha
    
    # Calculate strategy statistics
    avg_portfolio_return = np.mean(portfolio_returns)
    std_portfolio_return = np.std(portfolio_returns)
    
    # Print strategy performance metrics
    print(f'\nBalanced Strategy Performance:')
    print(f'Initial Wealth: {env.initial_wealth:.2f}')
    print(f'Final Wealth: {final_wealth:.2f}')
    print(f'Final Utility: {final_utility:.4f}')
    print(f'Average Portfolio Return: {avg_portfolio_return:.4f}')
    print(f'Portfolio Return Std: {std_portfolio_return:.4f}')