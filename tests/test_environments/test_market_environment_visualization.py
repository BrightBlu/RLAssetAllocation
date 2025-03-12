import pytest
import numpy as np
import matplotlib.pyplot as plt
from environments.market_environment import MarketEnvironment

@pytest.fixture
def env_config():
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

def test_random_strategy_visualization(env_config):
    """Test visualization of environment running process under random strategy."""
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Initialize environment
    env = MarketEnvironment(env_config)
    observation = env.reset()
    
    # Record data for each step
    steps = []
    actions = []
    risky_returns = []
    portfolio_returns = []
    wealth_history = [env.wealth]
    
    done = False
    while not done:
        # Randomly select action (asset weight)
        action = np.random.random()
        observation, reward, done, info = env.step(action)
        
        # Record step data
        steps.append(env.current_step)
        actions.append(action)
        risky_returns.append(info['risky_return'])
        portfolio_returns.append(info['portfolio_return'])
        wealth_history.append(env.wealth)
    
    # Create visualization plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Asset Allocation Plot
    ax1.plot(steps, actions, 'b-', label='Risky Asset Weight')
    ax1.plot(steps, [1-a for a in actions], 'r-', label='Risk-free Asset Weight')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Asset Weight')
    ax1.set_title('Asset Allocation Strategy')
    ax1.legend()
    ax1.grid(True)
    
    # 2. Returns Plot
    ax2.plot(steps, risky_returns, 'b-', label='Risky Asset Return')
    ax2.plot(steps, [env.risk_free_rate] * len(steps), 'r-', label='Risk-free Rate')
    ax2.plot(steps, portfolio_returns, 'g-', label='Portfolio Return')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Return Rate')
    ax2.set_title('Asset Returns Comparison')
    ax2.legend()
    ax2.grid(True)
    
    # 3. Wealth Evolution Plot
    ax3.plot(range(len(wealth_history)), wealth_history, 'g-')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Wealth')
    ax3.set_title('Wealth Evolution Over Time')
    ax3.grid(True)
    
    # 4. Performance Summary
    final_utility = (1 - np.exp(-env.alpha * env.wealth)) / env.alpha
    ax4.text(0.5, 0.5, f'Performance Summary:\nFinal Wealth: {env.wealth:.2f}\nFinal Utility: {final_utility:.2f}\nRisk Aversion (α): {env.alpha}',
             horizontalalignment='center', verticalalignment='center',
             transform=ax4.transAxes, fontsize=12)
    ax4.axis('off')
    
    plt.tight_layout()
    plt.savefig('random_strategy_results.png')
    plt.close()