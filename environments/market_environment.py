from typing import Tuple, Dict, Any
import numpy as np

class MarketEnvironment:
    """Market environment for dual-asset allocation problem.

    This environment simulates a discrete-time investment problem with risky and risk-free assets.
    - Risky asset returns follow a discrete distribution: Y_t ~ a*δ_p + b*δ_(1-p)
    - Risk-free asset has a fixed return rate r
    - The objective is to maximize the CARA utility of final wealth: U(W_T) = (1 - e^(-α*W_T))/α
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the market environment.

        Args:
            config: Configuration dictionary containing environment parameters:
                - risky_return_a: Parameter a for risky asset return distribution
                - risky_return_b: Parameter b for risky asset return distribution
                - risky_return_p: Probability p for risky asset return distribution
                - risk_free_rate: Risk-free return rate r
                - alpha: Risk aversion coefficient α
                - gamma: Time discount factor
                - T: Number of time steps T
                - initial_wealth: Initial wealth W_0
        """
        env_config = config.get('environment', {})
        self.risky_return_a = env_config.get('risky_return_a', 0.05)  # 5% return by default
        self.risky_return_b = env_config.get('risky_return_b', -0.01)  # -1% return by default
        self.risky_return_p = env_config.get('risky_return_p', 0.5)  # 50% chance by default
        self.risk_free_rate = env_config.get('risk_free_rate', 0.02)  # 2% return by default
        self.alpha = env_config.get('alpha', 0.5)
        self.gamma = env_config.get('gamma', 0.99)
        self.T = env_config.get('T', 10)
        self.initial_wealth = env_config.get('initial_wealth', 1)
        self.instant_reward = env_config.get('instant_reward', False)
        
        self.current_step = 0
        self.wealth = self.initial_wealth

    def reset(self) -> np.ndarray:
        """Reset the environment to initial state.

        Returns:
            Initial state observation containing [current wealth, remaining time steps]
        """
        self.current_step = 0
        self.wealth = self.initial_wealth
        return np.array([self.wealth, self.T - self.current_step])

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, Dict]:
        """Execute an action and return the result.

        Args:
            action: Proportion of wealth invested in risky asset, range [0,1]

        Returns:
            tuple containing:
            - Next state observation [new wealth, remaining time steps]
            - Reward value (based on CARA utility function)
            - Whether episode is done
            - Dictionary of additional information
        """
        # Generate risky asset return
        risky_return = self.risky_return_a if np.random.random() < self.risky_return_p else self.risky_return_b

        # Calculate portfolio return
        # action represents the amount of wealth invested in risky asset (can be negative for short selling)
        risky_investment = action
        risk_free_investment = self.wealth - risky_investment
        
        # Store old wealth for info
        old_wealth = self.wealth
        
        # Update total wealth
        self.wealth = risky_investment * (1 + risky_return) + risk_free_investment * (1 + self.risk_free_rate)
        
        # Calculate portfolio return
        portfolio_return = (self.wealth - old_wealth) / old_wealth
        
        self.current_step += 1

        # Check if episode is done
        done = self.current_step >= self.T

        # Calculate reward (based on CARA utility function)
        if done:
            if self.alpha * self.wealth <= -700:
                multiplier = np.float64(1e+304)
            else:
                multiplier = np.exp(-self.alpha * self.wealth)

            reward = (1 - multiplier) / self.alpha
        else:
            if self.instant_reward:
                reward =  1 / (1 + np.exp(-portfolio_return)) - 0.5
            else:
                reward = 0

        # Prepare observation and info
        observation = np.array([self.wealth, self.T - self.current_step], dtype=np.float32)
        info = {
            'portfolio_return': portfolio_return,
            'risky_return': risky_return,
            'wealth_change': self.wealth - old_wealth
        }

        return observation, reward, done, info

    def render(self) -> None:
        """Render the current state of the environment."""
        print(f'Step: {self.current_step}/{self.T}')
        print(f'Current Wealth: {self.wealth:.2f}')
