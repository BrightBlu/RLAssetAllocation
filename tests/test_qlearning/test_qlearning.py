import pytest
import numpy as np
from environments.market_environment import MarketEnvironment
from agents import qlearning_agent as qlearning_agent


@pytest.fixture
def default_config():
    return {
        'environment': {
            'risky_return_a': 0.2,
            'risky_return_b': -0.2,
            'risky_return_p': 0.5,
            'risk_free_rate': 0.02,
            'alpha': 0.5,
            'gamma': 1,
            'T': 10,
            'initial_wealth': 10000
        }
    }

@pytest.fixture
def env(default_config):
    return MarketEnvironment(default_config)


def optimal_x(env, t):


def optimal_q(env, wt, t):