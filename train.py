import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from environments.market_environment import MarketEnvironment


def load_config(config_path: str):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def train(env: MarketEnvironment, agent, n_episodes: int):
    """Train the agent for specified number of episodes."""


    metrics = agent.train(env, n_episodes)
    
    returns = metrics['returns']
    epsilons = metrics['epsilons']
    wealth_history = metrics['wealth_history']
    action_history = metrics['action_history']
    td_errors = metrics['td_errors']
    
    return returns, epsilons, wealth_history, action_history, td_errors

def get_next_run_dir(config_path):
    """Create and get the next run directory based on config name."""
    config_name = Path(config_path).stem
    base_dir = Path('results') / config_name
    base_dir.mkdir(parents=True, exist_ok=True)
    
    existing_runs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith(f'{config_name}_')]
    next_run = 1
    if existing_runs:
        last_run = max(int(d.name.split('_')[-1]) for d in existing_runs)
        next_run = last_run + 1
    
    run_dir = base_dir / f'{config_name}_{next_run:03d}'
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def save_results(returns, epsilons, wealth_history, action_history, td_errors, episode, save_dir):
    """Save training results and generate visualizations."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    np.save(save_dir / 'returns.npy', returns)
    np.save(save_dir / 'epsilons.npy', epsilons)
    np.save(save_dir / 'wealth_history.npy', wealth_history)
    np.save(save_dir / 'action_history.npy', action_history)
    np.save(save_dir / 'td_errors.npy', td_errors)
    
    # Create visualizations using the visualization module
    metrics = {
        'returns': returns,
        'wealth_history': wealth_history,
        'action_history': action_history,
        'td_errors': td_errors
    }
    
    from utils.visualization import plot_training_metrics
    plot_training_metrics(metrics, str(save_dir / f'training_metrics_ep{episode}.png'))
    


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Asset Allocation Reinforcement Learning Training')
    parser.add_argument('--config', type=str, default='configs/mc.json',
                        help='Configuration file path')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create run directory
    run_dir = get_next_run_dir(args.config)
    print(f'Results will be saved to: {run_dir}')
    
    # Set random seed
    np.random.seed(config.get('training', {}).get('random_seed', 42))
    
    # Initialize environment and agent
    env = MarketEnvironment(config)
    if config.get('agent', {}).get('type') == 'mc':
        from agents.mc_agent import Agent
    if config.get('agent', {}).get('type') =='sarsa':
        from agents.sarsa_agent import Agent
    if config.get('agent', {}).get('type') =='qlearning':
        from agents.qlearning_agent import Agent
    if config.get('agent', {}).get('type') =='dqn':
        from agents.dqn_agent import Agent
    if config.get('agent', {}).get('type') =='qlearning_nw_cws':
        from agents.qlearning_agent_no_wealth_const_weight_sign import Agent
    
    agent = Agent(config)
    
    # Training parameters
    n_episodes = config.get('training', {}).get('n_episodes', 1000)
    
    # Train agent
    print('Starting training...')
    returns, epsilons, wealth_history, action_history, td_errors = train(env, agent, n_episodes)
    
    # Save final results
    save_results(returns, epsilons, wealth_history, action_history, td_errors, n_episodes, run_dir)
    
    # Save trained agent and config
    agent.save(str(run_dir / 'model.npy'))
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    
    print('Training completed. Results saved to:', run_dir)

if __name__ == '__main__':
    main()