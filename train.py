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
    
    # Save intermediate results every 100 episodes
    # for episode in range(100, n_episodes + 1, 100):
    #     save_results(returns[:episode], epsilons[:episode],
    #                 wealth_history[:episode], action_history[:episode],
    #                 episode, Path('results/intermediate'))
    
    return returns, epsilons, wealth_history, action_history

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

def save_results(returns, epsilons, wealth_history, action_history, episode, save_dir):
    """Save training results and generate visualizations."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    np.save(save_dir / 'returns.npy', returns)
    np.save(save_dir / 'epsilons.npy', epsilons)
    np.save(save_dir / 'wealth_history.npy', wealth_history)
    np.save(save_dir / 'action_history.npy', action_history)
    
    # # Create visualizations
    # fig = plt.figure(figsize=(15, 10))
    
    # # Plot returns with moving average and standard deviation
    # ax1 = plt.subplot(221)
    # window_size = min(100, len(returns))  # Adjust window size based on available data
    # returns_array = np.array(returns)
    
    # if len(returns_array) >= window_size:
    #     ma = np.convolve(returns_array, np.ones(window_size)/window_size, mode='valid')
    #     std = np.array([np.std(returns_array[max(0, i-window_size):i]) for i in range(window_size, len(returns_array)+1)])
    #     episodes = np.arange(window_size-1, len(returns_array))
        
    #     ax1.plot(episodes, ma, 'b-', label='Moving Average', linewidth=2)
    #     ax1.fill_between(episodes, ma - std, ma + std, color='b', alpha=0.2, label='Standard Deviation')
    # else:
    #     # If we don't have enough data for moving average, just plot the raw returns
    #     episodes = np.arange(len(returns_array))
    #     ax1.plot(episodes, returns_array, 'b-', label='Returns', linewidth=2)
    # ax1.set_xlabel('Episodes')
    # ax1.set_ylabel('Cumulative Return')
    # ax1.set_title('Training Return Curve')
    # ax1.grid(True)
    # ax1.legend()
    
    # # Plot epsilon
    # ax2 = plt.subplot(222)
    # ax2.plot(epsilons)
    # ax2.set_xlabel('Episodes')
    # ax2.set_ylabel('Exploration Rate')
    # ax2.set_title('Exploration Rate Curve')
    # ax2.grid(True)
    
    # # Plot wealth trajectories
    # ax3 = plt.subplot(223)
    # for wealth in wealth_history[-10:]:  # Plot last 10 episodes
    #     ax3.plot(wealth)
    # ax3.set_xlabel('Time Steps')
    # ax3.set_ylabel('Wealth')
    # ax3.set_title('Last 10 Episodes Wealth Trajectories')
    # ax3.grid(True)
    
    # # Plot actions for last 10 episodes
    # ax4 = plt.subplot(224)
    # last_10_actions = action_history[-10:]
    # # Plot each episode's actions with improved visualization
    # colors = plt.cm.viridis(np.linspace(0, 1, len(last_10_actions)))
    # for i, actions in enumerate(last_10_actions):
    #     time_steps = np.arange(len(actions))
    #     ax4.plot(time_steps, actions, color=colors[i], linewidth=1.5, alpha=0.8, 
    #              label=f'Episode {len(action_history)-10+i+1}')
    # ax4.set_xlabel('Time Steps')
    # ax4.set_ylabel('Investment Position')
    # ax4.set_title('Investment Positions Over Time (Last 10 Episodes)')
    # ax4.grid(True, alpha=0.3)
    # ax4.legend(bbox_to_anchor=(1.05, 1), loc='upper left', framealpha=0.9)

    
    # plt.tight_layout()
    # plt.savefig(save_dir / f'training_results_ep{episode}.png')
    # plt.close()

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
    
    agent = Agent(config)
    
    # Training parameters
    n_episodes = config.get('training', {}).get('n_episodes', 1000)
    
    # Train agent
    print('Starting training...')
    returns, epsilons, wealth_history, action_history = train(env, agent, n_episodes)
    
    # Save final results
    save_results(returns, epsilons, wealth_history, action_history, n_episodes, run_dir)
    
    # Save trained agent and config
    agent.save(str(run_dir / 'model.npy'))
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)
    
    print('Training completed. Results saved to:', run_dir)

if __name__ == '__main__':
    main()