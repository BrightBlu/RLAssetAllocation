import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
from environments.market_environment import MarketEnvironment
from agents.mc_agent import MCAgent

def load_config(config_path: str):
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def train(env: MarketEnvironment, agent: MCAgent, n_episodes: int):
    """Train the agent for specified number of episodes."""
    returns = []
    epsilons = []
    
    for episode in range(n_episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # Select and execute action
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            
            # Store transition and update agent
            agent.store_transition(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
        
        # Update agent and record metrics
        metrics = agent.update()
        returns.append(metrics.get('episode_return', 0))
        epsilons.append(metrics.get('epsilon', 0))
        
        # Print progress
        if (episode + 1) % 100 == 0:
            print(f'Episode {episode + 1}/{n_episodes}')
            print(f'Average Return (last 100): {np.mean(returns[-100:]):.2f}')
            print(f'Epsilon: {agent.epsilon:.3f}\n')
    
    return returns, epsilons

def get_next_run_dir(config_path):
    """Create and get the next run directory based on config name."""
    # Extract config name without extension
    config_name = Path(config_path).stem
    base_dir = Path('results') / config_name
    
    # Create base directory if it doesn't exist
    base_dir.mkdir(parents=True, exist_ok=True)
    
    # Find next available run number
    existing_runs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith(f'{config_name}_')]
    next_run = 1
    if existing_runs:
        last_run = max(int(d.name.split('_')[-1]) for d in existing_runs)
        next_run = last_run + 1
    
    # Create and return new run directory
    run_dir = base_dir / f'{config_name}_{next_run:03d}'
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def visualize_training(returns, epsilons, save_path):
    """Visualize training progress."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot returns
    ax1.plot(returns)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Return')
    ax1.set_title('Training Returns')
    ax1.grid(True)
    
    # Plot epsilon
    ax2.plot(epsilons)
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Epsilon')
    ax2.set_title('Exploration Rate')
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Asset Allocation RL Training')
    parser.add_argument('--config', type=str, default='configs/default.json',
                        help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create run directory
    run_dir = get_next_run_dir(args.config)
    print(f'Saving results to: {run_dir}')
    
    # Initialize environment and agent
    env = MarketEnvironment(config)
    if config.get('agent', {}).get('type', '') == 'mc':
        print('Using Monte Carlo Agent')
        agent = MCAgent(config)
    
    # Training parameters
    n_episodes = config.get('training', {}).get('n_episodes', 1000)
    
    # Train agent
    print('Starting training...')
    returns, epsilons = train(env, agent, n_episodes)
    
    # Save results
    plot_path = run_dir / 'training_results.png'
    model_path = run_dir / 'model.npy'
    config_path = run_dir / 'config.json'
    
    # Visualize results
    visualize_training(returns, epsilons, plot_path)
    
    # Save trained agent and config
    agent.save(str(model_path))
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    
    print('Training completed. Results saved to:', run_dir)

if __name__ == '__main__':
    main()