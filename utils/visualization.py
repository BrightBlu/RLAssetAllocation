import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict

def plot_training_metrics(metrics: Dict[str, List], save_path: str = None, clip_returns: bool = False):
    """Plot training metrics including TD errors and returns/utilities."""
    # Plot overall metrics
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot TD errors if available
    window_size = 100
    td_errors = metrics['td_errors']
    if len(td_errors) > 0 and not isinstance(td_errors[0], list):
        smoothed_td_errors = np.convolve(td_errors, np.ones(window_size)/window_size, mode='valid')
        ax1.plot(smoothed_td_errors, label='Smoothed TD Errors')
        ax1.set_title('TD Errors over Episodes')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('TD Error')
        ax1.grid(True)
        ax1.legend()
    else:
        ax1.text(0.5, 0.5, 'No TD Errors Available', horizontalalignment='center', verticalalignment='center')
        ax1.set_title('TD Errors (Not Available)')
        ax1.grid(False)
    
    # Plot returns or utilities
    if 'utilities' in metrics:
        utilities = metrics['utilities']
        smoothed_utilities = np.convolve(utilities, np.ones(window_size)/window_size, mode='valid')
        ax2.plot(smoothed_utilities, label='Smoothed Utilities')
        ax2.set_title('Utilities over Episodes')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Utility')
    else:
        if clip_returns:
            returns = np.clip(metrics['returns'], -200, 200)
        else:
            returns = metrics['returns']

        smoothed_returns = np.convolve(returns, np.ones(window_size)/window_size, mode='valid')
        episodes = np.arange(window_size-1, len(returns))
        std = np.array([np.std(returns[max(0, i-window_size):i]) for i in range(window_size, len(returns)+1)])
        
        ax2.plot(episodes, smoothed_returns, 'b-', label='Moving Average', linewidth=2)
        ax2.fill_between(episodes, smoothed_returns - std, smoothed_returns + std, color='b', alpha=0.2, label='Standard Deviation')
        ax2.set_title('Returns over Episodes')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Return')
    ax2.grid(True)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        fig1.savefig(save_path)
    else:
        plt.show()
    
    plt.close(fig1)