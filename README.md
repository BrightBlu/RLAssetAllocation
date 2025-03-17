# Reinforcement Learning Asset Allocation Project

This project implements and evaluates various Temporal Difference (TD) learning algorithms for asset allocation in financial markets. The project is now complete with multiple algorithms implemented and extensive experimental results.

## Project Status

All planned algorithms have been implemented and evaluated:
- Q-Learning with different variants (wealth-based, constant/exponential weights)
- SARSA
- Deep Q-Network (DQN)
- Monte Carlo
- TD(0)

Extensive experiments have been conducted to evaluate and compare the performance of different algorithms under various market conditions.

## Environment Setup and Running Instructions

### 1. Python Environment Requirements
- Python 3.9 or higher
- Conda package manager

### 2. Environment Setup Steps

1. Clone the project locally:
```bash
git clone [repository_url]
cd RLAssetAllocation
```

2. Create and activate Conda environment:
```bash
# Create new Conda environment
conda create -n rl_asset python=3.9

# Activate environment
conda activate rl_asset
```

3. Install project dependencies:
```bash
conda install --file requirements.txt
```

### 3. Running Tests
Run all test cases:
```bash
python -m pytest
```

Run specific test file:
```bash
python -m pytest tests/test_environments/test_market_environment.py
```

### 4. Running Main Program
1. Check configuration file:
   - Multiple configuration files are available in `configs/` for different algorithms
   - Each config file contains specific hyperparameters and environment settings

2. Run training program:
```bash
# Run with specific configuration
python train.py --config configs/qlearning_with_wealth_exp_weight_no_sign.json
```

## Project Structure

```
├── agents/                # TD algorithm implementations
│   ├── dqn_agent          # Deep Q-Network
│   ├── mc_agent           # Monte Carlo
│   ├── qlearning_agent    # Q-Learning variants
│   ├── sarsa_agent        # SARSA
│   └── td0_agent          # TD(0)
├── environments/          # Environment implementation
│   ├── __init__.py        # Environment initialization
│   └── market_environment.py # Market environment
├── configs/               # Algorithm configurations
│   └── [algorithm_specific_configs].json
├── experiment/           # Experimental results
│   └── experiment_*/     # Results for each experiment
├── tests/               # Unit tests
└── utils/               # Utility functions
    └── visualization.py # Result visualization
```

## Main Components

1. **agents/**: Implements various TD algorithms with different features
   - Q-learning with wealth consideration
   - Q-learning with different weight schemes
   - SARSA implementation
   - DQN with neural network
   - Monte Carlo learning
   - TD(0) implementation

2. **environments/**: Market environment implementation
   - State space: market prices, positions, wealth
   - Action space: asset allocation decisions
   - Reward function: return on investment

3. **configs/**: Algorithm-specific configurations
   - Learning rates and discount factors
   - Exploration parameters
   - Neural network architectures (for DQN)

4. **experiment/**: Comprehensive experimental results
   - Training metrics
   - Performance plots
   - Action histories
   - Wealth trajectories

## Usage

1. Select Algorithm and Configuration:
   - Choose from implemented algorithms in `agents/`
   - Use corresponding config file from `configs/`
   - Adjust hyperparameters if needed

2. Run Training:
   - Execute training script with chosen config
   - Monitor training progress
   - Results are saved in `experiment/`

3. Analyze Results:
   - Use visualization tools in `utils/`
   - Compare performance across algorithms
   - Examine wealth growth and action patterns

## Experimental Results

The project includes extensive experiments comparing different algorithms:
- Performance comparison under various market conditions
- Analysis of wealth growth trajectories
- Evaluation of different weight schemes
- Impact of wealth consideration on decision making
- Comparison of learning stability across algorithms

Detailed results and analysis can be found in the `experiment/` directory.

