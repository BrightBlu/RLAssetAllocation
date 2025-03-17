# Reinforcement Learning Approach for Solving Asset Allocation Problem

This project implements and evaluates various Temporal Difference (TD) learning algorithms for asset allocation in financial markets. The project is now complete with multiple algorithms implemented and extensive experimental results.

## Authors
WU, Yuheng (yuheng.wu@connect.ust.hk) 21107083

FAN, Kwan Wai (kwfanaa@connect.ust.hk) 05037383

## Project Status
The repository is for the Assignment 1 of MAFS5370/MSBD6000M Spring 2025.

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
pip install -e .
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

3. View training results:
   - Training metrics, plots, and action histories are saved in `experiment/`
   - Results can be visualized using the provided visualization tools

4. Extract Q-Table and Policy:
   - After training, run the following command to extract Q-Table and Policy:

```bash
# Extract model results
python extract_results.py model.npy
```

5. Simulate Market Environment:
   To simulate market environment for checking, you can just go for the `environment_simulation.ipynb` file, there are some predefined examples.

## Project Structure

```
├── agents/                                   # RL algorithm implementations
│   ├── dqn_agent                             # Deep Q-Network
│   ├── mc_agent                              # Monte Carlo
│   ├── qlearning_agent                       # Q-Learning variants
│   ├── sarsa_agent                           # SARSA
│   └── td0_agent                             # TD(0)
├── configs/                                  # Algorithm configurations
│   └── [algorithm_specific_configs].json
├── environments/                             # Environment implementation
│   ├── __init__.py                           # Environment initialization
│   └── market_environment.py                 # Market environment
├── experiment/                               # Experimental results
│   └── experiment_*/                         # Results for each experiment
├── tests/                                    # Unit tests
│   └── test_agents/                          # Unit test cases for agents
│   └── test_environments/                    # Unit test cases for environments
├── utils/                                    # Utility functions
│   └── visualization.py                      # Result visualization
├── environment_simulation.ipynb              # To simulate market environment for checking
├── extract_results.py                        # To extract and analyze Q-table and policy information
└── train.py                                  # Main training program
```

## Main Components

1. **agents/**: Implements various TD algorithms with different features
   - Q-learning with wealth consideration
   - Q-learning with different weight schemes
   - SARSA implementation
   - DQN with neural network
   - Monte-Carlo learning
   - TD(0) implementation

2. **environments/**: Market environment implementation
   - State space: wealth and time
   - Action space: asset allocation decisions
   - Reward function: final return on investment with CARA

3. **configs/**: Algorithm-specific configurations
   - Learning rates and discount factors
   - Exploration parameters
   - Action Space and State Space settings
   - Instant Reward setting
   - Training episodes
   - Random seed

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
   - Results are saved in `results/`

3. Replicate Experiments:
   - Choos from given experiments in `experiment/`
   - Use config in `experiment/[choosen_experiment]/config.json`

3. Analyze Results:
   - Use visualization tools in `utils/`
   - Extrract Q-Table and Policy
   - Simulate market environment

## Experimental Results

The project includes extensive experiments results with following information:
- TD Error trajectories
- Reward trajectories
- Q-Table
- Final Policy
- Experiment configuration

Detailed results and analysis can be found in the `experiment/` directory.
