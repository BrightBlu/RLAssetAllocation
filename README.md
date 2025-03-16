# Reinforcement Learning Asset Allocation Project

## Environment Setup and Running Instructions

### 1. Python Environment Requirements
- Python 3.9 or higher version
- Recommended to use virtual environment for project dependencies management

### 2. Environment Setup Steps

1. Clone the project locally:
```bash
git clone [repository_url]
cd RLAssetAllocation
```

2. Create and activate virtual environment:

You can simply run install.bat to create virtual environment and install dependencies listed in requirements.txt
Note that you may need to adjust your python path in install.bat according to your OS environment setting.
Or you may use below commands:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment on Linux/Mac
source venv/bin/activate

# Activate virtual environment on Windows
venv\Scripts\activate
```

3. Install project dependencies:
```bash
venv\Scripts\pip install -r requirements.txt
```

### 3. Running Tests
Run all test cases:
```bash
venv\scripts\python -m pytest
```

Run specific test file:
```bash
venv\scripts\python -m pytest tests/test_environments/test_market_environment.py
```

### 4. Running Main Program
1. Check configuration file:
   - Set parameters in `configs/default.json`
   - Or create custom configuration file

2. Run training program:
```bash
# Run with default configuration
python main.py

# Run with custom configuration
python main.py --config configs/custom/my_config.json
```

## Project Structure

```
├── agents/          # Implementation of different TD algorithms
│   ├── q_learning/  # Q-learning algorithm
│   ├── sarsa/      # SARSA algorithm
│   └── dqn/        # Deep Q-Network algorithm
├── environments/    # Environment implementation
│   ├── base.py     # Base environment class
│   └── market.py   # Market environment implementation
├── configs/        # Configuration files
│   ├── default.json # Default configuration
│   └── custom/     # Custom configurations
├── tests/          # Unit tests
│   ├── test_environments/
│   └── test_agents/
├── utils/          # Utility functions
│   ├── data_processing.py
│   └── visualization.py
└── results/        # Experiment results
    ├── models/     # Model storage
    ├── logs/       # Training logs
    └── plots/      # Result visualization
```

## Main Components

1. **agents/**: Contains implementations of various TD algorithms
   - Q-learning
   - SARSA
   - DQN (Deep Q-Network)

2. **environments/**: Defines reinforcement learning environments
   - Asset market environment
   - State space definition
   - Action space definition
   - Reward function design

3. **configs/**: Configuration files in JSON format
   - Algorithm hyperparameters
   - Environment parameters
   - Training settings

4. **tests/**: Unit tests to ensure code quality
   - Environment tests
   - Algorithm tests

5. **utils/**: Auxiliary functions
   - Data processing
   - Result visualization

6. **results/**: Storage for experiment results
   - Trained models
   - Training logs
   - Performance charts

## Usage

1. Configure Environment:
   - Create or modify JSON configuration files in the `configs/` directory
   - Set algorithm parameters and environment parameters

2. Run Experiments:
   - Select algorithm and configuration
   - Execute training script
   - View results and logs

3. Evaluate Results:
   - Use visualization tools to analyze performance
   - Compare effects of different algorithms

