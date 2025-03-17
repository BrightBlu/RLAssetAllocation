import numpy as np
import pandas as pd
import argparse

def analyze_qtable(data):
    # Analyze optimal policy for each state
    state_optimal_actions = {}
    for state_key in data['q_table'].keys():
        wealth_idx, time_step = state_key
        q_values_dict = data['q_table'][state_key]
        
        # Handle case of empty Q-value dictionary
        if not q_values_dict:
            # Use default values for empty dictionary
            state_optimal_actions[state_key] = {
                'wealth_idx': wealth_idx,
                'time_step': time_step,
                'log_value': 0,  # Default value 0
                'max_q_value': 0.0,
                'is_default': True  # Mark as default value
            }
            continue
        
        # Find optimal action
        optimal_action = max(q_values_dict.items(), key=lambda x: x[1])
        # Keep discrete action in original form
        log_value = optimal_action[0]
        state_optimal_actions[state_key] = {
            'wealth_idx': wealth_idx,
            'time_step': time_step,
            'log_value': log_value,
            'max_q_value': optimal_action[1],
            'is_default': False  # Mark as non-default value
        }
    return state_optimal_actions

def main():
    # Set command line arguments
    parser = argparse.ArgumentParser(description='Extract and analyze Q-table and policy information from reinforcement learning model')
    parser.add_argument('model_path', type=str, help='Path to model.npy file')
    args = parser.parse_args()

    # Load model data
    try:
        data = np.load(args.model_path, allow_pickle=True).item()
    except Exception as e:
        print(f"Error loading model file: {e}")
        return

    # Analyze Q-table
    state_optimal_actions = analyze_qtable(data)

    # Convert results to DataFrame for viewing
    optimal_actions_df = pd.DataFrame.from_dict(state_optimal_actions, orient='index')
    optimal_actions_df = optimal_actions_df.sort_values(['time_step', 'wealth_idx'])

    print("\n=== Q-table Analysis Results ===")
    print("Optimal Policy Table:")
    print(optimal_actions_df)


if __name__ == '__main__':
    main()
    