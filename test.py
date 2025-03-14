import numpy as np
import pandas as pd

# 加载保存的Q表数据
data = np.load('results/qlearning/qlearning_010/model.npy', allow_pickle=True).item()

# 分析每个状态的最优策略
state_optimal_actions = {}
for state_key in data['q_table'].keys():
    wealth_idx, time_step = state_key
    q_values_dict = data['q_table'][state_key]
    
    # 处理空Q值字典的情况
    if not q_values_dict:
        # 对于空字典，使用默认值
        state_optimal_actions[state_key] = {
            'wealth_idx': wealth_idx,
            'time_step': time_step,
            'sign': 1,  # 默认为正
            'log_value': 0,  # 默认为0
            'max_q_value': 0.0,
            'is_default': True  # 标记为默认值
        }
        continue
    
    # 找出最优动作
    optimal_action = max(q_values_dict.items(), key=lambda x: x[1])
    # 保持离散动作的原始形式
    log_value = optimal_action[0]
    state_optimal_actions[state_key] = {
        'wealth_idx': wealth_idx,
        'time_step': time_step,
        'log_value': log_value,
        'max_q_value': optimal_action[1],
        'is_default': False  # 标记为非默认值
    }

# 将结果转换为DataFrame以便查看
optimal_actions_df = pd.DataFrame.from_dict(state_optimal_actions, orient='index')
optimal_actions_df = optimal_actions_df.sort_values(['time_step', 'wealth_idx'])
print("最优策略表：")
print(optimal_actions_df)
optimal_actions_df.to_csv('optimal_actions.csv')