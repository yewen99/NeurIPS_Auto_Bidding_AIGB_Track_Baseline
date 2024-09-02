from torch.utils.data import Dataset
import pandas as pd
import ast
import numpy as np
import torch
import os



class aigb_dataset(Dataset):
    def __init__(self, step_len, load_preprocessed_tain_data = True, R_min=0., R_max=1000., A_min=0, A_max=30, C_max=12, C_min=6, **kwargs) -> None:
        super().__init__()
        
        if load_preprocessed_tain_data:
            # load without cpa
            # states, actions, rewards, terminals = load_local_preprocessed_data_nips(train_data_path='/home/yewen001/CODE/ks/aigb/NeurIPS_Auto_Bidding_AIGB_Track_Baseline/data/preprocessed_train_data/train_data_all.npy')
            # load with cpa 
            # if not os.path.exists('/data/preprocessed_train_data/train_dict_s_a_r_t_c.npy'):
            #     preprocess_data_nips(train_data_dir='data/trajectory')
            states, actions, rewards, terminals, cpas = load_local_preprocessed_data_nips(train_data_path='/home/yewen001/CODE/ks/aigb/NeurIPS_Auto_Bidding_AIGB_Track_Baseline/data/preprocessed_train_data/train_dict_s_a_r_t_c.npy')
        else:
            states, actions, rewards, terminals, cpas = load_local_data_nips(
                train_data_path="data/trajectory/trajectory_data.csv")
            
        # self.states = states
        # process the state to AIGB simplified version
        # 1) original 16 dim version state
        # test_state = np.array([
        #     time_left, budget_left, historical_bid_mean, last_three_bid_mean,
        #     historical_LeastWinningCost_mean, historical_pValues_mean, historical_conversion_mean,
        #     historical_xi_mean, last_three_LeastWinningCost_mean, last_three_pValues_mean,
        #     last_three_conversion_mean, last_three_xi_mean,
        #     current_pValues_mean, current_pv_num, last_three_pv_num_total,
        #     historical_pv_num_total
        # ])
        # 2) simplify test state to reproduce the result of AIGB
        # budget_spend_speed = ((self.budget - self.remaining_budget)/self.budget) / (1 - time_left)
        # realtime_cost_efficiency = last_three_LeastWinningCost_mean / last_three_conversion_mean
        # avg_cost_efficiency = historical_LeastWinningCost_mean / historical_conversion_mean
        # test_state = np.array([time_left, budget_left, budget_spend_speed, realtime_cost_efficiency, avg_cost_efficiency])
        self.states = np.zeros((states.shape[0], 5))
        self.states[:,0] = states[:, 0]  # time left
        self.states[:,1] = states[:, 1]  # budget left
        eps = 1e-10  # to avoid NaN
        self.states[:,2] = (1 - states[:, 1]) / ((1 - states[:, 0])+ eps)  # budget_spend_speed
        self.states[:,3] = states[:, 8] / (states[:, 10] + eps)  # realtime_cost_efficiency
        self.states[:,4] = states[:, 4] / (states[:, 6] + eps)  # avg_cost_efficiency

        self.actions = actions  # action max:531 min:0.0
        self.rewards = rewards
        self.terminals = terminals
        self.cpas = cpas

        self.step_len = step_len
        self.num_of_states = states.shape[1]
        self.num_of_actions = actions.shape[1]

        # 分割序列
        # 每个序列的开头
        self.candidate_pos = (self.terminals == 0).nonzero()[0]
        self.candidate_pos += 1
        self.candidate_pos = [0] + self.candidate_pos.tolist()[:-1]
        # 后面再加上序列的结尾
        self.candidate_pos = self.candidate_pos + [self.states.shape[0]]

        
        self.R_min, self.R_max = R_min, R_max
        self.A_min, self.A_max = A_min, A_max
        self.C_max, self.C_min = C_max, C_min
        # all_R = []
        # for index in range(len(self.candidate_pos)-1):
        #     reward = torch.tensor(self.rewards[self.candidate_pos[index]:self.candidate_pos[index + 1], :],
        #                     dtype=torch.float32)
        #     cur_R = reward.sum()
        #     all_R.append(cur_R)
        #     if self.R_min > cur_R:
        #         self.R_min = cur_R
        #     if self.R_max < cur_R:
        #         self.R_max = cur_R
        # --> max=1557  min=0



    def __len__(self):
        return len(self.candidate_pos) - 1

    def __getitem__(self, index):
        # 获取序列
        state = torch.tensor(self.states[self.candidate_pos[index]:self.candidate_pos[index + 1], :],
                             dtype=torch.float32)
        action = torch.tensor(self.actions[self.candidate_pos[index]:self.candidate_pos[index + 1], :],
                              dtype=torch.float32)

        # clamp切断action使得训练稳定
        action = torch.clamp(action, max=self.A_max)

        reward = torch.tensor(self.rewards[self.candidate_pos[index]:self.candidate_pos[index + 1], :],
                              dtype=torch.float32)
        # action = action - 1
        # 当前序列的长度
        len_state = len(state)
        # 进行padding
        state = torch.nn.functional.pad(state, (0, 0, 0, self.step_len - len(state)), "constant", 0)
        action = torch.nn.functional.pad(action, (0, 0, 0, self.step_len - len(action)), "constant", 0)
        # 计算并归一化returns
        # 使用sigmoid会使得大部分(如一个1024的batch中，有977为1)return全为1，这导致该condition无意义
        # returns = reward.sum().sigmoid()
        returns = (reward.sum() - self.R_min)/(self.R_max - self.R_min)
        returns = torch.clamp(returns, max=1.0).reshape(1)

        cpas = (torch.tensor(self.cpas[self.candidate_pos[index]:self.candidate_pos[index + 1], :], dtype=torch.float32)).mean().reshape(1)
        cpas = (cpas - self.C_min) / (self.C_max - self.C_min)
        returns = torch.cat([returns, cpas])

        # 计算masks
        masks = torch.zeros(self.step_len)
        masks[:len_state] = 1
        masks = masks.bool()
        # 返回
        return state, action, returns, masks


# 加载本地数据
def load_local_data(data_version):
    states = pd.read_csv("simulation_platform/data/offline_trajectory/" + data_version + "/states.csv").values[:,
             0::]
    actions = pd.read_csv("simulation_platform/data/offline_trajectory/" + data_version + "/actions.csv").values[:,
              0::]
    rewards = pd.read_csv("simulation_platform/data/offline_trajectory/" + data_version + "/rewards.csv").values[:,
              0::]
    terminals = pd.read_csv("simulation_platform/data/offline_trajectory/" + data_version + "/terminal.csv").values[
                :,
                0::]
    return states, actions, rewards, terminals


def load_local_data_nips(train_data_path="data/traffic/training_data_rlData_folder/training_data_all-rlData.csv"):
    training_data = pd.read_csv(train_data_path)

    def safe_literal_eval(val):
        if pd.isna(val):
            return val  # 如果是NaN，返回NaN
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            print(ValueError)
            return val  # 如果解析出错，返回原值

    # 使用apply方法应用上述函数
    training_data["state"] = training_data["state"].apply(safe_literal_eval)
    training_data["terminal"] = training_data["timeStepIndex"] != 47
    training_data["terminal"] = training_data["terminal"].astype(int)
    
    states = np.array(training_data['state'].tolist())
    actions = training_data["action"].to_numpy().reshape(-1, 1)
    cpas = training_data["CPAConstraint"].to_numpy().reshape(-1, 1)
    rewards = training_data["reward"].to_numpy().reshape(-1, 1)
    terminals = training_data["terminal"].to_numpy().reshape(-1, 1)
    return states, actions, rewards, terminals, cpas

# def preprocessed_train_data(data_dir='./data/trajectory'):
#     for 
#     return preprocessed_train_data_file


def load_local_preprocessed_data_nips(train_data_path='./data/preprocessed_train_data/train_dict_s_a_r_t_c.npy'):
    training_data = np.load(train_data_path, allow_pickle=True).item()
    return training_data["states"], training_data["actions"], training_data["rewards"], training_data["terminals"], training_data["cpas"]
    
def preprocess_data_nips(train_data_dir='data/trajectory'):
    train_data = {"states":[], "actions": [], "rewards": [], "terminals": [], "cpas": []}
    for file in ['trajectory_data.csv', 'trajectory_data_extended_1.csv', 'trajectory_data_extended_2.csv']:
        s, a, r, t, c = load_local_data_nips(
                    train_data_path=os.path.join(train_data_dir, file))
        train_data["states"].append(s)
        train_data["actions"].append(a)
        train_data["rewards"].append(r)
        train_data["terminals"].append(t)
        train_data["cpas"].append(c)
    train_data["states"] = np.concatenate(train_data["states"], axis=0)
    train_data["actions"] = np.concatenate(train_data["actions"], axis=0)
    train_data["rewards"] = np.concatenate(train_data["rewards"], axis=0)
    train_data["terminals"] = np.concatenate(train_data["terminals"], axis=0)
    train_data["cpas"] = np.concatenate(train_data["cpas"], axis=0)
    np.save('data/preprocessed_train_data/train_dict_s_a_r_t_c.npy', train_data)
    