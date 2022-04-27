import torch
import os
import sys
current_path = os.path.abspath(__file__)
root_path = '/'.join(current_path.split('/')[:-2])
sys.path.append(root_path)
from Utils.model_utils import deserialize_model, create_model
from Utils import action_space

class Agent:
    # 定义一个采样智能体，它能够完成的事情有：加载模型，保存模型，发送数据，给定状态计算动作
    def __init__(self, policy_config):
        # ----------- 这个地方创建出来的net_work其实是一个策略网络 ------- deepcopy(self.policy_config['agent'][model_type][agent_name])
        self.net_work = create_model(policy_config)
        
    def synchronize_model(self, model_path):
        # ---------- 这个函数是用来同步本地模型的 ----------
        deserialize_model(self.net_work, model_path)

    def compute(self, agent_obs, action_list=None):
        # 这个是调用策略网络来进行计算的，网络使用的是Transformer结构，Encoder得到多个有意义的隐藏向量矩阵后，传入到Decoder，使用pointer network来进行解码操作
        with torch.no_grad():
            # 通过transformer进行了特征提取了之后，有两个head出来，一个是transformer decoder部分，得到调度列表，另外一个head出来的是v(s)的值，在推断阶段这个值不要
            # 因为还是遵循的CTDE的训练方式，每次决策之后，需要将所有智能体的backbone汇聚在一起进行V值的计算，由于action的长度有长有短，因此需要进行mask操作，统一到
            # 固定长度。如果是单个点进行决策，返回的log_probs表示的是联合概率的对数，action表示的是调度序列，mask表示的是固定长度的0-1向量
            # 按理来说，在eval mode的时候，每次决策都需要选择概率最大的动作的
            if action_list is not None:
                log_joint_prob = self.net_work(agent_obs, action_list)
                return log_joint_prob
            else:
                log_joint_prob, scheduling_action = self.net_work(agent_obs, action_list)
                return log_joint_prob, scheduling_action
            

    def compute_state_value(self, agent_obs):
        with torch.no_grad():
            state_value = self.net_work(agent_obs)
        return state_value


    def compute_action_and_state_value(self, agent_obs):
        with torch.no_grad():
            log_prob, action, state_value = self.net_work(agent_obs)
        return log_prob, action, state_value

class WolpAgent():
    def __init__(self, policy_config):
        self.net_work = create_model(policy_config)
        low_value = policy_config.get('action_low', 0)
        high_value = policy_config.get('action_high', 1)
        action_dim = policy_config.get('action_dim')
        self.action_space = action_space.Space(low_value, high_value, action_dim)
        self.k_nearest_point = policy_config.get('k_nearest_point', 3)
        self.selected_matrix = torch.LongTensor([[i for i in range(self.k_nearest_point)]])


    def add_critic_net(self, critic_net):
        self.critic_net = critic_net
        

    def compute(self, agent_obs):
        # ---------------- 使用了这个算法，默认更新算法为DDPG ------------
        with torch.no_grad():
            scheduling_action = self.net_work(agent_obs)
            return scheduling_action

        
    def compute_state_value(self, agent_obs, action):
        with torch.no_grad():
            state_value = self.critic_net(agent_obs, action)
        return state_value

    
    def search_action(self, agent_obs, action):
        # ------------ 传入的action的维度是batch_size * action_dim  ---------
        # ------------ 通过search_point函数，返回batch_size * k_nearest_point * action_dim 个点 -------
        k_nearest_action = self.action_space.search_point(action, self.k_nearest_point)
        # ------------- 把这个numpy数据变成Tensor --------------
        k_nearest_action_tensor = torch.FloatTensor(k_nearest_action)
        q_value_list = []
        for k in range(self.k_nearest_point):
            q_value_list.append(self.compute_state_value(agent_obs, k_nearest_action_tensor[:, k, :]))
        # ----------- 拼接，得到一个维度为batch_size * k 的q值矩阵 -----------
        concatenate_q_list = torch.cat(q_value_list, -1)
        # ----------- 使用argmax操作，获取最大的q值索引 --------------
        max_index = torch.argmax(concatenate_q_list, 1)
        batch_size = max_index.shape[0]
        selected_matrix = self.selected_matrix.repeat(batch_size, 1)
        pointer = selected_matrix == max_index.unsqueeze(-1)
        selected_action = k_nearest_action_tensor[pointer]
        return selected_action

    def synchronize_model(self, model_path):
        # ---------- 这个函数是用来同步本地模型的 ----------
        deserialize_model(self.net_work, model_path)