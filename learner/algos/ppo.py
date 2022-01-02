import torch
import torch.nn as nn
from torch.distributions import Categorical

class MAPPOTrainer:
    def __init__(self, net, optimizer, policy_config):
        self.policy_config = policy_config
        self.clip_epsilon = self.policy_config["clip_epsilon"]
        self.entropy_coef = self.policy_config['entropy_coef']
        self.clip_value = self.policy_config.get("clip_value", True)
        self.grad_clip = self.policy_config.get("grad_clip", None)
        self.dual_clip = self.policy_config.get("dual_clip", None)
        self.parameter_sharing = self.policy_config.get("parameter_sharing", False)
        self.network = net
        self.optimizer = optimizer
        self.critic_loss = self.huber_loss()
    

    def huber_loss(self, a, b, delta=1.0):
        gap = a - b
        flag_matrix = torch.abs(gap) <= delta
        mse_loss = 0.5 * gap ** 2
        other_branch = delta * (torch.abs(gap) - 0.5 * delta)
        return flag_matrix * mse_loss + (1-flag_matrix) * other_branch

    def mse_loss(self, a,b):
        return  0.5 * (a-b) **2



    def step(self, training_batch):
        '''
        传入的数据格式，并且所有的value都放到了gpu上面 
            training_batch['obs'] = concatenate_obs_dict
            training_batch['global_state'] = concatenae_global_dict
            training_batch['action_list'] = concatenate_action_list
            training_batch['old_action_log_probs'] = concatenate_old_action_log_probs
            training_batch['advantage'] = concatenate_advantage
            training_batch['target_value'] = concatenate_target_value
            training_batch['old_state_value'] = concatenate_old_state_value
            training_batch['return_valuel] = concatenate_return_value
        '''
        obs = training_batch['obs']
        global_state = training_batch['global_state']
        action_list = training_batch['action_list']
        old_action_log_probs = training_batch['old_action_log_probs']
        advantage = training_batch['advantage']
        target_value = training_batch['target_value']
        old_state_value = training_batch['old_state_value']
        return_value = training_batch['return_value']
        # 通过神经网络计算一下在当前网络下的状态值
        predict_state_value = self.network.cirtic(global_state)
        # ============ 由于使用了popart算法,因此这里需要对采样batch return进行正则化,更新出它的均值和方差 ===============
        self.network.update(return_value)
        # ============ 更新了均值和方差之后,需要对return value进行正则化,得到正则之后的v和Q值 =================
        normalize_state_value = self.network.normalize(return_value)
        # 这个地方使用value clip操作
        if self.clip_value:
            value_clamp_range = 0.2
            value_pred_clipped = old_state_value + (predict_state_value - old_state_value).clamp(-value_clamp_range, value_clamp_range)
            # ================= 由于这个value pred clipped的值是batch size * 2
            clipped_state_value_loss = self.critic_loss(normalize_state_value, value_pred_clipped)
            unclipped_state_value_loss = self.critic_loss(normalize_state_value, predict_state_value)
            value_loss_matrix = torch.max(clipped_state_value_loss, unclipped_state_value_loss)
            # ================= 将这个矩阵进行mean计算,得到两个head的loss
            
        else:
            # 这个地方是说,如果不适用value clip操作
            value_loss_matrix = self.critic_loss(normalize_state_value, predict_state_value)
        
        mean_loss = torch.mean(value_loss_matrix, 0)
        # ================== 将这两个head的loss进行backward之后,更新这个网络的参数即可 ===================
        state_value_loss_PF_head = mean_loss[0]
        state_value_loss_Edge_head = mean_loss[1]
        total_state_loss = state_value_loss_Edge_head + state_value_loss_PF_head
        total_state_loss.backward()
        # ------------------ 这个地方开始用来更新策略网络的参数, 使用PPO算法, 把多个智能体的观测叠加到batch维度上 ----------------------------
        action_log_probs, conditionla_entropy = self.network.actor(obs, action_list, False)
        importance_ratio = torch.exp(action_log_probs - old_action_log_probs)
        






