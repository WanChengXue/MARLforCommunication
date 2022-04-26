import torch
import torch.nn as nn
from algos.utils import soft_update, Discrete_space

def get_cls():
    return DDPGTrainer


# clip_epsilon: 0.2
# max_grad_norm: 10
# dual_clip: 3
# entropy_coef: 0.01
# agent_nums: *agent_nums
'''
net = {
    'policy_net': {
        default: Net
    },
    'critic_net':{
        default: Net
    }
}
'''
class DDPGTrainer:
    def __init__(self, net, target_net, optimizer, scheduler ,policy_config):
        self.policy_config = policy_config
        # ----- 这个地方进行value loss的clip操作 ------
        self.clip_value = self.policy_config.get("clip_value", True)
        # ----- 这个值是最大的grad ------
        self.grad_clip = self.policy_config.get("max_grad_norm", None)
        # -------- 下面两个值分别表示要不要开启popart算法，以及是不是用多个目标 -----
        self.popart_start = self.policy_config.get("popart_start", False)
        self.multi_objective_start = self.policy_config.get("multi_objective_start", False)     
        self.policy_net = net['policy']
        self.target_policy_net = target_net['policy']
        self.policy_optimizer = optimizer['policy']
        self.policy_scheduler = scheduler['policy']
        self.policy_name = list(self.policy_net.keys())[0]
        ##########  增量式修改优势值的均值和方差  #########
        self.advantage_mean = 0
        self.advantage_std = 1
        self.M = 0
        ##############################################
        self.critic_net = net['critic']
        self.target_critic_net = target_net['critic']
        self.critic_optimizer = optimizer['critic']
        self.critic_scheduler = scheduler['critic']
        self.critic_loss = self.huber_loss
        self.critic_name = list(self.critic_net.keys())[0]

    def huber_loss(self, a, b, delta=1.0):
        gap = a - b
        flag_matrix = (torch.abs(gap) <= delta).float()
        mse_loss = 0.5 * gap ** 2
        other_branch = delta * (torch.abs(gap) - 0.5 * delta)
        return flag_matrix * mse_loss + (1-flag_matrix) * other_branch


    def mse_loss(self, a,b):
        return  0.5 * (a-b) **2


    '''
    training_batch = {
        'state': {
            'real_part': R^{bs*user_nums*bs_antennas}
            'img_part': R^{bs*user_nums*bs_antennas}
        },
        'actions': R^{bs*user_nums},
        'old_action_log_probs': R^{bs*1},
        'old_network_value': R^{bs*1},
        'instant_reward': R^{bs*1}
    }

    '''


    def step(self, training_batch):
        info_dict = dict() 
        policy_dict = dict()
        critic_dict = dict()
        current_state = training_batch['state']
        actions = training_batch['actions']
        next_state = training_batch['next_state']
        done = training_batch['done']
        instant_reward = training_batch['instant_reward']
        # -------------- 通过神经网络计算一下在当前网络下的状态值 --------------
        if self.multi_objective_start:
            predict_state_value_PF, predict_state_value_Edge = self.critic_net(current_state)
            predict_state_value = torch.cat([predict_state_value_PF, predict_state_value_Edge], 1)
            # ----------------- 这个值是使用采样critic网络前向计算出出来的结果，主要用来做value clip ----------------
            old_network_value = training_batch['current_state_value'].squeeze(-1)
            target_state_value = training_batch['target_state_value'].squeeze(-1)
        else:
            # --------- 使用target网络计算next state的value -----------
            with torch.no_grad():
                next_action = self.target_policy_net[self.policy_name](next_state)
                next_q_state_value = self.target_critic_net[self.critic_name](next_state, next_action)
                target_state_value = instant_reward + next_q_state_value * done
            predict_state_value = self.critic_net[self.critic_name](current_state, actions)
            if self.popart_start:
                # ============ 由于使用了popart算法,因此这里需要对gae估计出来的target V值进行正则化,更新出它的均值和方差 ===============
                self.critic_net.update(target_state_value)
                # ============ 更新了均值和方差之后,需要对return value进行正则化,得到正则之后的v和Q值 =================
                normalize_state_value = self.critic_net.normalize(target_state_value)
                value_loss_matrix = self.critic_loss(normalize_state_value, predict_state_value)
            else:
                value_loss_matrix = self.critic_loss(target_state_value, predict_state_value)

        mean_loss = torch.mean(value_loss_matrix, 0)
        mse_loss = torch.mean(0.5*(predict_state_value-target_state_value)**2)
        total_state_loss = mean_loss

        if self.seperate_critic:
            self.critic_optimizer[self.critic_name].zero_grad()
            total_state_loss.backward()
            if self.grad_clip is not None:
                critic_dict['grad'] = nn.utils.clip_grad_norm_(self.critic_net[self.critic_name].parameters(), self.grad_clip)
            for name, value in self.critic_net[self.critic_name].named_parameters():
                critic_dict['Layer_max_grad_{}'.format(name)] = torch.max(value.grad).item()
            critic_dict['Value_loss'] = total_state_loss.item()
            critic_dict['Mse_loss'] = mse_loss.item()
            self.critic_optimizer[self.critic_name].step()
            self.critic_scheduler[self.critic_name].step()
            info_dict['Critic_loss'] = critic_dict
        # -------------- 策略网络进行更新 -------------
        current_action_value = self.policy_net[self.policy_name](current_state)
        q_value = self.policy_net[self.policy_name](current_state, current_action_value)
        policy_loss = -torch.mean(q_value)
        self.policy_optimizer[self.policy_name].zero_grad()
        policy_loss.backward()
        policy_dict['Policy_loss'] = policy_loss.item()
        policy_dict['Q_value_std'] = torch.std(q_value).item()
        if self.grad_clip is not None:
            max_grad_norm = 10
            policy_net_grad = nn.utils.clip_grad_norm_(self.policy_net[self.policy_name].parameters(), max_grad_norm)
            policy_dict['grad']= policy_net_grad.item()
        for name,value in self.policy_net[self.policy_name].named_parameters():
            policy_dict['Layer_{}_max_grad'.format(name)] = torch.max(value).item()
        info_dict['Policy_loss'] = policy_dict
        self.policy_optimizer[self.policy_name].step()
        self.policy_scheduler[self.policy_name].step()
        # ------------ 使用soft update更新target网络 --------

        return info_dict






