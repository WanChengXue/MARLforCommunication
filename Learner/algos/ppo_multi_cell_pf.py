import torch
import torch.nn as nn

def get_cls():
    return PPOTrainer


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
class PPOTrainer:
    def __init__(self, net, optimizer, scheduler ,policy_config, local_rank):
        self.policy_config = policy_config
        self.rank = local_rank
        self.clip_epsilon = self.policy_config["clip_epsilon"]
        self.entropy_coef = self.policy_config['entropy_coef']
        # ----- 这个地方进行value loss的clip操作 ------
        self.clip_value = self.policy_config.get("clip_value", True)
        # ----- 这个值是最大的grad ------
        self.grad_clip = self.policy_config.get("max_grad_norm", None)
        # ------ 这个是双梯度剪裁 -------
        self.dual_clip = self.policy_config.get("dual_clip", None)
        self.using_critic = self.policy_config.get("using_critic", True)
        # -------- 下面两个值分别表示要不要开启popart算法，以及是不是用多个目标 -----
        self.popart_start = self.policy_config.get("popart_start", False)
        self.multi_objective_start = self.policy_config.get("multi_objective_start", False)     
        # ------------ 这个表示是不是critic和policy在一起 -------------
        self.seperate_critic = self.policy_config.get('seperate_critic', True)
        self.parameter_sharing = self.policy_config.get('parameter_sharing', True)
        self.agent_nums = self.policy_config['agent_nums']
        self.policy_net = net['policy']
        self.policy_optimizer = optimizer['policy']
        self.policy_scheduler = scheduler['policy']
        self.policy_name = list(self.policy_net.keys())
        ##########  增量式修改优势值的均值和方差  #########
        self.advantage_mean = 0
        self.advantage_std = 1
        self.M = 0
        ##############################################
        # ------- 如果使用了critic，无论采用中心化critic还是s
        if self.using_critic:
            if self.seperate_critic:
                # -------- 如果critic和policy网络在一起，则不需要单独的critic，seperate_critic为False -----
                self.critic_net = net['critic']
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
        for i in range(5):
            current_state = training_batch['current_state']
            # ================= 使用了GAE估计出来了advantage value  ============
            actions = training_batch['actions']
            old_action_log_probs = training_batch['old_action_log_probs']
            # -------------- 通过神经网络计算一下在当前网络下的状态值 --------------
            if self.multi_objective_start:
                predict_state_value_PF, predict_state_value_Edge = self.critic_net(current_state)
                predict_state_value = torch.cat([predict_state_value_PF, predict_state_value_Edge], 1)
                advantages = training_batch['advantages']
                # ----------------- 这个值是使用采样critic网络前向计算出出来的结果，主要用来做value clip ----------------
                old_network_value = training_batch['current_state_value'].squeeze(-1)
                target_state_value = training_batch['target_state_value'].squeeze(-1)
            else:
                if self.using_critic:
                    predict_state_value = self.critic_net[self.critic_name](current_state['global_state'])
                else:
                    action_log_probs, predict_state_value = self.policy_net[self.policy_name](current_state['global_state'])
                advantages = training_batch['advantages']
                old_network_value = training_batch['current_state_value']
                target_state_value = training_batch['target_state_value'] 
            # 这个地方使用value clip操作
            if self.clip_value:
                value_clamp_range = 0.2
                value_pred_clipped = old_network_value + (predict_state_value - old_network_value).clamp(-value_clamp_range, value_clamp_range)
                # ================= 由于这个value pred clipped的值是batch size * 2
                if self.popart_start:
                    # ============ TODO 由于使用了popart算法,因此这里需要对gae估计出来的target V值进行正则化,更新出它的均值和方差 ===============
                    self.critic_net.update(target_state_value)
                    # ============ 更新了均值和方差之后,需要对return value进行正则化,得到正则之后的v和Q值 =================
                    normalize_state_value = self.critic_net.normalize(target_state_value)
                    # ------------ 计算正则化之后的target value和神经网络的输出之间的loss -----------------
                    clipped_state_value_loss = self.critic_loss(normalize_state_value, value_pred_clipped)
                    unclipped_state_value_loss = self.critic_loss(normalize_state_value, predict_state_value)
                else:
                    # ------------- 如果不使用popart，就不需要更新popart网络了 -----------
                    clipped_state_value_loss = self.critic_loss(target_state_value, value_pred_clipped)
                    unclipped_state_value_loss = self.critic_loss(target_state_value, predict_state_value)
                value_loss_matrix = torch.max(clipped_state_value_loss, unclipped_state_value_loss)
                # ================= 将这个矩阵进行mean计算,得到两个head的loss    
            else:
                # ------------- 这个地方是说,如果不适用value clip操作 -------------
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
            # ================== 将这两个head的loss进行backward之后,更新这个网络的参数即可 ===================
            if self.multi_objective_start:
                state_value_loss_PF_head = mean_loss[0]
                state_value_loss_Edge_head = mean_loss[1]
                total_state_loss = state_value_loss_Edge_head + state_value_loss_PF_head
            else:
                total_state_loss = mean_loss
            if self.seperate_critic:
                self.critic_optimizer[self.critic_name].zero_grad()
                total_state_loss.backward()
                if self.grad_clip is not None:
                    info_dict['Critic_loss/grad'] = nn.utils.clip_grad_norm_(self.critic_net[self.critic_name].parameters(), self.grad_clip)
                for name, value in self.critic_net[self.critic_name].named_parameters():
                    info_dict['Critic_model_grad/Layer_max_grad_{}'.format(name)] = torch.max(value.grad).item()
                info_dict['Critic_loss/Value_loss'] = total_state_loss.item()
                info_dict['Critic_loss/Mse_loss'] = mse_loss.item()
                self.critic_optimizer[self.critic_name].step()
                self.critic_scheduler[self.critic_name].step()
            # ------------------ 训练critic ---------------------------
            for agent_index in range(self.agent_nums):
                if self.parameter_sharing:
                    agent_name = self.policy_name[0]
                else:
                    agent_name = self.policy_name[agent_index]
                advantage_std = torch.std(advantages[agent_name], 0)
                advantage_mean = torch.mean(advantages[agent_name], 0)
                N = advantages[agent_name].shape[0]
                # -----------  增量式的修改均值和方差 ------------
                new_advantage_mean = self.advantage_mean + N/(N+self.M) *(advantage_mean - self.advantage_mean)
                new_advantage_var = self.M*(self.advantage_std**2 +(self.advantage_mean-new_advantage_mean)**2) + N * (advantage_std**2+(new_advantage_mean -advantage_mean)**2)
                self.advantage_mean = new_advantage_mean
                self.advantage_std = torch.sqrt(new_advantage_var/(self.M+N))
                self.M += N
                # ---------------------------------------------
                # advantage = advantages / self.advantage_std
                advantage_mean = 0
                advantage = (advantages[agent_name] - advantage_mean) / advantage_std
                # entropy_loss_list = []
                self.policy_optimizer[agent_name].zero_grad()
                action_log_probs, conditional_entropy = self.policy_net[agent_name](current_state['agent_obs']['agent_'+str(agent_index)], actions['agent_'+str(agent_index)], False)
                importance_ratio = torch.exp(action_log_probs - old_action_log_probs[agent_name])
                surr1 = importance_ratio * advantage
                info_dict['Policy_loss_{}/Surr1'.format('agent_'+str(agent_index))] = surr1.mean().item()
                # ================== 这个地方采用PPO算法，进行clip操作 ======================
                surr2 = torch.clamp(importance_ratio, 1.0-self.clip_epsilon, 1.0+self.clip_epsilon) * advantage
                info_dict['Policy_loss_{}/Surr2'.format('agent_'+str(agent_index))] = surr2.mean().item()
                surr = torch.min(surr1, surr2)   
                info_dict['Policy_loss_{}/Surr_min_1_and_2'.format('agent_'+str(agent_index))] = surr.mean().item()     
                if self.dual_clip is not None:
                    c = self.dual_clip
                    surr3 = torch.min(c*advantage, torch.zeros_like(advantage))
                    surr = torch.max(surr, surr3)
                policy_loss = torch.mean(-surr)
                entropy_loss = torch.mean(conditional_entropy)
                # ================== 需要添加entropy的限制 =================
                # total_policy_loss = policy_loss - self.entropy_coef * entropy_loss
                total_policy_loss = policy_loss
                total_policy_loss.backward()
                info_dict['Policy_loss_{}/Policy_loss'.format('agent_'+str(agent_index))] = policy_loss.item()
                info_dict['Policy_loss_{}/Entropy_loss'.format('agent_'+str(agent_index))] = entropy_loss.item()
                info_dict['Policy_loss_{}/Total_policy_loss'.format('agent_'+str(agent_index))] = total_policy_loss.item()    
                info_dict['Policy_loss_{}/Advantage_mean'.format('agent_'+str(agent_index))] = self.advantage_mean.item()
                info_dict['Policy_loss_{}/Advantage_std'.format('agent_'+str(agent_index))] = self.advantage_std.item()
                if self.grad_clip is not None:
                    max_grad_norm = 10
                    policy_net_grad = nn.utils.clip_grad_norm_(self.policy_net[agent_name].parameters(), max_grad_norm)
                    info_dict['Policy_loss_{}/grad'.format('agent_'+str(agent_index))]= policy_net_grad.item()
                for name,value in self.policy_net[agent_name].named_parameters():
                    if value.required_grad:
                        info_dict['Policy_model_grad_{}/Layer_{}_max_grad'.format('agent_'+str(agent_index), name)] = torch.max(value).item()
                
                self.policy_optimizer[agent_name].step()
                self.policy_scheduler[agent_name].step()
                # ------------------ 这个地方开始用来更新策略网络的参数, 使用PPO算法, 把多个智能体的观测叠加到batch维度上 ----------------------------
        return info_dict    






