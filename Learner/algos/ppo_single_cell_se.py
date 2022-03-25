import torch
import torch.nn as nn

def get_cls():
    return MAPPOTrainer


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
class MAPPOTrainer:
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
        self.policy_net = net['policy']
        self.policy_optimizer = optimizer['policy']
        self.policy_scheduler = scheduler['policy']
        self.policy_name = list(self.policy_net.keys())[0]
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
        current_state = training_batch['state']
        # ================= 使用了GAE估计出来了advantage value  ============
        
        actions = training_batch['actions']
        old_action_log_probs = training_batch['old_action_log_probs']
        # -------------- 通过神经网络计算一下在当前网络下的状态值 --------------
        if self.multi_objective_start:
            predict_state_value_PF, predict_state_value_Edge = self.critic_net(current_state)
            predict_state_value = torch.cat([predict_state_value_PF, predict_state_value_Edge], 1)
            advantages = training_batch['instant_reward'] - training_batch['old_network_value']
            # ----------------- 这个值是使用采样critic网络前向计算出出来的结果，主要用来做value clip ----------------
            old_network_value = training_batch['old_network_value'].squeeze(-1)
            target_state_value = training_batch['target_state_value'].squeeze(-1)
        else:
            if self.using_critic:
                predict_state_value = self.critic_net[self.critic_name](current_state)
            else:
                action_log_probs, predict_state_value = self.policy_net[self.policy_name](current_state)
            advantages = training_batch['instant_reward'] - training_batch['old_network_value']
            old_network_value = training_batch['old_network_value']
            target_state_value = training_batch['instant_reward'] 
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
                nn.utils.clip_grad_norm_(self.critic_net[self.critic_name].parameters(), self.grad_clip)
            self.critic_optimizer[self.critic_name].step()
            self.critic_scheduler[self.critic_name].step()
            advantage_std = torch.std(advantages, 0)
            advantage_mean = torch.mean(advantages, 0)
            advantage = (advantages - advantage_mean) / advantage_std
            # entropy_loss_list = []
            self.policy_optimizer[self.policy_name].zero_grad()
            action_log_probs = self.policy_net[self.policy_name](current_state, actions, False)
            importance_ratio = torch.exp(action_log_probs - old_action_log_probs)
            surr1 = importance_ratio * advantage
            # ================== 这个地方采用PPO算法，进行clip操作 ======================
            surr2 = torch.clamp(importance_ratio, 1.0-self.clip_epsilon, 1.0+self.clip_epsilon) * advantage
            surr = torch.min(surr1, surr2)        
            if self.dual_clip is not None:
                c = self.dual_clip
                surr3 = torch.min(c*advantage, torch.zeros_like(advantage))
                surr = torch.max(surr, surr3)
            policy_loss = torch.mean(-surr)
            # ================== 需要添加entropy的限制 =================
            total_policy_loss = policy_loss
            total_policy_loss.backward()    
            if self.grad_clip is not None:
                max_grad_norm = 10
                nn.utils.clip_grad_norm_(self.policy_net[self.policy_name].parameters(), max_grad_norm)
            self.policy_optimizer[self.policy_name].step()
            self.policy_scheduler[self.policy_name].step()
            
        else:
            # ------------ TODO 这个地方是critic和actor连在一起的时候使用，loss backward的时候需要保存计算图 --------------
            pass 
        # ------------------ 这个地方开始用来更新策略网络的参数, 使用PPO算法, 把多个智能体的观测叠加到batch维度上 ----------------------------
        return {
            'value_loss': total_state_loss.item(),
            # 'conditional_entropy': entropy_loss.item(),
            'advantage_std': advantage_std.cpu().numpy().tolist(),
            'advantage_mean': advantage_mean.cpu().numpy().tolist(),
            'policy_loss': policy_loss.item(),
            'total_policy_loss': total_policy_loss.item()
            }






