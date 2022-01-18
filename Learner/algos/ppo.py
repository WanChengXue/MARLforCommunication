import torch
import torch.nn as nn

def get_cls():
    return MAPPOTrainer


class MAPPOTrainer:
    def __init__(self, net, optimizer, policy_config, parameter_sharing):
        self.policy_config = policy_config
        self.clip_epsilon = self.policy_config["clip_epsilon"]
        self.entropy_coef = self.policy_config['entropy_coef']
        self.clip_value = self.policy_config.get("clip_value", True)
        self.grad_clip = self.policy_config.get("grad_clip", None)
        self.dual_clip = self.policy_config.get("dual_clip", None)
        self.popart_start = self.policy_config.get("popart_start", False)
        self.agent_nums = self.policy_config['agent_nums']
        self.parameter_sharing = parameter_sharing
        self.policy_net = net['policy_net']
        self.critic_net = net['critic_net']
        self.optimizer_critic = optimizer['critic']
        self.optimizer_actor = optimizer['actor']
        self.critic_loss = self.huber_loss
    

    def huber_loss(self, a, b, delta=1.0):
        gap = a - b
        flag_matrix = torch.abs(gap) <= delta
        mse_loss = 0.5 * gap ** 2
        other_branch = delta * (torch.abs(gap) - 0.5 * delta)
        return flag_matrix * mse_loss + (1-flag_matrix) * other_branch


    def mse_loss(self, a,b):
        return  0.5 * (a-b) **2


    def step(self, training_batch):
        current_state = training_batch['current_state']
        # ================= 使用了GAE估计出来了advantage value  ============
        advantages = training_batch['advantages']
        done = training_batch['done']
        # ----------------- 这个值是使用采样critic网络前向计算出出来的结果，主要用来做value clip ----------------
        old_state_value = training_batch['current_state_value']
        target_state_value = training_batch['target_state_value']
        actions = training_batch['actions']
        old_action_log_probs = training_batch['old_action_log_probs']
        # -------------- 通过神经网络计算一下在当前网络下的状态值 --------------
        predict_state_value = self.critic_net(current_state['global_state'])
        # 这个地方使用value clip操作
        if self.clip_value:
            value_clamp_range = 0.2
            value_pred_clipped = old_state_value + (predict_state_value - old_state_value).clamp(-value_clamp_range, value_clamp_range)
            # ================= 由于这个value pred clipped的值是batch size * 2
            if self.popart_start:
                # ============ 由于使用了popart算法,因此这里需要对gae估计出来的target V值进行正则化,更新出它的均值和方差 ===============
                self.critic_net.update(target_state_value)
                # ============ 更新了均值和方差之后,需要对return value进行正则化,得到正则之后的v和Q值 =================
                normalize_state_value = self.critic_net.normalize(target_state_value)
                # ------------ 计算正则化之后的target value和神经网络的输出之间的loss -----------------
                clipped_state_value_loss = self.critic_loss(normalize_state_value, value_pred_clipped)
                unclipped_state_value_loss = self.critic_loss(normalize_state_value, predict_state_value)
            else:
                # ------------- 如果不使用popart，就不需要更新popart网络了 -----------
                clipped_state_value_loss = self.critic_loss(target_state_value, value_pred_clipped)
                unclipped_state_value_loss = self.critic_net(target_state_value, predict_state_value)
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
        state_value_loss_PF_head = mean_loss[0]
        state_value_loss_Edge_head = mean_loss[1]
        total_state_loss = state_value_loss_Edge_head + state_value_loss_PF_head
        self.optimizer_critic.zero_grad()
        total_state_loss.backward()
        if self.grad_clip is not None:
            max_grad_norm = 10
            nn.utils.clip_grad_norm_(self.critic_net.parameters(), max_grad_norm)
        self.optimizer_critic.step()
        # ------------------ 这个地方开始用来更新策略网络的参数, 使用PPO算法, 把多个智能体的观测叠加到batch维度上 ----------------------------
        advantage_std = torch.std(advantages, 0)
        advantage = torch.sum(advantages / advantage_std, 1).unsqueeze(-1)
        if self.parameter_sharing:
            policy_loss = 0
            entropy_loss = 0
            policy_surr = 0
            self.optimizer_actor.zeros_grad()
            for index in range(self.agent_nums):
                agent_key = 'agent_' + str(index)
                agent_action_log_probs, agent_conditionla_entropy = self.policy_net(current_state['agent_obs'][agent_key], actions[agent_key], False)
                importance_ratio = torch.exp(agent_action_log_probs - old_action_log_probs)
                surr1 = importance_ratio * advantage
                # ================== 这个地方采用PPO算法，进行clip操作 ======================
                surr2 = torch.clamp(importance_ratio, 1.0-self.clip_epsilon, 1.0+self.clip_epsilon) * advantage
                surr = torch.min(surr1, surr2)        
                if self.dual_clip is not None:
                    c = self.dual_clip
                    surr3 = torch.min(c*advantage, torch.zeros_like(advantage))
                    surr = torch.max(surr, surr3)
                # ================== 这个advantage是一个矩阵，需要进行额外处理一下 ==============
                policy_loss = - torch.mean(surr) + policy_loss
                # ================== entropy loss =====================
                entropy_loss = torch.mean(agent_conditionla_entropy) + entropy_loss
                policy_surr = policy_loss + entropy_loss * self.entropy_coef + policy_surr
            policy_surr.backward()    
            if self.grad_clip is not None:
                max_grad_norm = 10
                nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_grad_norm)
            self.optimizer_actor.step()

            return {
                'value_loss': total_state_loss.item(),
                'conditional_entropy': entropy_loss.item(),
                'advantage_std': advantage_std.cpu().numpy(),
                'policy_loss': policy_loss.item(),
                'total_policy_loss': policy_surr.item()
            }
        else:
            raise NotImplementedError()





