# 这个智能体,是说将动作空间变成连续的,采用DPG算法来更新结果.
import torch
import torch.optim as optim

from Model.multi_head_model import Actor_Critic
from Tool.replay_buffer import ReplayBuffer
# 这个智能体是将输入的状态矩阵进行拆分, 假设这里的观测就只有一个矩阵,则主矩阵作为智能体的输入,其余两个矩阵作为干扰信息
class Agent:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if self.args.cuda else "cpu"
        self.sector_number = self.args.sector_number
        self.user_number =  self.args.user_numbers
        self.bs_antennas = self.args.bs_antennas
        self.agent_number = self.args.n_agents
        self.parameter_sharing = self.args.parameter_sharing
        # ==== 定义策略网络,以及critic网络,以及tensorboard的相关路径等 ====
        if self.parameter_sharing:
            self.Replay_buffer = [ReplayBuffer(self.args) for _ in range(self.agent_number)]
            self.actor_critic = Actor_Critic(self.args, (1, args.obs_matrix_number, args.obs_dim1, args.obs_dim2)).to(self.device)
            self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=args.actor_lr)
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=50)
            print(self.actor_critic)
        else:
            self.Replay_buffer = ReplayBuffer(self.args)
            self.actor_critic = [Actor_Critic(self.args, (1, args.obs_matrix_number, args.obs_dim1, args.obs_dim2)).to(self.device) for _ in range(self.agent_number)]
            self.optimizer = [optim.Adam(self.actor_critic[agent_index].parameters(), lr=args.actor_lr) for agent_index in range(self.agent_number)]
            for agent_index in range(self.agent_number):
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer[agent_index], T_max=50)
            print(self.actor_critic[0])
        self.actor_loss_path = ["Policy_loss/Agent_" + str(agent_index)  for agent_index in range(self.agent_number)]
        self.critic_loss_path = ["Critic_loss/Agent_" + str(agent_index)  for agent_index in range(self.agent_number)]
        self.critic_loss = torch.nn.MSELoss()
        self.writer = self.args.writer
        # 定义两个变量，用来记录loss
        self.average_reward = "Agent/Average_reward"
        self.update_count = 0
        # ============================================================

        self.max_norm_grad = self.args.max_norm_grad
        # ==== 定义6个参数，分别表示的actor，critic的lr，lr decay，min lr =====
        self.critic_lr = self.args.critic_lr
        self.critic_lr_decay = self.args.critic_lr_decay
        self.actor_lr = self.args.actor_lr
        self.actor_lr_decay = self.args.actor_lr_decay
        self.actor_min_lr = self.args.actor_min_lr
        self.critic_min_lr = self.args.critic_min_lr
        # ================================================================

    def Pick_action_Max_SE_batch(self,state_list):
        Scheduling_sequence = []
        prob = []
        v_value_list = []
        for agent_index in range(self.agent_number):
            net_input = torch.FloatTensor(state_list[agent_index]).to(self.device) 
            if self.parameter_sharing:
                batch_prob, scheduling_user, v_value = self.actor_critic(net_input) 
            else:
                batch_prob, scheduling_user, v_value = self.actor_critic[agent_index](net_input) 
            Agent_scheduling_sequence = scheduling_user[:,1:].cpu().numpy()
            Scheduling_sequence.append(Agent_scheduling_sequence)
            prob.append(batch_prob)
            v_value_list.append(v_value)
        return Scheduling_sequence, prob, v_value_list

    def training(self, reward, prob, v_Value):
        # batch_data = torch.FloatTensor(batch_data).to(self.device).transpose(2,3).reshape(-1, self.sector_number **2, self.user_number, self.bs_antennas*2)
        reward = torch.FloatTensor(reward).to(self.device).unsqueeze(-1)
        self.writer.add_scalar(self.average_reward, torch.mean(reward).item(), self.update_count)
        for agent_index in range(self.agent_number):
            if self.parameter_sharing:
                if agent_index == 0:
                    self.optimizer.zero_grad()
                    v_value = v_Value[agent_index]
                    v_loss = self.critic_loss(v_value, reward)
                    v_loss.backward(retain_graph=True)
                    policy_loss = -torch.mean((reward-v_value.detach()) *prob[agent_index])
                    policy_loss.backward(retain_graph=True)

                elif agent_index == self.agent_number -1:
                    v_value = v_Value[agent_index]
                    v_loss = self.critic_loss(v_value, reward)
                    v_loss.backward(retain_graph=True)
                    policy_loss = -torch.mean((reward-v_value.detach()) *prob[agent_index])
                    policy_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_norm_grad)
                    self.optimizer.step()
                else:
                    v_value = v_Value[agent_index]
                    v_loss = self.critic_loss(v_value, reward)
                    v_loss.backward(retain_graph=True)
                    policy_loss = -torch.mean((reward-v_value.detach()) *prob[agent_index])
                    policy_loss.backward(retain_graph=True)
            else:
                # 如果不使用参数共享策略
                self.optimizer[agent_index].zero_grad()
                v_value = v_Value[agent_index]
                v_loss = self.critic_loss(v_value, reward)
                v_loss.backward(retain_graph=True)
                policy_loss = -torch.mean((reward-v_value.detach()) *prob[agent_index])
                policy_loss.backward()
                # 进行参数的clip操作,然后更新网络参数
                torch.nn.utils.clip_grad_norm_(self.actor_critic[agent_index].parameters(), self.max_norm_grad)
                self.optimizer[agent_index].step()
            self.writer.add_scalar(self.critic_loss_path[agent_index], v_loss.item(), self.update_count)
            self.writer.add_scalar(self.actor_loss_path[agent_index], policy_loss.item(), self.update_count)
        self.update_count += 1