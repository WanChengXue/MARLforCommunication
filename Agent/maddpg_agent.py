# 这个智能体,是说将动作空间变成连续的,采用DPG算法来更新结果.
import torch
import numpy as np
import torch.optim as optim
from torch.distributions import Categorical

from Model.maddpg_model import Actor, Critic
from Tool.replay_buffer import ReplayBuffer

class Agent:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if self.args.cuda else "cpu"
        self.sector_number = self.args.sector_number
        self.user_number =  self.args.user_numbers
        self.bs_antennas = self.args.bs_antennas
        self.agent_nums = self.args.n_agents
        self.parameter_sharing = self.args.parameter_sharing

        # ==== 定义6个参数，分别表示的actor，critic的lr，lr decay，min lr =====
        self.critic_lr = self.args.critic_lr
        self.critic_lr_decay = self.args.critic_lr_decay
        self.actor_lr = self.args.actor_lr
        self.actor_lr_decay = self.args.actor_lr_decay
        self.actor_min_lr = self.args.actor_min_lr
        self.critic_min_lr = self.args.critic_min_lr
        # ================================================================

        # ==== 定义策略网络,以及critic网络,以及tensorboard的相关路径等 ====
        if self.parameter_sharing:
            self.Replay_buffer = [ReplayBuffer(self.args) for _ in range(self.agent_nums)]
            self.actor = Actor(self.args, (1, args.obs_matrix_number, args.obs_dim1, args.obs_dim2)).to(self.device)
            self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
            print(self.actor)
        else:
            self.Replay_buffer = ReplayBuffer(self.args)
            self.actor = [Actor(self.args, (1, args.obs_matrix_number, args.obs_dim1, args.obs_dim2)).to(self.device) for _ in range(self.agent_nums)]
            self.optimizer_actor = [optim.Adam(self.actor[agent_index].parameters(), lr=args.actor_lr) for agent_index in range(self.agent_nums)]
            print(self.actor[0])
        self.actor_loss_path = ["Policy_loss/Agent_" + str(agent_index)  for agent_index in range(self.user_number)]
        self.critic = Critic(self.args, (1, args.total_state_matrix_number, args.state_dim1, args.obs_dim2)).to(self.device)
        print(self.critic)
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        # 定义critic网络的loss function
        self.critic_loss = torch.nn.MSELoss()
        self.writer = self.args.writer
        # 定义两个变量，用来记录loss
        self.critic_loss_path = "Agent/Critic_loss"
        self.average_reward = "Agent/Average_reward"
        self.update_value_net_count = 0
        self.update_policy_net_count = 0
        # ============================================================

        self.max_norm_grad = self.args.max_norm_grad

        

    def Pick_action_Max_SE_batch(self,state_list):
        Scheduling_sequence = []
        action = []
        for agent_index in range(self.agent_nums):
            net_input = torch.FloatTensor(state_list[agent_index]).to(self.device).transpose(1,2)
            with torch.no_grad():
                if self.parameter_sharing:
                    batch_agent_prob = self.actor(net_input).unsqueeze(-1)
                else:
                    batch_agent_prob = self.actor[agent_index](net_input).unsqueeze(-1)
            # 采样,得到对应的调度结果
            batch_prob_matrix = torch.cat([1-batch_agent_prob, batch_agent_prob], -1)
            dist = Categorical(batch_prob_matrix)
            agent_scheduling_sequence = dist.sample() 
            Scheduling_sequence.append(agent_scheduling_sequence.cpu().numpy())
            action.append(batch_agent_prob.cpu().numpy())
        return Scheduling_sequence, action

    def training(self, batch_data, reward, action):
        critic_input = torch.FloatTensor(batch_data).to(self.device).transpose(2,3).reshape(-1, self.sector_number **2, self.user_number, self.bs_antennas*2)
        # 这个action的维度是batch size * agent number * action dim
        action = torch.FloatTensor(np.stack(action, axis=1)).to(self.device)
        reward = torch.FloatTensor(reward).to(self.device).unsqueeze(-1)
        self.writer.add_scalar(self.average_reward, torch.mean(reward).item(), self.update_value_net_count)
        reward  = (reward-torch.mean(reward)) / torch.max(reward-torch.mean(reward))
        # === 更新critic神经网络的参数 ====
        v_Value = self.critic(critic_input, action)
        v_loss = self.critic_loss(v_Value, reward)
        self.optimizer_critic.zero_grad()
        v_loss.backward()
        self.optimizer_critic.step()
        self.writer.add_scalar(self.critic_loss_path, v_loss.item(), self.update_value_net_count)
        self.update_value_net_count += 1
        self.critic_lr = max(self.critic_min_lr, self.critic_lr * (1-self.critic_lr_decay))

        # === 更新actor神经网络的参数 ====
        for agent_index in range(self.agent_nums):
            net_input = torch.FloatTensor(batch_data[:,agent_index,:,:]).to(self.device).transpose(1,2)
            if self.parameter_sharing:
                agent_action = self.actor(net_input).unsqueeze(-1)
            else:
                agent_action = self.actor[agent_index](net_input).unsqueeze(-1)
            # 组合成网络的input
            whole_action = action.clone()
            whole_action[:,agent_index,:,:] = agent_action
            agent_actor_loss = -torch.mean(self.critic(critic_input, whole_action))
            self.writer.add_scalar(self.actor_loss_path[agent_index], agent_actor_loss.item(), self.update_policy_net_count)
            if self.parameter_sharing:
                self.optimizer_critic.zero_grad()
                if agent_index == 0:
                    self.optimizer_actor.zero_grad()
                    agent_actor_loss.backward(retain_graph=True)

                elif agent_index == self.agent_nums - 1:
                    agent_actor_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_norm_grad)
                    self.optimizer_actor.step()
                else:
                    agent_actor_loss.backward(retain_graph=True)
                
            else:
                self.optimizer_critic.zero_grad()
                self.optimizer_actor[agent_index].zero_grad()
                agent_actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor[agent_index].parameters(), self.max_norm_grad)
                self.optimizer_actor[agent_index].step()
        self.update_policy_net_count += 1
        self.actor_lr = max(self.actor_min_lr, self.actor_lr * (1-self.actor_lr_decay))
        

