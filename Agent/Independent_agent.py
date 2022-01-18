# 这个智能体,是说将动作空间变成连续的,采用DPG算法来更新结果.
import torch
import torch.optim as optim
from torch.optim import optimizer
from Model.Single_cell_model_se import Actor, Critic
# from Model.model_SE import Actor, Critic
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
        # ==== 定义策略网络,以及critic网络,以及tensorboard的相关路径等 ====
        if self.parameter_sharing:
            self.Replay_buffer = [ReplayBuffer(self.args) for _ in range(self.agent_nums)]
            self.actor = Actor(self.args).to(self.device)
            self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer_actor, T_max=50)
            print("============ policy network 的网络结构为: ==========")
            print(self.actor)
            self.critic = Critic(self.args, (1, args.state_dim1, args.obs_dim2)).to(self.device)
            self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=args.critic_lr)
            torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer_critic, T_max=50)
            print("============ value network 的网络结构为: ===========")
            print(self.critic)
        else:
            self.Replay_buffer = ReplayBuffer(self.args)
            self.actor = [Actor(self.args).to(self.device) for _ in range(self.agent_nums)]
            self.optimizer_actor = [optim.Adam(self.actor[agent_index].parameters(), lr=args.actor_lr) for agent_index in range(self.agent_nums)]
            self.critic = [Critic(self.args, (1, args.state_dim1, args.obs_dim2)).to(self.device) for _ in range(self.agent_nums)]
            self.optimizer_critic = [optim.Adam(self.critic[agent_index].parameters(), lr=args.critic_lr) for agent_index in range(self.agent_nums)]
            for agent_index in range(self.agent_nums):
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer_critic[agent_index], T_max=50)
                torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer_actor[agent_index], T_max=50)
            print("============ policy network 的网络结构为: ==========")
            print(self.actor[0])
            print("============ value network 的网络结构为: ===========")
            print(self.critic[0])
        self.actor_loss_path = ["Policy_loss/Agent_" + str(agent_index)  for agent_index in range(self.agent_nums)]
        self.critic_loss_path = ["Critic_loss/Agent_" + str(agent_index)  for agent_index in range(self.agent_nums)]
        # 定义critic网络的loss function
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
        for agent_index in range(self.agent_nums):
            net_input = torch.FloatTensor(state_list[agent_index]).to(self.device)
            if self.parameter_sharing:
                batch_prob, scheduling_user = self.actor(net_input) 
            else:
                batch_prob, scheduling_user = self.actor[agent_index](net_input) 
            Agent_scheduling_sequence = scheduling_user[:,1:].cpu().numpy()
            Scheduling_sequence.append(Agent_scheduling_sequence)
            prob.append(batch_prob)
        return Scheduling_sequence, prob

    def training(self, data, reward, prob):
        # 传入的batch data是一个列表,长度为agent number, 每一个元素,其维度是batch size * user number * bs antennas
        # prob也是一个列表, 长度为agent number, reward是一个batch size的列向量.
        reward = torch.FloatTensor(reward).to(self.device).unsqueeze(-1)
        self.writer.add_scalar(self.average_reward, torch.mean(reward).item(), self.update_count)
        reward  = (reward - torch.mean(reward))/ (1e-6 + torch.max(reward))
        for agent_index in range(self.agent_nums):
            batch_data = torch.FloatTensor(data[:, agent_index, :, agent_index, :]).to(self.device)
            if self.parameter_sharing:
                if agent_index == 0:
                    self.optimizer_critic.zero_grad()
                    self.optimizer_actor.zero_grad()
                    v_value = self.critic(batch_data)
                    v_loss = self.critic_loss(v_value, reward)
                    v_loss.backward(retain_graph=True)
                    policy_loss = -torch.mean((reward-v_value.detach()) *prob[agent_index])
                    policy_loss.backward(retain_graph=True)
                elif agent_index == self.agent_nums -1:
                    v_value = self.critic(batch_data)
                    v_loss = self.critic_loss(v_value, reward)
                    v_loss.backward()
                    policy_loss = -torch.mean((reward-v_value.detach()) *prob[agent_index])
                    policy_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_norm_grad)
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_norm_grad)
                    self.optimizer_actor.step()
                    self.optimizer_critic.step()
                else:
                    v_value = self.critic(batch_data)
                    v_loss = self.critic_loss(v_value, reward)
                    v_loss.backward()
                    policy_loss = -torch.mean((reward-v_value.detach()) *prob[agent_index])
                    policy_loss.backward()
            else:
                # 如果不使用参数共享策略
                self.optimizer_actor[agent_index].zero_grad()
                self.optimizer_critic[agent_index].zero_grad()
                v_value = self.critic[agent_index](batch_data)
                v_loss = self.critic_loss(v_value, reward)
                v_loss.backward()
                policy_loss = -torch.mean((reward-v_value.detach()) *prob[agent_index])
                policy_loss.backward()
                # 进行参数的clip操作,然后更新网络参数
                torch.nn.utils.clip_grad_norm_(self.actor[agent_index].parameters(), self.max_norm_grad)
                torch.nn.utils.clip_grad_norm_(self.critic[agent_index].parameters(), self.max_norm_grad)
                self.optimizer_actor[agent_index].step()
                self.optimizer_critic[agent_index].step()
            self.writer.add_scalar(self.critic_loss_path[agent_index], v_loss.item(), self.update_count)
            self.writer.add_scalar(self.actor_loss_path[agent_index], policy_loss.item(), self.update_count)
        self.update_count += 1
        
