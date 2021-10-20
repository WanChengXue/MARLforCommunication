# 这个智能体,是说将动作空间变成连续的,采用DPG算法来更新结果.
import torch
import torch.optim as optim
from Model.CommNet_model import CommNet_Actor, Critic
from Tool.replay_buffer import ReplayBuffer

class Agent:
    def __init__(self, args):
        self.args = args
        self.device = "cuda" if self.args.cuda else "cpu"
        self.sector_number = self.args.sector_number
        self.user_number =  self.args.user_numbers
        self.bs_antennas = self.args.bs_antennas
        self.agent_number = self.args.n_agents
        self.parameter_sharing = self.args.parameter_sharing
        self.communication_turns = self.args.communication_turns
        
        self.actor = CommNet_Actor(self.args, (1, self.sector_number, self.user_number, self.bs_antennas*2)).to(self.device)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=args.actor_lr)
        print(self.actor)
        
        self.critic = Critic(self.args).to(self.device)
        print(self.critic)
        self.writer = self.args.writer
        # 定义两个变量，用来记录loss
        self.critic_loss_path = "Agent/Critic_loss"
        self.average_reward = "Agent/Average_reward"
        self.update_value_net_count = 0
        self.update_policy_net_count = 0
        # 定义replay_buffer
        if self.parameter_sharing:
            self.Replay_buffer = [ReplayBuffer(self.args) for _ in range(self.agent_number)]
            self.actor_loss_path = "Agent/Policy_loss"
        else:
            self.Replay_buffer = ReplayBuffer(self.args)
            self.actor_loss_path = ["Agent_" + str(index) + "/Policy_loss" for index in range(self.user_number)]
        self.max_norm_grad = self.args.max_norm_grad
        # ==== 定义6个参数，分别表示的actor，critic的lr，lr decay，min lr =====
        self.critic_lr = self.args.critic_lr
        self.critic_lr_decay = self.args.critic_lr_decay
        self.actor_lr = self.args.actor_lr
        self.actor_lr_decay = self.args.actor_lr_decay
        self.actor_min_lr = self.args.actor_min_lr
        self.critic_min_lr = self.args.critic_min_lr
        # ================================================================
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        # 定义critic网络的loss function
        self.critic_loss = torch.nn.MSELoss()
        self.batch_size = self.args.batch_size
        

    def Pick_action_Max_SE_batch(self,state_list):
        net_input = [torch.FloatTensor(state_list[agent_index]).to(self.device).transpose(1,2) for agent_index in range(self.agent_number)]
        batch_prob,  Scheduling_sequence= self.actor(net_input)
        return Scheduling_sequence, batch_prob

    def training(self, batch_data, reward, prob):
        batch_data = torch.FloatTensor(batch_data).to(self.device).transpose(2,3).reshape(self.batch_size, self.sector_number **2, self.user_number, self.bs_antennas*2)
        reward = torch.FloatTensor(reward).to(self.device).unsqueeze(-1)
        self.writer.add_scalar(self.average_reward, torch.mean(reward).item(), self.update_value_net_count)
        reward  = (reward-torch.mean(reward)) / (1e-6 + torch.max(reward-torch.mean(reward)))
        v_Value = self.critic(batch_data)
        v_loss =  self.critic_loss(v_Value, reward)
        self.optimizer_critic.zero_grad()
        v_loss.backward()
        self.optimizer_critic.step()
        # 更新policy网络, prob的维度是3×batch*1
        advantage_value = (reward-v_Value.detach()).unsqueeze(0).expand_as(prob)
        p_loss = -torch.mean((advantage_value * prob).reshape(-1,1))
        self.optimizer_actor.zero_grad()
        p_loss.backward()
        self.optimizer_actor.step()
        self.writer.add_scalar(self.actor_loss_path, p_loss.item(), self.update_policy_net_count)
        self.writer.add_scalar(self.critic_loss_path, v_loss.item(), self.update_value_net_count)
        self.update_policy_net_count += 1
        self.update_value_net_count += 1
        self.actor_lr = max(self.actor_min_lr, self.actor_lr * (1-self.actor_lr_decay))
        self.critic_lr = max(self.critic_min_lr, self.critic_lr * (1-self.critic_lr_decay))

