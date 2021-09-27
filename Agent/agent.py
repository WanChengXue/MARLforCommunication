from typing import Sequence
import torch
from torch._C import Value
import torch.optim as optim
from torch.nn.parameter import Parameter

# from priority_model import Actor, Critic
# from model import Actor, Critic
from Model.model_SE import Actor, Critic
from Tool.replay_buffer import ReplayBuffer

class Agent:
    def __init__(self, args, index):
        self.args = args
        self.device = "cuda" if self.args.cuda else "cpu"
        # 定义一个通用的智能体出来
        # self.actor = Transformer(self.args).to(self.device)
        self.actor = Actor(self.args).to(self.device)
        self.critic = Critic(self.args).to(self.device)
        self.index = index
        self.parameter_sharing = self.args.parameter_sharing
        self.writer = self.args.writer
        self.agent_number = self.args.n_agents
        # 定义两个变量，用来记录loss
        # self.actor_path = "Agent" + "_" + str(self.index)  + "/Q_value"
        self.actor_loss_path = "Agent_" + str(self.index) + "/Actor_loss"
        self.critic_loss_path = "Agent_" + str(self.index) + "/Critic_loss"
        self.update_value_net_count = 0
        self.update_policy_net_count = 0
        # 定义replay_buffer
        if self.parameter_sharing:
            self.Replay_buffer = [ReplayBuffer(self.args) for _ in range(self.agent_number)]
        else:
            self.Replay_buffer = ReplayBuffer(self.args)
        self.epsilon = self.args.epsilon
        self.Training = self.args.Training
        # 定义一些必要的参数
        self.gamma = self.args.gamma
        self.GAE_factor = self.args.GAE_factor
        self.max_norm_grad = self.args.max_norm_grad
        # 定义learning rate, learning rate decay
        self.critic_lr = self.args.critic_lr
        self.critic_lr_decay = self.args.critic_lr_decay
        self.actor_lr = self.args.actor_lr
        self.actor_lr_decay = self.args.actor_lr_decay
        self.actor_min_lr = self.args.actor_min_lr
        self.critic_min_lr = self.args.critic_min_lr
        self.optimizer_critic = optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        # 定义critic网络的loss function
        self.critic_loss = torch.nn.MSELoss()
        self.eps = 1e-12
        self.batch_size = self.args.batch_size
        self.sector_number = self.args.sector_number
        self.user_number =  self.args.user_numbers
        self.bs_antennas = self.args.bs_antennas

        

    def Pick_action_Max_SE(self, state):
        # 将网络的输入变成1*3*16*32
        net_input = torch.FloatTensor(state).to(self.device).transpose(0,1).unsqueeze(0)
        with torch.no_grad():
            scheduling_result, _ ,pad_mask = self.actor(net_input)    
        scheduling_number = torch.sum(pad_mask != 0)
        Scheduling_sequence = (scheduling_result.squeeze(0).cpu().numpy()-1).tolist()[:scheduling_number-1]
        return Scheduling_sequence

    def Pick_action_Max_SE_batch(self,state):
        net_input = torch.FloatTensor(state).to(self.device).transpose(1,2)
        # with torch.no_grad():
        _, batch_prob, pad_mask, scheduling_user = self.actor(net_input)
        # 首先将某些概率变成0，然后进行列叠加，得到联合概率
        prob = torch.sum(pad_mask * batch_prob, -1)
        Scheduling_sequence = scheduling_user[:,1:].cpu().numpy()
        # Scheduling_sequence = (scheduling_result.squeeze(0).cpu().numpy()-1).tolist()[:schedulig_number-1]
        return Scheduling_sequence, prob

    def Pick_action(self, state, rank_priority):
        # 这个函数用来决策动作，state是一个tuple，其中包括信道矩阵H，以及user的average reward向量
        channel_matrix = state[0]
        user_average_reward = state[1]
        channel_matrix = torch.FloatTensor(channel_matrix).to(self.device)
        user_average_reward = torch.FloatTensor(user_average_reward).to(self.device)
        modify_rank = torch.FloatTensor(rank_priority).to(self.device)
        # user_average_reward = torch.FloatTensor(user_average_reward).unsqueeze(1).to(self.device
        action, prob = self.actor(channel_matrix.unsqueeze(0), user_average_reward.unsqueeze(0))
        return action, prob

    def training_parameter_sharing(self, batch_data, reward, prob, release_graph, zero_grad):
        batch_data = torch.FloatTensor(batch_data).to(self.device).transpose(2,3).reshape(self.batch_size, self.sector_number **2, self.user_number, self.bs_antennas*2)
        reward = torch.FloatTensor(reward).to(self.device).unsqueeze(-1)
        reward  = reward / (1e-6 + torch.max(reward))
        v_Value = self.critic(batch_data)
        v_loss =  self.critic_loss(v_Value, reward)
        # 更新policy网络
        p_loss = -torch.mean((reward - v_Value.detach()) * prob.unsqueeze(-1))
        if zero_grad:   attention_model
            p_loss.backward()
        else:
            p_loss.backward(retain_graph=True)

        if zero_grad:
            self.optimizer_critic.zero_grad()
            
        if release_graph:
            v_loss.backward()
            self.optimizer_critic.step()
            self.optimizer_actor.step()
            self.writer.add_scalar(self.actor_loss_path, p_loss.item(), self.update_policy_net_count)
            self.writer.add_scalar(self.critic_loss_path, v_loss.item(), self.update_value_net_count)
            self.update_policy_net_count += 1
            self.update_value_net_count += 1
            self.actor_lr = max(self.actor_min_lr, self.actor_lr * (1-self.actor_lr_decay))
            self.critic_lr = max(self.critic_min_lr, self.critic_lr * (1-self.critic_lr_decay))
        

    def training(self, batch_data, reward, prob):
        batch_data = torch.FloatTensor(batch_data).to(self.device).transpose(2,3).reshape(self.batch_size, self.sector_number **2, self.user_number, self.bs_antennas*2)
        reward = torch.FloatTensor(reward).to(self.device).unsqueeze(-1)
        reward  = reward / (1e-6 + torch.max(reward))
        v_Value = self.critic(batch_data)
        v_loss =  self.critic_loss(v_Value, reward)
        self.optimizer_critic.zero_grad()
        v_loss.backward()
        self.optimizer_critic.step()
        # 更新policy网络
        p_loss = -torch.mean((reward - v_Value.detach()) * prob.unsqueeze(-1))
        self.optimizer_actor.zero_grad()
        p_loss.backward()
        self.optimizer_actor.step()
        self.writer.add_scalar(self.actor_loss_path, p_loss.item(), self.update_policy_net_count)
        self.writer.add_scalar(self.critic_loss_path, v_loss.item(), self.update_value_net_count)
        self.update_policy_net_count += 1
        self.update_value_net_count += 1
        self.actor_lr = max(self.actor_min_lr, self.actor_lr * (1-self.actor_lr_decay))
        self.critic_lr = max(self.critic_min_lr, self.critic_lr * (1-self.critic_lr_decay))


    def Learning(self, agent_index=None):
        # first smaple trajectory from replay buffer
        if self.parameter_sharing:
            Trajectories = self.Replay_buffer[agent_index].sample()
        else:
            Trajectories = self.Replay_buffer.sample()

        Channel = Trajectories['Channel']
        Average_reward = Trajectories['Average_reward']
        Global_channel = Trajectories['Global_channel']
        Global_reward = Trajectories['Global_reward']
        Action = Trajectories['Action']
        Pad = Trajectories['Pad']
        Instant_reward = Trajectories['instant_reward']
        Terminate = Trajectories['terminate']
        Probs =Trajectories['prob']
        Probs = torch.stack([torch.stack(Probs[i],0) for i in range(self.args.episodes)], 0).reshape(-1, 1)
        # Transfer variable to tensor 
        Channel = torch.FloatTensor(Channel).to(self.device).reshape(-1, self.args.obs_matrix_number, self.args.obs_dim1, self.args.obs_dim2)
        Average_reward = torch.FloatTensor(Average_reward).to(self.device).reshape(-1, self.args.state_dim1)
        Global_channel = torch.FloatTensor(Global_channel).to(self.device).reshape(-1, self.args.state_matrix_number, self.args.state_dim1, self.args.state_dim2)
        Global_reward = torch.FloatTensor(Global_reward).to(self.device).reshape(-1, self.args.n_agents, self.args.state_dim1)
        Instant_reward = torch.FloatTensor(Instant_reward).to(self.device).reshape(-1,1)
        Terminate = torch.FloatTensor(Terminate).to(self.device).reshape(-1)
        Pad = torch.FloatTensor(Pad).to(self.device).reshape(-1, self.args.state_dim1+1)
        # Probs = torch.FloatTensor(Probs).to(self.device).reshape(-1)
        Action = torch.LongTensor(Action).to(self.device).reshape(-1,  self.args.state_dim1+1)
        # reshape all tensor
        V_value = self.critic(Global_channel, Global_reward).detach()
        advantages = Instant_reward - V_value
        advantages = (advantages-torch.mean(advantages)) / torch.std(advantages) 
        if self.parameter_sharing:
            self.Update_critic_parameter_sharing(Global_channel, Global_reward ,Instant_reward, agent_index)
            policy_net_loss = -torch.mean(Probs * advantages)
            if agent_index == 0:
                self.optimizer_actor.zero_grad()
                self.policy_net_loss_value = policy_net_loss.item()
            self.policy_net_loss_value += policy_net_loss.item()
            if agent_index == self.agent_number -1:
                self.update_policy_net_count += 1
                policy_net_loss.backward()
                self.optimizer_actor.step()
                self.writer.add_scalar(self.actor_loss_path, self.policy_net_loss_value/self.agent_number, self.update_policy_net_count)
                self.actor_lr = (1-self.actor_lr_decay) * self.actor_lr
                self.critic_lr = (1-self.critic_lr_decay) * self.critic_lr
                # self.Soft_update()
                for agent_index in range(self.agent_number):
                    self.Replay_buffer[agent_index].reset_buffer()
            else:
                policy_net_loss.backward(retain_graph=True)
        else:
            self.Update_critic_alter(Global_channel, Global_reward, Instant_reward)
            # Update policy net
            self.update_policy_net_count += 1
            policy_net_loss = -torch.mean(Probs * advantages)
            # self.PPO_update_policy(Channel, Action, Average_reward, Probs, Pad, advantages)
            self.optimizer_actor.zero_grad()
            policy_net_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_norm_grad)
            self.optimizer_actor.step()
            self.writer.add_scalar(self.actor_loss_path, policy_net_loss.item(), self.update_policy_net_count)

            self.actor_lr = (1-self.actor_lr_decay) * self.actor_lr
            self.critic_lr = (1-self.critic_lr_decay) * self.critic_lr
            # self.Soft_update()
            self.Replay_buffer.reset_buffer()

    def Update_critic_parameter_sharing(self, Global_channel, Global_reward ,targets, agent_index):
        approximate_value = self.critic(Global_channel, Global_reward)
        Value_loss = self.critic_loss(targets, approximate_value)
        if agent_index == 0:
            self.optimizer_critic.zero_grad()
            self.critic_loss_value = Value_loss.item()
        self.critic_loss_value += Value_loss.item()
        if agent_index == self.agent_number - 1:
            self.update_value_net_count+= 1
            Value_loss.backward()
            self.optimizer_critic.step()
            self.writer.add_scalar(self.critic_loss_path, self.critic_loss_value/self.agent_number, self.update_value_net_count)
        
        else:
            Value_loss.backward(retain_graph=True)

    def Update_critic_alter(self, Global_channel, Global_reward, targets):
        for _ in range(self.args.update_times):
            approximate_value = self.critic(Global_channel, Global_reward)
            value_loss = self.critic_loss(targets, approximate_value)
            self.optimizer_critic.zero_grad()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_norm_grad)
            self.optimizer_critic.step()
            self.update_value_net_count += 1
            self.writer.add_scalar(self.critic_loss_path, value_loss.item(), self.update_value_net_count)

    def Store_transition(self, transition):
        self.Replay_buffer.store_episode(transition)

    def Stor_transition_parameter_sharing(self, transition, cell_index):
        self.Replay_buffer[cell_index].store_episode(transition)