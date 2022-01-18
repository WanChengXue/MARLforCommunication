import torch
import torch.optim as optim
from Model.attention_model import Policy, Critic
from Tool.replay_buffer import ReplayBuffer

# 这个地方的使用transofrmer作为encoder，具体实现按照原本论文来
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
            self.actor = Policy(self.args, (1, args.obs_matrix_number, args.obs_dim1, args.obs_dim2)).to(self.device)
            self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
            print(self.actor)
        else:
            self.Replay_buffer = ReplayBuffer(self.args)
            self.actor = [Policy(self.args, (1, args.obs_matrix_number, args.obs_dim1, args.obs_dim2)).to(self.device) for _ in range(self.agent_nums)]
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
        prob = []
        for agent_index in range(self.agent_nums):
            net_input = torch.FloatTensor(state_list[agent_index]).to(self.device).transpose(1,2)
            if self.parameter_sharing:
                _, batch_prob, pad_mask, scheduling_user = self.actor(net_input) 
            else:
                _, batch_prob, pad_mask, scheduling_user = self.actor[agent_index](net_input) 
            Agent_prob = torch.sum(pad_mask * batch_prob, -1)
            Agent_scheduling_sequence = scheduling_user[:,1:].cpu().numpy()
            Scheduling_sequence.append(Agent_scheduling_sequence)
            prob.append(Agent_prob)
        return Scheduling_sequence, prob


    def training(self, batch_data, reward, prob):
        batch_data = torch.FloatTensor(batch_data).to(self.device).transpose(2,3).reshape(-1, self.sector_number **2, self.user_number, self.bs_antennas*2)
        reward = torch.FloatTensor(reward).to(self.device).unsqueeze(-1)
        self.writer.add_scalar(self.average_reward, torch.mean(reward).item(), self.update_value_net_count)
        reward  = reward/ (1e-6 + torch.max(reward))
        v_Value = self.critic(batch_data)
        v_loss =  self.critic_loss(v_Value, reward)
        self.optimizer_critic.zero_grad()
        v_loss.backward()
        self.optimizer_critic.step()
        self.update_value_net_count += 1
        self.writer.add_scalar(self.critic_loss_path, v_loss.item(), self.update_value_net_count)
        # 更新policy网络
        for agent_index in range(self.agent_nums):
            p_loss = -torch.mean((reward - v_Value.detach()) * prob[agent_index].unsqueeze(-1))
            if self.parameter_sharing:
                if agent_index == 0:
                    self.optimizer_actor.zero_grad()
                    p_loss.backward(retain_graph=True)
                elif agent_index == self.agent_nums -1:
                    p_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_norm_grad)
                    self.optimizer_actor.step()
                else:
                    p_loss.backward(retain_graph=True)
            else:
                self.optimizer_actor[agent_index].zero_grad()
                p_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_norm_grad)
                self.optimizer_actor[agent_index].step()
            self.writer.add_scalar(self.actor_loss_path[agent_index], p_loss.item(), self.update_policy_net_count)
        
        self.update_policy_net_count += 1
        self.actor_lr = max(self.actor_min_lr, self.actor_lr * (1-self.actor_lr_decay))
        self.critic_lr = max(self.critic_min_lr, self.critic_lr * (1-self.critic_lr_decay))