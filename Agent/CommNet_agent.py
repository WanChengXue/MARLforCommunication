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
            self.actor = CommNet_Actor(self.args, (1, args.obs_matrix_number, args.obs_dim1, args.obs_dim2)).to(self.device)
            self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
            print(self.actor)
        else:
            self.Replay_buffer = ReplayBuffer(self.args)
            self.actor = [CommNet_Actor(self.args, (1, args.obs_matrix_number, args.obs_dim1, args.obs_dim2)).to(self.device) for _ in range(self.agent_nums)]
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
        self.communication_turns = self.args.communication_turns

    def Pick_action_Max_SE_batch(self,state_list):
        net_input = [torch.FloatTensor(state_list[agent_index]).to(self.device).transpose(1,2) for agent_index in range(self.agent_nums)]
        batch_size = net_input[0].shape[0]
        # ===== 首先进行通信交互 =====
        for turns in range(self.communication_turns):
            if turns == 0:
                if self.parameter_sharing:
                    h_list = [self.actor.precoding_agent(net_input[agent_index]) for agent_index in range(self.agent_nums)]
                    c_list = [torch.zeros_like(h_list[agent_index]) for agent_index in range(self.agent_nums)]
                else:
                    h_list = [self.actor[agent_index].precoding_agent(net_input[agent_index]) for agent_index in range(self.agent_nums)] 
                    c_list = [torch.zeros_like(h_list[agent_index]) for agent_index in range(self.agent_nums)]
            else:
                # 需要把c作为GRU的隐藏单元
                if self.parameter_sharing:
                    h_list = [self.actor.f_comm(h_list[agent_index],c_list[agent_index]) for agent_index in range(self.agent_nums)] 
                else:
                    h_list = [self.actor[agent_index].f_comm(h_list[agent_index], c_list[agent_index]) for agent_index in range(self.agent_nums)]
                sum_list = torch.sum(torch.stack(h_list, 0), 0)
                c_list = [(sum_list - h_list[agent_index])/(self.agent_nums-1) for agent_index in range(self.agent_nums)]
        # ===== 通信交互完成，使用Pointer网络进行决策=====
        scheduling_prob_list = []
        mask_list = []
        for agent_index in range(self.agent_nums):
            if self.parameter_sharing:
                Pointer_network_input = torch.tanh(self.actor.decoding_layer(h_list[agent_index])).reshape(batch_size, self.user_number, -1)
                _, scheduling_prob, selected_mask, mask = self.actor.pointer_agent(Pointer_network_input)
            else:
                Pointer_network_input = torch.tanh(self.actor[agent_index].decoding_layer(h_list[agent_index])).reshape(batch_size, self.user_number, -1)
                _, scheduling_prob, selected_mask, mask = self.actor[agent_index].pointer_agent(Pointer_network_input)
            scheduling_prob = torch.sum(scheduling_prob * selected_mask, -1).unsqueeze(-1)
            scheduling_prob_list.append(scheduling_prob.unsqueeze(0))
            mask_list.append(mask[:,1:].cpu().numpy())
        return mask_list, scheduling_prob_list
        

    def training(self, batch_data, reward, prob):
        batch_data = torch.FloatTensor(batch_data).to(self.device).transpose(2,3).reshape(-1, self.sector_number **2, self.user_number, self.bs_antennas*2)
        reward = torch.FloatTensor(reward).to(self.device).unsqueeze(-1)
        self.writer.add_scalar(self.average_reward, torch.mean(reward).item(), self.update_value_net_count)
        reward  = (reward-torch.mean(reward)) / (1e-6 + torch.max(reward-torch.mean(reward)))
        # ========== 更新critic网络的参数 =========
        v_Value = self.critic(batch_data)
        v_loss =  self.critic_loss(v_Value, reward)
        self.optimizer_critic.zero_grad()
        v_loss.backward()
        self.optimizer_critic.step()
        self.writer.add_scalar(self.critic_loss_path, v_loss.item(), self.update_value_net_count)
        # ========== 更新策略网络的参数 ===========
        advantage_value = reward-v_Value.detach()
        for agent_index in range(self.agent_nums):
            p_loss = -torch.mean(advantage_value * prob[agent_index])
            if self.parameter_sharing:
                if agent_index == 0:
                    self.optimizer_actor.zero_grad()
                    p_loss.backward(retain_graph=True)
                elif agent_index == self.agent_nums-1:
                    p_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_norm_grad)
                    self.optimizer_actor.step()
                else:
                    p_loss.backward(retain_graph=True)

            self.writer.add_scalar(self.actor_loss_path[agent_index], p_loss.item(), self.update_policy_net_count)
        # ======== 更新训练的learning rate =======
        self.update_policy_net_count += 1
        self.update_value_net_count += 1
        self.actor_lr = max(self.actor_min_lr, self.actor_lr * (1-self.actor_lr_decay))
        self.critic_lr = max(self.critic_min_lr, self.critic_lr * (1-self.critic_lr_decay))

