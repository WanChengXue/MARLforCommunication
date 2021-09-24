import copy

import numpy as np
import matplotlib.pyplot as plt

import os
import shutil
from tqdm import tqdm

from arguments import get_common_args, get_agent_args, get_transformer_args, get_MADDPG_args
from Env import Environment
import time
import copy
from multiprocessing import Pool
import pathlib

def multiprocessing_training(index):
    GPU_id = 0
    import torch
    from torch.utils.tensorboard import SummaryWriter
    # torch.cuda.set_device(GPU_id)
    # 开一个多进程

    user_number_list = ['10_user','20_user','30_user','40_user']
    velocity = ['3KM','30KM','90KM']
    user_index = user_number_list[index // 3]
    velocity_index = velocity[index % 3]
    # 修改用户的数量和用户移动速度
    user_number = int(user_index.split('_')[0])
    velocity_number = int(velocity_index.split('K')[0])
    common_args = get_common_args()
    common_args.user_numbers = user_number
    common_args.user_velocity = velocity_number
    common_args.parameter_sharing = False
    # common_args.attention_start = True
    agent_args = get_agent_args(common_args)
    transformer_args = get_transformer_args(agent_args)
    MADDPG_args = get_MADDPG_args(transformer_args)
    if MADDPG_args.transformer_start:
        from transformer_agent import Agent
    elif MADDPG_args.attention_start:
        from attention_agent import Agent
    else:
        from agent import Agent

    class Project:
        def __init__(self, args, Training=True):
            self.args = args
            self.args.Training = Training
            # 这个小区用户移动速度
            self.velocity = self.args.user_velocity
            # 定义小区的数目
            self.sector_number = self.args.sector_number
            self.cell_number = self.args.cell_number
            self.agent_number = self.cell_number * self.sector_number
            # 定义每个小区的用户数目
            self.total_user_antennas = self.args.total_user_antennas
            # 定义模型开关
            self.transformer_start = self.args.transformer_start
            # 定义算法开关
            self.rank_start = self.args.rank_start
            self.weighted_start = self.args.weighted_start
            self.priority_start = self.args.priority_start
            self.edge_max_start = self.args.edge_max_start
            self.attention_start = self.args.attention_start
            self.prev_policy_start = self.args.prev_policy_start
            self.PF_start = self.args.PF_start
            # 是否使用参数共享
            self.parameter_sharing = self.args.parameter_sharing
            self.data_folder = pathlib.Path(self.args.training_data_path)/(str(self.total_user_antennas) + '_user')/(str(self.velocity)+'KM')/'training_data_10_10.npy'
            self.args.data_folder = self.data_folder
            self.rewrite_and_make_folder()
            self.args.writer = SummaryWriter(self.args.vision_folder)
            # define the number of agents
            # define environment
            self.env = Environment(self.args)
            # self.load_model()
            # define agent list
            if self.parameter_sharing:
                self.agent = Agent(self.args, 0)
                if self.prev_policy_start:
                    self.prev_agent = Agent(self.args, 0)
                    self.prev_agent.actor.load_state_dict(self.agent.actor.state_dict())
            else:
                self.agent_list = [Agent(self.args, i) for i in range(self.agent_number)] 
                if self.prev_policy_start:
                    self.prev_agent_list = []
                    for i in range(self.agent_number):
                        cell_agent = Agent(self.args, i)
                        cell_agent.actor.load_state_dict(self.agent_list[i].actor.state_dict())
                        self.prev_agent_list.append(cell_agent)

        def rewrite_and_make_folder(self):
            if self.transformer_start:
                model_matrix_path = "./Transformer_folder"
            elif self.attention_start:
                model_matrix_path = "./Attention_folder"
            else:
                model_matrix_path = "./Pointer_network_folder"
            self.create_matrix_folder(model_matrix_path)

            if self.rank_start:
                algorithm_matrix_path = model_matrix_path + "/Rank"
            elif self.edge_max_start:
                algorithm_matrix_path = model_matrix_path + "/Edge_max"
            elif self.priority_start:
                algorithm_matrix_path = model_matrix_path + "Priority"
            elif self.weighted_start:
                weighted_matrix_path = model_matrix_path + "/Weighted"
                self.create_matrix_folder(weighted_matrix_path)
                self.weighted_ratio = self.args.weighted_ratio
                algorithm_matrix_path = weighted_matrix_path + "/weighted_ratio_" + str(self.weighted_ratio)
            elif self.PF_start:
                algorithm_matrix_path = model_matrix_path + "/PF"
            else:
                algorithm_matrix_path = model_matrix_path + "/Max_SE"
            self.create_matrix_folder(algorithm_matrix_path)

            # 定义parameter sharing 开关
            if self.parameter_sharing:
                Matrix_model_folder = algorithm_matrix_path + '/Sharing_model' 
                Matrix_vision_folder = algorithm_matrix_path + '/Sharing_exp'
                Matrix_result_folder = algorithm_matrix_path + '/Sharing_result'
                Matrix_figure_folder = algorithm_matrix_path + '/Sharing_figure' 
            else:
                Matrix_model_folder = algorithm_matrix_path + '/Model' 
                Matrix_vision_folder = algorithm_matrix_path + '/Exp'
                Matrix_result_folder = algorithm_matrix_path + '/Result'
                Matrix_figure_folder = algorithm_matrix_path + '/Figure' 
            self.create_matrix_folder(Matrix_model_folder)
            self.create_matrix_folder(Matrix_vision_folder)
            self.create_matrix_folder(Matrix_result_folder)
            self.create_matrix_folder(Matrix_figure_folder)

            self.model_folder = Matrix_model_folder +  '/' + str(self.total_user_antennas) + '_user_' + str(self.velocity) + 'KM'
            self.vision_folder = Matrix_vision_folder + '/' + str(self.total_user_antennas) + '_user_' + str(self.velocity) + 'KM'
            self.result_folder = Matrix_result_folder + '/' + str(self.total_user_antennas) + '_user_' + str(self.velocity) + 'KM'
            self.figure_folder = Matrix_figure_folder + '/' + str(self.total_user_antennas) + '_user_' + str(self.velocity) + 'KM'
            # 这个函数将对ArgumentParser中的一些参数进行重新写入,包括Exp,Model,result这三个文件夹

            if self.args.Training:
                self.create_folder(self.model_folder)
                self.create_folder(self.vision_folder)
                self.create_folder(self.result_folder)
                self.create_folder(self.figure_folder)
            else:
                Matrix_vision_folder = Matrix_vision_folder + '_eval'
                self.create_matrix_folder(Matrix_vision_folder)
                self.vision_folder = Matrix_vision_folder +  '/' + str(self.total_user_antennas) + '_user_' + str(self.velocity) + 'KM'
                self.create_folder(self.vision_folder)
                
            
            self.args.model_folder = self.model_folder
            self.args.vision_folder = self.vision_folder
            self.args.result_folder = self.result_folder
            

        def load_model(self):
            if os.path.exists(self.args.model_folder) and os.listdir(self.args.model_folder):
                if self.parameter_sharing:
                    policy_net_path = self.args.model_folder + '/'  +  'Agent_' + str(0) +'_policy_net.pkl'
                    value_net_path = self.args.model_folder + '/' +  'Agent_' + str(0) +'_value_net.pkl'
                    self.agent.actor.load_state_dict(torch.load(policy_net_path))
                    self.agent.critic.load_state_dict(torch.load(value_net_path))
                else:
                    for agent_id in range(self.agent_number):
                        policy_net_path = self.args.model_folder +  '/' +  'Agent_' + str(agent_id + 1) +'_policy_net.pkl'
                        value_net_path = self.args.model_folder + '/' + 'Agent_' + str(agent_id + 1) +'_value_net.pkl'
                        self.agent_list[agent_id].actor.load_state_dict(torch.load(policy_net_path))
                        self.agent_list[agent_id].critic.load_state_dict(torch.load(value_net_path)) 
            else:
                os.mkdir(self.args.model_folder)


        def create_matrix_folder(self,folder_name):
            if not os.path.exists(folder_name):
                os.mkdir(folder_name)


        def create_folder(self, folder_name):
            # create a folder, if folder exists, load model, else, create 
            if os.path.exists(folder_name):
                shutil.rmtree(folder_name)
            os.mkdir(folder_name)
        
        def Simulation_SE_only(self):
            # 这个函数就是专门正对于Max SE进行采样操作
            batch_average_reward = []
            for ep in tqdm(range(self.args.epoches)):
                # channel_data = self.env.Reset()
                self.env.Reset_batch()
                # 将这个channel_data进行划分，得到三个智能体的观测，以及critic的状态
                obs_list = self.env.get_agent_obs_SE_batch()
                action_list = []
                batch_prob_list = []
                # start_time = time.time()
                for sector_index in range(self.sector_number):
                    if self.parameter_sharing:
                        scheduling_users, prob = self.agent.Pick_action_Max_SE_batch(obs_list[sector_index])
                    else:
                        scheduling_users, prob = self.agent_list[sector_index].Pick_action_Max_SE_batch(obs_list[sector_index])
                    action_list.append(scheduling_users)
                    batch_prob_list.append(prob)
                # 3*batch_size * 20的0-1序列
                action_list = np.stack(action_list,axis=1)
                # end_time = time.time()
                # print("使用神经网络计算动作耗费的时间为：{}".format(end_time-start_time))
                # start_time = time.time()
                Instant_reward = self.env.calculate_batch_instant_rewrd(self.env.batch_data, action_list)
                # end_time = time.time()
                # print("计算batch的所有用户的SE耗费的时间为：{}".format(end_time-start_time))
                # 这个地方进行训练

                for sector_index in range(self.sector_number):
                    self.agent_list[sector_index].training(self.env.batch_data, Instant_reward, batch_prob_list[sector_index])
                
                batch_average_reward.append(np.mean(Instant_reward))
            plt.figure()
            plt.plot(batch_average_reward)
            plt.savefig(self.figure_folder + '/learning_curve.png')

        def Simulation_fairness(self):
            epoch_cumulative_reward =[]
            epoch_system_capacity = []
            epoch_edge_user_capacity = []
            epoch_real_reward = []
            optimal_reward = 0
            optimal_edge = 0
            optimal_capacity = 0
            for ep in tqdm(range(self.args.epoches)):
                episode_reward = []
                self.env.read_training_data()
                for _ in range(self.args.episodes):
                    # collect experience2
                    episodic_reward = self.generate_episode()
                    average_reward = np.mean(episodic_reward,0)
                    episode_reward.append(average_reward)
                # 这个地方对三个值进行平均化处理
                episode_average_reweard = np.mean(episode_reward,0)
                average_capacity = episode_average_reweard[0]
                min_average_capacity = episode_average_reweard[1]
                average_PF_sum = episode_average_reweard[2]
                average_real_reward = episode_average_reweard[3]
                epoch_cumulative_reward.append(average_PF_sum)
                epoch_system_capacity.append(average_capacity)
                epoch_edge_user_capacity.append(min_average_capacity)
                epoch_real_reward.append(average_real_reward)
                print("Epoch: {}, Current epoch average user reward is: {}, min average reward is: {}, PF sum is: {}, real reward is: {}".format(ep, average_capacity, min_average_capacity, average_PF_sum, average_real_reward))
                self.learn()
                if optimal_edge >=0.02:
                    if optimal_capacity < average_capacity:
                        optimal_capacity = average_capacity
                        self.save_model('o')
                else:
                    if optimal_capacity < average_capacity and optimal_edge < min_average_capacity:
                        optimal_capacity = average_capacity
                        optimal_edge = min_average_capacity
                        self.save_model('o')

                if ep % 50 == 0:
                    self.save_model(ep)

                if self.args.epsilon > self.args.min_epsilon:
                    self.args.epsilon -= self.args.epsilon_decay

                if self.args.actor_lr < self.args.actor_lr_decay:
                    self.args.actor_lr = self.args.actor_lr_decay
                else:
                    self.args.actor_lr = self.args.actor_lr - self.args.actor_lr_decay

                if self.args.critic_lr < self.args.critic_lr_decay:
                    self.args.critic_lr = self.args.critic_lr_decay
                else:
                    self.args.critic_lr = self.args.critic_lr - self.args.critic_lr_decay
                # 更新prev网络中的参数
                if self.prev_policy_start:
                    if ep % 20 == 0:
                        if self.parameter_sharing:
                            self.prev_agent.actor.load_state_dict(self.agent.actor.state_dict())
                    else:
                        for i in range(self.agent_number):
                            self.prev_agent_list[i].actor.load_state_dict(self.agent_list[i].actor.state_dict())
        
            self.plot_figure(np.array(epoch_real_reward), np.array(epoch_cumulative_reward).squeeze(), np.array(epoch_system_capacity).squeeze(),np.array(epoch_edge_user_capacity).squeeze())
        
        def learn(self):
            if self.parameter_sharing:
                for agent_index in range(self.agent_number):
                    self.agent.Learning(agent_index)
            else:
                for agent_id in range(self.agent_number):
                    self.agent_list[agent_id].Learning()

        def generate_episode(self):
            # generate a trajectory
            Channel, Average_reward, Global_channel, Global_reward, Action, Pad = [],[],[],[],[],[]
            instant_reward, terminate = [], []
            probs = [[] for _ in range(self.agent_number)]
            self.env.Reset()
            terminated = False
            episode_reward = []
            while not terminated:
                episode_channel, episode_average_reward = self.env.get_agent_obs()
                episode_global_channel, episode_global_reward = self.env.get_state()
                rank = copy.deepcopy(self.env.user_rank)
                # 是否需要上一次的策略来辅助判断
                if self.prev_policy_start:
                    current_actions = []
                    current_pad = []
                    # 这个地方使用当前智能体的参数进行决策
                    for agent_id in range(self.agent_number):
                        if self.rank_start:
                            cell_rank = rank[agent_id]
                        else:
                            cell_rank = [1 for _ in range(self.total_user_antennas)]
                        if self.parameter_sharing:
                            active_agent = self.agent
                        else:
                            active_agent = self.agent_list[agent_id]
                        action, _ = active_agent.Pick_action([episode_channel[agent_id], episode_average_reward[agent_id]], cell_rank)
                        # 这个地方补全action
                        current_actions.append(action)
                        current_pad.append([])
                    
                    current_average_capacity, current_average_min_users, current_PF_sum = self.env.Calculate_reward_with_sequence(current_actions)
                    # 这个地方使用是一个时刻智能体的参数进行决策
                    prev_actions = []
                    prev_pad = []
                    for agent_id in range(self.agent_number):
                        if self.rank_start:
                            cell_rank = rank[agent_id]
                        else:
                            cell_rank = [1 for _ in range(self.total_user_antennas)]
                        if self.parameter_sharing:
                            active_agent = self.prev_agent
                        else:
                            active_agent = self.prev_agent_list[agent_id]
                        action, _ = active_agent.Pick_action([episode_channel[agent_id], episode_average_reward[agent_id]], cell_rank)
                        # 这个地方补全action
                        prev_actions.append(action)
                        prev_pad.append([])
                    prev_average_capacity, prev_average_min_users, prev_PF_sum = self.env.Calculate_reward_with_sequence(current_actions)
                    if current_average_min_users < prev_average_min_users:
                        # 如果说边缘用户的SE下降了，就使用上一个策略的决策序列
                        actions = prev_actions
                        pad = prev_pad
                    else:
                        actions = current_actions
                        pad = current_pad
                    # 按照这个调度序列去实际调度
                    # 然后对prob进行添加
                    for agent_id in range(self.agent_number):
                        if self.rank_start:
                            cell_rank = rank[agent_id]
                        else:
                            cell_rank = [1 for _ in range(self.total_user_antennas)]
                        if self.parameter_sharing:
                            active_agent = self.agent
                        else:
                            active_agent = self.agent_list[agent_id]
                        action = copy.deepcopy(actions[agent_id])
                        prob = active_agent.Calculate_prob([episode_channel[agent_id], episode_average_reward[agent_id]], cell_rank, action)
                        probs[agent_id].append(prob)
                else:
                    actions = []
                    pad = []
                    for agent_id in range(self.agent_number):
                        if self.rank_start:
                            cell_rank = rank[agent_id]
                        else:
                            cell_rank = [1 for _ in range(self.total_user_antennas)]
                        if self.parameter_sharing:
                            active_agent = self.agent
                        else:
                            active_agent = self.agent_list[agent_id]
                        action, prob = active_agent.Pick_action([episode_channel[agent_id], episode_average_reward[agent_id]], cell_rank)
                        probs[agent_id].append(prob)
                        # 这个地方补全action
                        actions.append(action)
                        pad.append([])
                average_capacity, average_min_users, PF_sum, terminated = self.env.Step(actions)
                if self.rank_start:
                    algrithm_instant_reward = average_capacity
                elif self.weighted_start:
                    algrithm_instant_reward = self.weighted_ratio * average_capacity + (1-self.weighted_ratio) * average_min_users
                elif self.edge_max_start:
                    algrithm_instant_reward = average_min_users
                elif self.priority_start:
                    if 0.05 * average_capacity > average_min_users:
                        algrithm_instant_reward = 0
                    else:
                        algrithm_instant_reward = average_min_users
                else:
                    algrithm_instant_reward = PF_sum
                instant_reward.append(algrithm_instant_reward)
                episode_reward.append([average_capacity, average_min_users, PF_sum, algrithm_instant_reward])
                terminate.append(terminated)
                # instant_reward.append(PF_sum)
                Channel.append(episode_channel)
                Average_reward.append(episode_average_reward)
                Global_channel.append(episode_global_channel)
                Global_reward.append(episode_global_reward)
                for agent_id in range(self.agent_number):
                    action_length = len(actions[agent_id])
                    actions[agent_id] = actions[agent_id] + (self.args.state_dim1 + 1-action_length) * [-1]
                    pad[agent_id] +=(1 + action_length) * [1]
                    pad[agent_id] += (self.args.state_dim1-action_length) * [0]
                Action.append(actions)
                Pad.append(pad)
                
            for agent_id in range(self.agent_number):
                episode_batch = {}
                episode_batch['Channel'] = np.array(Channel)[:,agent_id,:,:]
                episode_batch['Average_reward'] = np.array(Average_reward)[:,agent_id,:]
                episode_batch['Global_channel'] = np.array(Global_channel)
                episode_batch['Global_reward'] = np.array(Global_reward)
                episode_batch['Action'] = (np.array(Action) + 1)[:,agent_id,:]
                episode_batch['Pad'] = np.array(Pad)[:,agent_id,:]
                episode_batch['instant_reward'] = np.array(instant_reward)
                episode_batch['terminate'] = np.array(terminate)
                episode_batch['prob'] = probs[agent_id]
                if self.parameter_sharing:
                    self.agent.Stor_transition_parameter_sharing(episode_batch, agent_id)
                else:
                    self.agent_list[agent_id].Store_transition(episode_batch)
            
            return episode_reward

        def save_model(self, ep):
            # save model parameters
            if self.parameter_sharing:
                policy_net_path = self.args.model_folder + '/'  + str(ep) + '_Agent_' + str(0) +'_policy_net.pkl'
                value_net_path = self.args.model_folder + '/' + str(ep) + '_Agent_' + str(0) +'_value_net.pkl'
                torch.save(self.agent.actor.state_dict(), policy_net_path)
                torch.save(self.agent.critic.state_dict(), value_net_path)
            else:
                for agent_id in range(self.agent_number):
                    policy_net_path = self.args.model_folder +  '/' + str(ep) +  '_Agent_' + str(agent_id + 1) +'_policy_net.pkl'
                    value_net_path = self.args.model_folder + '/' + str(ep) + '_Agent_' + str(agent_id + 1) + '_value_net.pkl'
                    torch.save(self.agent_list[agent_id].actor.state_dict(), policy_net_path)
                    torch.save(self.agent_list[agent_id].critic.state_dict(), value_net_path)

        def plot_figure(self, Iteration_result, PF_sum, system_capacity, edge_user_capacity):
            plt.figure()
            plt.plot(np.arange(self.args.epoches)+1, Iteration_result)
            save_path = self.args.result_folder + '/' + 'Iteration_result.png'
            plt.savefig(save_path)
            plt.close()
            reward_save_path = self.args.result_folder + '/' + 'Iteration_result.npy'
            np.save(reward_save_path, np.array(Iteration_result))

            plt.figure()
            plt.plot(np.arange(self.args.epoches)+1, PF_sum)
            save_path_PF = self.args.result_folder + '/' + 'PF_sum.png'
            plt.savefig(save_path_PF)
            plt.close()
            PF_save_path = self.args.result_folder + '/' + 'PF_sum.npy'
            np.save(PF_save_path, PF_sum)

            plt.figure()
            plt.plot(np.arange(self.args.epoches)+1, system_capacity)
            save_path_capacity = self.args.result_folder + '/' + 'system_result.png'
            plt.savefig(save_path_capacity)
            plt.close()
            system_capacity_save_path = self.args.result_folder + '/' + 'system_result.npy'
            np.save(system_capacity_save_path, system_capacity)
            
            plt.figure()
            plt.plot(np.arange(self.args.epoches)+1, edge_user_capacity)
            save_path_edge_user = self.args.result_folder + '/' + 'edge_user_result.png'
            plt.savefig(save_path_edge_user)
            plt.close()
            edge_user_capacity_save_path = self.args.result_folder + '/' + 'edge_user_result.npy'
            np.save(edge_user_capacity_save_path, edge_user_capacity)

    # test = Project(Evaluation=True)
    # test = Project()
    # test.Simulation()

    def training_cell(args):
        test = Project(args)
        test.Simulation_SE_only()
        # print(args.user_numbers)

    
        
    training_cell(MADDPG_args)



# pool = Pool(3)
# process_exp = [0,3,6]
# for process_id in process_exp:
#     pool.apply_async(multiprocessing_training, (process_id,))
# pool.close()
# pool.join()

multiprocessing_training(4)
