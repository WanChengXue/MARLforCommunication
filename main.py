import copy

import numpy as np
import matplotlib.pyplot as plt

import os
import shutil
from tqdm import tqdm

from Tool.arguments import get_common_args, get_agent_args, get_transformer_args, get_MADDPG_args
from Env.Env import Environment
import time
import copy
from multiprocessing import Pool
import pathlib
from Env.Instant_Reward import calculate_instant_reward

def multiprocessing_training(index):
    GPU_id = 0
    import torch
    from torch.utils.tensorboard import SummaryWriter
    torch.cuda.set_device(GPU_id)
    
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
    common_args.parameter_sharing = True
    if common_args.cuda:
        torch.cuda.manual_seed_all(3122)
    else:
        torch.manual_seed(3122)
    np.random.seed(3122)
    # common_args.attention_start = True
    agent_args = get_agent_args(common_args)
    transformer_args = get_transformer_args(agent_args)
    MADDPG_args = get_MADDPG_args(transformer_args)
    if MADDPG_args.transformer_start:
        from Agent.transformer_agent import Agent
    elif MADDPG_args.attention_start:
        from Agent.attention_agent import Agent
    else:
        from Agent.agent import Agent

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
            self.testing_data_folder = pathlib.Path(self.args.training_data_path)/(str(self.total_user_antennas) + '_user')/(str(self.velocity)+'KM')/'testing_data_10_10.npy'
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
                    if self.parameter_sharing:
                        if sector_index == 0:
                            zero_grad = True
                        else:
                            zero_grad = False

                        if sector_index == (self.sector_number-1):
                            release_graph = True
                        else:
                            release_graph = False
                        self.agent.training_parameter_sharing(self.env.batch_data, Instant_reward, batch_prob_list[sector_index], release_graph, zero_grad)
                    else:
                        self.agent_list[sector_index].training(self.env.batch_data, Instant_reward, batch_prob_list[sector_index])
                
                batch_average_reward.append(np.mean(Instant_reward))

            plt.figure()
            plt.plot(batch_average_reward)
            plt.savefig(self.figure_folder + '/learning_curve.png')
            # 保存数据到本地
            reward_training_path = self.result_folder + '/Training_rewrad.npy'
            np.save(reward_training_path, np.array(batch_average_reward))
            # 保存模型到本地
            self.save_model(ep)
            self.testing_model()

        def testing(self):
            self.load_model()
            self.testing_model()


        def testing_model(self):
            testing_data = np.load(self.testing_data_folder).transpose(4,0,1,2,3)
            obs = []
            agent_infer_sequence = []
            for sector_index in range(self.agent_number):
                # 每一个元素都是batch_size*20*3*32
                obs.append(testing_data[:,sector_index,:,:,:])
                if self.parameter_sharing:
                    scheduling_users, _ = self.agent.Pick_action_Max_SE_batch(obs[sector_index])
                else:
                    scheduling_users, _ = self.agent_list[sector_index].Pick_action_Max_SE_batch(obs[sector_index])
                agent_infer_sequence.append(scheduling_users)
            agent_infer_sequence = np.stack(agent_infer_sequence, axis=1)
            infer_SE = self.env.calculate_batch_instant_rewrd(testing_data, agent_infer_sequence)
            infer_SE_save_path = self.result_folder + '/infer_SE.npy'
            np.save(infer_SE_save_path, np.array(infer_SE))
            infer_sequence_save_path = self.result_folder + '/infer_sequence.npy'
            np.save(infer_sequence_save_path, np.array(agent_infer_sequence))

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


    def training_cell(args):
        test = Project(args)
        test.Simulation_SE_only()
        # print(args.user_numbers

    
        
    training_cell(MADDPG_args)


multiprocessing_training(4)
