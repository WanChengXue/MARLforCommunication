import copy

import numpy as np
import matplotlib.pyplot as plt

import os
import shutil
from tqdm import tqdm

from arguments import get_common_args, get_agent_args, get_transformer_args, get_MADDPG_args
from Env import Environment


from multiprocessing import Pool

def multiprocessing_training(index):
    GPU_id = index // 3
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
    common_args.parameter_sharing = False
    common_args.TTI_length = 200
    # common_args.transformer_start = True
    agent_args = get_agent_args(common_args)
    transformer_args = get_transformer_args(agent_args)
    MADDPG_args = get_MADDPG_args(transformer_args)
    MADDPG_args.epsilon = 0.1
    if MADDPG_args.transformer_start:
        from transformer_agent import Agent
    elif MADDPG_args.attention_start:
        from attention_agent import Agent
    else:
        from agent import Agent

    class Project:
        def __init__(self, args, Training=False):
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
            # 是否使用参数共享
            self.parameter_sharing = self.args.parameter_sharing
            self.data_folder = self.args.data_folder
            self.rewrite_and_make_folder()
            self.args.writer = SummaryWriter(self.args.vision_folder)
            # define the number of agents
            # define environment
            self.env = Environment(self.args)
            # self.load_model()
            # define agent list
            if self.parameter_sharing:
                self.agent = Agent(self.args, 0)
            else:
                self.agent_list = [Agent(self.args, i) for i in range(self.agent_number)] 

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
            else:
                algorithm_matrix_path = model_matrix_path + "/PF"
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
                
            self.data_folder = self.data_folder + str(self.total_user_antennas) + '_user/' + str(self.velocity) + 'KM'
            self.args.model_folder = self.model_folder
            self.args.vision_folder = self.vision_folder
            self.args.result_folder = self.result_folder
            self.args.data_folder = self.data_folder

        def load_model(self, ep):
            if os.path.exists(self.args.model_folder) and os.listdir(self.args.model_folder):
                if self.parameter_sharing:
                    policy_net_path = self.args.model_folder + '/'  + str(ep) +  '_Agent_' + str(0) +'_policy_net.pkl'
                    value_net_path = self.args.model_folder + '/' +  str(ep) + '_Agent_' + str(0) +'_value_net.pkl'
                    self.agent.actor.load_state_dict(torch.load(policy_net_path))
                    self.agent.critic.load_state_dict(torch.load(value_net_path))
                else:
                    for agent_id in range(self.agent_number):
                        policy_net_path = self.args.model_folder +  '/' + str(ep) + '_Agent_' + str(agent_id + 1) +'_policy_net.pkl'
                        value_net_path = self.args.model_folder + '/' + str(ep) + '_Agent_' + str(agent_id + 1) +'_value_net.pkl'
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
        
        def Simulation(self):
            self.env.read_training_data()
            self.load_model('o')
            self.generate_episode()

            # self.plot_figure(np.array(epoch_real_reward), np.array(epoch_cumulative_reward).squeeze(), np.array(epoch_system_capacity).squeeze(),np.array(epoch_edge_user_capacity).squeeze())

        def generate_episode(self):
            # generate a trajectory
            Channel, Average_reward, Global_channel, Global_reward, Action, Pad = [],[],[],[],[],[]
            instant_reward, terminate = [], []
            probs = [[] for _ in range(self.agent_number)]
            self.env.Reset()
            terminated = False
            episode_reward = []
            edge_result = []
            pF_sum_result = []
            system_result = []
            for i in tqdm(range(self.args.TTI_length)):
                episode_channel, episode_average_reward = self.env.get_agent_obs()
                episode_global_channel, episode_global_reward = self.env.get_state()
                actions = []
                pad = []
                rank = copy.deepcopy(self.env.user_rank)
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
                system_result.append(average_capacity * self.total_user_antennas * self.agent_number)
                edge_result.append(average_min_users)
                pF_sum_result.append(PF_sum)
                if average_capacity == 0:
                    print(actions)
            save_edge_result = self.result_folder + "/RL_edge.npy"
            np.save(save_edge_result, np.array(edge_result))
            save_system_result = self.result_folder + "/RL_system_npy"
            np.save(save_system_result, np.array(system_result))
            save_PF_result = self.result_folder + "/RL_pf_sum.npy"
            np.save(save_PF_result, np.array(pF_sum_result))
                
    # test = Project(Evaluation=True)
    # test = Project()
    # test.Simulation()

    def training_cell(args):
        test = Project(args)
        test.Simulation()
        # print(args.user_numbers)

    
        
    training_cell(MADDPG_args)



# pool = Pool(12)
# for process_id in range(12):
#     # multiprocessing_training(process_id)
#     pool.apply_async(multiprocessing_training, (process_id,))
# pool.close()
# pool.join()

# multiprocessing_training(0)
pool = Pool(3)
process_exp = [0,3,6]
for process_id in process_exp:
    pool.apply_async(multiprocessing_training, (process_id,))
pool.close()
pool.join()

