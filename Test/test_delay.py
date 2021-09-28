import torch
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import matplotlib.pyplot as plt

import os
import shutil
from tqdm import tqdm

from Tool.arguments import get_common_args, get_agent_args, get_MADDPG_args
from Env.Env_delay import Environment
from Agent.agent import Agent

import multiprocessing
from multiprocessing import Pool

class Project:
    def __init__(self, args, Training=False):
        self.args = args
        self.args.Training = Training
        # 这个小区用户移动速度
        self.velocity = self.args.user_velocity
        # 定义小区的数目
        self.agent_number = self.args.cell_number
        # 定义每个小区的用户数目
        self.total_user_antennas = self.args.total_user_antennas
        # 定义三个文件夹
        self.model_folder = self.args.model_folder
        self.vision_folder = self.args.eval_vision_folder
        self.result_folder = self.args.result_folder
        self.data_folder = self.args.data_folder
        self.rewrite_argument()
        # create three folders which is used to store model, figure and tensorboard data
        if Training:
            # 如果是训练模式,将创建三个文件夹
            # self.create_folder(self.args.result_folder)
            self.create_folder(self.vision_folder)
            self.create_folder(self.model_folder)
            self.create_folder(self.result_folder)
        else:
            self.create_folder(self.vision_folder)
            # create a summary writer, which is used to record the loss value of individual agent
        self.args.writer = SummaryWriter(self.args.vision_folder)
        # define the number of agents
        # define environment
        self.env = Environment(self.args)
        # self.load_model()
        # define agent list
        self.agent_list = [Agent(self.args, i) for i in range(self.agent_number)]

    def rewrite_argument(self):
        # 这个函数将对ArgumentParser中的一些参数进行重新写入,包括Exp,Model,result这三个文件夹
        self.model_folder = self.model_folder + '/' + str(self.total_user_antennas) + '_user_' + str(self.velocity) + 'KM_delay'
        self.vision_folder = self.vision_folder + '/' + str(self.total_user_antennas) + '_user_' + str(self.velocity) + 'KM_delay'
        self.result_folder = self.result_folder + '/' + str(self.total_user_antennas) + '_user_' + str(self.velocity) + 'KM_delay'
        self.data_folder = self.data_folder + str(self.total_user_antennas) + '_user/' + str(self.velocity) + 'KM/channel_data'
        self.args.model_folder = self.model_folder
        self.args.vision_folder = self.vision_folder
        self.args.result_folder = self.vision_folder
        self.args.data_folder = self.data_folder

    def load_model(self):
        if os.path.exists(self.args.model_folder) and os.listdir(self.args.model_folder):
            for agent_id in range(self.agent_number):
                policy_net_path = self.args.model_folder +  '/' +  'Agent_' + str(agent_id + 1) +'_policy_net.pkl'
                critic_net_path = self.args.model_folder + '/' + 'Agent_' + str(agent_id + 1) +'_value_net.pkl'
                self.agent_list[agent_id].actor.load_state_dict(torch.load(policy_net_path))
                self.agent_list[agent_id].critic.load_state_dict(torch.load(critic_net_path)) 

    def create_folder(self, folder_name):
        # create a folder, if folder exists, load model, else, create 
        if os.path.exists(folder_name):
            shutil.rmtree(folder_name)
        os.mkdir(folder_name)

    def Evaluation(self):
        self.load_model()
        test_file = self.env.read_testing_data()
        file_number = len(test_file)
        cell_reward = []
        for file_index in tqdm(range(file_number)):
            self.env.Evaluate_mode(test_file[file_index])
            instant_reward = self.generate_episode()
            cell_reward.append(instant_reward)
        save_path = self.result_folder + '/' + 'RL_result.npy'
        np.save(save_path, np.array(cell_reward))

    def generate_episode(self):
        # generate a trajectory
        instant_reward, terminate = [],[]
        terminated = False
        while not terminated:
            episode_channel, episode_average_reward = self.env.get_agent_obs()
            episode_global_channel, episode_global_reward = self.env.get_state()
            actions = []
            for agent_id in range(self.agent_number):
                active_agent = self.agent_list[agent_id]
                action, prob = active_agent.Pick_action([episode_channel[agent_id], episode_average_reward[agent_id]])
                # 这个地方补全action
                actions.append(action)
            reward, terminated = self.env.Step(actions)
            instant_reward.append(reward)
        print(np.sum(instant_reward)/1100/30)
        return np.array(instant_reward)

def testing_cell(args):
    test = Project(args)
    test.Evaluation()
    # print(args.user_numbers)

def multiprocessing_testing(index):
    # 开一个多进程
    user_number = ['10_user','20_user','30_user','40_user']
    velocity = ['3KM','30KM','90KM']
    args_list = []
    for user_index in user_number:
        for velocity_index in velocity:
            # 修改用户的数量和用户移动速度
            user_number = int(user_index.split('_')[0])
            velocity_number = int(velocity_index.split('K')[0])
            common_args = get_common_args(user_number)
            common_args.user_numbers = user_number
            common_args.user_velocity = velocity_number
            agent_args = get_agent_args(common_args)
            MADDPG_args = get_MADDPG_args(agent_args)
            MADDPG_args.total_TTI_length = MADDPG_args.total_TTI_length - MADDPG_args.delay_time
            args_list.append(MADDPG_args)
    
    # testing_length = len(args_list)
    # # 计算CPU个数，开启进程数
    # workers = np.minimum(multiprocessing.cpu_count() - 1, testing_length)
    # # workers = 1
    # # # 开进程池
    # pool = Pool(workers)
    # for args_index in range(testing_length):
    #     pool.apply_async(training_cell, (args_list[args_index],))
    # pool.close()
    # pool.join()
    testing_cell(args_list[index])

# index = input("please input a number: ")
# multiprocessing_training(int(index))
multiprocessing_testing(0)
multiprocessing_testing(1)
multiprocessing_testing(2)
multiprocessing_testing(3)