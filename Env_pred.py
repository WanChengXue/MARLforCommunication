# 这个函数用来模仿通信环境的变化
import numpy as np
import os
import copy
from tqdm import tqdm
from scipy.io import loadmat
import shutil
from ast import literal_eval

class Environment:
    def __init__(self, args):
        self.args = args
        self.agent_number = self.args.cell_number
        self.user_num = self.args.user_numbers
        self.bs_antenna_number = self.args.bs_antennas
        self.total_antennas = self.args.total_user_antennas 
        self.tau = self.args.tau
        self.TTI_length = self.args.TTI_length
        self.transmit_power = self.args.transmit_power
        self.noise_power = self.args.noise_power
        self.data_folder = self.args.data_folder
        self.koopman_predict_folder = self.args.koopman_predict_folder
        self.training_set_length = self.args.episodes
        self.read_training_data()
        self.delay_TTI = self.args.delay_time


    def read_training_data(self):
        # variable file_index is the index of current file
        self.file_index = 0
        self.training_set = []
        self.training_koopman_set = []
        file_list = sorted(os.listdir(self.data_folder))
        # Read chanel data set
        koopman_file_list = sorted(os.listdir(self.koopman_predict_folder))
        for i in range(self.training_set_length):
            temp_file = []
            koopman_temp_file = []
            for agent_index in range(self.agent_number):
                temp_file.append(self.data_folder + "/" + file_list[i*self.agent_number + agent_index])
                koopman_temp_file.append(self.koopman_predict_folder + "/" + koopman_file_list[i*self.agent_number + agent_index])
            self.training_set.append(temp_file)
            self.training_koopman_set.append(koopman_temp_file)
    
    def read_testing_data(self):
        # variable file_index is the index of current file
        testing_set = []
        testing_koopman_set = []
        file_list = sorted(os.listdir(self.data_folder))
        koopman_file_list = sorted(os.listdir(self.koopman_predict_folder))
        total_file = int(len(file_list) / self.agent_number)
        # Read chanel data set
        for i in range(total_file):
            temp_file = []
            koopman_temp_file = []
            for agent_index in range(self.agent_number):
                temp_file.append(self.data_folder + "/" + file_list[i*self.agent_number + agent_index])
                koopman_temp_file.append(self.koopman_predict_folder + "/" + koopman_file_list[i*self.agent_number + agent_index])
            testing_set.append(temp_file)
            testing_koopman_set.append(koopman_temp_file)
        return testing_set, testing_koopman_set

    def Reset(self):
        # TTI_count represent the time index of current channel file
        self.TTI_count = 0
        # Read specific channel data based on file_index 
        agent_data_list = self.training_set[self.file_index]
        agent_data_koopmam_list = self.training_koopman_set[self.file_index]
        self.episode_data = []
        self.koopman_episode_data = []
        for agent_index in range(self.agent_number):
                self.episode_data.append(np.load(agent_data_list[agent_index]))
                self.koopman_episode_data.append(np.load(agent_channel[agent_index]))
        self.file_index += 1
        self.Calculate_average_reward()
        self.Read_TTI_data()
        
    def Evaluate_mode(self, agent_data_list):
        self.TTI_count = 0
        self.episode_data = []
        self.koopman_episode_data = []
        for agent_index in range(self.agent_number):
                self.episode_data.append(np.load(agent_data_list[agent_index]))
                self.koopman_episode_data.append(np.load(agent_channel[agent_index]))
        self.file_index += 1
        self.Calculate_average_reward()
        self.Read_TTI_data()

    def Read_TTI_data(self):
        # Since multi-agent environment, we need return observations and global state
        self.TTI_data = []
        self.TTI_delay_data = []
        self.source_channel = []
        self.TTI_koopman_channel = []
        for agent in range(self.agent_number):
            activate_channel = self.episode_data[agent][:,:,:,self.TTI_count]
            delay_activate_channel = self.episode_data[agent][:,:,:,self.TTI_count+self.delay_TTI]
            koopman_pred_channel = self.koopman_episode_data[agent][:,:,:,self.TTI_count]
            self.source_channel.append(koopman_pred_channel)
            self.TTI_data.append(activate_channel[:,:,0:self.bs_antenna_number]+ 1j*activate_channel[:,:,self.bs_antenna_number:])
            self.TTI_delay_data.append(delay_activate_channel[:,:,0:self.bs_antenna_number]+1j*delay_activate_channel[:,:,self.bs_antenna_number:])
            self.TTI_koopman_channel.append(koopman_pred_channel[:,:,0:self.bs_antenna_number] + 1j * koopman_pred_channel[:,:,self.bs_antenna_number:])
        self.TTI_count += 1

    def Calculate_average_reward(self, instant_reward=None):
        if not instant_reward:
            self.average_reward = np.zeros((self.agent_number, self.total_antennas))
        else:
            last_average_reward = self.average_reward.copy()
            current_average_reward = last_average_reward * self.tau + (1-self.tau) * np.array(instant_reward)
            self.average_reward = current_average_reward

    def Calculate_precoding_matrix(self):
        # This function is used to calculate precoding matrix. If action of current cell is satisfied stream schedule rule
        # then this cell will have precoding matrix, otherwise, the precoding matrix is setted as None
        if self.is_reasonable:
            precoding_matrix = []
            for cell_index in range(self.agent_number):
                if self.cell_schedule_user_number[cell_index] != 0:
                    cell_channel_matrix = self.koopman_select_data[cell_index][:, cell_index, :]
                    pseudo_inverse = np.linalg.pinv(cell_channel_matrix)
                    cell_norm = np.linalg.norm(pseudo_inverse, 2 ,0)
                    cell_precoding_matrix = pseudo_inverse / cell_norm
                    precoding_matrix.append(cell_precoding_matrix)
                else:
                    precoding_matrix.append(None)
        else:
            precoding_matrix = [None for cell_index in range(self.agent_number)]
        self.precoding_matrix = precoding_matrix

    def Select_channel_data(self, action):
        if self.is_reasonable:
            selected_channel = []
            delay_selected_channel = []
            koopman_selected_channel = []
            for cell_index in range(self.agent_number):
                cell_action = action[cell_index]
                selected_channel.append(self.TTI_data[cell_index][np.array(cell_action).astype(bool),:,:])
                delay_selected_channel.append(self.TTI_delay_data[cell_index][np.array(cell_action).astype(bool),:,:])
                koopman_selected_channel.append(self.TTI_koopman_channel[cell_index][np.array(cell_action).astype(bool),:,:])
        else:
            selected_channel = [None for cell_index in range(self.agent_number)]
            delay_selected_channel = [None for cell_index in range(self.agent_number)]
            koopman_selected_channel = [None for cell_index in range(self.agent_number)]
        self.select_data = selected_channel
        self.delay_select_data = delay_selected_channel
        self.koopman_select_data = koopman_selected_channel

    def Action_reasonable(self, action):
        # Define a list, which is used to decide whether is reasonable of arbitrary cell action
        is_reasonable = True
        cell_schedule_user_number = []
        power = []
        # 此处需要遍历三个cell哦
        for cell_index in range(self.agent_number):
            cell_action = action[cell_index]
            schedule_number = np.sum(cell_action)
            cell_schedule_user_number.append(schedule_number)
            if schedule_number == 0:
                power.append(0)
                is_reasonable = False
            else:
                power.append(self.transmit_power/schedule_number)
            if np.sum(cell_action)>self.args.max_stream:
                is_reasonable =False
        self.is_reasonable = is_reasonable
        self.cell_schedule_user_number = cell_schedule_user_number
        self.power = power
        
    
    def Calculate_user_sum_rate(self, action):
        users_sum_rate = [[] for cell_index in range(self.agent_number)]
        user_count = [0 for cell_index in range(self.agent_number)]
        schedule_user_number = copy.deepcopy(self.cell_schedule_user_number)
        schedule_user_set =  [[i for i in range(schedule_user_number[cell_index])]for cell_index in range(self.agent_number)]
        # for cell_index in range(self.agent_number):
        #     cell_schedule_user_set = [user for user in range(self.total_antennas) if action[cell_index][user]== 1]
        #     schedule_user_set.append(cell_schedule_user_set)
        cell_index_list = [i for i in range(self.agent_number)]
        if self.is_reasonable:
            for cell_index in range(self.agent_number):
                cell_action = action[cell_index]
                cell_count = user_count[cell_index]
                cell_select_data = self.delay_select_data[cell_index]
                cell_precoding_matrix = self.precoding_matrix[cell_index]
                # 如果这个cell什么用户都没有调度,则其users_sum_rate就是一个全0的向量
                if schedule_user_number[cell_index] == 0:
                    for user_index in range(self.total_antennas):
                        users_sum_rate[cell_index].append(0)
                else:
                    for user in range(self.total_antennas):
                        # traverse all actions, if action has selected by the policy net, SINR will be calculated, otherwise, directly add zero
                        if cell_action[user] == 1:
                            # 得到的是一个(基站天线,)的向量
                            antenna_channel = cell_select_data[cell_count,cell_index,:]
                            # 计算分子和分母部分, 分母分成两个部分，一个是当前小区之内的干扰，另外一个是相邻小区的干扰j 
                            Molecular = self.power[cell_index] * np.abs(np.sum(antenna_channel * cell_precoding_matrix[:,cell_count])) **2
                            Intra_interference_user = schedule_user_set[cell_index].copy()
                            Intra_interference_user.remove(cell_count)
                            if len(Intra_interference_user) == 0:
                                Intra_interference_value = 0
                            else:
                                Intra_precoding_matrix = cell_precoding_matrix[:, np.array(Intra_interference_user)]
                                # 如果长度是大于等于1， 则显然其需要先计算一个向量
                                Intra_interference_value = self.power[cell_index] * np.sum(np.abs(antenna_channel[np.newaxis, :].dot(Intra_precoding_matrix)) **2)
                            # ========= 这个部分计算Inter_cell interference ============
                            Inter_interference_cell = cell_index_list.copy()
                            Inter_interference_cell.remove(cell_index)
                            Inter_interference_value = 0
                            for inter_cell_index in Inter_interference_cell:
                                if schedule_user_number[inter_cell_index] == 0:
                                    # 如果说这个inter_cell没有任何用户进行调度,则其干扰就为0
                                    cell_interference_value = 0
                                else:
                                    # 将信道向量抽出来
                                    Inter_antenna_channel = cell_select_data[cell_count,inter_cell_index,:]
                                    # 干扰小区的预编码矩阵
                                    Inter_cell_precoding_matrix = self.precoding_matrix[inter_cell_index]
                                    # 直接和precoding向量相乘
                                    cell_interference_value = self.power[inter_cell_index] * np.sum(np.abs(Inter_antenna_channel[np.newaxis, :].dot(Inter_cell_precoding_matrix)) **2)
                                Inter_interference_value += cell_interference_value
                            Dominator = self.noise_power + Intra_interference_value + Inter_interference_value
                            user_sum_rate = np.log2(1+Molecular/Dominator)
                            users_sum_rate[cell_index].append(user_sum_rate)
                            cell_count += 1
                        else:
                            users_sum_rate[cell_index].append(0)
        else:
            # 这个表示的当前小区没有数据进行发送，因此也就直接将所有用户的instant reward设置为0
            for cell_index in range(self.agent_number):
                for user in range(self.total_antennas):
                    users_sum_rate[cell_index].append(0)
        return users_sum_rate

    def Calculate_reward(self, action):
        # Traversal all cell, and calculate the instant rate of individual user
        self.Calculate_precoding_matrix()
        reward = self.Calculate_user_sum_rate(action)
        # 计算capacity
        capacity = np.sum(reward)
        return capacity

    def Step(self, sequence):
        # 这个sequence是一个序列,需要变成0-1 string
        action = [[0 for i in range(self.total_antennas)] for cell_index in range(self.agent_number)]
        for cell_index in range(self.agent_number):
            cell_action = sequence[cell_index]
            for user_index in range(self.total_antennas):
                if user_index in cell_action:
                    action[cell_index][user_index] = 1

        terminated = False
        self.Action_reasonable(action)
        self.Select_channel_data(action)
        instant_reward = self.Calculate_reward(action)
        if self.args.Training:
            if self.TTI_count == self.TTI_length:
                terminated = True
            else:
                self.Read_TTI_data()
        else:
            if self.TTI_count == self.args.total_TTI_length:
                terminated = True
            else:
                self.Read_TTI_data()
        return instant_reward, terminated

    def get_agent_obs(self):
        # apart channel matrix and average reward
        channel = []
        average_reward = []
        for agent_id in range(self.agent_number):
            agent_channel = []
            for index in range(self.agent_number):
                # 添加三个信道矩阵,其中意思表达的是当前基站收到的信号是什么,最好加上一个one hot编码,表示的是当前智能体的index
                agent_channel.append(copy.deepcopy(self.source_channel[index][:,agent_id,:]))
            channel.append(agent_channel)
            average_reward.append(copy.deepcopy(self.average_reward[agent_id, :]))
        return channel, average_reward

    def get_state(self):
        global_channel = []
        global_reward = copy.deepcopy(self.average_reward)
        for agent_id in range(self.agent_number):
            for index in range(self.agent_number):
                global_channel.append(copy.deepcopy(self.source_channel[agent_id][:, index, :]))
        return global_channel, global_reward 