# 这个函数用来模仿通信环境的变化
import numpy as np
import os
import copy
from numpy.lib.polynomial import RankWarning
from tqdm import tqdm
from scipy.io import loadmat
import shutil
from multiprocessing import Pool
import multiprocessing
from Instant_Reward import calculate_instant_reward

class Environment:
    def __init__(self, args):
        self.args = args
        self.user_num = self.args.user_numbers
        self.sector_number = self.args.sector_number
        self.cell_number = self.args.cell_number
        self.agent_number = self.cell_number * self.sector_number
        self.bs_antenna_number = self.args.bs_antennas
        self.total_antennas = self.args.total_user_antennas 
        self.tau = self.args.tau
        self.TTI_length = self.args.TTI_length
        self.transmit_power = self.args.transmit_power 
        self.noise_spectrum_density = self.args.noise_spectrum_density
        self.system_bandwidth = self.args.system_bandwidth
        self.subcarriers_numbers = self.args.subcarrier_numbers
        self.subcarrier_gaps = self.args.subcarrier_gaps
        # 计算单个载波的频带宽度
        self.subcarrier_bandwidth = self.system_bandwidth / self.subcarriers_numbers - self.subcarrier_gaps
        self.noise_power = self.noise_spectrum_density * self.subcarrier_bandwidth
        self.data_folder = self.args.data_folder
        self.training_set_length = self.args.episodes
        # 当使用PF调度的时候，所有用户的average sum rate进行初始化
        self.rank_button = self.args.rank_start
        self.read_training_data()


    def Divide_group(self, priority_array):
        # 这个函数是将优先级列表分成二级列表
        rank_level = []
        group_dict = {}
        max_rank = []
        for cell_index in range(self.agent_number):
            cell_priority = copy.deepcopy(priority_array[cell_index])
            cell_dict = {}
            cell_level = []
            for user_index, user_priority in enumerate(cell_priority):
                if str(user_priority) in cell_dict.keys():
                    cell_dict[str(user_priority)].append(user_index)
                else:
                    # 如果这个优先级不存在，则创建
                    cell_dict[str(user_priority)] = [user_index]
            max_rank.append(user_priority)
            group_dict[str(cell_index)] = cell_dict
            rank_level.append(cell_level)
        return group_dict

    def read_training_data(self):
        # variable file_index is the index of current file
        self.file_index = 0
        self.training_set = []
        file_list = sorted(os.listdir(self.data_folder))
        # Read chanel data set
        for i in range(self.training_set_length):
            temp_file = []
            for agent_index in range(self.agent_number):
                temp_file.append(self.data_folder + "/" + file_list[i*self.agent_number + agent_index])
            self.training_set.append(temp_file)
        
    def read_testing_data(self):
        # variable file_index is the index of current file
        testing_set = []
        file_list = sorted(os.listdir(self.data_folder))
        total_file = int(len(file_list) / self.agent_number)
        # Read chanel data set
        for i in range(total_file):
            temp_file = []
            for agent_index in range(self.agent_number):
                temp_file.append(self.data_folder + "/" + file_list[i*self.agent_number + agent_index])
            testing_set.append(temp_file)
        return testing_set

    def Reset(self):
        # TTI_count represent the time index of current channel file
        self.TTI_count = 0
        # Read specific channel data based on file_index 
        agent_data_list = self.training_set[self.file_index]
        self.episode_data = []
        for agent_index in range(self.agent_number):
                self.episode_data.append(np.load(agent_data_list[agent_index]))
        self.file_index += 1
        self.user_rank = [[1 for _ in range(self.total_antennas)] for cell_index in range(self.agent_number)]
        self.Read_TTI_data()
        self.Calculate_average_reward()
        
        
    def Calculate_init_average_user_sum_rate(self):
        # 这个函数用来计算再初始时刻，每个用户的平均sum rate，具体就是将第一个TTI的数据拿过来，挨个计算一次
        Init_user_average_reward = np.zeros((self.agent_number, self.total_antennas)) 
        for cell_index in range(self.agent_number):
            # 将用户的信道拿出来
            cell_id = cell_index // self.sector_number
            sector_id = cell_index % self.sector_number
            user_channel = self.TTI_data[cell_index][:,cell_id, sector_id, :]
            # 遍历所有的用户即可
            for user_index in range(self.total_antennas):
                single_user_channel = user_channel[user_index,:].reshape(1,self.bs_antenna_number)
                sum_rate = np.log(1+np.linalg.norm(single_user_channel)**2 * self.transmit_power/self.noise_power)
                Init_user_average_reward[cell_index, user_index] = sum_rate
        return Init_user_average_reward

    def Evaluate_mode(self, agent_data_list):
        self.TTI_count = 0
        self.episode_data = []
        for agent_index in range(self.agent_number):
                self.episode_data.append(np.load(agent_data_list[agent_index]))
        self.file_index += 1
        self.Read_TTI_data()
        self.Calculate_average_reward()
        

    def Read_TTI_data(self):
        # Since multi-agent environment, we need return observations and global state
        self.TTI_data = []
        self.source_channel = []
        for agent in range(self.agent_number):
            activate_channel = self.episode_data[agent][:,:,:,:,self.TTI_count]
            self.source_channel.append(activate_channel)
            self.TTI_data.append(activate_channel[:,:,:,0:self.bs_antenna_number]+ 1j*activate_channel[:,:,:,self.bs_antenna_number:])
        self.TTI_count += 1

    def Calculate_average_reward(self, instant_reward=None):
        if not instant_reward:
            self.average_reward = self.Calculate_init_average_user_sum_rate()
            self.real_average_reward = np.zeros((self.agent_number, self.total_antennas))
        else:
            last_average_reward = copy.deepcopy(self.average_reward)
            current_average_reward = last_average_reward * self.tau + (1-self.tau) * np.array(instant_reward)
            self.real_average_reward = self.real_average_reward * self.tau + (1-self.tau) * np.array(instant_reward)
            self.average_reward = current_average_reward

    def Calculate_precoding_matrix(self):
        # This function is used to calculate precoding matrix. If action of current cell is satisfied stream schedule rule
        # then this cell will have precoding matrix, otherwise, the precoding matrix is setted as None
        # select data 的维度是3*3×scheduled_number * 32
        if self.is_reasonable:
            precoding_matrix = []
            for cell_index in range(self.agent_number):
                if self.cell_schedule_user_number[cell_index] != 0:
                    sector_id = cell_index % self.sector_number
                    cell_id = cell_index // self.sector_number
                    cell_channel_matrix = self.select_data[cell_index][:,cell_id, sector_id, :]
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
            for cell_index in range(self.agent_number):
                cell_action = action[cell_index]
                selected_channel.append(self.TTI_data[cell_index][np.array(cell_action).astype(bool),:,:,:])
        else:
            selected_channel = [None for _ in range(self.agent_number)]
        self.select_data = selected_channel
    
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

        # users_sum_rate是一个二维列表,数目与扇区的数目是一样的
        users_sum_rate = [[] for cell_index in range(self.agent_number)]
        # schedule_user_number表示的是每一个扇区中调度的用户数目
        schedule_user_number = copy.deepcopy(self.cell_schedule_user_number)
        # schedule_user_set是一个二维列表,第一个维度和扇区数目是一样的,第二个维度是给调度的用户进行重新记上索引
        schedule_user_set =  [[i for i in range(schedule_user_number[cell_index])]for cell_index in range(self.agent_number)]
        # 对所有的扇区进行标号
        cell_index_list = [i for i in range(self.agent_number)]
        if self.is_reasonable:
            for cell_index in range(self.agent_number):
                cell_action = action[cell_index]
                # 取出第i个小区第j个sector的信道数据
                sector_id = cell_index % self.sector_number
                cell_id = cell_index // self.sector_number
                # 两个变量，分别表示的cell的索引以及sector的索引
                cell_select_data = self.select_data[cell_index]
                cell_precoding_matrix = self.precoding_matrix[cell_index]
                cell_schedule_user_set = schedule_user_set[cell_index]
                cell_power = self.power[cell_index]
                # 如果这个cell什么用户都没有调度,则其users_sum_rate就是一个全0的向量
                # 分成四部分计算，分别是扇区内部的有效信号，intra-sector interference， inter-sector interference , inter-cell interference
                if schedule_user_number[cell_index] == 0:
                    # 如果当前这个扇区什么用户都没有调度,则所有用户的SE都给0
                    for _ in range(self.total_antennas):
                        users_sum_rate[cell_index].append(0)
                else:
                    # 这个count表示的是调度用户的索引
                    count = 0
                    for user in range(self.total_antennas):
                        # traverse all actions, if action has selected by the policy net, SINR will be calculated, otherwise, directly add zero
                        if cell_action[user] == 1:
                            # 先把分子部分拿出来
                            target_sector_channel = cell_select_data[:,cell_id, sector_id, :]
                            target_user_sector_channel = target_sector_channel[count,:].reshape(1, self.bs_antenna_number)
                            signal_power = np.linalg.norm(target_user_sector_channel.dot(cell_precoding_matrix[:, count])) **2 * cell_power
                            # 计算intra-sector之间的干扰
                            intra_sector_interference = []
                            # 将当前用户移出去
                            intra_sector_schedule_index = copy.deepcopy(cell_schedule_user_set)
                            intra_sector_schedule_index.remove(count)
                            # 如果这个扇区只是调度了一个用户,则intra-sector干扰就是0
                            if len(intra_sector_schedule_index) == 0:
                                intra_sector_interference.append(0)
                            else:
                                # 如果这个扇区调度了其余的用户,计算扇区内部的干扰
                                for other_user_index in intra_sector_schedule_index:
                                    user_intra_sector_interference = np.linalg.norm(target_user_sector_channel.dot(cell_precoding_matrix[:,other_user_index])) ** 2 * cell_power
                                    intra_sector_interference.append(user_intra_sector_interference)
                            # 计算inter_sector之间的干扰
                            inter_sector_interference = []
                            inter_sector_index = copy.deepcopy(cell_index_list)
                            inter_sector_index.remove(cell_index)
                            for inter_sector in inter_sector_index:
                                # 将这个小区的发射功率拿出来
                                inter_sector_power = self.power[inter_sector]
                                # 先拿出干扰小区的预编码矩阵来
                                inter_sector_precoding_matrix = self.precoding_matrix[inter_sector]
                                # 将inter_sector变成小区,扇区索引
                                inter_cell_id = inter_sector // self.sector_number
                                inter_sector_id = inter_sector % self.sector_number
                                # 将其余扇区到本扇区的信道向量拿出来
                                inter_sector_channel = cell_select_data[:, inter_cell_id, inter_sector_id, :]
                                inter_user_sector_channel = inter_sector_channel[count,:].reshape(1, self.bs_antenna_number)
                                sector_interference = np.linalg.norm(inter_user_sector_channel.dot(inter_sector_precoding_matrix),2)**2
                                inter_sector_interference.append(inter_sector_power*sector_interference)
                            # 计算SINR
                            SINR = signal_power / (self.noise_power + np.sum(intra_sector_interference) + np.sum(inter_sector_interference))
                            # 计算这个用户的SE
                            SE = np.log(1+SINR)
                            users_sum_rate[cell_index].append(SE)
                            count += 1
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
        PF_factor = np.array(reward) / self.average_reward
        PF_sum = np.sum(PF_factor)
        self.Calculate_average_reward(reward)
        # 最后的instant的reward由两个部分组成，一个是系统的容量，另外一个是每个小区最糟糕的那几个用户的容量的平均值
        capacity = np.sum(reward)
        min_user_arrary = np.min(self.real_average_reward,1)
        average_min_users = np.mean(min_user_arrary)
        average_capacity = capacity / (self.agent_number * self.total_antennas)
        # weight_reward = self.weight_factor * average_capacity + (1-self.weight_factor) * average_min_users
        return average_capacity, average_min_users, PF_sum

    def Step(self, sequence):
        # 这个sequence是一个序列,需要变成0-1 string
        action = [[0 for i in range(self.total_antennas)] for cell_index in range(self.agent_number)]
        for cell_index in range(self.agent_number):
            cell_action = sequence[cell_index]
            for user_index in range(self.total_antennas):
                if user_index in cell_action:
                    action[cell_index][user_index] = 1

        terminated = False
        if self.rank_button:
            self.update_user_rank(action)
        self.Action_reasonable(action)
        self.Select_channel_data(action)
        average_capacity, average_min_users, PF_sum = self.Calculate_reward(action)
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
        return average_capacity, average_min_users, PF_sum, terminated

    def Calculate_reward_with_sequence(self, sequence):
        action = [[0 for i in range(self.total_antennas)] for cell_index in range(self.agent_number)]
        for cell_index in range(self.agent_number):
            cell_action = sequence[cell_index]
            for user_index in range(self.total_antennas):
                if user_index in cell_action:
                    action[cell_index][user_index] = 1

        if self.rank_button:
            self.update_user_rank(action)
        self.Action_reasonable(action)
        self.Select_channel_data(action)
        self.Calculate_precoding_matrix()
        reward = self.Calculate_user_sum_rate(action)
        PF_factor = np.array(reward) / self.average_reward
        PF_sum = np.sum(PF_factor)
        capacity = np.sum(reward)
        min_user_arrary = np.min(self.real_average_reward,1)
        average_min_users = np.mean(min_user_arrary)
        average_capacity = capacity / (self.agent_number * self.total_antennas)
        return average_capacity, average_min_users, PF_sum

    def get_agent_obs(self):
        # apart channel matrix and average reward
        channel = []
        average_reward = []
        for agent_id in range(self.agent_number):
            agent_channel = []
            for index in range(self.agent_number):
                # 添加九个信道矩阵,其中意思表达的是当前基站收到的信号是什么,最好加上一个one hot编码,表示的是当前智能体的index
                cell_id = index // self.sector_number
                sector_id = index % self.sector_number
                agent_channel.append(copy.deepcopy(self.source_channel[agent_id][:,cell_id,sector_id,:]))
            channel.append(agent_channel)
            average_reward.append(copy.deepcopy(self.average_reward[agent_id, :]))
        return channel, average_reward

    def get_state(self):
        global_channel = []
        global_reward = copy.deepcopy(self.average_reward)
        for agent_id in range(self.agent_number):
            for index in range(self.agent_number):
                cell_id = index // self.sector_number
                sector_id = index % self.sector_number
                global_channel.append(copy.deepcopy(self.source_channel[agent_id][:,cell_id,sector_id,:]))
        return global_channel, global_reward 

    def update_user_rank(self, action):
        # 这个地方是根据神经网络决策出来的动作来更新rank向量
        for cell_index in range(self.agent_number):
            for user_antenna in range(self.total_antennas):
                if action[cell_index][user_antenna] == 1 and self.user_rank[cell_index][user_antenna]== 1:
                    self.user_rank[cell_index][user_antenna] -= 1
            cell_rank = self.user_rank[cell_index]
            if np.max(cell_rank) == 0:
                cell_rank = (np.array(cell_rank) + 1).tolist()
                self.user_rank[cell_index] = cell_rank
        
