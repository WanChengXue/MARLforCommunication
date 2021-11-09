# 这个函数用来模仿通信环境的变化
import numpy as np
import os
import copy
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
        self.transmit_power = [self.args.transmit_power] * self.agent_number
        self.noise_power = self.args.noise_spectrum_density
        # self.system_bandwidth = self.args.system_bandwidth
        # self.subcarriers_numbers = self.args.subcarrier_numbers
        # self.subcarrier_gaps = self.args.subcarrier_gaps
        # # 计算单个载波的频带宽度
        # self.subcarrier_bandwidth = self.system_bandwidth / self.subcarriers_numbers - self.subcarrier_gaps
        # self.noise_power = self.noise_spectrum_density * self.subcarrier_bandwidth
        self.data_folder = self.args.data_folder
        # 当使用PF调度的时候，所有用户的average sum rate进行初始化
        self.rank_button = self.args.rank_start
        # 读取训练数据
        self.training_data_path = self.args.data_folder
        self.training_data = np.load(self.training_data_path)
        self.total_training_sample = self.training_data.shape[-1]
        self.legal_range = [self.args.min_stream, self.args.max_stream]

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
        self.count = 0
        # TTI_count represent the time index of current channel file
        batch_index = np.random.choice(self.total_training_sample, self.args.TTI_length, replace=False)
        self.batch_index = batch_index
        
        TTI_data = self.random_sample_data()
        self.current_state = TTI_data
        return TTI_data

    def random_sample_data(self):
        TTI_index = self.batch_index[self.count]
        self.count += 1
        return self.training_data[:,:,:,:,TTI_index]
        

    def Reset_batch(self):
        batch_index = np.random.choice(self.total_training_sample, self.args.TTI_length, replace=False)
        batch_data = [self.training_data[:,:,:,:,index] for index in batch_index]
        self.batch_data= np.stack(batch_data, axis=0)

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
        
    def calculate_batch_instant_rewrd(self, batch_channel, batch_action):
        batch_instant_reward = []
        sample_number = batch_channel.shape[0]
        for index in range(sample_number):
            batch_instant_reward.append(calculate_instant_reward(batch_channel[index], batch_action[index], self.legal_range, self.noise_power, self.transmit_power))
        return batch_instant_reward


    def Step(self, sequence):
        # 这个sequence是一个序列,需要变成0-1 string
        action = np.zeros((self.agent_number, self.total_antennas))
        for cell_index in range(self.agent_number):
            cell_action = sequence[cell_index]
            for user_index in range(self.total_antennas):
                if user_index in cell_action:
                    action[cell_index][user_index] = 1

        terminated = False
        instant_reward = calculate_instant_reward(self.current_state, action, self.legal_range, self.noise_power, self.transmit_power)
        
        if self.args.Training:
            self.current_state = self.random_sample_data()
            if self.count == self.TTI_length-1:
                terminated = True
        else:
            if self.count == self.args.total_TTI_length:
                terminated = True
            
        return instant_reward, terminated

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



    def get_agent_obs_SE(self):
        obs = []
        for sector_index in range(self.agent_number):
            # 每个元素都是20 × 3 × 32
            obs.append(self.current_state[sector_index, :, :, :])
        return obs

    def get_agent_obs_SE_batch(self):
        obs = []
        if self.args.ablation_experiment:
            for sector_index in range(self.agent_number):
                # 每一个元素都是batch_size*20*3*32
                if self.args.independent_learning:
                    obs.append(self.batch_data[:, sector_index,sector_index,:,:])
                else:
                    sub_obs = []
                    for sub_sector_index in range(self.agent_number):
                        if sector_index == sub_sector_index:
                            sub_obs.append(self.batch_data[:, sector_index, :, sub_sector_index, :])
                        else:
                            sub_obs.append(np.zeros(self.user_num, self.total_antennas))

                    obs.append(np.stack(sub_obs, 1))
                # obs.append(self.batch_data[:,sector_index,:,:,:])
        elif self.args.multi_head_input:
            for sector_index in range(self.agent_number):
                sub_obs = []
                extra_obs = []
                sub_sector_index = sector_index
                for _ in range(self.agent_number):
                    index = sub_sector_index % self.agent_number
                    sub_obs.append(self.batch_data[:, sector_index, :, index, :])
                    if index == sector_index:
                        sub_sector_index += 1
                        continue
                    else:
                        # 其中extra_obs中的每一个元素的维度都是batch_size * 20 * 3* 32
                        extra_obs.append(self.batch_data[:, index, :, :, :].transpose(0,2,1,3))
                        sub_sector_index += 1
                # stack之后得到的数据维度为batch size * 3 * user number
                stack_sub_obs = np.stack(sub_obs, 1)
                obs.append(np.concatenate([stack_sub_obs] + extra_obs, 1))    

        else:
            for sector_index in range(self.agent_number):
                if self.args.independent_learning:
                    obs.append(self.batch_data[:, sector_index, :, sector_index, :])
                else:
                    obs.append(self.batch_data[:,sector_index,:,:,:])
        return obs