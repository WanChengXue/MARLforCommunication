# 这个地方采用贪婪策略结合PF调度计算出最优的调度序列
from matplotlib.pyplot import axis
import numpy as np
from tqdm import tqdm
import os 
import shutil
from multiprocessing import Pool
import copy
import json
import time
from Env.Instant_reward_single_cell import calculate_instant_reward
import pathlib
from Tool.arguments import get_common_args 
# 这个地方是三个扇区单独做greedy，需要改动Instant reward函数
import Env.Instant_Reward as NIR
class Greedy:
    def __init__(self, args, channel_matrix):
        self.args=args
        self.sector_number = self.args.sector_number
        self.cell_number = self.args.cell_number
        self.agent_number = self.cell_number * self.sector_number
        self.bs_antenna_number = self.args.bs_antennas
        self.user_numbers = self.args.user_numbers
        self.transmit_power = [self.args.transmit_power] * self.agent_number
        self.noise_spectrum_density = self.args.noise_spectrum_density
        self.system_bandwidth = self.args.system_bandwidth
        self.subcarriers_numbers = self.args.subcarrier_numbers
        self.subcarrier_gaps = self.args.subcarrier_gaps
        # 计算单个载波的频带宽度
        self.subcarrier_bandwidth = self.system_bandwidth / self.subcarriers_numbers - self.subcarrier_gaps
        self.noise_power = self.noise_spectrum_density * self.subcarrier_bandwidth
        self.legal_range = [self.args.min_stream, self.args.max_stream]
        self.transmit_power = [self.args.transmit_power] * self.agent_number
        self.channel_matrix = channel_matrix

    def greedy_add_users(self, last_action, sector_index):
        max_se = 0
        optimal_user = None
        for user_index in range(self.user_numbers):
            current_action = copy.deepcopy(last_action)
            if last_action[sector_index, user_index] == 0:
                # 如果当前用户没有被选到，则动作进行跳转
                # 调用函数计算给定的信道矩阵，以及在当前的调度序列下面的SE
                current_action[sector_index, user_index] = 1
                sum_SE = calculate_instant_reward(self.channel_matrix[sector_index, :, sector_index, :], current_action[sector_index, :][np.newaxis,:], self.legal_range, self.noise_power, self.transmit_power[sector_index])
                if max_se < sum_SE:
                    max_se = sum_SE
                    optimal_user = user_index
        return max_se, optimal_user
                
    def recyle_add_user(self):
        current_scheduling_sequence = np.zeros((self.sector_number, self.user_numbers), dtype=int)
        current_max_se = 0
        # 最开始的时候，三个小区，随机选出三个用户出来
        random_center_user = np.random.choice(self.user_numbers, self.sector_number)
        for sector_index in range(self.sector_number):
            current_scheduling_sequence[sector_index, random_center_user[sector_index]] = 1

        for sector_index in range(self.sector_number):
            while True:
                # 判断循环结束的标志是下一次遍历无法添加新用户的时候，则跳出
                SE_after_new_user_append, new_user_id = self.greedy_add_users(current_scheduling_sequence, sector_index)
                if SE_after_new_user_append >= current_max_se:
                    current_scheduling_sequence[sector_index, new_user_id] = 1
                    current_max_se = SE_after_new_user_append
                else:
                    # 如果当前这一轮没有办法添加新的用户进去，就直接break循环
                    current_max_se = 0
                    break
        actual_se = NIR.calculate_instant_reward(self.channel_matrix, current_scheduling_sequence, self.legal_range, self.noise_power, self.transmit_power)
        return current_scheduling_sequence, actual_se

def simulation(args):
    # 这个函数用来遍历一下所有的TTI测试信道数据，包括了将instant SE和调度序列进行存储两种功能
    file_path = args.testing_path
    channel_data = np.load(file_path)
    TTI_length = channel_data.shape[-1]
    scheduling_sequence = []
    SE = []
    for TTI in tqdm(range(TTI_length)):
        agent = Greedy(args, channel_data[:,:,:,:,TTI])
        greedy_scheduling_sequence, max_se = agent.recyle_add_user()
        scheduling_sequence.append(greedy_scheduling_sequence)
        SE.append(max_se)
    # 路径格式，data_part/preprocess_data/Greedy_result/用户数目/移动速度/
    Sum_se_path = args.greedy_folder / 'Individual_greedy_sum_SE'
    Scheduling_path = args.greedy_folder / 'Individual_greedy_shceduling_sequence'
    np.save(Sum_se_path, np.array(SE))
    np.save(Scheduling_path, np.stack(scheduling_sequence, axis=0))


# def main():
#     from arguments import get_common_args, get_agent_args, get_MADDPG_args
#     user_number = ['10_user','20_user','30_user','40_user']
#     velocity = ['3KM','30KM','90KM']
#     args_list = []
#     for user_index in user_number:
#         for velocity_index in velocity:
#             common_args = get_common_args()
#             common_args.data_folder = common_args.data_folder + user_index +'/' + velocity_index
#             common_args.greedy_folder = common_args.greedy_folder + user_index + '/' + velocity_index + '/'  + 'Greedy_PF_result/'
#             common_args.TTI_length = 200
#             common_args.user_numbers = int(user_index.split('_')[0])
#             common_args.user_velocity = int(velocity_index.split('K')[0])
#             agent_args = get_agent_args(common_args)
#             args = get_MADDPG_args(agent_args)
#             args_list.append(args)
#     testing_length = len(args_list)
#     pool = Pool(testing_length)

#     for i in range(testing_length):
#         pool.apply_async(Greedy_solver, (args_list[i],))
#     pool.close()
#     pool.join()

def main():
    user_number_list = ['10_user','20_user','30_user','40_user']
    velocity = ['3KM','30KM','90KM']
    # for index in range(12):
    index = 4
    user_index = user_number_list[index // 3]
    velocity_index = velocity[index % 3]
    # 修改用户的数量和用户移动速度
    user_number = int(user_index.split('_')[0])
    velocity_number = int(velocity_index.split('K')[0])
    common_args = get_common_args()
    common_args.user_numbers = user_number
    common_args.user_velocity = velocity_number
    common_args.testing_path = pathlib.Path(common_args.training_data_path)/(str(common_args.user_numbers) + '_user')/(str(common_args.user_velocity)+'KM')/'testing_data_10_10.npy'
    common_args.greedy_folder = pathlib.Path(common_args.greedy_folder)/(str(common_args.user_numbers) + '_user')/(str(common_args.user_velocity)+'KM')
    # 如果文件不存在就创建
    common_args.greedy_folder.mkdir(parents=True, exist_ok=True)
    simulation(common_args)

if __name__=='__main__':
    main()

