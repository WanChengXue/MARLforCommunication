import numpy as np
from tqdm import tqdm
import sys
import os
current_path = os.path.abspath(__file__)
root_path = '/'.join(current_path.split('/')[:-2])
sys.path.append(root_path)

from Utils import create_folder
from Utils.config_parse import load_yaml
from Env.Instant_Reward import Single_cell_instant_reward
import copy
import multiprocessing
import shutil
import pathlib
import random

class Greedy:
    def __init__(self, env_dict):
        self.env_dict = env_dict
        # ================ 定义工程参数 =================
        self.user_nums = self.env_dict['user_nums']
        self.sector_nums = self.env_dict['sector_nums']
        self.cell_nums = self.env_dict['cell_nums']
        self.agent_nums = self.env_dict['agent_nums']
        self.bs_antenna_nums = self.env_dict['bs_antenna_nums']
        self.total_antenna_nums = self.env_dict['total_antenna_nums']
        self.transmit_power = self.env_dict['transmit_power']
        self.noise_power = self.env_dict['noise_power']
        self.velocity = self.env_dict['velocity']
        # ---------- 文件的路径采用绝对位置 -------------
        self.save_data_folder = self.generate_abs_path(self.env_dict['save_data_folder'] + '/' + str(self.user_nums) +'_user/'+str(self.velocity)+'KM')
        self.single_cell_reward_calculator = Single_cell_instant_reward(self.transmit_power, self.noise_power, self.user_nums, self.bs_antenna_nums)
        self.max_decoder_time = self.env_dict['max_stream_nums']
        self.eps = 1e-9
        self.current_average_SE = np.zeros((self.sector_nums, self.total_antenna_nums))

    def load_eval_file(self, eval_file_number):
        self.construct_save_path() 
        # ----------- 载入测试文件 ------------
        loaded_file_name = self.save_data_folder + '/eval_channel_file_' + str(eval_file_number) + '.npy'
        self.simulation_channel = (np.load(loaded_file_name)).squeeze()
        self.TTI_length = self.simulation_channel.shape[-1]
        self.filter_factor = 1/self.TTI_length


    def load_training_file(self, trainiing_file_number):
        self.construct_training_data_save_path()
        loaded_file_name = self.save_data_folder + '/training_channel_file_' + str(trainiing_file_number) + '.npy'
        self.simulation_channel = (np.load(loaded_file_name)).squeeze()
        self.TTI_length = self.simulation_channel.shape[-1]
        self.filter_factor = 1/self.TTI_length

    def generate_abs_path(self, related_path):
        file_path = os.path.abspath(__file__)
        root_path = '/'.join(file_path.split('/')[:-3])
        return os.path.join(root_path, related_path)

    def construct_save_path(self):
        # ----------- 构建结果保存路径 -------------
        related_path = 'data_part/Greedy_result/single_cell_scenario_PF/' + str(self.user_nums) +'_user/'+str(self.velocity)+'KM'
        self.abs_result_save_prefix = self.generate_abs_path(related_path)
        create_folder(self.abs_result_save_prefix)

    def construct_training_data_save_path(self):
        related_path = 'data_part/demonstration_data/single_cell_scenario_PF/' + str(self.user_nums) +'_user/'+str(self.velocity)+'KM'
        self.abs_result_save_prefix = self.generate_abs_path(related_path)
        create_folder(self.abs_result_save_prefix)

    def greedy_add_users(self, sector_index, last_action, average_SE_matrix):
        # 这个函数是遍历一次，找一个用户出来
        max_PF_sum = 0
        optimal_user = None
        for user_index in range(self.user_nums):
            current_action = copy.deepcopy(last_action)
            if last_action[user_index] == 0:
                # 如果当前用户没有被选到，则动作进行跳转
                # 调用函数计算给定的信道矩阵，以及在当前的调度序列下面的SE
                current_action[user_index] = 1
                SE_array = self.single_cell_reward_calculator.calculate_instant_reward(self.channel_matrix[sector_index,:,sector_index,:], current_action)
                PF_matrix = SE_array / (average_SE_matrix + self.eps)
                sum_PF = np.sum(PF_matrix).item() / (self.total_antenna_nums)
                if max_PF_sum < sum_PF:
                    max_PF_sum = sum_PF
                    optimal_user = user_index
        return max_PF_sum, optimal_user
                
    def recyle_add_user(self):
        action_sector_dict = dict()
        for sector_index in range(self.sector_nums):
            action_sector_dict['sector_{}'.format(sector_index)] = []

        current_scheduling_sequence = np.zeros((self.sector_nums, self.user_nums), dtype=int)
        max_PF_list = np.zeros(self.sector_nums)
        for sector_index in range(self.sector_nums):
            current_max_PF = 0
            step = 0
            while True:
                # 判断循环结束的标志是下一次遍历无法添加新用户的时候，则跳出
                PF_after_new_user_append, new_user_id = self.greedy_add_users(sector_index, current_scheduling_sequence[sector_index,:], self.current_average_SE[sector_index, :])
                if PF_after_new_user_append >= current_max_PF and step<=self.max_decoder_time:
                    action_sector_dict['sector_{}'.format(sector_index)].append(new_user_id+1)
                    current_scheduling_sequence[sector_index, new_user_id] = 1
                    current_max_PF = PF_after_new_user_append
                    step += 1
                else:
                    # 如果当前这一轮没有办法添加新的用户进去，就直接break循环
                    max_PF_list[sector_index] = current_max_PF
                    # ------------ 将action_sector_dict用0填充 -------
                    if step < self.max_decoder_time:
                        action_sector_dict['sector_{}'.format(sector_index)] = action_sector_dict['sector_{}'.format(sector_index)] + [0 for i in range(self.max_decoder_time-step)]
                    # --------- 计算一下在当前最优调度序列下的SE向量 ----------
                    SE_array = self.single_cell_reward_calculator.calculate_instant_reward(self.channel_matrix[sector_index,:,sector_index,:], current_scheduling_sequence[sector_index,:])
                    # --------- 修改average SE矩阵 ------------
                    self.current_average_SE[sector_index, :] = (1-self.filter_factor) * self.current_average_SE[sector_index, :] + self.filter_factor * SE_array.squeeze(-1)
                    break
        return action_sector_dict, max_PF_list


    def simulation(self, file_index):
        self.load_eval_file(file_index)
        TTI_length = self.simulation_channel.shape[-1]
        # TTI_length = 10
        scheduling_sequence = dict()
        PF = dict()
        scheduling_sequence_saved_path = dict()
        PF_result_saved_path = dict()
        for sector_index in range(self.sector_nums):
            scheduling_sequence['sector_{}'.format(sector_index)] = []
            PF['sector_{}'.format(sector_index)] = []
            scheduling_sequence_saved_path['sector_{}'.format(sector_index)] = self.abs_result_save_prefix + '/' +str(file_index)+ '_sector_{}_scheduling_sequence.npy'.format(sector_index)
            PF_result_saved_path['sector_{}'.format(sector_index)] = self.abs_result_save_prefix + '/' +str(file_index)+ '_sector_{}_PF_sum_result.npy'.format(sector_index)
        average_se_saved_path = self.abs_result_save_prefix + '/' +str(file_index)+ '_sector_{}_average_se_result.npy'.format(sector_index)
        for TTI in tqdm(range(TTI_length)):
            self.channel_matrix = self.simulation_channel[:,:,:,:,TTI]
            greedy_scheduling_sequence, max_pf = self.recyle_add_user()
            for sector_index in range(self.sector_nums):
                scheduling_sequence['sector_{}'.format(sector_index)].append(greedy_scheduling_sequence['sector_{}'.format(sector_index)])
                PF['sector_{}'.format(sector_index)].append(max_pf[sector_index])
        # ------------------ 将来步
        for sector_index in range(self.sector_nums):
            np.save(scheduling_sequence_saved_path['sector_{}'.format(sector_index)], np.stack(scheduling_sequence['sector_{}'.format(sector_index)], 0))
            np.save(PF_result_saved_path['sector_{}'.format(sector_index)], np.array(PF['sector_{}'.format(sector_index)]))
        np.save(average_se_saved_path, self.current_average_SE)

    def simulation_training_file(self, file_index):
        self.load_training_file(file_index)
        TTI_length = self.simulation_channel.shape[-1]
        # TTI_length = 10
        scheduling_sequence = dict()
        PF = dict()
        scheduling_sequence_saved_path = dict()
        PF_result_saved_path = dict()
        for sector_index in range(self.sector_nums):
            scheduling_sequence['sector_{}'.format(sector_index)] = []
            PF['sector_{}'.format(sector_index)] = []
            scheduling_sequence_saved_path['sector_{}'.format(sector_index)] = self.abs_result_save_prefix + '/' +str(file_index)+ '_sector_{}_scheduling_sequence.npy'.format(sector_index)
            PF_result_saved_path['sector_{}'.format(sector_index)] = self.abs_result_save_prefix + '/' +str(file_index)+ '_sector_{}_se_sum_result.npy'.format(sector_index)
        for TTI in tqdm(range(TTI_length)):
            self.channel_matrix = self.simulation_channel[:,:,:,:,TTI]
            greedy_scheduling_sequence, max_se = self.recyle_add_user()
            for sector_index in range(self.sector_nums):
                scheduling_sequence['sector_{}'.format(sector_index)].append(greedy_scheduling_sequence['sector_{}'.format(sector_index)])
                PF['sector_{}'.format(sector_index)].append(max_se[sector_index])
        # ------------------ 将来步
        for sector_index in range(self.sector_nums):
            np.save(scheduling_sequence_saved_path['sector_{}'.format(sector_index)], np.array(scheduling_sequence['sector_{}'.format(sector_index)]))
            np.save(PF_result_saved_path['sector_{}'.format(sector_index)], np.array(PF['sector_{}'.format(sector_index)]))

def start_process_training_data(file_index, config_dict):
    test_greedy = Greedy(config_dict['env'])
    # test_greedy.simulation_training_file(file_index)
    test_greedy.simulation(file_index)


if __name__=='__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='/Learner/configs/config_multi_cell_pointer_network.yaml', help='yaml format config')
    args = parser.parse_args()
    # ------------- 构建绝对地址 --------------
    # Linux下面是用/分割路径，windows下面是用\\，因此需要修改
    abs_path = '/'.join(os.path.abspath(__file__).split('/')[:-2])
    # abs_path = '/'.join(os.path.abspath(__file__).split('\\')[:-2])
    concatenate_path = abs_path + args.config_path
    config_dict = load_yaml(concatenate_path)
    test_greedy = Greedy(config_dict['env']) 
    # for i in range(50):
    #     test_greedy.simulation(i)
    # start_process_training_data(0, config_dict)
    pool = multiprocessing.Pool(processes = 12)
    for i in range(50):
        pool.apply_async(start_process_training_data, (i, config_dict, ))   #维持执行的进程总数为processes，当一个进程执行完毕后会添加新的进程进去
    pool.close()
    pool.join() 
