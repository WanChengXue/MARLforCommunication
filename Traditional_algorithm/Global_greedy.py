import numpy as np
from tqdm import tqdm
import sys
import os
current_path = os.path.abspath(__file__)
root_path = '/'.join(current_path.split('/')[:-2])
sys.path.append(root_path)

from Utils import create_folder
from Utils.config_parse import load_yaml
from Env.Instant_Reward import Multi_cell_instant_reward
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
        self.max_decoder_time = self.env_dict['max_stream_nums']
        self.min_decoder_time = self.env_dict['min_stream_nums']
        # ---------- 文件的路径采用绝对位置 -------------
        self.save_data_folder = self.generate_abs_path(self.env_dict['save_data_folder'] + '/' + str(self.user_nums) +'_user/'+str(self.velocity)+'KM')
        self.cyclic_index_matrix = np.array([[(i+j)%self.agent_nums for i in range(self.agent_nums)] for j in range(self.agent_nums)])
        self.reward_calculator = Multi_cell_instant_reward(self.transmit_power, self.noise_power, self.cyclic_index_matrix, self.sector_nums, self.user_nums, self.bs_antenna_nums)

    def load_eval_file(self, eval_file_number):
        self.construct_save_path()
        # ----------- 载入测试文件 ------------
        loaded_file_name = self.save_data_folder + '/eval_channel_file_' + str(eval_file_number) + '.npy'
        self.simulation_channel = (np.load(loaded_file_name)).squeeze()

    def load_training_file(self, trainiing_file_number):
        self.construct_training_data_save_path()
        loaded_file_name = self.save_data_folder + '/training_channel_file_' + str(trainiing_file_number) + '.npy'
        self.simulation_channel = (np.load(loaded_file_name)).squeeze()

    def generate_abs_path(self, related_path):
        file_path = os.path.abspath(__file__)
        root_path = '/'.join(file_path.split('/')[:-3])
        return os.path.join(root_path, related_path)

    def construct_training_data_save_path(self):
        related_path = 'data_part/demonstration_data/multi_cell_scenario/' + str(self.user_nums) +'_user/'+str(self.velocity)+'KM'
        self.abs_result_save_prefix = self.generate_abs_path(related_path)
        create_folder(self.abs_result_save_prefix)

    def construct_save_path(self):
        # ----------- 构建结果保存路径 -------------
        related_path = 'data_part/Greedy_result/global_greedy/' + str(self.user_nums) +'_user/'+str(self.velocity)+'KM'
        self.abs_result_save_prefix = self.generate_abs_path(related_path)
        create_folder(self.abs_result_save_prefix)

    def greedy_add_users(self, last_action, step_dict):
        # 这个函数是遍历一次，找一个用户出来
        max_se = 0
        optimal_sector = None
        optimal_user = None
        meeting_constraint = self.all_sector_meet_min_scheduling(step_dict)
        # ----------------- 如果说所有小区的调度用户数目都满足最小限制，则跳过，否则就生成一个矩阵来追踪所有的调度结果 ----------------
        if not meeting_constraint:
            tracking_matrix = np.zeros((self.sector_nums, self.total_antenna_nums))
        for sector_index in range(self.sector_nums):
            # ---------- 如果说这个小区已经调度了最大用户数目的用户，就直接跳过这个循环 ---------
            if step_dict['sector_{}'.format(sector_index)] >= self.max_decoder_time or not self.reasonable_sector_scheduling(step_dict, sector_index):
                continue
            else:
                for user_index in range(self.user_nums):
                    current_action = copy.deepcopy(last_action)
                    if last_action[sector_index, user_index] == 0:
                        # 如果当前用户没有被选到，则动作进行跳转
                        # 调用函数计算给定的信道矩阵，以及在当前的调度序列下面的SE
                        current_action[sector_index, user_index] = 1
                        SE_array = self.reward_calculator.calculate_instant_reward(self.channel_matrix, current_action)
                        sum_SE = np.sum(np.sum(SE_array)).item() /(self.sector_nums * self.user_nums)
                        if meeting_constraint:
                            # ----------如果满足这个限制条件，那就自然执行了 -----------
                            if max_se < sum_SE:
                                max_se = sum_SE
                                optimal_sector = sector_index
                                optimal_user = user_index
                        else:
                            tracking_matrix[sector_index, user_index] = sum_SE
        if not meeting_constraint:
            # ----------- 寻找到追踪矩阵中，value最大的索引 --------------
            flatten_index = np.argmax(tracking_matrix)
            optimal_sector = flatten_index // self.total_antenna_nums
            optimal_user = flatten_index % self.total_antenna_nums
            max_se = tracking_matrix[optimal_sector, optimal_user]
        return max_se, optimal_sector, optimal_user


    def reasonable_sector_scheduling(self, scheduling_dict, sector_index):
        # ----------- 传入一个调度字典，如果某个sector已经调度了min stream number个用户后，其余sector一个没有调度，则当前sector不调度了-----
        resonable_sector_scheduling = True
        if scheduling_dict['sector_'+str(sector_index)] >=self.min_decoder_time:
            for sector_key in scheduling_dict.keys():
                if sector_key == 'sector_' + str(sector_index):
                    continue
                else:
                    if scheduling_dict[sector_key] <= self.min_decoder_time:
                        resonable_sector_scheduling = False
                        break
        return resonable_sector_scheduling
            
                
    def all_sector_resonable_scheduling(self, scheduling_dict):
        # --------- 传入一个调度字典，如果所有的sector的step都小于self.max_decode_time则返回True -------
        resonable_scheduling = True
        for sector_key in scheduling_dict.keys():
            if scheduling_dict[sector_key] <= self.max_decoder_time:
                pass
            else:
                resonable_scheduling = False
                break
        return resonable_scheduling

    def all_sector_meet_min_scheduling(self, scheduling_dict):
        # ---------- 传入一个调度字典，如果scheduling dict中所有的小区都满足了最小用户调度需求，就返回True，否则返回False -----------
        meeting_constraint = True
        for sector_key in scheduling_dict.keys():
            if scheduling_dict[sector_key] <self.min_decoder_time:
                meeting_constraint = False
                break
        return meeting_constraint

    def recyle_add_user(self):
        action_sector_dict = dict()
        step_dict = dict()
        for sector_index in range(self.sector_nums):
            action_sector_dict['sector_{}'.format(sector_index)] = []
            step_dict['sector_{}'.format(sector_index)] = 0

        current_scheduling_sequence = np.zeros((self.sector_nums, self.user_nums), dtype=int)
        current_max_se = 0
        while True:
            # 如果说所有的sector都调度了max decoder user个用户，调用greedy_add_users返回0，因此直接break了
            SE_after_new_user_append, new_user_sector_id, new_user_id = self.greedy_add_users(current_scheduling_sequence, step_dict)
            if self.all_sector_meet_min_scheduling(step_dict):
                if SE_after_new_user_append >= current_max_se and self.all_sector_resonable_scheduling(step_dict):
                    # -------- 满足的条件，新加入用户之后，系统性能必须上升，然后所有的sector的当前调度用户数目必须小于最大调度数目 ---------
                    current_scheduling_sequence[new_user_sector_id, new_user_id] = 1
                    action_sector_dict['sector_{}'.format(new_user_sector_id)].append(new_user_id+1)
                    step_dict['sector_{}'.format(new_user_sector_id)] += 1
                    current_max_se = SE_after_new_user_append
                else:
                    # 如果当前这一轮没有办法添加新的用户进去，就直接break循环
                    break
            else:
                # ------------ 如果说不满足每一个小区最低调度用户要求，就不管这个添加了新用户之后，SE必须上升的constraint ------------
                current_scheduling_sequence[new_user_sector_id, new_user_id] = 1
                action_sector_dict['sector_{}'.format(new_user_sector_id)].append(new_user_id+1)
                step_dict['sector_{}'.format(new_user_sector_id)] += 1
                current_max_se = SE_after_new_user_append

        # --------- 调度完成，将action_sector_dict中一些key添加0 ------
        action_sector_list = []
        for sector_index in range(self.sector_nums):
            if step_dict['sector_{}'.format(sector_index)] < self.max_decoder_time:
                action_sector_list.append(action_sector_dict['sector_{}'.format(sector_index)] + [0 for i in range(self.max_decoder_time-step_dict['sector_{}'.format(sector_index)])])
        return np.array(action_sector_list), current_max_se


    def simulation_training_file(self, file_index):
        self.load_training_file(file_index)
        TTI_length = self.simulation_channel.shape[-1]
        scheduling_sequence = []
        SE = []
        scheduling_sequence_saved_path = self.abs_result_save_prefix + '/' +str(file_index)+ '_scheduling_sequence.npy'
        SE_result_saved_path = self.abs_result_save_prefix + '/' +str(file_index)+ '_se_sum_result.npy'
        for TTI in tqdm(range(TTI_length)):
            self.channel_matrix = self.simulation_channel[:,:,:,:,TTI]
            greedy_scheduling_sequence, max_se = self.recyle_add_user()
            scheduling_sequence.append(greedy_scheduling_sequence)
            SE.append(max_se)
        # ------------------ 将来步
        np.save(scheduling_sequence_saved_path, np.stack(scheduling_sequence, 0))
        np.save(SE_result_saved_path, np.array(SE))

    def simulation(self, file_index):
        self.load_eval_file(file_index)
        TTI_length = self.simulation_channel.shape[-1]
        # TTI_length = 10
        scheduling_sequence = []
        SE = []
        scheduling_sequence_saved_path = self.abs_result_save_prefix + '/' +str(file_index)+ '_scheduling_sequence.npy'
        SE_result_saved_path = self.abs_result_save_prefix + '/' +str(file_index)+ '_se_sum_result.npy'
        for TTI in tqdm(range(TTI_length)):
            self.channel_matrix = self.simulation_channel[:,:,:,:,TTI]
            greedy_scheduling_sequence, max_se = self.recyle_add_user()
            scheduling_sequence.append(greedy_scheduling_sequence)
            SE.append(max_se)
        # ------------------ 将来步
        np.save(scheduling_sequence_saved_path, np.stack(scheduling_sequence, 0))
        np.save(SE_result_saved_path, np.array(SE))


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
    # test_greedy = Greedy(config_dict['env']) 
    # for i in range(50):
    #     test_greedy.simulation(i)
    # ----------- 开进程池 --------
    # start_process_training_data(0, config_dict)
    pool = multiprocessing.Pool(processes = 12)
    for i in range(50):
        pool.apply_async(start_process_training_data, (i, config_dict, ))   #维持执行的进程总数为processes，当一个进程执行完毕后会添加新的进程进去
    pool.close()
    pool.join()   #调用join之前，先调用close函数，否则会出错。执行完close后不会有新的进程加入到pool,join函数等待所有子进程结束

