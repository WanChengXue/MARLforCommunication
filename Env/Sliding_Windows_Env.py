# 这个函数用来模仿通信环境的变化
import numpy as np
import os
import copy
import pathlib
import random

import gym
from Instant_Reward import calculate_instant_reward
from utils.config_parse import parse_config

class Environment(gym.Env):
    '''
    定义一个环境，这个环境通过随机数种子，任意选择一个载波（1-50）信道，然后从中截取一段固定长度的TTI信道作为模拟环境
    信道的起始位置TTI也不是固定的，需要通过这个随机数种子计算出来。
    '''
    def __init__(self, config_path, random_seed=None):
        self.config_dict = parse_config(config_path)
        # ================ 定义工程参数 =================
        self.user_nums = self.config_dict['user_nums']
        self.sector_nums = self.config_dict['sector_nums']
        self.cell_nums = self.config_dict['cell_nums']
        self.agent_nums = self.user_nums * self.sector_nums
        self.bs_antenna_nums = self.config_dict['bs_antenna_nums']
        self.total_antenna_nums = self.user_nums * self.bs_antenna_nums
        self.sliding_windows_length = self.config_dict['sliding_windows_length']
        self.transmit_power = self.config_dict['transmit_power']
        self.noise_power = self.config_dict['noise_power']
        self.velocity = self.config_dict['velocity']
        self.eval_mode = self.config_dict['eval_mode']
        self.save_data_folder = self.config_dict['save_data_folder']
        self.subcarrier_nums = self.config_dict['subcarrier_nums']
        self.delay_time_window = self.config_dict['delay_time_window']
        self.training_data_total_TTI_length = self.config_dict['training_data_total_TTI_length']
        self.eval_data_total_TTI_length = self.config_dict['eval_data_total_TTI_length']
        self.root_path = pathlib.Path('/'.join(os.path.realpath(__path__).split('/')[:-2]))
        self.min_user_average_se  = self.config_dict['min_user_average_se']
        self.max_user_pf_value = self.config_dict['max_user_pf_value']
        # ================ 组合出来的到的路径就是数据进行读取的地址 ===================
        self.load_data_path = self.root_path / self.save_data_folder / (str(self.user_nums)+'_user') / (str(self.velocity)+'KM')
        # ======================================================================
        if random_seed is None:
            self.random_seed = random.randint(0, 1000000)
        else:
            self.random_seed  = random_seed
        self.TTI_count = 0
        
        self.training_data_path = self.args.data_folder
        self.training_data = np.load(self.training_data_path)
        self.total_training_sample = self.training_data.shape[-1]
        self.legal_range = [self.args.min_stream, self.args.max_stream]

    def load_training_data(self):
        # 载入训练数据,根据随机数种子来看，首先是对随机数取余数，看看读取哪个载波
        load_subcarrier_index = self.random_seed % self.subcarrier_nums
        loaded_file_name = self.load_data_path / ('training_channel_file_' +str(load_subcarrier_index) + '.npy')
        channel_data = np.load(loaded_file_name)
        # 需要随机生成一个随机数，作为开始采样的位置
        max_start_TTI = self.training_data_total_TTI_length - self.sliding_windows_length
        start_TTI = np.random.choice(max_start_TTI, 1).item()
        end_TTI = start_TTI + self.sliding_windows_length
        self.simulation_channel = channel_data[:,:,:,:,:,:,:,start_TTI:end_TTI]

    def load_eval_data(self):
        # 载入eval数据集
        load_subcarrier_index = self.random_seed % self.subcarrier_nums
        loaded_file_name = self.load_data_path / ('eval_channel_file_' + str(load_subcarrier_index) + '.npy')
        self.simulation_channel = np.load(loaded_file_name)

    def reset(self, random_seed = None):
        if random_seed is not None:
            self.random_seed = random_seed

        # 定义所有小区所有用户的初始用户奖励向量
        init_se = np.zeros((self.cell_nums, self.user_nums))
        channel_matrix = self.simulation_channel[:,:,:,:,:,:,:,self.TTI_count]
        if self.eval_mode:
            # 如果是评估模式,就加载评估数据
            self.load_eval_data()
        else:
            self.load_training_data()
        self.current_channel_matrix = channel_matrix
        self.current_average_se = init_se
        state = dict()
        state['channel_matrix'] = copy.deepcopy(channel_matrix)
        state['average_se'] = copy.deepcopy(init_se)
        return state
        
    def decide_clip_operation(self):
        # 这个函数用来判断当前决策需不需要进行clip操作，评判标准是所有用户的平均容量都大于1了
        if (self.current_average_se > self.min_user_average_se).all():
            return True
        else:
            return False

    def step(self, action_list):
        instant_se = calculate_instant_reward(self.current_state, action_list)
        # 计算PF因子，这里会出现一个问题哈，算法最开始运行的那段时间，reward会特别的大，因此需要进行clamp操作，就给1。只有当所有的用户平均容量都到了1以上，才进行解锁
        if self.decide_clip_operation():
            proportional_factor = np.clip(instant_se / (1e-6 + self.current_average_se), 0, self.max_user_pf_value)
        else:
            proportional_factor = instant_se / (1e-6 + self.current_average_se)

        # 更新平均se矩阵
        self.current_average_se = (1-1/self.delay_time_window) * self.current_average_se + 1/self.delay_time_window * instant_se
        self.TTI_count += 1
        # 会碰到越界的情况，如果说当前的数据已经结束了，那么读取新的状态会报错
        terminate = self.is_terminal()
        if not terminate:
            self.current_state = self.simulation_channel[:,:,:,:,:,:,:,self.TTI_count]
        next_state = dict()
        next_state['channel_matrix'] = copy.deepcopy(self.current_state)
        next_state['averae_se'] = copy.deepcopy(self.current_average_se)
        return next_state, proportional_factor, terminate


    def is_terminal(self):
        if self.eval_mode:
            if self.TTI_count == self.eval_data_total_TTI_length:
                return True
            else:
                return False
        else:
            if self.TTI_count == self.sliding_windows_length:
                return True
            else:
                return False



    