# 这个函数用来模仿通信环境的变化
import numpy as np
import os
import sys
current_path = os.path.abspath(__file__)
root_path = '/'.join(current_path.split('/')[:-2])
sys.path.append(root_path)
import copy
import pathlib
import random

import gym
from Instant_Reward import calculate_instant_reward
from Utils.config_parse import parse_config

class Environment(gym.Env):
    '''
    定义一个环境，这个环境通过随机数种子，任意选择一个载波（1-50）信道，然后从中截取一段固定长度的TTI信道作为模拟环境
    信道的起始位置TTI也不是固定的，需要通过这个随机数种子计算出来。
    '''
    def __init__(self, config_path, random_seed=None):
        self.config_dict = parse_config(config_path)
        self.env_dict = self.config_dict['env']
        # ================ 定义工程参数 =================
        self.user_nums = self.env_dict['user_nums']
        self.sector_nums = self.env_dict['sector_nums']
        self.cell_nums = self.env_dict['cell_nums']
        self.agent_nums = self.env_dict['agent_nums']
        self.bs_antenna_nums = self.env_dict['bs_antenna_nums']
        self.total_antenna_nums = self.user_nums * self.bs_antenna_nums
        self.sliding_windows_length = self.env_dict['sliding_windows_length']
        self.transmit_power = self.env_dict['transmit_power']
        self.noise_power = self.env_dict['noise_power']
        self.velocity = self.env_dict['velocity']
        self.save_data_folder = self.env_dict['save_data_folder']
        self.sub_carrier_nums = self.env_dict['sub_carrier_nums']
        self.delay_time_window = self.env_dict['delay_time_window']
        self.training_data_total_TTI_length = self.env_dict['training_data_total_TTI_length']
        self.eval_data_total_TTI_length = self.env_dict['eval_data_total_TTI_length']
        self.root_path = pathlib.Path('/'.join(os.path.realpath(__file__).split('/')[:-2]))
        self.min_user_average_se  = self.env_dict['min_user_average_se']
        self.max_user_pf_value = self.env_dict['max_user_pf_value']
        self.eval_mode = self.config_dict['eval_mode']
        # ======================================================================
        if random_seed is None:
            self.random_seed = random.randint(0, 1000000)
        else:
            self.random_seed  = random_seed
        self.TTI_count = 0
        
        self.legal_range = [self.env_dict['min_stream_nums'], self.env_dict['max_stream_nums']]
        # ------------ 生成一个cyclic index matrix -------------
        self.cyclic_index_matrix = np.array([[(i+j)%self.agent_nums for i in range(self.agent_nums)] for j in range(self.agent_nums)])

    def load_training_data(self):
        # 载入训练数据,根据随机数种子来看，首先是对随机数取余数，看看读取哪个载波
        load_subcarrier_index = self.random_seed % self.sub_carrier_nums
        loaded_file_name = self.save_data_folder + '/training_channel_file_' +str(load_subcarrier_index) + '.npy'
        channel_data = np.load(loaded_file_name)
        # 需要随机生成一个随机数，作为开始采样的位置
        max_start_TTI = self.training_data_total_TTI_length - self.sliding_windows_length
        start_TTI = np.random.choice(max_start_TTI, 1).clip(0, max_start_TTI-1).item()
        # ---------- 这个地方将start_TTI clamp在0- max_start_TTI - 1之间
        end_TTI = start_TTI + self.sliding_windows_length
        # ------- 通过squeeze函数之后，得到的仿真信道维度为，3 * 20 * 3 *16 * TTI,表示目的扇区 * 用户数目 * 源扇区 * 基站天线数目
        self.simulation_channel = (channel_data[:,:,:,:,:,:,:,start_TTI:end_TTI]).squeeze()

    def load_eval_data(self):
        # 载入eval数据集
        eval_file_number = self.config_dict.get('eval_file_number', random.randint(0, self.sub_carrier_nums-1))
        loaded_file_name = self.save_data_folder + '/eval_channel_file_' + str(eval_file_number) + '.npy'
        # ============== TODO 这个地方不是很完善，对于测试文件来说，需要测试所有的文件 ====================
        self.simulation_channel = (np.load(loaded_file_name)).squeeze()

    def reset(self, random_seed = None):
        if random_seed is not None:
            self.random_seed = random_seed
        if self.eval_mode:
            # 如果是评估模式,就加载评估数据
            self.load_eval_data()
        else:
            self.load_training_data()
        # 定义所有小区所有用户的初始用户奖励向量
        init_se = np.zeros((self.sector_nums, self.user_nums, 1))
        channel_matrix = self.simulation_channel[:,:,:,:,self.TTI_count]
        init_scheduling_count = np.zeros((self.sector_nums, self.user_nums, 1)) 
        self.current_channel_matrix = channel_matrix
        self.current_average_se = init_se
        self.current_scheduling_count = init_scheduling_count
        state = dict()
        # ========== 定义全局状态，以及每一个智能体的状态 =============
        state['global_state'] = dict()
        state['global_state']['global_channel_matrix'] = copy.deepcopy(channel_matrix)
        state['global_state']['global_average_reward'] = copy.deepcopy(init_se)
        state['global_state']['global_scheduling_count'] = copy.deepcopy(init_scheduling_count)
        state['agent_obs'] = dict()
        # ---------- 定义每一个智能体的状态 ---------------
        for agent_index in range(self.agent_nums):
            agent_key = "agent_" + str(agent_index)
            agent_obs = dict()
            # -------- 对于channel matrix而言，一定是主cell的信道放在最前面，然后干扰的信道放在后面 --------------
            agent_channel_matrix = channel_matrix[:,:, agent_index, :]
            # -------- 调整一下位置，123，231，312 -----------
            agent_obs['channel_matrix'] = agent_channel_matrix[self.cyclic_index_matrix[agent_index, :], :]
            agent_obs['average_reward'] = init_se[agent_index, :, :]
            agent_obs['scheduling_count'] = init_scheduling_count[agent_index, :, :]
            state['agent_obs'][agent_key] = agent_obs
        self.current_state = state
        return state
        
    def decide_clip_operation(self):
        # 这个函数用来判断当前决策需不需要进行clip操作，评判标准是所有用户的平均容量都大于1了
        if (self.current_average_se > self.min_user_average_se).all():
            return False
        else:
            return True

    def guide_reward(self):
        # ---------------- 这个地方是写引导奖励部分，基于规则设计 -----------------、
        '''
        1.当一个用户初次被调用，会给一个比较大的奖励
        2.基于调度数目，在过去1000次调度中，这个用户的调度次数比较少，那么计数奖励会稍微奖励
        3.边缘用户奖励上升，会给一个额外的奖励
        '''
        pass

    def specialize_reward(self, PF_matrix):
        # ----------------- 传入PF矩阵和，然后根据调度结果，计算出边缘用户的平均SE，将所有小区最差的那个用户拿出来 -------------
        PF_sum = np.sum(PF_matrix)
        # ----------------- 这样子调用了之后，得到的是一个3*1的向量，然后对这个向量计算平均值，得到的就是边缘用户的SE ----------
        sector_min_edge_average_SE = np.min(self.current_average_se, 1)
        average_sector_edge_SE = np.mean(sector_min_edge_average_SE)
        return PF_sum, average_sector_edge_SE

    def convert_action_list_to_scheduling_mask(self, action_list):
        '''这个函数传入一个列表，然后转变为一个bool矩阵'''
        random_action_mask = []
        for i in range(self.agent_nums):
            sector_mask = np.zeros(self.user_nums)
            for user_index in action_list[i].squeeze():
                if user_index != 0:
                    sector_mask[user_index-1] = 1
            # 添加-1，使得整个列表长度为20
            random_action_mask.append(sector_mask)
        # print(np.stack(random_action,0))
        concatenate_mask =np.expand_dims(np.stack(random_action_mask, 0), -1)
        return concatenate_mask
        
    def step(self, action_list):
        scheduling_mask = self.convert_action_list_to_scheduling_mask(action_list)
        instant_se = calculate_instant_reward(self.current_state['global_state']['global_channel_matrix'], scheduling_mask, self.noise_power, self.transmit_power)
        # 计算PF因子，这里会出现一个问题哈，算法最开始运行的那段时间，reward会特别的大，因此需要进行clamp操作，就给1。只有当所有的用户平均容量都到了1以上，才进行解锁
        if self.decide_clip_operation():
            # ---------------- 这个地方仅仅是将PF值限制到0-0.25之间，当所有的用户的平均性能都大于0.1，就没有这个限制了 ---------------
            proportional_factor = np.clip(instant_se / (1e-6 + self.current_average_se), 0, self.max_user_pf_value)
        else:
            proportional_factor = instant_se / (1e-6 + self.current_average_se)
        # 更新平均se矩阵
        self.current_average_se = (1-1/self.delay_time_window) * self.current_average_se + 1/self.delay_time_window * instant_se
        self.TTI_count += 1
        # 更新所有用户的调度数目情况
        self.current_scheduling_count += scheduling_mask
        terminate = self.is_terminal()
        if not terminate:
            channel_matrix = self.simulation_channel[:,:,:,:,self.TTI_count]
        else:
            channel_matrix = self.simulation_channel[:,:,:,:,self.TTI_count-1]
        next_state = dict()
        next_state['global_state'] = dict()
        next_state['global_state']['global_channel_matrix'] = copy.deepcopy(channel_matrix)
        next_state['global_state']['global_average_reward'] = copy.deepcopy(self.current_average_se)
        next_state['global_state']['global_scheduling_count'] = copy.deepcopy(self.current_scheduling_count)
        next_state['agent_obs'] = dict()
        # ---------- 定义每一个智能体的状态 ---------------
        for agent_index in range(self.agent_nums):
            agent_key = "agent_" + str(agent_index)
            agent_obs = dict()
            # -------- 对于channel matrix而言，一定是主cell的信道放在最前面，然后干扰的信道放在后面 --------------
            agent_channel_matrix = channel_matrix[:,:, agent_index, :]
            # -------- 调整一下位置，123，231，312 -----------
            agent_obs['channel_matrix'] = agent_channel_matrix[self.cyclic_index_matrix[agent_index, :], :]
            agent_obs['average_reward'] = self.current_average_se[agent_index, :, :]
            agent_obs['scheduling_count'] = self.current_scheduling_count[agent_index, :, :]
            next_state['agent_obs'][agent_key] = agent_obs
            self.current_state = next_state
        self.current_state = copy.deepcopy(next_state)
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



# -------------- 测试一下环境 --------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='/Learner/configs/config_pointer_network.yaml', help='yaml format config')
    args = parser.parse_args()
    # ------------- 构建绝对地址 --------------
    # Linux下面是用/分割路径，windows下面是用\\，因此需要修改
    abs_path = '/'.join(os.path.abspath(__file__).split('/')[:-2])
    # abs_path = '/'.join(os.path.abspath(__file__).split('\\')[:-2])
    concatenate_path = abs_path + args.config_path
    test_env = Environment(concatenate_path) 
    obs = test_env.reset()
    # 随机生成动作，然后测试step函数, 调度的动作形式为：
    '''
    random_action = [
        [2,5,6,1,7,0,0...],
        [4,6,1,0,0,0,...],
        [2,6,1,5,0,...]
    ]
    由于0表示的是终止用户的标志，因此，需要对所有用户减去1，之后，需要得到一个mask矩阵
    '''
    while True:
        random_action = []
        random_action_mask = []
        for i in range(test_env.agent_nums):
            scheduling_users = np.random.randint(test_env.legal_range[0],test_env.legal_range[1], 1) 
            sector_action = np.random.choice(test_env.user_nums, scheduling_users, replace=False) 
            sector_mask = np.zeros(test_env.user_nums)
            for user_index in sector_action:
                sector_mask[user_index] = 1
            # 添加-1，使得整个列表长度为20
            sector_action = np.append(sector_action, np.zeros(test_env.user_nums-scheduling_users)-1)
            random_action.append(sector_action)
            random_action_mask.append(sector_mask)
        # print(np.stack(random_action,0))
        concatenate_mask =np.expand_dims(np.stack(random_action_mask, 0), -1)
        next_state, reward, terminate = test_env.step(concatenate_mask)
        if terminate:
            break
    print(next_state['global_state']['global_scheduling_count'])
    print(next_state['global_state']['global_average_reward'])