from curses import termname
import random
import numpy as np
import os
import sys
current_path = os.path.abspath(__file__)
root_path = '/'.join(current_path.split('/')[:-2])
sys.path.append(root_path)
import random
import gym
from Instant_Reward import Single_cell_instant_reward

class Environment(gym.Env):
    '''
    定义一个环境，这个环境通过随机数种子，任意选择一个载波（1-50）信道，然后从中截取一段固定长度的TTI信道作为模拟环境
    信道的起始位置TTI也不是固定的，需要通过这个随机数种子计算出来, 这个环境是用来处理PF调度的
    '''
    def __init__(self, env_dict, random_seed=None):
        self.env_dict = env_dict
        # ================ 定义工程参数 =================
        self.user_nums = self.env_dict['user_nums']
        self.sector_nums = self.env_dict['sector_nums']
        self.bs_antenna_nums = self.env_dict['bs_antenna_nums']
        self.total_antenna_nums = self.env_dict['total_antenna_nums']
        self.sliding_windows_length = self.env_dict['sliding_windows_length']
        self.transmit_power = self.env_dict['transmit_power']
        self.noise_power = self.env_dict['noise_power']
        self.velocity = self.env_dict['velocity']
        # ---------- 文件的路径采用绝对位置 -------------
        self.eval_mode = self.env_dict.get('eval_mode',False)
        if self.eval_mode:
            self.sliding_windows_length = self.env_dict['eval_TTI']
        self.save_data_folder = self.generate_abs_path(self.env_dict['save_data_folder'] + '/' + str(self.user_nums) +'_user/'+str(self.velocity)+'KM')
        if not self.eval_mode:
            self.demonstration_data_folder = self.generate_abs_path(self.env_dict['demonstration_data_folder'] + '/' + str(self.user_nums) +'_user/'+str(self.velocity)+'KM')
        self.sub_carrier_nums = self.env_dict['sub_carrier_nums']
        self.delay_time_window = self.env_dict['delay_time_window']
        self.training_data_total_TTI_length = self.env_dict['training_data_total_TTI_length']
        self.min_user_average_se  = self.env_dict['min_user_average_se']
        self.max_user_pf_value = self.env_dict['max_user_pf_value']
    
        # ======================================================================
        if random_seed is None:
            self.random_seed = random.randint(0, 1000000)
        else:
            self.random_seed  = random_seed
        self.legal_range = [self.env_dict['min_stream_nums'], self.env_dict['max_stream_nums']]
        # ------------ 生成一个cyclic index matrix -------------
        self.reward_calculator = Single_cell_instant_reward(self.transmit_power, self.noise_power, self.user_nums, self.bs_antenna_nums)
        self.max_user_pf_value = self.env_dict['max_user_pf_value']
        # -------- 为了防止分母除以0 -------
        self.eps = 1e-9
        self.filter_factor = 1/self.sliding_windows_length

    def generate_abs_path(self, related_path):
        file_path = os.path.abspath(__file__)
        root_path = '/'.join(file_path.split('/')[:-3])
        return os.path.join(root_path, related_path)


    def load_training_data(self):
        # 载入训练数据,根据随机数种子来看，首先是对随机数取余数，看看读取哪个载波
        self.training_file_index = self.random_seed % self.sub_carrier_nums
        loaded_file_name = self.save_data_folder + '/training_channel_file_' +str(self.training_file_index) + '.npy'
        channel_data = np.load(loaded_file_name)
        # 需要随机生成一个随机数，作为开始采样的位置
        # max_start_TTI = self.training_data_total_TTI_length - self.sliding_windows_length
        # start_TTI = random.randint(0,max_start_TTI-1)
        # # ---------- 这个地方将start_TTI clamp在0- max_start_TTI - 1之间
        # end_TTI = start_TTI + self.sliding_windows_length
        # 随机生成一个矩阵进行采样
        self.random_tti = random.sample(range(0,self.training_data_total_TTI_length), self.sliding_windows_length+1)
        # ------- 通过squeeze函数之后，得到的仿真信道维度为，3 * 20 * 3 *16 * TTI,表示目的扇区 * 用户数目 * 源扇区 * 基站天线数目
        self.simulation_channel = (channel_data[:,:,:,:,:,:,:,self.random_tti]).squeeze()
        # ---------- current_TTI 表示当前的TTI时间点 -----------
        self.current_TTI = 0
        # ---------------- 下面两个变量分别表示每一个用户平均se和在过去一段时间内每一个用户被调度了多少次构成的矩阵 -------
        self.average_user_se = np.zeros((self.sector_nums, self.total_antenna_nums, 1))
        self.user_scheduling_counts = np.zeros((self.sector_nums, self.total_antenna_nums, 1))

    def load_eval_data(self, file_index):
        # 载入eval数据集
        if file_index is None:
            eval_file_number = self.env_dict.get('eval_file_number', random.randint(0, self.sub_carrier_nums-1))
        else:
            eval_file_number = file_index
        loaded_file_name = self.save_data_folder + '/eval_channel_file_' + str(eval_file_number) + '.npy'
        # ============== TODO 这个地方不是很完善，对于测试文件来说，需要测试所有的文件 ====================
        self.simulation_channel = (np.load(loaded_file_name)).squeeze()
        # ---------- current_TTI 表示当前的TTI时间点 -----------
        self.current_TTI = 0
        # ---------------- 下面两个变量分别表示每一个用户平均se和在过去一段时间内每一个用户被调度了多少次构成的矩阵 -------
        self.average_user_se = np.zeros((self.sector_nums, self.total_antenna_nums, 1))
        self.user_scheduling_counts = np.zeros((self.sector_nums, self.total_antenna_nums, 1))


    def reset(self, file_index=None):
        if self.eval_mode:
            # 如果是评估模式,就加载评估数据
            self.load_eval_data(file_index)
        else:
            self.load_training_data()
        # ------------- 构建输入的状态集 ------------
        '''
            state = {
                'agent_obs':{
                    'default':{
                        'real_part': bs * 20 * 16
                        'img_part': bs * 20 * 16
                    }
                }
                
            }
        '''
        return self.construct_state()
    
    def construct_state(self):
        state = dict()
        TTI = min(self.current_TTI, self.sliding_windows_length-1)
        # ========== 三个小区数据进行展开来，得到三倍的数据  =============
        state['real_part'] = np.stack([self.simulation_channel[sector_index, :, sector_index, 0:self.bs_antenna_nums, TTI] for sector_index in range(self.sector_nums)],0)
        # -------- 信道部分concatenate之后，得到的维度为3*20*16 -----------
        state['img_part'] = np.stack([self.simulation_channel[sector_index, :, sector_index, self.bs_antenna_nums: , TTI]for sector_index in range(self.sector_nums)],0)
        state['average_user_se'] = self.average_user_se
        state['user_scheduling_counts'] = self.user_scheduling_counts
        return state


    def convert_action_list_to_scheduling_mask(self, action_list):
        '''这个函数传入一个用户调度列表，然后转变为一个bool向量, 传入的是16的向量'''
        bool_mask = np.zeros(self.total_antenna_nums)
        for scheduling_user in action_list:
            if scheduling_user == 0:
                break
            else:
                bool_mask[scheduling_user-1] = 1
        return bool_mask

    def read_action_from_demonstration(self):
        # ---------- 这个函数表示直接从demonstration中将动作读取出来 ---------
        actions_list =[]
        reward_list = []
        for sector_index in range(self.sector_nums):
            action_file_name = self.demonstration_data_folder + '/' + str(self.training_file_index) + '_sector_' + str(sector_index) + '_scheduling_sequence.npy'
            instant_reward_file_name = self.demonstration_data_folder + '/' + str(self.training_file_index) + '_sector_' + str(sector_index) + '_se_sum_result.npy'
            # --------- 载入数据 --------
            sector_demonstration_actions = np.load(action_file_name)[self.random_tti, :]
            sector_demonstration_rewards = np.load(instant_reward_file_name)[self.random_tti]
            actions_list.append(sector_demonstration_actions)
            reward_list.append(sector_demonstration_rewards[:, np.newaxis])
        return np.concatenate(actions_list, 0), np.concatenate(reward_list, 0)


    def step(self, action_list):
        SE_list = []
        for sector_index in range(self.sector_nums):
            scheduling_mask = self.convert_action_list_to_scheduling_mask(action_list[sector_index,:])
            # ------------ 根据调度结果，修改所有用户被调度次数的矩阵 ----------
            self.user_scheduling_counts[sector_index, :, :] += scheduling_mask[:,np.newaxis]
            active_instant_se = self.reward_calculator.calculate_instant_reward(self.simulation_channel[sector_index,:,sector_index,: ,self.current_TTI], scheduling_mask)
            SE_list.append(active_instant_se)
        self.current_TTI += 1
        if self.current_TTI == self.sliding_windows_length:
            terminate = [True for i in range(self.sector_nums)]
        else:
            terminate = [False for i in range(self.sector_nums)]
        # ---------- 将这个列表变成array类型的数据 -------
        concatenate_se = np.stack(SE_list, 0)
        PF_matrix = self.calculate_pf_and_udpate_average_se(concatenate_se)
        instant_reward = [np.sum(concatenate_se,1), np.sum(PF_matrix,1)]
        # ---------- 构建next state -----------------
        next_state = self.construct_state()
        return next_state, instant_reward, terminate


    def calculate_pf_and_udpate_average_se(self, instant_reward_list):
        # --------- 这个函数用来计算一下PF值 -------
        '''
            在调度初期的时候，这个所有用户的平均SE都很小，接近0，因此在最开始的时候PF值会特别大
        '''
        current_PF_matrix = instant_reward_list / (self.average_user_se + self.eps)
        # ------- 对这个PF_matrix 进行clip操作, PF矩阵中所有的值都是0-2之间 ----------
        cliped_current_PF_matrix = np.clip(current_PF_matrix * self.filter_factor,0,self.max_user_pf_value)
        # ------- 更新一下所有用户的平均SE -------
        self.average_user_se = (1-self.filter_factor) * self.average_user_se + self.filter_factor * instant_reward_list
        if self.eval_mode:
            return current_PF_matrix
        else:
            return cliped_current_PF_matrix

    @property
    def get_user_average_se_matrix(self):
        return self.average_user_se


    @property
    def get_user_scheduling_count_matrix(self):
        return self.user_scheduling_counts

# -------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='/Learner/configs/config_single_cell_pointer_network.yaml', help='yaml format config')
    args = parser.parse_args()
    # ------------- 构建绝对地址 --------------
    # Linux下面是用/分割路径，windows下面是用\\，因此需要修改
    abs_path = '/'.join(os.path.abspath(__file__).split('/')[:-2])
    # abs_path = '/'.join(os.path.abspath(__file__).split('\\')[:-2])
    concatenate_path = abs_path + args.config_path
    from Utils.config_parse import parse_config
    config = parse_config(concatenate_path)
    test_env = Environment(config['env']) 
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
    test_action = np.array([[16,  9, 12, 10,  8, 17,  7, 18, 15, 13, 19,  5, 11,  0,  0,  0],
        [13,  1, 14,  5, 12,  7, 19, 20,  6, 11,  4,  9,  8,  3, 16, 15],
        [16,  9, 12, 10,  8, 17,  7, 18, 15, 13, 19,  5, 11,  0,  0,  0]
        ])

    # test_action = [np.array([
    #     [0,  0,  0, 0,  0,  0,  0,  0,  0, 0,  0,  0, 0,  0,  0,  0],
    #     [16,  9, 12, 10,  8, 17,  7, 18, 15, 13, 19,  5, 11,  0,  0,  0],
    #     [13,  1, 14,  5, 12,  7, 19, 20,  6, 11,  4,  9,  8,  3, 16, 15],
    #     [13,  1, 14,  5, 12,  7, 19, 20,  6, 11,  4,  9,  8,  3, 16, 15],
    #     [16,  9, 12, 10,  8, 17,  7, 18, 15, 13, 19,  5, 11,  0,  0,  0],
    #     [16,  9, 12, 10,  8, 17,  7, 18, 15, 13, 19,  5, 11,  0,  0,  0]
    # ])]
    test_env.step(test_action)
    # convert_list = convert_action_list_to_scheduling_mask(test_action)
    # action_label = np.array(
    #     [
    #         [ 0,  0,  0, 0,  1,  0, 1,  1,  1, 1,  1,  1, 1,  0,  1, 1,  1,  1,1,0],
    #         [ 1,  0,  1, 1,  1,  1, 1,  1,  1, 0,  1,  1, 1,  1,  1, 1,  0,  0,1,1],
    #         [ 0,  0,  0, 0,  1,  0, 1,  1,  1, 1,  1,  1, 1,  0,  1, 1,  1,  1,1,0]
    #     ]
    # )
    # print(convert_list)
