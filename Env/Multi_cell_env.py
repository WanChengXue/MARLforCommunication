# 这个函数用来模仿通信环境的变化
import random
import numpy as np
import os
import sys
current_path = os.path.abspath(__file__)
root_path = '/'.join(current_path.split('/')[:-2])
sys.path.append(root_path)
import random
import gym
from Instant_Reward import calculate_instant_reward

class Environment(gym.Env):
    '''
    定义一个环境，这个环境通过随机数种子，任意选择一个载波（1-50）信道，然后从中截取一段固定长度的TTI信道作为模拟环境
    信道的起始位置TTI也不是固定的，需要通过这个随机数种子计算出来。
    '''
    def __init__(self, env_dict, random_seed=None):
        self.env_dict = env_dict
        # ================ 定义工程参数 =================
        self.user_nums = self.env_dict['user_nums']
        self.sector_nums = self.env_dict['sector_nums']
        self.cell_nums = self.env_dict['cell_nums']
        self.agent_nums = self.env_dict['agent_nums']
        self.bs_antenna_nums = self.env_dict['bs_antenna_nums']
        self.total_antenna_nums = self.env_dict['total_antenna_nums']
        self.sliding_windows_length = self.env_dict['sliding_windows_length']
        self.transmit_power = self.env_dict['transmit_power']
        self.noise_power = self.env_dict['noise_power']
        self.velocity = self.env_dict['velocity']
        # ---------- 文件的路径采用绝对位置 -------------
        self.save_data_folder = self.generate_abs_path(self.env_dict['save_data_folder'] + '/' + str(self.user_nums) +'_user/'+str(self.velocity)+'KM')
        self.sub_carrier_nums = self.env_dict['sub_carrier_nums']
        self.delay_time_window = self.env_dict['delay_time_window']
        self.training_data_total_TTI_length = self.env_dict['training_data_total_TTI_length']
        self.eval_data_total_TTI_length = self.env_dict['eval_data_total_TTI_length']
        self.min_user_average_se  = self.env_dict['min_user_average_se']
        self.max_user_pf_value = self.env_dict['max_user_pf_value']
        self.eval_mode = self.env_dict.get('eval_mode',False)
        # ======================================================================
        if random_seed is None:
            self.random_seed = random.randint(0, 1000000)
        else:
            self.random_seed  = random_seed
        self.legal_range = [self.env_dict['min_stream_nums'], self.env_dict['max_stream_nums']]
        # ------------ 生成一个cyclic index matrix -------------
        self.cyclic_index_matrix = np.array([[(i+j)%self.agent_nums for i in range(self.agent_nums)] for j in range(self.agent_nums)])
    
    def generate_abs_path(self, related_path):
        file_path = os.path.abspath(__file__)
        root_path = '/'.join(file_path.split('/')[:-3])
        return os.path.join(root_path, related_path)

    def load_training_data(self):
        # 载入训练数据,根据随机数种子来看，首先是对随机数取余数，看看读取哪个载波
        load_subcarrier_index = self.random_seed % self.sub_carrier_nums
        loaded_file_name = self.save_data_folder + '/training_channel_file_' +str(load_subcarrier_index) + '.npy'
        channel_data = np.load(loaded_file_name)
        # 需要随机生成一个随机数，作为开始采样的位置
        max_start_TTI = self.training_data_total_TTI_length - self.sliding_windows_length
        start_TTI = random.randint(0,max_start_TTI-1)
        # ---------- 这个地方将start_TTI clamp在0- max_start_TTI - 1之间
        end_TTI = start_TTI + self.sliding_windows_length
        # ------- 通过squeeze函数之后，得到的仿真信道维度为，3 * 20 * 3 *16 * TTI,表示目的扇区 * 用户数目 * 源扇区 * 基站天线数目
        self.simulation_channel = (channel_data[:,:,:,:,:,:,:,start_TTI:end_TTI]).squeeze()

    def load_eval_data(self):
        # 载入eval数据集
        eval_file_number = self.env_dict.get('eval_file_number', random.randint(0, self.sub_carrier_nums-1))
        loaded_file_name = self.save_data_folder + '/eval_channel_file_' + str(eval_file_number) + '.npy'
        # ============== TODO 这个地方不是很完善，对于测试文件来说，需要测试所有的文件 ====================
        self.simulation_channel = (np.load(loaded_file_name)).squeeze()

    def reset(self):
        if self.eval_mode:
            # 如果是评估模式,就加载评估数据
            self.load_eval_data()
        else:
            self.load_training_data()
        # ------------- 构建输入的状态集 ------------
        '''
            state={
                'global_channel_matrix':
                    {
                        'real_part': R^{bs*user_num*bs_antennas},
                        'img_part': R^{bs*user_num*bs_antennas}
                    },
                'agent_obs':
                    {
                        'agent_0':
                            {
                                'channel_matrix':
                                    {
                                        'main_matrix':
                                            {'real_part':R^{bs*user_num*bs_antennas}, 'img_part':R^{bs*user_num*bs_antennas}},
                                        'interference_matrix:
                                            {
                                                'interference_cell_1: {'real_part': xxx, 'img_part': xxx},
                                                'interference_cell_2:{'real_part': xxx, 'img_part':xxx}
                                            }
                                    }
                            }
                    }
            }
        '''
        state = dict()
        # ========== 定义全局状态，以及每一个智能体的状态 =============
        # ----------- 定义全局状态 ---------------
        state['global_channel_matrix'] = dict()
        state['global_channel_matrix']['real_part'] = self.simulation_channel[:,:,:,0:self.bs_antenna_nums,:].transpose(4,0,2,1,3).reshape(self.sliding_windows_length, self.sector_nums**2, self.user_nums, self.bs_antenna_nums)
        state['global_channel_matrix']['img_part'] = self.simulation_channel[:,:,:,self.bs_antenna_nums:,:].transpose(4,0,2,1,3).reshape(self.sliding_windows_length, self.sector_nums**2, self.user_nums, self.bs_antenna_nums)
        # ------------ 构建agent obs ----------
        state['agent_obs'] = dict()
        for agent_index in range(self.agent_nums):
            agent_key = "agent_" + str(agent_index)
            agent_obs = dict()
            agent_obs['channel_matrix'] = dict()
            agent_obs['channel_matrix']['main_matrix'] = dict()
            agent_obs['channel_matrix']['interference_matrix'] = dict()
            agent_obs['channel_matrix']['main_matrix']['real_part'] = self.simulation_channel[agent_index,:,agent_index,0:self.bs_antenna_nums,:].transpose(2,0,1)
            agent_obs['channel_matrix']['main_matrix']['img_part'] = self.simulation_channel[agent_index,:,agent_index,self.bs_antenna_nums:,:].transpose(2,0,1)
            for interference_cell_index in self.cyclic_index_matrix[agent_index,:][1:]:
                interference_cell_channel = dict()
                interference_cell_channel['real_part'] = self.simulation_channel[interference_cell_index,:,agent_index,0:self.bs_antenna_nums,:].transpose(2,0,1)
                interference_cell_channel['img_part'] = self.simulation_channel[interference_cell_index,:,agent_index,self.bs_antenna_nums:,:].transpose(2,0,1)
                agent_obs['channel_matrix']['interference_matrix']['interference_cell_{}'.format(interference_cell_index)] = interference_cell_channel
            state['agent_obs'][agent_key] = agent_obs
        return state


    def convert_action_list_to_scheduling_mask(self, action_list):
        '''这个函数传入一个用户调度列表，然后转变为一个bool矩阵, 传入的矩阵是3*16的向量'''
        bool_mask = np.zeros((self.agent_nums, self.total_antenna_nums))
        for agent_index in range(self.agent_nums):
            for scheduling_user in action_list[agent_index,:]:
                if scheduling_user == 0:
                    break
                else:
                    bool_mask[agent_index, scheduling_user-1] = 1
        return bool_mask
        
    def step(self, action_list):
        # ---------------------------- 这个函数传入一个调度列表，维度是3个bs*20的列表 ------------------------------
        # self.current_state['global_state']['global_channel_matrix'].reshape(self.sector_nums, self.sector_nums, self.user_nums, self.bs_antenna_nums*2).transpose(0,2,1,3)
        # -------- 首先把这个动作列表变成numpy类型，bs×3*20的矩阵 ---------
        numpy_type_action_list = np.stack(action_list, 1)
        SE_list = []
        for i in range(self.sliding_windows_length):
            # ----------- 将传入的动作矩阵变成0-1矩阵 -----------
            scheduling_mask = self.convert_action_list_to_scheduling_mask(numpy_type_action_list[i,:,:])
            active_instant_se = calculate_instant_reward(self.simulation_channel[:,:,:,:,i], scheduling_mask.squeeze(), self.noise_power, self.transmit_power, self.cyclic_index_matrix)
            SE_list.append(sum(sum(active_instant_se))/(self.user_nums*self.cell_nums))
        return SE_list

# -------------- 测试一下环境 --------------
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='/Learner/configs/config_multi_cell_pointer_network.yaml', help='yaml format config')
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
    test_action = np.array([[[16,  9, 12, 10,  8, 17,  7, 18, 15, 13, 19,  5, 11,  0,  0,  0],
        [13,  1, 14,  5, 12,  7, 19, 20,  6, 11,  4,  9,  8,  3, 16, 15],
        [16,  9, 12, 10,  8, 17,  7, 18, 15, 13, 19,  5, 11,  0,  0,  0]
        ],
        [[16,  9, 12, 10,  8, 17,  7, 18, 15, 13, 19,  5, 11,  0,  0,  0],
        [13,  1, 14,  5, 12,  7, 19, 20,  6, 11,  4,  9,  8,  3, 16, 15],
        [16,  9, 12, 10,  8, 17,  7, 18, 15, 13, 19,  5, 11,  0,  0,  0]
        ]])
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
