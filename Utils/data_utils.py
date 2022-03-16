import torch
import numpy as np
import copy


class TrainingSet:
    def __init__(self, replay_buffer_config):
        self.replay_buffer_config = replay_buffer_config
        # ---------- 这个参数表示的就是batch size -------------
        self.batch_size = self.replay_buffer_config['batch_size']
        # ---------- 这个参数表示有多少个智能体 ---------------
        self.agent_nums = self.replay_buffer_config['agent_nums']
        # ---------- 这个参数表示传入的信道矩阵有多少个用户参与 ----------
        self.seq_len = self.replay_buffer_config['seq_len']
        # ---------- 这个参数表示这个replay buffer最大有多少条数据 ---------
        self.max_capacity = self.replay_buffer_config['max_capacity']
        # ---------- 这个参数表示最多进行多少次解码操作---------------------
        self.max_decoder_time = self.replay_buffer_config['max_decoder_time']
        self.bs_antenna_nums = self.replay_buffer_config['bs_antenna_nums']
        self.transmit_antenna_dim = 2* self.bs_antenna_nums
        self.init_replay_buffer()
        
    def init_replay_buffer(self):
        # ------------- 这个函数是用来初始化一个replaybuffer ----------
        '''
        training_batch = {
            'current_state':{
                'agent_obs': {'agent1': value, 'agent2': value}
                'global_state': ['global_channel_matrix','global_average_reward','global_average_reward']
                }
            'target_value': {'agent1': value, 'agent2': value}
            'advantage': {'agent1': value, 'agent2': value}
            'action': {'agent1': value, 'agent2': value}
            'old_action_log_probs':{'agent1': value, 'agent2': value}
            'next_state':{
                'agent_obs': {'agent1': value, 'agent2': value}
                'global_state': ['global_channel_matrix','global_average_reward','global_average_reward']
            }
        }
        agent_obs['agent1'] = {
            'channel_matrix': 信道矩阵,维度为batch size * n_channel * user number * 32
            'average_reward': batch size * user number * 1
            'scheduling_count': batch size * user number * 1
        }
        target_value['agent1'] : batch_size * 2
        advantage_value['agent1']: batch_size * 2
        action['agent1']: batch_size * user_number 
        old_action_log_porbs['agent1']: batch_size * 1
        '''
        self.data_buffer = dict()
        # --------------- 给buffer提前开好空间用来存放 -------------
        self.data_buffer['target_state_value'] = np.zeros((self.max_capacity, 2, 1))
        self.data_buffer['instant_reward'] = np.zeros((self.max_capacity, 2, 1))
        self.data_buffer['current_state_value'] = np.zeros((self.max_capacity, 2, 1))
        self.data_buffer['old_network_value'] = np.zeros((self.max_capacity, 2, 1))
        self.data_buffer['advantages'] = np.zeros((self.max_capacity, 2, 1))
        self.data_buffer['done'] = np.zeros((self.max_capacity, 1))
        self.data_buffer['old_action_log_probs'] = dict()
        self.data_buffer['actions'] = dict()
        self.data_buffer['current_state'] = dict()
        self.data_buffer['next_state'] = dict()
        # --------------- 定义一下状态中的key ------------
        self.data_buffer['current_state']['global_state'] = dict()
        self.data_buffer['next_state']['global_state'] = dict()
        self.data_buffer['current_state']['agent_obs'] = dict()
        self.data_buffer['next_state']['agent_obs'] = dict()
        # --------------- 定义状态中的全局观测 --------------
        self.data_buffer['current_state']['global_state']['global_channel_matrix'] = np.zeros((self.max_capacity, self.agent_nums **2, self.seq_len, self.transmit_antenna_dim))
        self.data_buffer['current_state']['global_state']['global_average_reward'] = np.zeros((self.max_capacity, self.agent_nums, self.seq_len, 1))
        self.data_buffer['current_state']['global_state']['global_scheduling_count'] = np.zeros((self.max_capacity, self.agent_nums, self.seq_len, 1))
        self.data_buffer['next_state']['global_state']['global_channel_matrix'] = np.zeros((self.max_capacity, self.agent_nums **2, self.seq_len, self.transmit_antenna_dim))
        self.data_buffer['next_state']['global_state']['global_average_reward'] = np.zeros((self.max_capacity, self.agent_nums, self.seq_len, 1))
        self.data_buffer['next_state']['global_state']['global_scheduling_count'] = np.zeros((self.max_capacity, self.agent_nums, self.seq_len, 1))
        # -------------- 定义状态中的每个智能体的单独观测 ------------------------------------------------------------------------------
        for index in range(self.agent_nums):
            agent_key = 'agent_' + str(index)
            agent_obs = dict()
            agent_obs['channel_matrix'] = np.zeros((self.max_capacity, self.agent_nums, self.seq_len, self.transmit_antenna_dim))
            agent_obs['average_reward'] = np.zeros((self.max_capacity, self.seq_len, 1))
            agent_obs['scheduling_count'] = np.zeros((self.max_capacity, self.seq_len, 1))
            self.data_buffer['current_state']['agent_obs'][agent_key] = copy.deepcopy(agent_obs)
            self.data_buffer['next_state']['agent_obs'][agent_key] = copy.deepcopy(agent_obs)
            # ------------ 初始化剩下的变量--------------、
            self.data_buffer['old_action_log_probs'][agent_key] = np.zeros((self.max_capacity, 1))
            self.data_buffer['actions'][agent_key] = np.zeros((self.max_capacity, self.max_decoder_time, 1))
        # ---------------------------------------------------------------------------------------------------------------------------
        # ------------- 定义一个变量，用来记录当前填了多少条数据进来 ------------------------
        self.cursor = 0

    def clear(self):
        self.init_replay_buffer()
    
    @property
    def full_buffer(self):
        if self.cursor >= self.max_capacity:
            return True
        else:
            return False

    def append_instance(self, instance, logger):
        # ------------- 这个地方是添加数据进去，这个instance是一个列表 --------------
        instant_number = len(instance)
        logger.info("------------- 此次添加的数据个数为 {} ----------------".format(instant_number))
        for sample_index in range(instant_number):
            local_index = self.cursor % self.max_capacity
            self.data_buffer['target_state_value'][local_index, :, :] = instance[sample_index]['target_state_value']
            self.data_buffer['instant_reward'][local_index, :, :] = instance[sample_index]['instant_reward']
            self.data_buffer['current_state_value'][local_index, :, :] = instance[sample_index]['current_state_value']
            self.data_buffer['advantages'][local_index, :, :] = instance[sample_index]['advantages']
            self.data_buffer['old_network_value'][local_index, :, :] = instance[sample_index]['old_network_value']   
            self.data_buffer['done'][local_index] = instance[sample_index]['done']
            self.data_buffer['current_state']['global_state']['global_channel_matrix'][local_index, :, :, :] = instance[sample_index]['current_state']['global_state']['global_channel_matrix'] 
            self.data_buffer['current_state']['global_state']['global_average_reward'][local_index, :, :] = instance[sample_index]['current_state']['global_state']['global_average_reward']
            self.data_buffer['current_state']['global_state']['global_scheduling_count'][local_index,:, :] = instance[sample_index]['current_state']['global_state']['global_scheduling_count']
            self.data_buffer['next_state']['global_state']['global_channel_matrix'][local_index, :, :, :] = instance[sample_index]['next_state']['global_state']['global_channel_matrix'] 
            self.data_buffer['next_state']['global_state']['global_average_reward'][local_index, :, :] = instance[sample_index]['next_state']['global_state']['global_average_reward']
            self.data_buffer['next_state']['global_state']['global_scheduling_count'][local_index,:, :] = instance[sample_index]['next_state']['global_state']['global_scheduling_count']
            for index in range(self.agent_nums):
                agent_key = 'agent_' + str(index)
                self.data_buffer['current_state']['agent_obs'][agent_key]['channel_matrix'][local_index, :, :, :] = instance[sample_index]['current_state']['agent_obs'][agent_key]['channel_matrix']
                self.data_buffer['current_state']['agent_obs'][agent_key]['average_reward'][local_index, :, :] = instance[sample_index]['current_state']['agent_obs'][agent_key]['average_reward']
                self.data_buffer['current_state']['agent_obs'][agent_key]['scheduling_count'][local_index, :, :] = instance[sample_index]['current_state']['agent_obs'][agent_key]['scheduling_count']
                self.data_buffer['next_state']['agent_obs'][agent_key]['channel_matrix'][local_index, :, :, :] = instance[sample_index]['next_state']['agent_obs'][agent_key]['channel_matrix']
                self.data_buffer['next_state']['agent_obs'][agent_key]['average_reward'][local_index, :, :] = instance[sample_index]['next_state']['agent_obs'][agent_key]['average_reward']
                self.data_buffer['next_state']['agent_obs'][agent_key]['scheduling_count'][local_index, :, :] = instance[sample_index]['next_state']['agent_obs'][agent_key]['scheduling_count']
                self.data_buffer['old_action_log_probs'][agent_key][local_index, :] = instance[sample_index]['old_action_log_probs'][agent_key]
                self.data_buffer['actions'][agent_key][local_index, :, :] = instance[sample_index]['actions'][agent_key]
            self.cursor += 1

    def slice(self):
        # ----------------- 这个函数随机从replaybuffer中选择出来一个batch ---------------
        current_data_point = min(self.max_capacity, self.cursor)
        random_batch = np.random.choice(current_data_point, self.batch_size, replace=False)
        sample_dict = dict()
        sample_dict['target_state_value'] = self.data_buffer['target_state_value'][random_batch, :, :]
        sample_dict['instant_reward'] = self.data_buffer['instant_reward'][random_batch, :, :]
        sample_dict['advantages'] = self.data_buffer['advantages'][random_batch, :, :]
        sample_dict['done'] = self.data_buffer['done'][random_batch, :]
        sample_dict['old_network_value'] = self.data_buffer['old_network_value'][random_batch, :, :]
        sample_dict['current_state_value'] = self.data_buffer['current_state_value'][random_batch, :, :]
        sample_dict['old_action_log_probs'] = dict()
        sample_dict['actions'] = dict()
        sample_dict['current_state'] = dict()
        sample_dict['current_state']['global_state'] = dict()
        sample_dict['current_state']['agent_obs'] = dict()
        sample_dict['next_state'] = dict()
        sample_dict['next_state']['global_state'] = dict()
        sample_dict['next_state']['agent_obs'] = dict()

        sample_dict['current_state']['global_state']['global_channel_matrix'] = self.data_buffer['current_state']['global_state']['global_channel_matrix'][random_batch, :, :, :]
        sample_dict['current_state']['global_state']['global_average_reward'] = self.data_buffer['current_state']['global_state']['global_average_reward'][random_batch, :, :]
        sample_dict['current_state']['global_state']['global_scheduling_count'] = self.data_buffer['current_state']['global_state']['global_scheduling_count'][random_batch, :, :]
        sample_dict['next_state']['global_state']['global_channel_matrix'] = self.data_buffer['next_state']['global_state']['global_channel_matrix'][random_batch, :, :, :]
        sample_dict['next_state']['global_state']['global_average_reward'] = self.data_buffer['next_state']['global_state']['global_average_reward'][random_batch, :, :]
        sample_dict['next_state']['global_state']['global_scheduling_count'] = self.data_buffer['next_state']['global_state']['global_scheduling_count'][random_batch, :, :]
        for index in range(self.agent_nums):
            agent_key = 'agent_' + str(index)
            sample_dict['current_state']['agent_obs'][agent_key] = dict()
            sample_dict['next_state']['agent_obs'][agent_key] = dict()
            sample_dict['current_state']['agent_obs'][agent_key]['channel_matrix'] = self.data_buffer['current_state']['agent_obs'][agent_key]['channel_matrix'][random_batch, :, :, :]
            sample_dict['current_state']['agent_obs'][agent_key]['average_reward'] = self.data_buffer['current_state']['agent_obs'][agent_key]['average_reward'][random_batch, :, :]
            sample_dict['current_state']['agent_obs'][agent_key]['scheduling_count'] = self.data_buffer['current_state']['agent_obs'][agent_key]['scheduling_count'][random_batch, :, :]
            sample_dict['next_state']['agent_obs'][agent_key]['channel_matrix'] = self.data_buffer['next_state']['agent_obs'][agent_key]['channel_matrix'][random_batch, :, :, :]
            sample_dict['next_state']['agent_obs'][agent_key]['average_reward'] = self.data_buffer['next_state']['agent_obs'][agent_key]['average_reward'][random_batch, :, :]
            sample_dict['next_state']['agent_obs'][agent_key]['scheduling_count'] = self.data_buffer['next_state']['agent_obs'][agent_key]['scheduling_count'][random_batch, :, :]
            sample_dict['old_action_log_probs'][agent_key] = self.data_buffer['old_action_log_probs'][agent_key][random_batch, :]
            sample_dict['actions'][agent_key] = self.data_buffer['actions'][agent_key][random_batch, :, :]
        return sample_dict

def conver_data_format_to_torch_interference(obs_dict):
    # 这个函数是将使用rollout和环境交互后得到的数据传入到网络做预处理，总的来说就是放入到torch上面
    ''' 
        obs_dict["global_state"] = {}
            # ------- 这个需要进行permute操作
            obs_dict["global_state"]['global_channel_matrix']维度为sector_number * total_antennas * sector_number * (2*transmit_antennas)
            obs_dict["global_state"]['global_average_reward']维度为sector_number * total_antennas * 1
            obs_dict["global_state"]['global_scheduling_count']维度为sector_number * total_antennas * 1

        obs_dict['agent_obs] = {}
            obs_dict['agent_obs']['agent_0'] = {}
            obs_dict['agent_obs']['agent_1'] = {}
            ...

        obs_dict["agent_obs"]['agent_0']['channel_matrix']维度为sector_number * total_antennas * (2*transmit_antennas)
        obs_dict["agent_obs"]['agent_0']['average_reward']维度为total_antennas * 1
        obs_dict["agent_obs"]['agent_0']['scheduling_count']维度为total_antennas * 1

        return: 原样子返回，唯一需要注意的就是，global state里面的信道矩阵需要置换一下维度 ，然后添加batch size维度，所有数据变成CPU类型，最后需要将第一第二个维度进行叠加
    '''
    torch_format_dict = dict()
    torch_format_dict['global_state'] = dict()
    torch_format_dict['global_state']['global_channel_matrix'] = torch.FloatTensor(obs_dict['global_state']['global_channel_matrix']).unsqueeze(0)
    torch_format_dict['global_state']['global_average_reward'] = torch.FloatTensor(obs_dict['global_state']['global_average_reward']).unsqueeze(0)
    torch_format_dict['global_state']['global_scheduling_count'] = torch.FloatTensor(obs_dict['global_state']['global_scheduling_count']).unsqueeze(0)
    torch_format_dict['agent_obs'] = dict()
    for agent_key in obs_dict['agent_obs'].keys():
        torch_format_dict['agent_obs'][agent_key] = dict()
        torch_format_dict['agent_obs'][agent_key]['channel_matrix'] = torch.FloatTensor(obs_dict['agent_obs'][agent_key]['channel_matrix']).unsqueeze(0)
        torch_format_dict['agent_obs'][agent_key]['average_reward'] = torch.FloatTensor(obs_dict['agent_obs'][agent_key]['average_reward']).unsqueeze(0)
        torch_format_dict['agent_obs'][agent_key]['scheduling_count'] = torch.FloatTensor(obs_dict['agent_obs'][agent_key]['scheduling_count']).unsqueeze(0)
    return torch_format_dict

def convert_data_format_to_torch_training(training_batch, device_index):
    # 上面那个replaybuffer中有一个slice函数，这个函数就是切一个batch数据出来，然后，这个函数就是将上面那个batch的数据变成torch类型的数据
    torch_format_data = dict()
    torch_format_data['target_state_value'] = torch.FloatTensor(training_batch['target_state_value']).to(device_index)
    torch_format_data['instant_reward'] = torch.FloatTensor(training_batch['instant_reward']).to(device_index)
    torch_format_data['advantages'] = torch.FloatTensor(training_batch['advantages']).to(device_index)
    torch_format_data['done'] = torch.FloatTensor(training_batch['done']).to(device_index)
    torch_format_data['old_network_value'] = torch.FloatTensor(training_batch['old_network_value']).to(device_index)
    torch_format_data['current_state_value'] = torch.FloatTensor(training_batch['current_state_value']).to(device_index)
    torch_format_data['current_state'] = dict()
    torch_format_data['current_state']['global_state'] = dict()
    torch_format_data['current_state']['agent_obs'] = dict()
    torch_format_data['next_state'] = dict()
    torch_format_data['next_state']['global_state'] = dict()
    torch_format_data['next_state']['agent_obs'] = dict()
    torch_format_data['actions'] = dict()
    torch_format_data['old_action_log_probs'] = dict()
    torch_format_data['current_state']['global_state']['global_channel_matrix'] = torch.FloatTensor(training_batch['current_state']['global_state']['global_channel_matrix']).to(device_index)
    torch_format_data['current_state']['global_state']['global_average_reward'] = torch.FloatTensor(training_batch['current_state']['global_state']['global_average_reward']).to(device_index)
    torch_format_data['current_state']['global_state']['global_scheduling_count'] = torch.FloatTensor(training_batch['current_state']['global_state']['global_scheduling_count']).to(device_index)
    torch_format_data['next_state']['global_state']['global_channel_matrix'] = torch.FloatTensor(training_batch['next_state']['global_state']['global_channel_matrix']).to(device_index)
    torch_format_data['next_state']['global_state']['global_average_reward'] = torch.FloatTensor(training_batch['next_state']['global_state']['global_average_reward']).to(device_index)
    torch_format_data['next_state']['global_state']['global_scheduling_count'] = torch.FloatTensor(training_batch['next_state']['global_state']['global_scheduling_count']).to(device_index)
    for agent_key in training_batch['actions'].keys():
        torch_format_data['current_state']['agent_obs'][agent_key] = dict()
        torch_format_data['next_state']['agent_obs'][agent_key] = dict()
        torch_format_data['current_state']['agent_obs'][agent_key]['channel_matrix'] = torch.FloatTensor(training_batch['current_state']['agent_obs'][agent_key]['channel_matrix']).to(device_index)
        torch_format_data['current_state']['agent_obs'][agent_key]['average_reward'] = torch.FloatTensor(training_batch['current_state']['agent_obs'][agent_key]['average_reward']).to(device_index)
        torch_format_data['current_state']['agent_obs'][agent_key]['scheduling_count'] = torch.FloatTensor(training_batch['current_state']['agent_obs'][agent_key]['scheduling_count']).to(device_index)
        torch_format_data['next_state']['agent_obs'][agent_key]['channel_matrix'] = torch.FloatTensor(training_batch['next_state']['agent_obs'][agent_key]['channel_matrix']).to(device_index)
        torch_format_data['next_state']['agent_obs'][agent_key]['average_reward'] = torch.FloatTensor(training_batch['next_state']['agent_obs'][agent_key]['average_reward']).to(device_index)
        torch_format_data['next_state']['agent_obs'][agent_key]['scheduling_count'] = torch.FloatTensor(training_batch['next_state']['agent_obs'][agent_key]['scheduling_count']).to(device_index)
        # ------------------- 动作和概率的对数进行转换 ------------------------
        torch_format_data['actions'][agent_key] = torch.LongTensor(training_batch['actions'][agent_key]).to(device_index)
        torch_format_data['old_action_log_probs'][agent_key] = torch.FloatTensor(training_batch['old_action_log_probs'][agent_key]).to(device_index)
    return torch_format_data