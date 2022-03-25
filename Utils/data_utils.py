from random import sample
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
            'channel_matrix': 信道矩阵,维度为batch size * n_channel * user number * 64
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
    def buffer_size(self):
        return self.cursor
    
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
        random_batch = np.random.choice(current_data_point-1, self.batch_size, replace=False)
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


# ====================== Multi_cell_one_step replay buffer ================
class TraningSet_one_step_multi_cell:
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
        self.transmit_antenna_dim = self.replay_buffer_config['bs_antenna_nums']
        self.cyclic_index_matrix = np.array([[(i+j)%self.agent_nums for i in range(self.agent_nums)] for j in range(self.agent_nums)])
        self.init_replay_buffer()
        
    def init_replay_buffer(self):
        # ------------- 这个函数是用来初始化一个replaybuffer ----------
        '''
        data_dict = [{'state': copy.deepcopy(state), 'instant_reward':instant_SE_sum_list}]
        # ------------- old_action_log_probs是一个字典，每一个key的维度都bsx1 ---------
        data_dict[-1]['old_action_log_probs'] = dict()
        # ------------ actions也是一个字典，每一个key的维度都是bsxuser_nums ------------
        data_dict[-1]['actions'] = dict()
        for agent_index in range(self.agent.agent_nums):
            agent_key = "agent_" + str(agent_index)
            data_dict[-1]['old_action_log_probs'][agent_key] = joint_log_prob[:,agent_index,:]
            data_dict[-1]['actions'][agent_key] = actions[:,agent_index,:]
        # ------------- net work output 的维度为bsx1 -----------
        data_dict[-1]['old_network_value'] = net_work_output

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
        self.data_buffer = dict()
        # --------------- 给buffer提前开好空间用来存放 -------------
        self.data_buffer['instant_reward'] = np.zeros((self.max_capacity, 1))
        self.data_buffer['old_network_value'] = np.zeros((self.max_capacity, 1))
        # ---------------- 定义一些字典用来存档action，action_log_probs, state --------
        self.data_buffer['actions'] = dict()
        self.data_buffer['old_action_log_probs'] = dict()
        self.data_buffer['state'] = dict()
        self.data_buffer['state']['global_channel_matrix'] = dict()
        self.data_buffer['state']['global_channel_matrix']['real_part'] = np.zeros((self.max_capacity, self.agent_nums*self.agent_nums, self.seq_len, self.transmit_antenna_dim))
        self.data_buffer['state']['global_channel_matrix']['img_part'] = np.zeros((self.max_capacity, self.agent_nums*self.agent_nums, self.seq_len, self.transmit_antenna_dim))
        self.data_buffer['state']['agent_obs'] = dict()
        # -------------- 定义状态中的每个智能体的单独观测 ------------------------------------------------------------------------------
        for agent_index in range(self.agent_nums):
            agent_key = 'agent_' + str(agent_index)
            agent_obs = dict()
            agent_obs['channel_matrix'] = dict()
            agent_obs['channel_matrix']['main_matrix'] = dict()
            agent_obs['channel_matrix']['main_matrix']['real_part'] = np.zeros((self.max_capacity, self.seq_len, self.transmit_antenna_dim))
            agent_obs['channel_matrix']['main_matrix']['img_part'] = np.zeros((self.max_capacity, self.seq_len, self.transmit_antenna_dim))
            agent_obs['channel_matrix']['interference_matrix'] = dict()
            for interference_cell_index in self.cyclic_index_matrix[agent_index,:][1:]:
                interference_cell_channel = dict()
                interference_cell_channel['real_part'] = np.zeros((self.max_capacity, self.seq_len, self.transmit_antenna_dim))
                interference_cell_channel['img_part'] = np.zeros((self.max_capacity, self.seq_len, self.transmit_antenna_dim))
                agent_obs['channel_matrix']['interference_matrix']['interference_cell_{}'.format(interference_cell_index)] = interference_cell_channel
            self.data_buffer['state']['agent_obs'][agent_key] = agent_obs
            # ------------ 初始化剩下的变量--------------、
            self.data_buffer['old_action_log_probs'][agent_key] = np.zeros((self.max_capacity, 1))
            self.data_buffer['actions'][agent_key] = np.zeros((self.max_capacity, self.max_decoder_time))
        # ---------------------------------------------------------------------------------------------------------------------------
        # ------------- 定义一个变量，用来记录当前填了多少条数据进来 ------------------------
        self.cursor = 0

    def clear(self):
        self.init_replay_buffer()
    
    @property
    def buffer_size(self):
        return self.cursor
    
    @property
    def full_buffer(self):
        if self.cursor >= self.max_capacity:
            return True
        else:
            return False

    def append_instance(self, instance, logger):
        # ------------- 这个地方是添加数据进去，这个instance是一个列表 -------------- 
        for sample_index in range(len(instance)):
            for bs_index in range(instance[sample_index]['instant_reward'].shape[0]):
                local_index = self.cursor % self.max_capacity
                self.data_buffer['instant_reward'][local_index,:] = instance[sample_index]['instant_reward'][bs_index,:]
                self.data_buffer['old_network_value'][local_index,:] = instance[sample_index]['old_network_value'][bs_index,:]
                #  ------------- 对state中global state进行添加 ----------------
                self.data_buffer['state']['global_channel_matrix']['real_part'][local_index,:,:,:] = instance[sample_index]['state']['global_channel_matrix']['real_part'][bs_index,:,:,:]
                self.data_buffer['state']['global_channel_matrix']['img_part'][local_index,:,:,:] = instance[sample_index]['state']['global_channel_matrix']['img_part'][bs_index,:,:,:]
                for agent_index in range(self.agent_nums):
                    agent_key = 'agent_' + str(agent_index)
                    self.data_buffer['state']['agent_obs'][agent_key]['channel_matrix']['main_matrix']['real_part'][local_index,:,:] = instance[sample_index]['state']['agent_obs'][agent_key]['channel_matrix']['main_matrix']['real_part'][bs_index,:,:]
                    self.data_buffer['state']['agent_obs'][agent_key]['channel_matrix']['main_matrix']['img_part'][local_index,:,:] = instance[sample_index]['state']['agent_obs'][agent_key]['channel_matrix']['main_matrix']['img_part'][bs_index,:,:]
                    for interference_cell_index in self.cyclic_index_matrix[agent_index,:][1:]:
                        self.data_buffer['state']['agent_obs'][agent_key]['channel_matrix']['interference_matrix']['interference_cell_{}'.format(interference_cell_index)]['real_part'][local_index,:,:] = instance[sample_index]['state']['agent_obs'][agent_key]['channel_matrix']['interference_matrix']['interference_cell_{}'.format(interference_cell_index)]['real_part'][bs_index,:,:]
                        self.data_buffer['state']['agent_obs'][agent_key]['channel_matrix']['interference_matrix']['interference_cell_{}'.format(interference_cell_index)]['img_part'][local_index,:,:] = instance[sample_index]['state']['agent_obs'][agent_key]['channel_matrix']['interference_matrix']['interference_cell_{}'.format(interference_cell_index)]['img_part'][bs_index,:,:]
                    self.data_buffer['old_action_log_probs'][agent_key][local_index,:] = instance[sample_index]['old_action_log_probs'][agent_key][bs_index,:]
                    self.data_buffer['actions'][agent_key][local_index,:] = instance[sample_index]['actions'][agent_key][bs_index,:]
                self.cursor += 1
        # instant_number = len(instance) * instance[sample_index]['instant_reward'].shape[0]
        # logger.info("------------- 此次添加的数据个数为 {} ----------------".format(instant_number))

    def slice(self):
        # ----------------- 这个函数随机从replaybuffer中选择出来一个batch ---------------
        current_data_point = min(self.max_capacity, self.cursor)
        random_batch = sample(range(0,current_data_point), self.batch_size)
        sample_dict = dict()
        sample_dict['instant_reward'] = self.data_buffer['instant_reward'][random_batch,:]
        sample_dict['old_network_value'] = self.data_buffer['old_network_value'][random_batch,:]
        sample_dict['actions'] = dict()
        sample_dict['old_action_log_probs'] = dict()
        sample_dict['state'] = dict()
        sample_dict['state']['global_channel_matrix'] = dict()
        sample_dict['state']['global_channel_matrix']['real_part'] = self.data_buffer['state']['global_channel_matrix']['real_part'][random_batch,:,:]
        sample_dict['state']['global_channel_matrix']['img_part'] = self.data_buffer['state']['global_channel_matrix']['img_part'][random_batch,:,:]
        sample_dict['state']['agent_obs'] = dict()
        # -------------- 定义状态中的每个智能体的单独观测 ------------------------------------------------------------------------------
        for agent_index in range(self.agent_nums):
            agent_key = 'agent_' + str(agent_index)
            agent_obs = dict()
            agent_obs['channel_matrix'] = dict()
            agent_obs['channel_matrix']['main_matrix'] = dict()
            agent_obs['channel_matrix']['main_matrix']['real_part'] = self.data_buffer['state']['agent_obs'][agent_key]['channel_matrix']['main_matrix']['real_part'][random_batch,:,:]
            agent_obs['channel_matrix']['main_matrix']['img_part'] = self.data_buffer['state']['agent_obs'][agent_key]['channel_matrix']['main_matrix']['img_part'][random_batch,:,:]
            agent_obs['channel_matrix']['interference_matrix'] = dict()
            for interference_cell_index in self.cyclic_index_matrix[agent_index,:][1:]:
                interference_cell_channel = dict()
                interference_cell_channel['real_part'] = self.data_buffer['state']['agent_obs'][agent_key]['channel_matrix']['interference_matrix']['interference_cell_{}'.format(interference_cell_index)]['real_part'][random_batch,:,:]
                interference_cell_channel['img_part'] = self.data_buffer['state']['agent_obs'][agent_key]['channel_matrix']['interference_matrix']['interference_cell_{}'.format(interference_cell_index)]['img_part'][random_batch,:,:]
                agent_obs['channel_matrix']['interference_matrix']['interference_cell_{}'.format(interference_cell_index)] = interference_cell_channel
            sample_dict['state']['agent_obs'][agent_key] = agent_obs
            # ------------ 初始化剩下的变量--------------、
            sample_dict['old_action_log_probs'][agent_key] = self.data_buffer['old_action_log_probs'][agent_key][random_batch,:]
            sample_dict['actions'][agent_key] = self.data_buffer['actions'][agent_key][random_batch,:]
        return sample_dict


# ======================= Single_cell_one_step replay buffer ==================
class TraningSet_one_step_single_cell:
    def __init__(self, replay_buffer_config):
        self.replay_buffer_config = replay_buffer_config
        # ---------- 这个参数表示的就是batch size -------------
        self.batch_size = self.replay_buffer_config['batch_size']
        # ---------- 这个参数表示传入的信道矩阵有多少个用户参与 ----------
        self.seq_len = self.replay_buffer_config['seq_len']
        # ---------- 这个参数表示这个replay buffer最大有多少条数据 ---------
        self.max_capacity = self.replay_buffer_config['max_capacity']
        # ---------- 这个参数表示最多进行多少次解码操作---------------------
        self.max_decoder_time = self.replay_buffer_config['max_decoder_time']
        self.transmit_antenna_dim = self.replay_buffer_config['bs_antenna_nums']
        self.init_replay_buffer()
        
    def init_replay_buffer(self):
        # ------------- 这个函数是用来初始化一个replaybuffer ----------
        '''
            data_dict = {
                'state':{
                    'real_part':
                    'img_part':
                },
                'actions':
                'old_network_value':
                'instant_reward'
            }
        '''
        self.data_buffer = dict()
        # --------------- 给buffer提前开好空间用来存放 -------------
        self.data_buffer['instant_reward'] = np.zeros((self.max_capacity, 1))
        self.data_buffer['old_network_value'] = np.zeros((self.max_capacity, 1))
        # ---------------- 定义一些字典用来存档action，action_log_probs, state --------
        self.data_buffer['actions'] = np.zeros((self.max_capacity, self.max_decoder_time), dtype=int)
        self.data_buffer['old_action_log_probs'] = np.zeros((self.max_capacity, 1))
        self.data_buffer['state'] = dict()
        self.data_buffer['state']['real_part'] = np.zeros((self.max_capacity, self.seq_len, self.transmit_antenna_dim))
        self.data_buffer['state']['img_part'] = np.zeros((self.max_capacity, self.seq_len, self.transmit_antenna_dim))
        # ------------- 定义一个变量，用来记录当前填了多少条数据进来 ------------------------
        self.cursor = 0

    def clear(self):
        self.init_replay_buffer()
    
    @property
    def buffer_size(self):
        return self.cursor
    
    @property
    def full_buffer(self):
        if self.cursor >= self.max_capacity:
            return True
        else:
            return False

    def append_instance(self, instance, logger):
        # ------------- 这个地方是添加数据进去，这个instance是一个列表 -------------- 
        for sample_index in range(len(instance)):
            for bs_index in range(instance[sample_index]['instant_reward'].shape[0]):
                local_index = self.cursor % self.max_capacity
                self.data_buffer['instant_reward'][local_index,:] = instance[sample_index]['instant_reward'][bs_index,:]
                self.data_buffer['old_network_value'][local_index,:] = instance[sample_index]['old_network_value'][bs_index,:]
                self.data_buffer['actions'][local_index, :] = instance[sample_index]['actions'][bs_index,:]
                self.data_buffer['old_action_log_probs'][local_index, :] = instance[sample_index]['old_action_log_probs'][bs_index, :]
                self.data_buffer['state']['real_part'][local_index, :, :] = instance[sample_index]['state']['real_part'][bs_index,:,:]
                self.data_buffer['state']['img_part'][local_index, :, :] = instance[sample_index]['state']['img_part'][bs_index,:,:]
                self.cursor += 1
        instant_number = len(instance) * instance[sample_index]['instant_reward'].shape[0]
        logger.info("------------- 此次添加的数据个数为 {} ----------------".format(instant_number))

    def slice(self):
        # ----------------- 这个函数随机从replaybuffer中选择出来一个batch ---------------
        current_data_point = min(self.max_capacity, self.cursor)
        random_batch = sample(range(0,current_data_point), self.batch_size)
        sample_dict = dict()
        sample_dict['instant_reward'] = self.data_buffer['instant_reward'][random_batch,:]
        sample_dict['old_network_value'] = self.data_buffer['old_network_value'][random_batch,:]
        sample_dict['actions'] = self.data_buffer['actions'][random_batch, :]
        sample_dict['old_action_log_probs'] = self.data_buffer['old_action_log_probs'][random_batch, :]
        sample_dict['state'] = dict()
        sample_dict['state']['real_part'] = self.data_buffer['state']['real_part'][random_batch, :, :]
        sample_dict['state']['img_part'] = self.data_buffer['state']['img_part'][random_batch,:, :]
        return sample_dict


def convert_data_format_to_torch_interference(obs_dict):
    # 这个函数是将使用rollout和环境交互后得到的数据传入到网络做预处理，总的来说就是放入到torch上面
    torch_format_dict = dict()
    for key,value in obs_dict.items():
        if isinstance(value, dict):
            torch_format_dict[key] = convert_data_format_to_torch_interference(value)
        else:
            torch_format_dict[key] = torch.FloatTensor(value)
    return torch_format_dict

def convert_data_format_to_torch_training(training_batch, device_index, long_tensor=False):
    # 上面那个replaybuffer中有一个slice函数，这个函数就是切一个batch数据出来，然后，这个函数就是将上面那个batch的数据变成torch类型的数据
    torch_format_dict = dict()
    for key,value in training_batch.items():
        if isinstance(value, dict):
            if key == 'actions':
                torch_format_dict[key] = convert_data_format_to_torch_training(value, device_index, long_tensor=True)
            else:
                torch_format_dict[key] = convert_data_format_to_torch_training(value, device_index)
        else:
            if long_tensor:
                torch_format_dict[key] = torch.LongTensor(value).to(device_index)
            else:
                torch_format_dict[key] = torch.FloatTensor(value).to(device_index)
    return torch_format_dict