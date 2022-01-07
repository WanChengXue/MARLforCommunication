import torch
import numpy as np
import copy

from torch._C import device

class TrainingSet:
    def __init__(self, batch_size, max_capacity=10000):
        self.batch_size = batch_size
        self.max_capacity = max_capacity
        self.data_list = []
        
    def clear(self):
        self.data_list.clear()
    
    def len(self):
        return len(self.data_list)

    def append_instance(self, instance):
        self.data_list.extend([i.data for i in instance])

    def get_batched_obs(self, obs_list):
        sample = obs_list[0]
        if isinstance(sample, dict):
            batched_obs = dict()
            for key in sample:
                batched_obs[key] = self.get_batched_obs([obs[key] for obs in obs_list])
        elif isinstance(sample, (list, tuple)):
            batched_obs = [self.get_batched_obs(o) for o in zip(*obs_list)]
        else:
            batched_obs = np.asarray(obs_list)
            # TODO: 一维向量需不需要expand_dims(x, -1)
            if len(batched_obs.shape) == 1:
                batched_obs = np.expand_dims(batched_obs, -1)
        return batched_obs

    def slice(self, index_list, remove=False):
        pass

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

        return: 原样子返回，唯一需要注意的就是，global state里面的信道矩阵需要置换一下维度 ，然后添加batch size维度，所有数据变成CPU类型的就可以了2
    '''
    torch_format_dict = dict()
    torch_format_dict['global_state'] = dict()
    torch_format_dict['global_state']['channel_matrix'] = torch.FloatTensor(obs_dict['global_state']['global_channel_matrix']).permute(0,2,1,3).unsqueeze(0)
    torch_format_dict['global_state']['average_reward'] = torch.FloatTensor(obs_dict['global_state']['global_average_reward']).unsqueeze(0)
    torch_format_dict['global_state']['scheduling_count'] = torch.FloatTensor(obs_dict['global_state']['global_scheduling_count']).unsqueeze(0)
    torch_format_dict['agent_obs'] = dict()
    for agent_key in obs_dict['agent_obs'].keys():
        torch_format_dict['agent_obs'][agent_key] = dict()
        torch_format_dict['agent_obs'][agent_key]['channel_matrix'] = torch.FloatTensor(obs_dict['agent_obs'][agent_key]['channel_matrix']).unsqueeze(0)
        torch_format_dict['agent_obs'][agent_key]['average_reward'] = torch.FloatTensor(obs_dict['agent_obs'][agent_key]['average_reward']).unsqueeze(0)
        torch_format_dict['agent_obs'][agent_key]['scheduling_count'] = torch.FloatTensor(obs_dict['agent_obs'][agent_key]['scheduling_count']).unsqueeze(0)
    return torch_format_dict

def convert_data_format_to_torch_training(training_batch, parameter_sharing, device_index):
    # 这个函数是用来将从plasma中获取到的数据变成torch形式, 由于默认使用的共享网络参数，因此是CTDE方式进行的训练，当然也可以开三个智能体就是了。
        '''
        training_batch = {
            'obs': {'agent1': value, 'agent2': value}
            'global_state': ['global_channel_matrix','global_average_reward','global_average_reward']
            'target_value': {'agent1': value, 'agent2': value}
            'advantage': {'agent1': value, 'agent2': value}
            'action': {'agent1': value, 'agent2': value}
            'old_action_log_probs':{'agent1': value, 'agent2': value}
        }
        obs['agent1'] = {
            'channel_matrix': 信道矩阵,维度为batch size * n_channel * user number * 32
            'average_reward': batch size * user number * 1
            'scheduling_count': batch size * user number * 1
        }
        target_value['agent1'] : batch_size * 2
        advantage_value['agent1']: batch_size * 2
        action['agent1']: batch_size * user_number 
        old_action_log_porbs['agent1']: batch_size * 1
        user_number:表示的是一个小区里面有多少个用户, 32表示基站接收天线有16根,然后实数部分和复数部分拼接
        return:
            concatenate_obs_dict, concatenate_global_dict, concatenate_action_list, concatenate_old_action_log_probs, concatenate_advantage, concatenate_target_value
            concatenate_obs_dict['chennel_matrix'] : (batch_size * agent_number) * n_channel * user_number * 32
            concatenate_obs_dict['average_reward] : (batch_size * agent_number) * user_number * 32
            concatenate_obs_dict['scheduling_count] : (batch_size * agent_number)  * user number * 32
            concatenate_global_dict['global_channel_matrxi'] : (batch_size * agent_nubmer) * (n_channel ** 2) * user_number * 32
            concatenate_globa_dict['global_average_reward]: (batch_size * agent_number) * agent_number * user_number * 1
            concatenate_global_dict['global_scheduling_count] : (batch_size * agent_number) * agent_number * user_number * 1
            concatenate_action_list : (batch_size * agent_number ) * user_number
            concatenate_old_action_log_probs : (batch_size * agent_number) * 1
            concatenate_advantage: (batch_szie * agent_number) *2
            concatenate_target_value: (batch_size * agent_number) * 2
        '''
        obs = training_batch['obs']
        global_state = training_batch['global_state']
        # 这个target_value表示的是使用了GAE估计出来的Target V
        target_value = training_batch['target_value']
        advantage = training_batch['advantage']
        action = training_batch['action']
        # 这个值是表示旧策略下，给定状态，执行了动作A的概率值,这个是用来计算IF的
        old_action_log_probs = training_batch['old_action_log_probs']
        # 这个是通过采样神经网络参数前向计算得到的V值
        old_state_value = training_batch['old_state_value']
        return_value = training_batch['return_value']
        if parameter_sharing:
            # ================== 首先将观测矩阵进行合并,变成一个字典 ================
            concatenate_obs_dict = dict()
            concatenate_obs_dict['channel_matrix'] = []
            concatenate_obs_dict['average_reward'] = []
            concatenate_obs_dict['scheduling_count'] = []
            # ================== 将全局观测矩阵进行合并,变成一个字典 ===============
            concatenate_global_dict = dict()
            concatenate_global_dict['global_channel_matrix'] = []
            concatenate_global_dict['global_average_reward'] = []
            concatenate_global_dict['global_scheduling_count'] = []
            # ================== 接下来几个变量,分别表示将动作进行合并,log prob等等
            concatenate_action_list = []
            concatenate_old_action_log_probs = []
            concatenate_advantage = []
            concatenate_target_value = []
            concatenate_old_state_value = []
            concatenate_return_value = []
            for agent_index in obs.keys():
                # ========================== 这个地方是将每一个智能体的观测分开,然后填入到一个字典中
                concatenate_obs_dict['channel_matrix'].append(torch.FloatTensor(obs[agent_index]['channel_matrix']).to(device_index))
                concatenate_obs_dict['average_reward'].append(torch.FloatTensor(obs[agent_index]['average_reward']).to(device_index))
                concatenate_obs_dict['scheduling_count'].append(torch.FloatTensor(obs[agent_index]['scheduling_count']).to(device_index))
                # ========================== 对全局状态进行添加,按理来说,所有智能体的全局状态都是一样的,因此这个地方是直接复制了三份存入
                concatenate_global_dict['global_channel_matrix'].append(torch.FloatTensor(copy.deepcopy(global_state['global_channel_matrix'])).to(device_index))
                concatenate_global_dict['global_average_reward'].append(torch.FloatTensor(copy.deepcopy(global_state['global_average_reward'])).to(device_index))
                concatenate_global_dict['global_scheduling_count'].append(torch.FloatTensor(copy.deepcopy(global_state['global_scheduling_count'])).to(device_index))
                # =========================== 接下来就是对其它变量进行拼接,因为其它变量都是列表,因此不需要额外定义字典了
                concatenate_action_list.append(torch.LongTensor(action[agent_index]).to(device_index))
                concatenate_old_action_log_probs.append(torch.FloatTensor(old_action_log_probs[agent_index]).to(device_index))
                concatenate_advantage.append(torch.FloatTensor(advantage[agent_index]).to(device_index))
                concatenate_target_value.append(torch.FloatTensor(target_value[agent_index]).to(device_index))
                concatenate_old_state_value.append(torch.FloatTensor(old_state_value[agent_index].to(device_index)))
                concatenate_return_value.append(torch.FloatTensor(return_value[agent_index].to(device_index)))
            # ================= 使用cat命令,对上面所有的矩阵进行拼接 =====================
            concatenate_obs_dict['channel_matrix'] = torch.cat(concatenate_obs_dict['channel_matrix'], 0)
            concatenate_obs_dict['average_reward'] = torch.cat(concatenate_obs_dict['average_reward'], 0)
            concatenate_obs_dict['scheduling_count'] = torch.cat(concatenate_obs_dict['scheduling_count'], 0)
            concatenate_global_dict['global_channel_matrix'] = torch.cat(concatenate_global_dict['global_channel_matrix'], 0)
            concatenate_global_dict['global_average_reward'] = torch.cat(concatenate_global_dict['global_average_reward'], 0) 
            concatenate_global_dict['global_scheduling_count'] = torch.cat(concatenate_global_dict['global_scheduling_count'], 0)
            concatenate_action_list = torch.cat(concatenate_action_list, 0)
            concatenate_old_action_log_probs = torch.cat(concatenate_old_action_log_probs, 0)
            concatenate_advantage = torch.cat(concatenate_advantage, 0)
            concatenate_target_value = torch.cat(concatenate_target_value, 0)
            concatenate_old_state_value = torch.cat(concatenate_old_state_value, 0)
            torch_format_data = dict()
            torch_format_data['obs'] = concatenate_obs_dict
            torch_format_data['global_state'] = concatenate_global_dict
            torch_format_data['action_list'] = concatenate_action_list
            torch_format_data['old_action_log_probs'] = concatenate_old_action_log_probs
            torch_format_data['advantage'] = concatenate_advantage
            torch_format_data['target_value'] = concatenate_target_value
            torch_format_data['old_state_value'] = concatenate_old_state_value
            torch_format_data['return_value'] = concatenate_return_value
        else:
            pass
            # 这个地方是所有的智能体不共享策略网络的处理过程,暂时空着
        return torch_format_data