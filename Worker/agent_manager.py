import pathlib
import pickle
import lz4.frame as frame
import torch
import zmq
import random
import os
import sys

import numpy as np

from copy import deepcopy
current_path = os.path.abspath(__file__)
root_path = '/'.join(current_path.split('/')[:-2])
sys.path.append(root_path)
from Worker.policy_fetcher import fetcher
from Worker.agent import Agent, WolpAgent
from Utils.data_utils import convert_data_format_to_torch_interference, OUNoise

class AgentManager:
    def __init__(self, config_dict, context, statistic, logger, process_uid, port_num=None):
        ''' 
            由于是多智能体环境，可能需要创建多个智能体构成list，如果使用了paramerter sharing的话，就只创建一个智能体，否则，创建多个智能体构成一个list
        '''
        self.config_dict = config_dict
        self.statistic = statistic
        self.agent_nums = self.config_dict['env']['agent_nums']
        self.policy_config = self.config_dict['policy_config']
        self.PF_scheduling = self.policy_config.get('PF_scheduling', False)
        self.data_sender = context.socket(zmq.PUSH)
        self.parameter_sharing = self.policy_config['parameter_sharing']
        self.homogeneous_agent = self.policy_config['homogeneous_agent']
        self.using_wolpagent = self.policy_config.get('using_wolpagent', False)
        self.eval_mode = self.policy_config.get('eval_mode', False)
        self.model_info = None
        self.logger = logger
        # ------------ 连接数据发送服务 --------------------------------
        if not self.eval_mode:
            self.choose_target_port(port_num)
            # ------------ 创建一个获取最新策略的fetcher --------------------
            self.policy_fetcher = fetcher(context, self.config_dict, statistic, process_uid, logger)
            # ------------ 构建智能体 --------------------------------------
        self.construct_agent()
        self.logger.info("------------ 完成AgentManager的构建 -----------")

        
    def _add_critic_net_for_wolpagent(self):
        for agent_name in self.agent_name_list:
            self.agent['policy'][agent_name].add_critic_net(self.agent['critic'][agent_name].net_work)


    def construct_agent(self):
        # --------- 构建一下智能体 -----------
        self.agent = dict()
        for model_type in self.policy_config['agent'].keys():
            self.agent[model_type] = dict()
            for agent_name in self.policy_config['agent'][model_type].keys():
                policy_config = deepcopy(self.policy_config['agent'][model_type][agent_name])
                if self.using_wolpagent and model_type == 'policy':
                    self.agent[model_type][agent_name] = WolpAgent(policy_config)
                else:
                    self.agent[model_type][agent_name] = Agent(policy_config)
        self.agent_name_list = list(self.agent['policy'].keys())    
        # ----------- 初始化WolpAgent的critic net --------
        if self.using_wolpagent:
            self._add_critic_net_for_wolpagent()


    # def construct_ou_explorator(self):
    #     # -------------- 这个函数是用来构建ou探索用的，仅限于DDPG系列的算法 --------------
    #     self.ou_explorator = dict()
    #     for agent_name in self.policy_config['agent']['policy'].keys():
    #         ou_config = self.policy_config['agent']['ou_config']
    #         self.ou_explorator[agent_name] = OUNoise(ou_config)
    #         self.ou_explorator[agent_name].reset_state()

    def choose_target_port(self, port_num=None):
        # 定义这个worker的数据发送到哪一个data server,首先计算一下每台机器会有多少个数据服务
        data_server_per_machine = self.policy_config['device_number_per_machine'] * self.policy_config['server_number_per_device']
        # 由于可能用了多台机器进行训练，因此要看一下是不是多机多卡场景
        total_data_server = len(self.policy_config['machine_list']) * data_server_per_machine
        # 随机选择一个数据服务进行连接
        choose_data_server_index = random.randint(0, total_data_server-1)
        # 计算机器的索引，确定连接的ip，端口
        machine_index = choose_data_server_index // data_server_per_machine
        if port_num is None:
            port_num = choose_data_server_index % data_server_per_machine
        target_ip = self.policy_config['machine_list'][machine_index]
        target_port = self.policy_config['start_data_server_port'] + port_num
        self.logger.info("==================== 此智能体要发送数据到: {}, 连接的端口为: {} =================".format(target_ip, target_port))
        self.data_sender.connect("tcp://{}:{}".format(target_ip, target_port))
        
    def construct_eval_data_save_path(self,file_index):
        # ------- 构建instant reward和scheduling result两种数据的路径 -------
        if self.agent_nums == 1:
            instant_saved_path = dict()
            scheduling_result_path = dict()
            for sector_index in range(self.config_dict['env']['sector_nums']):
                if self.PF_scheduling:
                    instant_saved_path['sector_{}'.format(sector_index)] = dict()
                    instant_saved_path['sector_{}'.format(sector_index)]['average_se'] = os.path.join(self.policy_config['result_save_path'], '{}_sector_{}_average_se_result.npy'.format(file_index, sector_index))
                    instant_saved_path['sector_{}'.format(sector_index)]['PF_sum'] = os.path.join(self.policy_config['result_save_path'], '{}_sector_{}_PF_sum_result.npy'.format(file_index, sector_index))
                else:
                    instant_saved_path['sector_{}'.format(sector_index)] = os.path.join(self.policy_config['result_save_path'], '{}_sector_{}_se_sum_result.npy'.format(file_index, sector_index))
                scheduling_result_path['sector_{}'.format(sector_index)] = os.path.join(self.policy_config['result_save_path'], '{}_sector_{}_scheduling_sequence.npy'.format(file_index, sector_index))
        else:
            scheduling_result_path = dict()
            for sector_index in range(self.config_dict['env']['sector_nums']):
                scheduling_result_path['sector_{}'.format(sector_index)] = os.path.join(self.policy_config['result_save_path'], '{}_sector_{}_scheduling_sequence.npy'.format(file_index, sector_index))
            instant_saved_path =  os.path.join(self.policy_config['result_save_path'], '{}_se_sum_result.npy'.format(file_index))
        return instant_saved_path, scheduling_result_path


    def send_data(self, data):
        # -------- 将采样数据发送出去 -------------
        if self.eval_mode:
            # ------ 将采样数据保存到本地 ----- 
            file_index = int(data['file_index'])
            instant_saved_path, scheduling_result_path = self.construct_eval_data_save_path(file_index)
            if self.agent_nums == 1:
                for sector_index in range(self.config_dict['env']['sector_nums']):
                    # --------- 拆分奖励向量和动作向量 ----------
                    if self.PF_scheduling:
                        # ------- 单小区PF调度传入的数据的action维度为1000*3*16 -------
                        sector_action = data['actions'][:,sector_index,:]
                        sector_average_se = data['instant_reward']['average_se'][sector_index,:]
                        sector_PF_sum = data['instant_reward']['PF_sum'][:,sector_index,:]
                        np.save(scheduling_result_path['sector_{}'.format(sector_index)], sector_action)
                        np.save(instant_saved_path['sector_{}'.format(sector_index)]['average_se'], sector_average_se)
                        np.save(instant_saved_path['sector_{}'.format(sector_index)]['PF_sum'], sector_PF_sum)
                    else:
                        sector_instant_reward = data['instant_reward'][sector_index*self.config_dict['env']['eval_TTI']:(1+sector_index)*self.config_dict['env']['eval_TTI'],:]
                        sector_action = data['actions'][sector_index*self.config_dict['env']['eval_TTI']:(1+sector_index)*self.config_dict['env']['eval_TTI'],:]
                        np.save(scheduling_result_path['sector_{}'.format(sector_index)], sector_action)
                        np.save(instant_saved_path['sector_{}'.format(sector_index)], sector_instant_reward)
            else:
                np.save(instant_saved_path, data['instant_reward'])
                for sector_index in range(self.config_dict['env']['sector_nums']):
                    np.save(scheduling_result_path['sector_{}'.format(sector_index)], data['actions']['agent_{}'.format(sector_index)])
        else:
            compressed_data = frame.compress(pickle.dumps(data))
            self.data_sender.send(compressed_data)

    def compute_single_agent(self, obs, demonstration_ations=None):
        torch_format_data = convert_data_format_to_torch_interference(obs)
        if demonstration_ations is None:
            if self.using_wolpagent:
                action = self.agent['policy'][self.agent_name_list[0]].compute(torch_format_data)
                # --------------- 需要做OU探索 --------------
                if self.eval_mode:
                    action = self.agent['policy'][self.agent_name_list[0]].search_action(torch_format_data, action.numpy())
                else:
                    batch_size = action.shape[0]
                    action_dim = action.shape[1]
                    noise = 0.5 * torch.randn(batch_size, action_dim)
                    noised_action = action + noise
                    clamped_action = torch.clamp(noised_action, min=0.0, max=1.0)
                    # ---------------- 进行k近邻搜索,batch size * action dim ------------
                    action = self.agent['policy'][self.agent_name_list[0]].search_action(torch_format_data, clamped_action.numpy())
            else:
                action_log_prob, action= self.agent['policy'][self.agent_name_list[0]].compute(torch_format_data)
                state_value = self.agent['critic'][self.agent_name_list[0]].compute_state_value(torch_format_data)
        else:
            if self.using_wolpagent:
                action = demonstration_ations.astype(np.float64)
            else:
                torch_format_action = torch.LongTensor(demonstration_ations)
                action_log_prob = self.agent['policy'][self.agent_name_list[0]].compute(torch_format_data, torch_format_action)
                state_value = self.agent['critic'][self.agent_name_list[0]].compute_state_value(torch_format_data)

        if self.eval_mode:
            return action.numpy()
            
        
        if demonstration_ations is None:
            if self.using_wolpagent:
                return action.numpy()
            else:
                return action_log_prob.numpy(), action.numpy(), state_value.numpy()
        else:
            if self.using_wolpagent:
                return action.numpy()
            else:
                return action_log_prob.numpy(), state_value.numpy()
        
    def compute_state_value(self, obs):
        torch_format_data = convert_data_format_to_torch_interference(obs)
        state_value = self.agent['critic'][self.agent_name_list[0]].compute_state_value(torch_format_data)
        return state_value.numpy()
        
    def compute_multi_agent(self, obs):
        # -------- 这个函数是用使用神经网络计算动作，以及动作对应的概率 ---------
        # 首先将这个obs_dict变成pytorch支持的数据，由于采样的时候，统一使用cpu就可以了，不需要用 GPU
        torch_format_data = convert_data_format_to_torch_interference(obs)
        # ----- 需要将动作构成列表，然后回传，以及将对应的log prob和prob -------
        joint_log_prob_list = []
        actions = []
        for agent_key in torch_format_data['agent_obs'].keys():
            active_agent_obs = torch_format_data['agent_obs'][agent_key]
            if self.parameter_sharing:
                agent_log_prob, agent_action = self.agent['policy'][self.agent_name_list[0]].compute(active_agent_obs)
            else:
                agent_log_prob, agent_action = self.agent['policy'][agent_key].compute(active_agent_obs)
            # ----------- 这个地方需要将数据变成numpy类型 ------------
            joint_log_prob_list.append(agent_log_prob.numpy())
            actions.append(agent_action.numpy())
        if self.eval_mode:
            return actions
        # -------- 这个地方计算一下当前的状态值 ---------------
        with torch.no_grad():
            state_value = self.agent['critic'][self.agent_name_list[0]].compute_state_value(torch_format_data['global_channel_matrix'])
        if isinstance(state_value, tuple):
            state_value = np.array(item.numpy() for item in state_value)
        else:
            state_value = state_value.numpy()
        return joint_log_prob_list, actions, state_value

    def read_model_information_from_yaml_file(self):
        model_path = dict()
        for model_type in self.policy_config['agent'].keys():
            model_path[model_type] = dict()
            for agent_name in self.policy_config['agent'][model_type].keys():
                model_path[model_type][agent_name] = self.policy_config['agent'][model_type][agent_name]['model_path']
        return model_path

    def synchronize_model(self):
        # ---------- 在训练模式下，先将模型的最新信息获取下来 -----------
        if self.eval_mode:
            model_path = self.read_model_information_from_yaml_file()
            for model_type in model_path.keys():
                for model_name in model_path[model_type].keys():
                    self.agent[model_type][model_name].synchronize_model(model_path[model_type][model_name])
        else:
            model_info = self.policy_fetcher.reset()
            if model_info is not None:
                self.logger.info("----------- 模型重置，使用model_fetcher到的模型数据:{} -----------".format(model_info))
                # ---------- 当获取到了最新的模型后，进行模型之间的同步 ------------
                for model_type in self.policy_fetcher.model_path.keys():
                    for model_name in self.policy_fetcher.model_path[model_type]:
                        self.agent[model_type][model_name].synchronize_model(self.policy_fetcher.model_path[model_type][model_name])
                if self.using_wolpagent:
                    self._add_critic_net_for_wolpagent()
        # else:
        #     self.logger.info("------------- agent调用reset函数之后没有获取到新模型,检测fetcher函数 ------------")

    def reset(self):
        # ---------- 模型重置 ------------------
        self.synchronize_model()

    def step(self):
        # -------------- 这个函数是在采样函数的时候使用，每一次rollout的时候都会step，把最新的模型拿过来 ---------
        '''
        model_info的样子为:
            如果使用参数共享
                model_info["policy_path"]: string
            否则:
                model_info["policy_path"]["agent_1"]: string
                ......
            model_info["critic_path"]: string
        '''
        if self.eval_mode:
            # ------ 如果说是eval mode，那么就不需要更新模型了，直接返回就可以了 -------
            return
        else:
            self.synchronize_model()

    def get_model_info(self):
        return self.model_info

    def normalize_state_value(self, input_matrix):
        normalize_PF_value = self.global_critic.popart_head_PF.normalize(input_matrix[:,0])
        normalize_Edge_value = self.global_critic.popart_head_Edge.normalize(input_matrix[:,1])
        return [normalize_PF_value, normalize_Edge_value]

    def denormalize_state_value(self, input_matrix):
        # ------------ 传入的数据input_matrix是一个长度为2的列表，其中每一个元素的维度都是1*1
        denormalize_PF_value = self.global_critic.popart_head_PF.denormalize(input_matrix[0])
        denormalize_Edge_value = self.global_critic.popart_head_Edge.denormalize(input_matrix[1])
        return [denormalize_PF_value, denormalize_Edge_value]

if __name__ == '__main__':
    import argparse
    from Worker.statistics import StatisticsUtils
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='/Learner/configs/config_single_cell_ddpg.yaml', help='yaml format config')
    args = parser.parse_args()
    # ------------- 构建绝对地址 --------------
    # Linux下面是用/分割路径，windows下面是用\\，因此需要修改
    # abs_path = '/'.join(os.path.abspath(__file__).split('/')[:-2])
    abs_path = '/'.join(os.path.abspath(__file__).split('/')[:-2])
    concatenate_path = abs_path + args.config_path
    context = zmq.Context()
    statistic = StatisticsUtils()
    from Utils.config_parse import parse_config
    from Utils import setup_logger
    config_dict = parse_config(concatenate_path)
    logger_path = pathlib.Path(config_dict['log_dir']+ '/sampler/test_agent_log')
    logger = setup_logger('Test_agent', logger_path)
    test_agent_manager = AgentManager(config_dict, context, statistic, logger)
    test_agent_manager.reset()