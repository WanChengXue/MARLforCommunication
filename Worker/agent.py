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
from Utils.model_utils import deserialize_model, create_model
from Utils.data_utils import convert_data_format_to_torch_interference

class Agent:
    # 定义一个采样智能体，它能够完成的事情有：加载模型，保存模型，发送数据，给定状态计算动作
    def __init__(self, policy_config):
        # ----------- 这个地方创建出来的net_work其实是一个策略网络 ------- deepcopy(self.policy_config['agent'][model_type][agent_name])
        self.net_work = create_model(policy_config)
        
    def synchronize_model(self, model_path):
        # ---------- 这个函数是用来同步本地模型的 ----------
        deserialize_model(self.net_work, model_path)

    def compute(self, agent_obs):
        # 这个是调用策略网络来进行计算的，网络使用的是Transformer结构，Encoder得到多个有意义的隐藏向量矩阵后，传入到Decoder，使用pointer network来进行解码操作
        with torch.no_grad():
            # 通过transformer进行了特征提取了之后，有两个head出来，一个是transformer decoder部分，得到调度列表，另外一个head出来的是v(s)的值，在推断阶段这个值不要
            # 因为还是遵循的CTDE的训练方式，每次决策之后，需要将所有智能体的backbone汇聚在一起进行V值的计算，由于action的长度有长有短，因此需要进行mask操作，统一到
            # 固定长度。如果是单个点进行决策，返回的log_probs表示的是联合概率的对数，action表示的是调度序列，mask表示的是固定长度的0-1向量
            # 按理来说，在eval mode的时候，每次决策都需要选择概率最大的动作的
            log_joint_prob, scheduling_action = self.net_work(agent_obs)
            return log_joint_prob, scheduling_action

    def compute_state_value(self, agent_obs):
        with torch.no_grad():
            state_value = self.net_work(agent_obs)
        return state_value


    def compute_action_and_state_value(self, agent_obs):
        with torch.no_grad():
            log_prob, action, state_value = self.net_work(agent_obs)
        return log_prob, action, state_value

class AgentManager:
    def __init__(self, config_dict, context, statistic, logger, process_uid):
        ''' 
            由于是多智能体环境，可能需要创建多个智能体构成list，如果使用了paramerter sharing的话，就只创建一个智能体，否则，创建多个智能体构成一个list
        '''
        self.config_dict = config_dict
        self.statistic = statistic
        self.agent_nums = self.config_dict['env']['agent_nums']
        self.policy_config = self.config_dict['policy_config']
        self.data_sender = context.socket(zmq.PUSH)
        self.parameter_sharing = self.policy_config['parameter_sharing']
        self.homogeneous_agent = self.policy_config['homogeneous_agent']
        self.eval_mode = self.policy_config['eval_mode']
        self.model_info = None
        self.logger = logger
        # ------------ 连接数据发送服务 --------------------------------
        self.choose_target_port()
        # ------------ 创建一个获取最新策略的fetcher --------------------
        self.policy_fetcher = fetcher(context, self.config_dict, statistic, process_uid, logger)
        # ------------ 构建智能体 --------------------------------------
        self.construct_agent()
        self.logger.info("------------ 完成AgentManager的构建 -----------")
        

    def construct_agent(self):
        # --------- 构建一下智能体 -----------
        self.agent = dict()
        for model_type in self.policy_config['agent'].keys():
            self.agent[model_type] = dict()
            for agent_name in self.policy_config['agent'][model_type].keys():
                policy_config = deepcopy(self.policy_config['agent'][model_type][agent_name])
                self.agent[model_type][agent_name] = Agent(policy_config)
        self.agent_name_list = list(self.agent['policy'].keys())         

    def choose_target_port(self):
        # 定义这个worker的数据发送到哪一个data server,首先计算一下每台机器会有多少个数据服务
        data_server_per_machine = self.policy_config['device_number_per_machine'] * self.policy_config['server_number_per_device']
        # 由于可能用了多台机器进行训练，因此要看一下是不是多机多卡场景
        total_data_server = len(self.policy_config['machine_list']) * data_server_per_machine
        # 随机选择一个数据服务进行连接
        choose_data_server_index = random.randint(0, total_data_server-1)
        # 计算机器的索引，确定连接的ip，端口
        machine_index = choose_data_server_index // data_server_per_machine
        port_num = choose_data_server_index % data_server_per_machine
        target_ip = self.policy_config['machine_list'][machine_index]
        target_port = self.policy_config['start_data_server_port'] + port_num
        self.logger.info("==================== 此智能体要发送数据到: {}, 连接的端口为: {} =================".format(target_ip, target_port))
        self.data_sender.connect("tcp://{}:{}".format(target_ip, target_port))
        

    def send_data(self, data):
        # -------- 将采样数据发送出去 -------------
        compressed_data = frame.compress(pickle.dumps(data))
        if self.eval_mode:
            with open("sampler_data", 'wb') as f:
                f.write(compressed_data)
        else:
            self.data_sender.send(compressed_data)

    def compute_single_agent(self, obs):
        torch_format_data = convert_data_format_to_torch_interference(obs)
        action_log_prob, action = self.agent['policy'][self.agent_name_list[0]].compute(torch_format_data)
        state_value = self.agent['critic'][self.agent_name_list[0]].compute_state_value(torch_format_data)
        return action_log_prob.numpy(), action.numpy(), state_value.numpy()

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
        # -------- 这个地方计算一下当前的状态值 ---------------
        with torch.no_grad():
            state_value = self.agent['critic'][self.agent_name_list[0]].compute_state_value(torch_format_data['global_channel_matrix'])
        if isinstance(state_value, tuple):
            state_value = np.array(item.numpy() for item in state_value)
        else:
            state_value = state_value.numpy()
        return joint_log_prob_list, actions, state_value

    def synchronize_model(self):
        # ---------- 在训练模式下，先将模型的最新信息获取下来 -----------
        model_info = self.policy_fetcher.reset()
        if model_info is not None:
            self.logger.info("----------- 模型重置，使用model_fetcher到的模型数据:{} -----------".format(model_info))
            # ---------- 当获取到了最新的模型后，进行模型之间的同步 ------------
            for model_type in self.policy_fetcher.model_path.keys():
                for model_name in self.policy_fetcher.model_path[model_type]:
                    self.agent[model_type][model_name].synchronize_model(self.policy_fetcher.model_path[model_type][model_name])
        # else:
        #     self.logger.info("------------- agent调用reset函数之后没有获取到新模型,检测fetcher函数 ------------")

    def reset(self):
        # ---------- 模型重置 ------------------
        if self.eval_mode:
            # 如果是测试模式，就加载模型
            # ---------- 如果是测试模型，就从本地读取模型即可 ---------
            assert self.model_info is None, '------ 测试模式下不接收learner的数据 ----------'
            model_path = self.policy_fetcher['agent']['policy'].keys()
            for model_name in model_path.keys():
                self.agent['policy'][model_name].synchronize_mode(model_path[model_name])
            # ---------- 测试模式不需要critic网络的参与 ------------------
        else:
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
    parser.add_argument('--config_path', type=str, default='/Learner/configs/config_pointer_network.yaml', help='yaml format config')
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