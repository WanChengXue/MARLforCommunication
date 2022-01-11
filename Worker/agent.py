import pathlib
import pickle
import lz4.frame as frame
import torch
import zmq
import random
import os
import sys
current_path = os.path.abspath(__file__)
root_path = '/'.join(current_path.split('/')[:-2])
sys.path.append(root_path)
from Worker.policy_fetcher import fetcher
from Utils import setup_logger
from Utils.model_utils import deserialize_model, create_policy_model, create_critic_net
from Utils.data_utils import conver_data_format_to_torch_interference

class Agent:
    # 定义一个采样智能体，它能够完成的事情有：加载模型，保存模型，发送数据，给定状态计算动作
    def __init__(self, config_dict, statistic):
        self.config_dict = config_dict
        self.policy_id = self.config_dict['policy_id']
        self.statistic = statistic
        # ----------- 这个地方创建出来的net_work其实是一个策略网络 -------
        self.net_work = create_policy_model(self.config_dict['learners'])
        self.mdoel_info = None
        self.eval_mode = self.config_dict['eval_mode']


    def load_model(self, model_info):
        # 载入模型，这个函数只有在训练的时候，才会被智能体调用来更新自己的网络参数
        if model_info is not None:
            self.mdoel_info = model_info
            model_path = model_info['path']
            deserialize_model(self.net_work, model_path)
            
    def synchronize_model(self, model_path):
        # ---------- 这个函数是用来同步本地模型的，只再eval的时候使用 ----------
        pass
        # deserialize_model(self.net_work, model_path)


    def compute(self, agent_obs):
        # 这个是调用策略网络来进行计算的，网络使用的是Transformer结构，Encoder得到多个有意义的隐藏向量矩阵后，传入到Decoder，使用pointer network来进行解码操作
        with torch.no_grad():
            # 通过transformer进行了特征提取了之后，有两个head出来，一个是transformer decoder部分，得到调度列表，另外一个head出来的是v(s)的值，在推断阶段这个值不要
            # 因为还是遵循的CTDE的训练方式，每次决策之后，需要将所有智能体的backbone汇聚在一起进行V值的计算，由于action的长度有长有短，因此需要进行mask操作，统一到
            # 固定长度。如果是单个点进行决策，返回的log_probs表示的是联合概率的对数，action表示的是调度序列，mask表示的是固定长度的0-1向量
            log_joint_prob, scheduling_action = self.net_work(agent_obs, inference_mode = True)
            return log_joint_prob, scheduling_action


class AgentManager:
    def __init__(self, config_dict, context, statistic):
        ''' 
            由于是多智能体环境，可能需要创建多个智能体构成list，如果使用了paramerter sharing的话，就只创建一个智能体，否则，创建多个智能体构成一个list
        '''
        self.config_dict = config_dict
        self.statistic = statistic
        self.agent_nums = self.config_dict['env']['agent_nums']
        self.policy_config = self.config_dict['learners']
        self.data_sender = context.socket(zmq.PUSH)
        self.parameter_sharing = self.config_dict['parameter_sharing']
        self.eval_mode = self.config_dict['eval_mode']
        self.model_info = None
        # ------------ 连接数据发送服务 --------------------------------
        self.choose_target_port()
        # ------------ 创建一个获取最新策略的fetcher --------------------
        self.agent_fetcher = fetcher(context, self.config_dict, statistic)
        # ------------ 构建智能体 --------------------------------------
        self.construct_agent()
        

    def construct_agent(self):
        if self.parameter_sharing:
            self.agent = Agent(self.config_dict, self.statistic)
        else:
            self.agent = dict()
            for agent_index in range(self.agent_nums):
                agent_key = "agent_" + str(agent_index)
                self.agent[agent_key] = Agent(self.config_dict, self.statistic)
        self.global_critic = create_critic_net(self.config_dict['learners'])


    def choose_target_port(self):
        # 定义这个worker的数据发送到哪一个data server,首先计算一下每台机器会有多少个数据服务
        data_server_per_machine = self.policy_config['gpu_num_per_machine'] * self.policy_config['data_server_to_learner_num']
        # 由于可能用了多台机器进行训练，因此要看一下是不是多机多卡场景
        total_data_server = len(self.policy_config['machines']) * data_server_per_machine
        # 随机选择一个数据服务进行连接
        choose_data_server_index = random.randint(0, total_data_server-1)
        # 计算机器的索引，确定连接的ip，端口
        machine_index = choose_data_server_index // data_server_per_machine
        port_num = choose_data_server_index % data_server_per_machine
        target_ip = self.policy_config['machines'][machine_index]
        target_port = self.policy_config['learner_port_start'] + port_num
        # TODO,需要考虑一下如何给worker进行log的存放
        logger_path = pathlib.Path(self.config_dict['log_dir'] + "/agent_manager_server")
        self.logger = setup_logger('AgentManager_log', logger_path)
        self.logger.info("==================== 此智能体要发送数据到: {}, 连接的端口为: {} =================".format(target_ip, target_port))
        self.data_sender.connect("tcp://{}:{}".format(target_ip, target_port))
        

    def send_data(self, data):
        # -------- 将采样数据发送出去 -------------
        compressed_data = frame.compress(pickle.dumps(data))
        if self.eval_mode:
            # ----- TODO 如果说eval mode，就将这个数据保存到本地就好 ------
            pass
        else:
            self.data_sender.send(compressed_data)


    def compute(self, obs):
        # -------- 这个函数是用使用神经网络计算动作，以及动作对应的概率 ---------
        # 首先将这个obs_dict变成pytorch支持的数据，由于采样的时候，统一使用cpu就可以了，不需要用 GPU
        torch_format_data = conver_data_format_to_torch_interference(obs)
        # ----- 需要将动作构成列表，然后回传，以及将对应的log prob和prob -------
        joint_log_prob_list = []
        actions = []
        for agent_key in torch_format_data['agent_obs'].keys():
            active_agent_obs = torch_format_data['agent_obs'][agent_key]
            if self.parameter_sharing:
                agent_log_prob, agent_action = self.agent.compute(active_agent_obs)
            else:
                agent_log_prob, agent_action = self.agent[agent_key].compute(active_agent_obs)
            # ----------- 这个地方需要将数据变成numpy类型 ------------
            joint_log_prob_list.append(agent_log_prob.numpy())
            actions.append(agent_action.numpy())
        # -------- 这个地方计算一下当前的状态值 ---------------
        with torch.no_grad():
            state_value_PF, state_value_Edge = self.global_critic(torch_format_data['global_state'])
        return joint_log_prob_list, actions, [state_value_PF.numpy(), state_value_Edge.numpy()]

    def reset(self):
        # ---------- 模型重置 ------------------
        if self.eval_mode:
            # 如果是测试模式，就加载模型
            assert self.model_info is None
            model_path = self.policy_config['model_path']
            if self.parameter_sharing:
                self.agent.synchronize_model(model_path['policy_path'])
            else:
                # ---- 当采用了非参数共享算法，需要给每一个智能体的网络进行模型加载 ----
                for agent_key in self.agent.keys():
                    self.agent[agent_key].synchronize_model(model_path['policy_path'])
            deserialize_model(self.global_critic, model_path['critic_path'])

        else:
            # 如果是非测试模式，需要使用fether进行获取最新的策略，然后调用agent里面的load model函数
            model_info = self.agent_fetcher.reset()
            if self.parameter_sharing:
                self.agent.load_model(model_info['policy_path'])
            else:
                # ----- 如果采样非参数共享，需要通过循环的方式给每一个智能体进行模型加载 -----
                for agent_key in self.agent.keys():
                    self.agent[agent_key].load_model(model_info['policy_path'][agent_key])
            # ------------- 非测试模式下，global critic的参数也是需要同步进行更新的 -----------------
            deserialize_model(self.global_critic, model_info['critic_path'])

    def step(self):
        # -------------- 这个函数是在采样函数的时候使用，每一次rollout的时候都会step，把最新的模型拿过来 ---------
        if self.eval_mode:
            # ------ 如果说是eval mode，那么就不需要更新模型了，直接返回就可以了 -------
            return
        else:
            model_info = self.agent_fetcher.step()
            self.model_info = model_info
            if self.parameter_sharing:
                self.agent.load_model(model_info['policy_path'])
            else:
                for agent_key in self.agent.keys():
                    self.agent[agent_key].load_model(model_info['policy_path'][agent_key])
            deserialize_model(self.global_critic, model_info['critic_path'])

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
    config_dict = parse_config(concatenate_path)
    test_agent_manager = AgentManager(config_dict, context, statistic)
    test_agent_manager.reset()