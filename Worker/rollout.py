from Env.Sliding_Windows_Env import Environment
from Worker.gae import gae_estimator
from Worker.agent import AgentManager
from utils import setup_logger
import copy
import numpy as np

class rollout_sampler:
    def __init__(self, config_path, config_dict, statistic, context):
        self.config_dict = config_dict
        self.policy_id = self.config_dict['policy_id']
        self.policy_config = self.config_dict['learners']
        self.statistic = statistic
        logger_name = self.config_dict['log_dir'] + '/rollout_log'
        self.logger = setup_logger(logger_name)

        # 定义强化学习需要的一些参数
        self.gamma = self.policy_config["gamma"]
        self.tau = self.policy_config["tau"]
        self.traj_len = self.policy_config["traj_len"]

        # 环境声明, 传入的config dict的路径
        self.env = Environment(config_path)
        # 收集数据放入到字典中
        self.data_dict = dict()
        # 声明一个智能体
        self.agent = AgentManager(self.config_dict, self.env, context, self.statistic)

    def pack_data(self, bootstrp_value, traj_data):
        '''
            这个函数表示将采样得到的数据进行打包，然后发送回去
        '''
        pass

    def run_one_episode(self):
        '''
        这个函数表示这个worker随机生成一个环境，然后使用当前策略进行交互收集数据
        obs的数据格式为：
            obs['global_channel_matrix']维度为 sector_number * total_antennas * sector_number * (2*transmit_antennas)
            obs['global_average_reward']维度为 sector_number * total_antennas * 1
            obs['global_scheduling_cout']维度为 sector_number * total_antennas * 1
            obs['agent_obs']是一个字典，有三个key agent_0,agent_1,agent_2
                obs['agent_0']['channel_matrix']维度为sector_number * total_antennas * (2*transmit_antennas)
                obs['agent_0']['average_reward']维度为total_antennas * 1
                obs['agent_0']['scheduling_count']维度为total_antennas * 1
        '''
        obs = self.env.reset()
        # --------- 首先同步最新 config server上面的模型 ------
        self.agent.reset()
        done = False
        while not done:
            self.agent.step()
            joint_log_prob, joint_prob, actions = self.agent.compute(obs['agent_obs'])
            