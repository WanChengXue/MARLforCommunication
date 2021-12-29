from utils.env_utils import make_env
from Worker.gae import gae_estimator
from Worker.agent import agent
from utils import setup_logger
import copy
import numpy as np

class rollout_sampler:
    def __init__(self, config_dict, statistic, context):
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

        # 环境声明
        self.env = make_env(self.config_dict)
        # 收集数据放入到字典中
        self.data_dict = dict()
        # 声明一个智能体
        self.agent = 