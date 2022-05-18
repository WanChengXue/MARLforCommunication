import os
import sys
import pathlib
import zmq
import random
from tqdm import tqdm
current_path = os.path.abspath(__file__)
root_path = '/'.join(current_path.split('/')[:-2])
sys.path.append(root_path)

import importlib
from Worker.gae import gae_estimator
from Worker.agent_manager import AgentManager
import copy
import numpy as np

class rollout_sampler:
    def __init__(self, config_dict):
        self.config_dict = config_dict
        self.policy_name = self.config_dict['policy_name']
        self.env_name = self.config_dict['env']['id']
        self.env = importlib.import_module(self.env_name).Environment(self.config_dict['env'])

    def read_action(self):
        action_file_path = os.
        

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='/Learner/configs/config_single_cell_SL_eval_config.yaml', help='yaml format config')
    args = parser.parse_args()
    # ------------- 构建绝对地址 --------------
    # Linux下面是用/分割路径，windows下面是用\\，因此需要修改
    # abs_path = '/'.join(os.path.abspath(__file__).split('/')[:-2])
    abs_path = '/'.join(os.path.abspath(__file__).split('/')[:-2])
    concatenate_path = abs_path + args.config_path
    from Utils.config_parse import parse_config
    context = zmq.Context()
    from Worker.statistics import StatisticsUtils
    from Utils import setup_logger
    import uuid
    process_uid = str(uuid.uuid4())
    config_dict = parse_config(concatenate_path)
    logger_path = pathlib.Path(config_dict['log_dir']+ '/sampler/test_rollout_' + process_uid[:6])
    logger = setup_logger('Rollout_agent_'+process_uid[:6], logger_path)
    statistic = StatisticsUtils()
    roll_out_test = rollout_sampler(parse_config(concatenate_path), statistic, context, logger, process_uid[0:6])
    # roll_out_test.run_one_episode_single_step()
    # roll_out_test.run_one_episode()
    roll_out_test.read_data_from_folder()
