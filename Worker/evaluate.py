import os
import sys
import pathlib
import zmq


current_path = os.path.abspath(__file__)
root_path = '/'.join(current_path.split('/')[:-2])
sys.path.append(root_path)

import importlib
from Worker.gae import gae_estimator
from Worker.agent import AgentManager
import copy
import numpy as np

class rollout_sampler:
    def __init__(self, config_dict, statistic, context, logger, process_uid):
        self.config_dict = config_dict
        self.policy_name = self.config_dict['policy_name']
        self.policy_config = self.config_dict['policy_config']
        self.popart_start = self.policy_config['training_parameters'].get("popart_start", False)
        self.statistic = statistic
        self.logger = logger
        # -------------- 定义环境相关的参数 --------------
        self.traj_len = self.policy_config["traj_len"]
        self.env_name = self.config_dict['env']['id']
        self.env = importlib.import_module(self.env_name).Environment(self.config_dict['env'])
        # 收集数据放入到字典中
        self.data_dict = dict()
        # 声明一个智能体
        self.agent = AgentManager(self.config_dict, context, self.statistic, self.logger, process_uid)


    def run_one_episode_single_step(self):
        '''
            --------- 在evaluate环境下，重置智能体，只需要将智能体load本地模型即可 ----------
        '''
        self.logger.info("======================== 重置环境 =======================")
        self.agent.reset()
        state = self.env.reset()
        # --------- 首先同步最新 config server上面的模型 ------
        joint_log_prob, actions, net_work_output = self.agent.compute(state)
        instant_SE_sum_list = self.env.step(actions)
        # ------------ instant_SE_sum_list的维度为bs×1 ------------
        data_dict = [{'state': copy.deepcopy(state), 'instant_reward':np.array(instant_SE_sum_list)}]
        # ------------- old_action_log_probs是一个字典，每一个key的维度都bs×1 ---------
        data_dict[-1]['old_action_log_probs'] = dict()
        # ------------ actions也是一个字典，每一个key的维度都是bs×user_nums ------------
        data_dict[-1]['actions'] = dict()
        for agent_index in range(self.agent.agent_nums):
            agent_key = "agent_" + str(agent_index)
            data_dict[-1]['old_action_log_probs'][agent_key] = joint_log_prob[agent_index]
            data_dict[-1]['actions'][agent_key] = actions[agent_index]
        # ------------- net work output 的维度为bs×1 -----------
        data_dict[-1]['current_state_value'] = net_work_output
        self.agent.send_data(data_dict)
        return instant_SE_sum_list

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='/Learner/configs/config_pointer_network.yaml', help='yaml format config')
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
    config_dict = parse_config(concatenate_path)
    logger_path = pathlib.Path(config_dict['log_dir']+ '/sampler/testrollout')
    logger = setup_logger('rollout_agent', logger_path)
    statistic = StatisticsUtils()
    roll_out_test = rollout_sampler(concatenate_path, parse_config(concatenate_path), statistic, context, logger)
    roll_out_test.run_one_episode()
