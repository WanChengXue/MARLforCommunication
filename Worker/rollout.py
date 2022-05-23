from ensurepip import bootstrap
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
    def __init__(self, config_dict, statistic, context, logger, process_uid, port_num=None):
        self.config_dict = config_dict
        self.policy_name = self.config_dict['policy_name']
        self.policy_config = self.config_dict['policy_config']
        self.popart_start = self.policy_config['training_parameters'].get("popart_start", False)
        self.demonstration_threshold = self.policy_config.get('demonstration_threshold', 0)
        self.statistic = statistic
        self.logger = logger
        self.eval_mode = self.policy_config.get('eval_mode', False)
        self.using_wolpagent = self.policy_config.get('using_wolpagent', False)
        if not self.eval_mode:
            # 定义强化学习需要的一些参数
            self.gamma = self.policy_config["gamma"]
            self.tau = self.policy_config["tau"]
            # -------------- 定义环境相关的参数 --------------
            self.traj_len = self.policy_config["traj_len"]
        self.env_name = self.config_dict['env']['id']
        self.env = importlib.import_module(self.env_name).Environment(self.config_dict['env'])
        self.agent_nums = self.config_dict['env']['agent_nums']
        # 声明一个智能体
        # self.eps = 1e-4
        self.eps = 0
        self.agent = AgentManager(self.config_dict, context, self.statistic, self.logger, process_uid, port_num)
        self.multiagent_scenario = self.config_dict['env'].get('multiagent_scenario', False)
    def pack_data(self, bootstrap_value, traj_data):
        '''
            这个函数表示将采样得到的数据进行打包，然后发送回去，传入的traj_data是一个列表，每一个列表的element都是一个字典，其数据结构为:
            traj_data[i]
            {
                "current_state": 当前时刻的状态，很复杂，也是一个嵌套字典类型的数据结构
                "instant_reward": 瞬时奖励，是一个字典，形式为{"agent_0": value, "agent_1": value}
                "actions": 智能体的决策动作，是一个字典，和instant reward是一样的
                "old_action_log_probs": 智能体的决策动作的联合概率的对数, dict类型
                "done": 这个是当前时刻i是不是terminate状态的标志
                "current_state_value": 这个是使用global critic网络估计出来的v向量，是一个list，包含两个元素，state_value_PF, state_value_Edge
                "denormalize_current_state_value": 这个是通过denormalize出来之后的状态值，也就是实际的v值,(只有在popart开启之后才使用)
                "next_state": 下一个时刻的状态
            }
            # 使用gae之后，添加两个key，advantages，以及target state value
            默认只有采样结束后才会将数据进行打包操作，因此这个bootstrap_value的样子是[[0,0]]
        '''
        # -------------- 首先我需要使用GAE算法进行估计出advantage值，target_state_value，然后放到原来的字典里面返回 -------------------
        modified_traj_data = gae_estimator(traj_data, self.gamma, self.tau, bootstrap_value, self.multiagent_scenario)
        # -------------- 打包，发送到dataserver -------------
        self.agent.send_data(modified_traj_data)

    def run_one_episode(self):
        '''
        这个函数表示这个worker随机生成一个环境，然后使用当前策略进行交互收集数据, obs的数据格式见
        '''
        # --------- 首先同步最新 config server上面的模型 ------
        self.logger.info("==================== 智能体重置，对于训练模式，就是同步configserver上面的模型，测试模式就是加载本地模型 ====================")
        if self.eval_mode:
            self.agent.reset()
            data_dict = dict()
            # eval_file_number = self.config_dict['env']['eval_file_number']
            eval_file_number = 50
            for file_index in tqdm(range(eval_file_number)):
                state = self.env.reset(file_index=file_index)
                action_list = []
                instant_reward_list = []
                instant_SE_list  = []
                while True:
                    actions = self.agent.compute_single_agent(state)
                    # -------------- 给定动作计算出对应的instant reward, 这个返回的是瞬时PF值，需要额外处理得到PF和，以及边缘用户的SE ------------
                    next_state, instant_reward, done = self.env.step(actions)
                    if self.agent_nums == 1:
                        instant_reward_list.append(instant_reward[1])
                        instant_SE_list.append(instant_reward[0])
                        action_list.append(np.array(actions))
                    else:
                        data_dict[-1]['instant_reward'] = dict()
                        data_dict[-1]['actions'] = dict()
                        for agent_index in range(self.agent.agent_nums):
                            agent_key = "agent_" + str(agent_index)
                            # ------------ 这个actions[agent_index]的维度是一个长度为16的向量，需要变成16*1
                            data_dict[-1]['actions'][agent_key] = actions[agent_index][:,np.newaxis]
                        data_dict[-1]['instant_reward'] = np.array(instant_reward[1])
                    state = next_state
                    if done[0]:
                        break
                data_dict['actions'] = np.stack(action_list, 0)
                data_dict['instant_reward'] = dict()
                data_dict['instant_reward']['average_se'] = self.env.get_user_average_se_matrix
                data_dict['instant_reward']['PF_sum']= np.stack(instant_reward_list)
                data_dict['instant_reward']['instant_SE'] = np.stack(instant_reward_list)
                data_dict['file_index'] = str(file_index)
                self.agent.send_data(data_dict)
        else:
            self.logger.info("======================== 重置环境 =======================")
            state = self.env.reset()
            self.agent.reset()
            self.logger.info('------------- 完成模型的reset,开始采样 ------------------')
            data_dict = []
            # ---------- 这里有两个list，第一个表示的是瞬时SE构成的列表，第二个表示的是PF和构成的列表 ----------
            instant_SE_sum_list = []
            PF_sum_list = []
            while True:
                joint_log_prob, actions, net_work_output = self.agent.compute_single_agent(state)
                # -------------- 此处需要给这个current_state_value 进行denormalizeing操作 ----------
                if self.popart_start:
                    current_state_value = self.agent.denormalize_state_value(net_work_output)
                else:
                    current_state_value = net_work_output
                # -------------- 给定动作计算出对应的instant reward, 这个返回的是瞬时PF值，需要额外处理得到PF和，以及边缘用户的SE ------------
                next_state, instant_reward, done = self.env.step(actions)
                # -------------- 构建一个字典，将current state, action, instant reward, old action log probs, done, current state value, next state 放入到字典 --------------
                instant_SE_sum_list.append(instant_reward[0])
                PF_sum_list.append(instant_reward[1])
                data_dict.append({'current_state': copy.deepcopy(state)})
                if not self.multiagent_scenario:
                    data_dict[-1]['instant_reward'] = instant_reward[1] # 3*1
                    data_dict[-1]['actions'] = actions 
                    data_dict[-1]['old_action_log_probs'] = joint_log_prob
                    data_dict[-1]['done'] = np.array(done)[:,np.newaxis]
                    data_dict[-1]['current_state_value'] = current_state_value
                else:
                    data_dict[-1]['old_action_log_probs'] = dict()
                    data_dict[-1]['actions'] = dict()
                    for agent_index in range(self.agent.agent_nums):
                        agent_key = "agent_" + str(agent_index)
                        data_dict[-1]['old_action_log_probs'][agent_key] = joint_log_prob[agent_key].squeeze()
                        # ------------ 这个actions[agent_index]的维度是一个长度为16的向量
                        data_dict[-1]['actions'][agent_key] = actions[agent_key].squeeze()
                    data_dict[-1]['done'] = done
                    data_dict[-1]['instant_reward'] = instant_reward[1]
                    # data_dict[-1]['instant_reward'] = np.array([PF_sum, edge_average_SE])[:,np.newaxis]
                    data_dict[-1]['current_state_value'] = current_state_value.squeeze(0)
                    data_dict[-1]['next_state'] = copy.deepcopy(next_state)
                state = next_state
                # -----------------------------------------------------------------------------------------
                terminate = False
                if self.multiagent_scenario:
                    if done:
                        terminate = True
                else:
                    if done[0]:
                        terminate = True
                    # ------------ 数据打包，然后发送，bootstrap value就给0吧，计算出来的current_sate_value为3*1，仅针对单小区场景 ----------------
                if terminate or len(data_dict) == self.policy_config['traj_len']:
                    objective_number = current_state_value.shape[1]
                    batch_size = current_state_value.shape[0]
                    # ------ 如果说采样结束，terminal state的v值就是0
                    if terminate:
                        bootstrap_value = self.agent.compute_state_value(next_state)
                    else:
                        bootstrap_value = np.zeros((batch_size, objective_number))
                    self.logger.info('---------- worker数据开始打包发送到dataserver -------------')
                    self.pack_data(bootstrap_value, data_dict)
                    self.agent.step()
                    data_dict = []
                if terminate:
                    break

            mean_instant_SE_sum = np.mean(instant_SE_sum_list).item()
            # mean_edge_average_SE = np.mean(edge_average_capacity_list).item()
            mean_PF_sum = np.mean(PF_sum_list).item()
            user_average_se_matrix = np.mean(self.env.get_user_average_se_matrix,0).squeeze()
            # ------- 这个返回的user_average_se_matrix的维度是20的用户平均SE矩阵 ---------
            return mean_instant_SE_sum, mean_PF_sum, user_average_se_matrix

    def run_one_episode_single_step(self):
        '''
        这个函数表示这个worker随机生成一个环境，然后使用当前策略进行交互收集数据, 整个环境进行one step 决策就停止了 
        '''
        self.logger.info("======================== 重置环境 =======================")
        if self.eval_mode:
            self.logger.info("================= 使用eval智能体进行验证 =============")
            self.agent.reset()
            # eval_file_number = self.config_dict['env']['eval_file_number']
            eval_file_number = 1
            for file_index in tqdm(range(eval_file_number)):
                state = self.env.reset(file_index=file_index)
                if self.agent_nums == 1:
                    actions = self.agent.compute_single_agent(state)
                else:
                    actions = self.agent.compute_multi_agent(state)
                instant_SE_sum_list = self.env.step(actions)
                data_dict = {'instant_reward': np.array(instant_SE_sum_list)}
                if self.agent_nums == 1:
                    data_dict['actions'] = actions
                else:
                    data_dict['actions'] = dict()
                    for agent_index in range(self.agent.agent_nums):
                        agent_key = "agent_" + str(agent_index)
                        data_dict['actions'][agent_key] = actions[agent_index]
                data_dict['file_index'] = str(file_index)
                self.agent.send_data(data_dict)

        else:
            self.agent.reset()
            state = self.env.reset()
            # ------- 定义变量，要不要从demonstration中拿数据来训练 ------
            if random.random() <= self.demonstration_threshold:
                actions, instant_SE_sum_list = self.env.read_action_from_demonstration()
                if self.agent_nums == 1:
                    joint_log_prob, net_work_output = self.agent.compute_single_agent(state, actions)
                else:
                    joint_log_prob, net_work_output = self.agent.compute_multi_agent(state, actions)
                data_dict = [{'state': copy.deepcopy(state), 'instant_reward': instant_SE_sum_list}]
                self.demonstration_threshold = max(0, self.demonstration_threshold-self.eps)
            else:
                # --------- 首先同步最新 config server上面的模型 ------
                if self.agent_nums == 1:
                    if self.using_wolpagent:
                        actions = self.agent.compute_single_agent(state)
                    else:
                        joint_log_prob, actions, net_work_output = self.agent.compute_single_agent(state)
                else:
                    joint_log_prob, actions, net_work_output = self.agent.compute_multi_agent(state)
                instant_SE_sum_list = self.env.step(actions)
                # ------------ instant_SE_sum_list的维度为bs×1 ------------
                data_dict = [{'state': copy.deepcopy(state), 'instant_reward':np.array(instant_SE_sum_list)}]


            if self.agent_nums == 1:
                # ---------- 如果说只有一一个用户，不需要套字典了 --------------
                if not self.using_wolpagent:
                    data_dict[-1]['old_action_log_probs'] = joint_log_prob
                data_dict[-1]['actions'] = actions
            else:
                # ------------- old_action_log_probs是一个字典，每一个key的维度都bs×1 ---------
                data_dict[-1]['old_action_log_probs'] = dict()
                # ------------ actions也是一个字典，每一个key的维度都是bs×user_nums ------------
                data_dict[-1]['actions'] = dict()
                for agent_index in range(self.agent.agent_nums):
                    agent_key = "agent_" + str(agent_index)
                    data_dict[-1]['old_action_log_probs'][agent_key] = joint_log_prob[agent_index]
                    data_dict[-1]['actions'][agent_key] = actions[agent_index]
                # ------------- net work output 的维度为bs×1 -----------
            if not self.using_wolpagent:
                data_dict[-1]['current_state_value'] = net_work_output
            self.agent.send_data(data_dict)
            return instant_SE_sum_list
            

    def read_data_from_folder(self):
        # ----------------- 这个函数是说,从本地读取数据发送到dataserver ----------------
        if self.eval_mode:
            self.logger.info("================= 使用eval智能体进行验证 =============")
            self.agent.reset()
            eval_file_number = self.config_dict['env']['eval_file_number']
            # eval_file_number = 1
            for file_index in tqdm(range(eval_file_number)):
                state = self.env.reset(file_index=file_index)
                if self.agent_nums == 1:
                    actions = self.agent.compute_single_agent(state)
                else:
                    actions = self.agent.compute_multi_agent(state)
                instant_SE_sum_list = self.env.step(actions)
                data_dict = {'instant_reward': np.array(instant_SE_sum_list)}
                if self.agent_nums == 1:
                    data_dict['actions'] = actions
                else:
                    data_dict['actions'] = dict()
                    for agent_index in range(self.agent.agent_nums):
                        agent_key = "agent_" + str(agent_index)
                        data_dict['actions'][agent_key] = actions[agent_index]
                data_dict['file_index'] = str(file_index)
                self.agent.send_data(data_dict)
        else:
            state = self.env.reset(load_eval=True)
            actions, instant_SE_sum_list = self.env.read_action_from_testing_set()
            data_dict = [{'state': copy.deepcopy(state), 'instant_reward': instant_SE_sum_list}]
            if self.agent_nums == 1:
                # ---------- 如果说只有一一个用户，不需要套字典了 --------------
                data_dict[-1]['actions'] = actions
            else:
                # ------------ actions也是一个字典，每一个key的维度都是bs×user_nums ------------
                data_dict[-1]['actions'] = dict()
                for agent_index in range(self.agent.agent_nums):
                    agent_key = "agent_" + str(agent_index)
                    data_dict[-1]['actions'][agent_key] = actions[agent_index]
            self.agent.send_data(data_dict)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='/Learner/configs/config_multi_cell_PF_pointer_network.yaml', help='yaml format config')
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
    config_dict = parse_config(concatenate_path,obj='Worker')
    logger_path = pathlib.Path(config_dict['log_dir']+ '/sampler/test_rollout_' + process_uid[:6])
    logger = setup_logger('Rollout_agent_'+process_uid[:6], logger_path)
    statistic = StatisticsUtils()
    roll_out_test = rollout_sampler(parse_config(concatenate_path, obj='Worker'), statistic, context, logger, process_uid[0:6])
    # roll_out_test.run_one_episode_single_step()
    roll_out_test.run_one_episode()
    # roll_out_test.read_data_from_folder()
