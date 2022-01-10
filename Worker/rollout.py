from Env.Sliding_Windows_Env import Environment
from Worker.gae import gae_estimator
from Worker.agent import AgentManager
from Utils import setup_logger
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

    def pack_data(self, bootstrap_value, traj_data):
        '''
            这个函数表示将采样得到的数据进行打包，然后发送回去，传入的traj_data是一个列表，每一个列表的element都是一个字典，其数据结构为：
            traj_data[i]
            {
                "current_state": 当前时刻的状态，很复杂，也是一个嵌套字典类型的数据结构
                "instant_reward": 瞬时奖励，是一个字典，形式为{"agent_0": value, "agent_1": value}
                "actions": 智能体的决策动作，是一个字典，和instant reward是一样的
                "old_action_log_probs": 智能体的决策动作的联合概率的对数, dict类型
                "done": 这个是当前时刻i是不是terminate状态的标志
                "current_state_value": 这个是使用global critic网络估计出来的v向量，是一个list，包含两个元素，state_value_PF, state_value_Edge
                "denormalize_current_state_value": 这个是通过denormalize出来之后的状态值，也就是实际的v值
                "next_state": 下一个时刻的状态
            }
            # 使用gae之后，添加两个key，advantages，以及target state value
            默认只有采样结束后才会将数据进行打包操作，因此这个bootstrap_value的样子是[[0,0]]
        '''
        # -------------- 首先我需要使用GAE算法进行估计出advantage值，target_state_value，然后放到原来的字典里面返回 -------------------
        modified_traj_data = gae_estimator(traj_data, self.gamma, self.tau, bootstrap_value)
        # -------------- 打包，发送到dataserver -------------
        self.agent.send_data(modified_traj_data)

    def run_one_episode(self):
        '''
        这个函数表示这个worker随机生成一个环境，然后使用当前策略进行交互收集数据, obs的数据格式见
        '''
        state = self.env.reset()
        # --------- 首先同步最新 config server上面的模型 ------
        self.agent.reset()
        done = False
        data_dict = []
        while not done:
            self.agent.step()
            joint_log_prob, actions, current_state_value = self.agent.compute(state)
            denormalize_state_value = self.agent.denormalize_state_value(current_state_value)
            # -------------- 此处需要给这个current_state_value 进行denormalizeing操作 ----------
            # -------------- 给定动作计算出对应的instant reward ------------
            next_state, instant_rewards, done = self.env.step(actions)
            # -------------- 构建一个字典，将current state, action, instant reward, old action log probs, done, current state value, next state 放入到字典 --------------
            data_dict.append({'current_state': copy.deepcopy(state)})
            data_dict[-1]['instant_reward'] = dict()
            data_dict[-1]['old_action_log_probs'] = dict()
            data_dict[-1]['actions'] = dict()
            for agent_index in range(self.agent.agent_nums):
                agent_key = "agent_" + str(agent_index)
                data_dict[-1]['instant_reward'][agent_key] = instant_rewards[agent_index]
                data_dict[-1]['old_action_log_porbs'][agent_key] = joint_log_prob[agent_index]
                data_dict[-1]['actions'][agent_key] = actions[agent_index]
            data_dict[-1]['done'] = done
            data_dict[-1]['current_state_value'] = current_state_value
            data_dict[-1]['denormalize_current_state_value'] = np.concatenate(denormalize_state_value, 1)
            data_dict[-1]['next_state'] = copy.deepcopy(next_state)
            state = next_state
            # -----------------------------------------------------------------------------------------
        # ------------ 数据打包，然后发送，bootstrap value就给0吧 ----------------
        bootstrap_value = np.zeros((1,2))
        self.pack_data(bootstrap_value, data_dict)

