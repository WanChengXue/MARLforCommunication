# 这个函数用来存放episode的数据
import numpy as np
import threading


class ReplayBuffer:
    def __init__(self, args):
        self.args = args
        self.n_agents = self.args.n_agents
        self.size = self.args.max_buffer_size
        self.episode_limit = self.args.TTI_length
        # memory management
        self.index = 0
        # 定义状态和局部观测的维度，且两个都是矩阵
        self.obs_dim1 = self.args.obs_dim1
        self.obs_dim2 = self.args.obs_dim2
        self.state_dim1 = self.args.state_dim1
        self.state_dim2 = self.args.state_dim2
        self.action_dim = self.args.state_dim1 + 1
        self.obs_matrix_number = self.args.obs_matrix_number 
        self.state_matrix_number = self.args.state_matrix_number 
        # create the buffer to store info, 这个表示提前开辟好一个位置,然后将经验进行替换就可以了
        self.buffers = {'Channel': np.empty([self.size, self.episode_limit, self.obs_matrix_number ,self.obs_dim1, self.obs_dim2]),
                        'Average_reward':np.empty([self.size, self.episode_limit ,self.obs_dim1]),
                        'Global_channel': np.empty([self.size, self.episode_limit, self.state_matrix_number, self.state_dim1, self.state_dim2]),
                        'Global_reward': np.empty([self.size, self.episode_limit, self.n_agents, self.state_dim1]),
                        'Action': np.empty([self.size, self.episode_limit, self.action_dim]),
                        'Pad': np.empty([self.size, self.episode_limit, self.action_dim]),
                        'instant_reward': np.empty([self.size, self.episode_limit]),
                        'terminate': np.empty([self.size, self.episode_limit]),
                        'prob': []
                        }

        self.lock = threading.Lock()

        # store the episode
    def store_episode(self, episode_batch):
        # batch_size = episode_batch['o'].shape[0]  # episode_number
        with self.lock:
            # store the informations
            self.buffers['Channel'][self.index] = episode_batch['Channel']
            self.buffers['Average_reward'][self.index] = episode_batch['Average_reward']
            self.buffers['Global_channel'][self.index] = episode_batch['Global_channel']
            self.buffers['Global_reward'][self.index] = episode_batch['Global_reward']
            self.buffers['Action'][self.index] = episode_batch['Action']
            self.buffers['Pad'][self.index] = episode_batch['Pad']
            self.buffers['instant_reward'][self.index] = episode_batch['instant_reward']
            self.buffers['terminate'][self.index] = episode_batch['terminate']
            self.buffers['prob'].append(episode_batch['prob'])
            # 这个地方的with lock操作表示在这个进程结束之前,其余的进程都是不可以运行的, 相当于lock.acquire() 以及结束之后运行lock.release()操作
            self.index += 1

    def sample(self):
        return self.buffers

    def reset_buffer(self):
        self.index = 0
        self.buffers = {'Channel': np.empty([self.size, self.episode_limit, self.obs_matrix_number ,self.obs_dim1, self.obs_dim2]),
                        'Average_reward':np.empty([self.size, self.episode_limit ,self.obs_dim1]),
                        'Global_channel': np.empty([self.size, self.episode_limit, self.state_matrix_number, self.state_dim1, self.state_dim2]),
                        'Global_reward': np.empty([self.size, self.episode_limit, self.n_agents, self.state_dim1]),
                        'Action': np.empty([self.size, self.episode_limit, self.action_dim]),
                        'Pad': np.empty([self.size, self.episode_limit, self.action_dim]),
                        'instant_reward': np.empty([self.size, self.episode_limit]),
                        'terminate': np.empty([self.size, self.episode_limit]),
                        'prob': []
                        }

