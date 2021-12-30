from os import stat
import pickle
import time
import torch
from torch.autograd.grad_mode import inference_mode
import zmq
import random
from Worker.policy_fetcher import fetcher
from utils import setup_logger
from utils.model_utils import create_model, deserialize_model
from utils.data_utils import convert_data_format_to_torch
class Agent:
    # 定义一个采样智能体，它能够完成的事情有：加载模型，保存模型，发送数据，给定状态计算动作
    def __init__(self, config_dict, env, context, statistic):
        self.config_dict = config_dict
        self.policy_id = self.config_dict['policy_id']
        self.fetcher = fetcher(context, self.config_dict, statistic)
        self.env = env
        self.net_work = create_model(self.config_dict, self.policy_id)
        self.mdoel_info = None
        self.data_sender = context.socket(zmq.PUSH)
        self.policy_config = self.config_dict['learners']
        self.eval_mode = self.config_dict['eval_mode']
        # TODO,需要考虑一下如何给worker进行log的存放
        self.logger = setup_logger()
        # 定义这个worker的数据发送到哪一个data server,首先计算一下每台机器会有多少个数据服务
        data_server_per_machine = self.policy_config['gpu_num_per_machine'] * self.policy_config['data_server_to_learner_num']
        # 由于可能用了多台机器进行训练，因此要看一下是不是多机多卡场景
        total_data_server = len(self.policy_config['machines']) * data_server_per_machine
        # 随机选择一个数据服务进行连接
        choose_data_server_index = random.randint(0, total_data_server -1)
        # 计算机器的索引，确定连接的ip，端口
        machine_index = choose_data_server_index // data_server_per_machine
        port_num = choose_data_server_index % data_server_per_machine
        target_ip = self.policy_config['machines'][machine_index]
        target_port = self.policy_config['learner_port_start'] + port_num
        self.logger.info("==================== 此智能体要发送数据到: {}, 连接的端口为: {} =================".format(target_ip, target_port))
        self.data_sender.connet("tcp://{}:{}".format(target_ip, target_port))

    def load_model(self, model_info):
        # 载入模型，这个函数只有在训练的时候，才会被智能体调用来更新自己的网络参数
        if model_info is not None:
            self.mdoel_info = model_info
            model_path = model_info['path']
            deserialize_model(self.net_work, model_path)


    def reset(self):
        # 对于reset函数而言，无论是eval还是train模式都是需要的，不同的地方在于，eval模式下直接读取模型路径，加载，train模式下是通过fether获取
        if self.eval_mode:
            model_path = self.policy_config['model_path']
            self.logger.info("================= 此次为测试模式，模型初始化的时候直接从本地读取模型，模型的路径为: {} =============".format(model_path))
            deserialize_model(self.net_work, model_path)
        else:
            model_info = self.fetcher.reset()
            self.load_model(model_info)
            

    def step(self):
        # 这个step函数也是在training模式下才会使用，具体是在每一次rollout的时候，都需要调用一次
        if self.eval_mode:
            return 
        else:
            model_info = self.fetcher.step()
            self.load_model(self.net_work, model_info)

    def get_model_info(self):
        return self.mdoel_info

    def send_data(self, data):
        # 测试模式跑在本地，因此将数据存到本地就可以了，一般都是服务器跑一下，TODO
        if self.eval_mode:
            pass
        else:
            self.data_sender.send(pickle.dumps(data))

    def compute(self, state):
        # 这个是调用策略网络来进行计算的，网络使用的是Transformer结构，Encoder得到多个有意义的隐藏向量矩阵后，传入到Decoder，使用pointer network来进行解码操作
        with torch.no_grad():
            # 通过transformer进行了特征提取了之后，有两个head出来，一个是transformer decoder部分，得到调度列表，另外一个head出来的是v(s)的值，在推断阶段这个值不要
            # 因为还是遵循的CTDE的训练方式，每次决策之后，需要将所有智能体的backbone汇聚在一起进行V值的计算，由于action的长度有长有短，因此需要进行mask操作，统一到
            # 固定长度。如果是单个点进行决策，返回的log_probs表示的是联合概率的对数，action表示的是调度序列，mask表示的是固定长度的0-1向量
            log_probs, action, mask = self.net_work(convert_data_format_to_torch(state), inference_mode = True)
            return action, log_probs, mask