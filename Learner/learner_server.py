# 这个函数用来更新参数
import argparse
import importlib
import os
import pickle
import pyarrow.plasma as plasma
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import zmq
import pathlib

import os
import sys
current_path = os.path.abspath(__file__)
root_path = '/'.join(current_path.split('/')[:-2])
sys.path.append(root_path)


from Learner.basic_server import basic_server
from Utils import setup_logger
from Utils.data_utils import convert_data_format_to_torch_training
from Utils.model_utils import serialize_model, deserialize_model

from Utils.plasma import generate_plasma_id
from Utils.model_utils import create_policy_model, create_critic_net

def get_algorithm_cls(name):
    # 这个函数返回的是一个类，返回一个具体的算法类
    return importlib.import_module("Learner.algos.{}".format(name)).get_cls()


class learner_server(basic_server):
    # 这个函数是用来对网络进行参数更新
    def __init__(self, args):
        super(learner_server, self).__init__(args.config_path)
        self.policy_config = self.config_dict['learners']
        self.policy_id = self.policy_config['policy_id']
        # 这个参数是用来控制,这个算法是不是让所有的智能体共享策略网络的参数
        self.parameter_sharing = self.config_dict['parameter_sharing']
        self.global_rank = args.rank
        self.world_size = args.world_size
        self.local_rank = self.global_rank % self.policy_config['gpu_num_per_machine']
        self.eval_mode = self.config_dict['eval_mode']
        logger_name =  pathlib.Path(self.config_dict['log_dir'] +  '/learner_log/{}_{}'.format(self.local_rank, self.policy_id))
        self.logger = setup_logger('Learner_log', logger_name)
        self.logger.info("================ 开始构造lerner server，这个server的global rank是: {} =========".format(self.global_rank))
        self.learning_rate = self.policy_config['learning_rate']
        # 开始初始化模型
        self.logger.info("============== global rank {}开始创建模型 ==========".format(self.global_rank))
        # --------------------- 开始创建网络,定义两个optimizer，一个优化actor，一个优化critic ------------------
        self.optimizer = {}
        if self.parameter_sharing:
            self.policy_net = create_policy_model(self.policy_config)
            self.optimizer['actor'] = torch.optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        else:
            self.policy_net = dict()
            self.optimizer['actor'] = dict()
            for agent_index in range(self.config_dict['env']['agent_nums']):
                agent_key = "agent_" + str(agent_index)
                self.policy_net[agent_key] = create_policy_model(self.policy_config)
                self.optimizer['actor'][agent_key] =  torch.optim.Adam(self.policy_net[agent_key].parameters(), lr=self.learning_rate)
        self.global_critic = create_critic_net(self.policy_config)
        self.optimizer['critic'] = torch.optim.Adam(self.global_critic.parameters(), lr=self.learning_rate)
        self.net = {'policy_net': self.policy_net, 'critic_net': self.global_critic}
        if self.global_rank == 0 and self.eval_mode:
            # 如果说是第一张卡，并且使用的是评估模式，就不需要训练网络了，直接加载即可
            self.logger.info("=============== 评估模式直接加载模型，模型的路径为:{} =============".format(self.policy_config['model_path']))
            deserialize_model(self.net, self.policy_config['model_path'])
        else:
            # torch分布式训练相关
            if self.parameter_sharing:
                self.local_rank = 'cpu'
                # dist.init_process_group(init_method=args.init_method, backend="gloo", rank=self.global_rank, world_size=self.world_size)
                self.policy_net.to(self.local_rank).train()
                # self.net = DDP(self.net, device_ids=[self.local_rank])
                torch.manual_seed(19930119)
                self.logger.info("============== 完成模型的创建 =========")
                algo_cls = get_algorithm_cls(self.policy_config.get("algorithm", "ppo"))
                self.algo = algo_cls(self.net, self.optimizer, self.policy_config, self.parameter_sharing)
            else:
                # ------------------------ TODO 这里是每个智能体使用不同的模型 -------------------
                self.logger.error("--------- 暂时不支持单卡多模型分布式的训练 --------------")

        self.total_training_steps = 0
        if self.global_rank == 0:
            #------------- 模型保存路径"/home/amax/Desktop/chenliang/distributed_swap/p2p_plf_max_average_SE" -------------
            path = self.policy_config['p2p_path']
            #------------- 发送新的模型给config server，然后config server再发布给所有的worker -------------
            self.model_sender = self.context.socket(zmq.PUSH)
            self.model_sender.connect("tcp://{}:{}".format(self.config_dict["config_server_address"], self.config_dict["config_server_model_update_port"]))
            # ------------- 每次模型发送出去后，下一次发送的时间间隔 -------------
            self.latest_update_interval = self.config_dict['latest_update_interval']
            self.next_model_update_time = time.time()
            # ------------- 最新模型的保存路径prefix: /home/amax/Desktop/chenliang/distributed_swap/p2p_plf_max_average_SE/model_max_average_SE ---------------------
            self.latest_model_path_prefix = os.path.join(path, self.policy_config['p2p_filename'])
            # ------------- 最新模型url的保存前缀为: http://42.186.72.223:6020/p2p_plf_max_average_SE/model_max_average_SE
            self.latest_model_url_prefix = os.path.join(self.policy_config['p2p_url'], self.policy_config['p2p_filename'])
            self.logger.info("=========== 将初始模型发送给config server =========")
            self.send_model(self.total_training_steps)
        # ------------- 由于一个plasma对应多个data_server，因此需要循环的从这个plasma id列表中选择 -------------
        self.plasma_id_list = []
        for i in range(self.policy_config['data_server_to_learner_num']):
            plasma_id = plasma.ObjectID(generate_plasma_id(self.global_rank, i))
            self.plasma_id_list.append(plasma_id)

        # ------------- 连接plasma 服务，这个地方需要提前启动这个plasma服务，然后让client进行连接 -------------
        self.plasma_client = plasma.connect(self.policy_config['plasma_server_location'], 2)
        # ------------- 这个列表是等待数据的时间 -------------
        self.wait_data_times = []
        # ------------- 定义一个变量，观察在一分钟之内参数更新了的次数 -------------
        self.training_steps = 0
        # ------------- 定义变量，每一次训练需要的时间 -------------
        self.training_time_list = []
        # ------------- 定义一下模型热更新的时间 -------------
        self.warm_up_time = time.time() + self.config_dict['warmup_time']
        # ------------- 定义一个变量，这个是每过一分钟，就朝着log server发送数据的 -------------
        self.next_check_time = time.time()

    def send_model(self, training_steps):
        if time.time()> self.next_model_update_time:
            # ------------- 这两个变量，分别表示的是worker访问的最新模型的url地址，以及这个模型保存在计算机的路径 -------------
            '''
            url_path的样子:
                如果使用了parameter sharing:
                    url_path['policy_url']: string
                如果没有使用parameter sharing:
                    url_path['policy_url']['agent_0']: string 
                    .......
                url_path['critic_url']: string
            '''
            url_path = serialize_model(self.latest_model_path_prefix, self.latest_model_url_prefix, self.net, self.config_dict['p2p_cache_size'], self.logger)
            model_info = {'policy_id': self.policy_id, 'url': url_path}
            self.next_model_update_time += self.latest_update_interval
            self.logger.info("============ 发送模型给config server, 发送的信息为: {}, 当前的模型已经更新了{}次 ==========".format(model_info, training_steps+1))
            self.model_sender.send(pickle.dumps(model_info))
    
    def update_parameters(self, training_batch):
        # 这个函数是通过传入训练数据，更新net的参数
        if self.global_rank == 0:
            if time.time() > self.warm_up_time:
                # 这个is_warmup是说参数暂时不更新，就是为了让智能体见识一下更多的数据
                training_batch = convert_data_format_to_torch_training(training_batch, self.local_rank)
                info  = self.algo.step(training_batch)
                self.logger.info("--------------- 完成一次参数更新, 返回的数据为 {} -----------------".format(info))
                ''' 
                    {'value_loss': 36.82817840576172, 
                    'conditional_entropy': -87.09746551513672, 
                    'advantage_std': array([1.8773540e+01, 3.7253532e-03], dtype=float32), 
                    'policy_loss': 10.446737289428711, 
                    'total_policy_loss': 9.575762748718262}
                '''
                # 将日志发送到log server上面
                # TODO, 日志发送操作
                self.send_statistic(info, prefix="model")
            else:
                self.logger.info("============== 模型不更新，在预热阶段 ===========")

    def learn(self):
        start_time = time.time()
        # 首先读取数据，从plasma client里
        raw_data = self.plasma_client.get(self.plasma_id_list[0])
        if self.global_rank == 0:
            # 添加一下等待数据所需的时间
            self.wait_data_times.append(time.time() - start_time)
        all_data = pickle.loads(raw_data)
        del raw_data
        self.update_parameters(all_data)
        self.training_steps += 1
        self.total_training_steps += 1
        # 这个地方其实就是将plasma id list中第一个id放到列表的最后面
        current_plasma_id = self.plasma_id_list.pop(0)
        self.plasma_id_list.append(current_plasma_id)
        # 发布新的模型
        if self.global_rank == 0:
            self.logger.info("============ 开始发布新模型 =========")
            self.send_model(self.total_training_steps)
            end_time = time.time()
            self.training_time_list.append(end_time - start_time)


    def run(self):
        # 这个函数是运行的主函数
        self.logger.info("================ learner: {} 开始运行 =============".format(self.global_rank))
        while True:
            self.learn()
            if time.time() > self.next_check_time:
                self.next_check_time += 60
                if self.global_rank == 0:
                    self.send_log({"learner_server/training_times_per_mins/{}".format(self.policy_id): self.training_steps})
                    self.send_log({"learner_server/training_consuming_times_per_turn/{}".format(self.policy_id): sum(self.training_time_list)/len(self.training_time_list)})
                    self.send_log({"learner_server/wait_data_time_per_mins/{}".format(self.policy_id): sum(self.wait_data_times)})
                self.training_steps = 0
                self.training_time_list = []
                self.wait_data_times = []

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', default= 0, type=int, help="rank of current process")
    parser.add_argument('--world_size', default=1, type=int, help='total gpu card')
    parser.add_argument('--init_method', default='tcp://120.0.0.1:23456')
    parser.add_argument('--config_path', type=str, default='/Learner/configs/config_pointer_network.yaml', help='yaml format config')
    args = parser.parse_args()
    abs_path = '/'.join(os.path.abspath(__file__).split('/')[:-2])
    concatenate_path = abs_path + args.config_path
    args.config_path = concatenate_path
    learner_server_obj = learner_server(args)
    learner_server_obj.run()