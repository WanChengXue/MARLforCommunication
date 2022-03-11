import argparse
import time
import pickle
import importlib
import pyarrow.plasma as plasma
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import zmq
import pathlib
import os
import sys
import queue
from copy import deepcopy

current_path = os.path.abspath(__file__)
root_path = '/'.join(current_path.split('/')[:-2])
sys.path.append(root_path)


from Learner.basic_server import basic_server
from Utils import create_folder, setup_logger
from Utils.model_utils import serialize_model, deserialize_model, create_model
from Utils.plasma import generate_plasma_id
from Utils.data_utils import convert_data_format_to_torch_training

def get_algorithm_cls(name):
    # 这个函数返回的是一个类，返回一个具体的算法类
    return importlib.import_module("Learner.algos.{}".format(name)).get_cls()


class learner_server(basic_server):
    # 这个函数是用来对网络进行参数更新
    def __init__(self, args):
        super(learner_server, self).__init__(args.config_path)
        self.policy_config = self.config_dict['policy_config']
        # ----------- 这个world size表示的是有多少张卡 ------------
        self.world_size = args.world_size
        # ------------ 这个变量表示这是第几个learner ----------
        self.policy_name = self.config_dict['policy_name']
        # ------------ 这个global_rank表示的是这个learner使用的是第几张卡，绝对索引 ----------
        self.global_rank = args.rank
        self.local_rank = self.global_rank % self.policy_config['device_number_per_machine']
        # ------------ 这个eval_mode表示这个learner是进行training，还是evaluate ——--------
        self.eval_mode = self.policy_config['eval_mode']
        logger_path = pathlib.Path(self.config_dict['log_dir'] + '/learner_log/{}_{}'.format(self.local_rank, self.policy_name)) 
        self.logger = setup_logger('LearnerServer_log_{}'.format(self.local_rank), logger_path)
        self.logger.info("============== 开始构造learner server，这个server的全局id是：{}, 具体到某一个机器上的id是: {} ===========".format(self.global_rank, self.local_rank))
        # ----------------- 开始初始化模型 ---------------------
        if self.global_rank ==0 and self.eval_mode:
            # ---------- 如果说这个learner是第一个进程，并且采用的是评估模式 -----------
            # self.logger.info("============== 评估模式直接加载模型，模型的路径为:{} ============".format(self.model_path))
            # deserialize_model(self.model, self.model_path)
            # ----------- TODO 这个地方的逻辑是，当发现是采用测试模式的时候，并且这是第0张卡，就开启一个work端就好了，其余的代码都不需要执行
            pass
        elif self.global_rank != 0 and self.eval_mode:
            self.logger.info("---------- 在测试模式下面，讲第0张卡对应的进程打开，其余learner server全部关闭 ----------")
            exit()
        else:
            self.logger.info('================ 开始初始化模型 ===========')
        # ------------ 默认是多机多卡，然后这个local rank表示的是某台机器上卡的相对索引 ----------
        self.machine_index = self.global_rank // self.policy_config['device_number_per_machine']
        self.parameter_sharing = self.policy_config['parameter_sharing']
        self.homogeneous_agent = self.policy_config['homogeneous_agent']
        self.logger.info("============== global rank {}开始创建模型 ==========".format(self.global_rank))
        # --------------------- 开始创建网络,定义两个optimizer，一个优化actor，一个优化critic ------------------
        self.construct_model()
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
            # ------------ 确保这个path是存在的 ----------------------
            pathlib.Path(path).mkdir(parents=True, exist_ok=True)
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

    
    def construct_model(self):
        self.optimizer = {}
        self.model = {}
        self.scheduler = {}
        # ------- 这个字典只用来保存模型路径，只有在测试的时候会用到 -------------------
        self.model_path = {}
        for model_type in self.policy_config['agent'].keys():
            self.optimizer[model_type] = dict()
            self.model[model_type] = dict()
            self.scheduler[model_type] = dict()
            self.model_path[model_type] = dict()
            for agent_name in self.policy_config['agent'][model_type].keys():
                model_config = deepcopy(self.policy_config['agent'][model_type][agent_name])
                self.model[model_type][agent_name] = create_model(model_config)

        if self.homogeneous_agent and self.parameter_sharing:
            model_config = {}
            model_config['model_name'] = self.policy_config['agent']['policy']['model_name']['default']
            model_config['state_dim'] = self.policy_config['agent']['policy']['state_dim']['default']
            model_config['action_dim'] = self.policy_config['agent']['policy']['action_dim']['default']
            self.model['policy'] = {'default': create_model(model_config)}
            self.optimizer['policy'] = {'default': torch.optim.Adam(self.model['policy']['default'].parameters(), lr=float(self.policy_config['agent']['policy']['learning_rate']))}
            self.model_path['policy'] = {'default': self.policy_config['agent']['policy']['model_path']['default']}
            self.scheduler['policy'] = {'default': CosineAnnealingWarmRestarts(self.optimizer['policy']['default'], self.policy_config['T_zero'])}
        else:
            # ----------- 如果采用的是异构智能体，则 ------------
            for agent_index in range(self.config_dict['env']['agent_nums']):
                self.model['policy'] = {}
                agent_name = 'agent_{}'.format(agent_index)
                model_config['model_name'] = self.policy_config['agent']['policy']['model_name'][agent_name]
                model_config['state_dim'] = self.policy_config['agent']['policy']['state_dim'][agent_name]
                model_config['action_dim'] = self.policy_config['agent']['policy']['action_dim']['agent_{}'.format(agent_index)]
                self.model['policy'][agent_name] = create_model(model_config)
                self.optimizer['policy'][agent_name] = torch.optim.Adam(self.model['policy'][agent_name].parameters(), lr=float(self.policy_config['agent']['policy']['learning_rate']))
                self.model_path['policy'][agent_name] = self.policy_config['agent']['policy'][agent_name]['model_path']
                self.scheduler['policy'][agent_name] = CosineAnnealingWarmRestarts(self.optimizer['policy'][agent_name], self.policy_config['T_zero'])
        if self.training_type == 'RL':
            # -------- 这个地方建立critic网络 TODO ------------
            pass
        # ----------- 训练模式, 使用DDP进行包装  --------------
        dist.init_process_group(init_method=self.policy_config["ddp_root_address"], backend="nccl",rank=self.global_rank, world_size=self.world_size)
        # ----- 把模型放入到设备上 ---------
        for model_type in self.model: 
            for sub_model in self.model[model_type]:
                self.model[model_type][sub_model].to(self.local_rank).train()
                self.model[model_type][sub_model] = DDP(self.model[model_type][sub_model], device_ids=[self.local_rank])
        torch.manual_seed(194864146)
        self.logger.info('----------- 完成模型的创建 ---------------')
        # ----------- 调用更新算法 ---------------
        algo_cls = get_algorithm_cls(self.policy_config['algorithm'])
        self.algo = algo_cls(self.model, self.optimizer, self.scheduler, self.policy_config)


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
            model_info = {'policy_name': self.policy_name, 'url': url_path}
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
                    self.send_log({"learner_server/training_times_per_mins/{}".format(self.policy_name): self.training_steps})
                    self.send_log({"learner_server/training_consuming_times_per_turn/{}".format(self.policy_name): sum(self.training_time_list)/len(self.training_time_list)})
                    self.send_log({"learner_server/wait_data_time_per_mins/{}".format(self.policy_name): sum(self.wait_data_times)})
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