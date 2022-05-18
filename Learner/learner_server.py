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
        logger_path = pathlib.Path(self.config_dict['log_dir'] + '/learner_log/{}_{}'.format(self.policy_name,self.local_rank)) 
        self.logger = setup_logger('LearnerServer_log_{}'.format(self.local_rank), logger_path)
        self.logger.info("============== 开始构造learner server，这个server的全局id是:{}, 具体到某一个机器上的id是: {} ===========".format(self.global_rank, self.local_rank))
        # ------------ 默认是多机多卡，然后这个local rank表示的是某台机器上卡的相对索引 ----------
        self.machine_index = self.global_rank // self.policy_config['device_number_per_machine']
        self.logger.info("============== global rank {}开始创建模型 ==========".format(self.global_rank))
        # --------------------- 开始创建网络,定义两个optimizer，一个优化actor，一个优化critic ------------------
        self.construct_model()
        self.total_training_steps = 0
        self.training_steps_per_mins = 0
        if self.global_rank == 0:
            # ------------ 多卡场景下的训练，只有第0张卡的learner才会存模型下来 ----------
            # ---------------- 发送新的模型的路径给configserver，然后configserver会将模型信息下发给所有的worker -----------
            self.model_sender = self.context.socket(zmq.PUSH)
            self.model_sender.connect("tcp://{}:{}".format(self.config_dict['config_server_address'], self.config_dict['config_server_model_from_learner']))
            # ---------------- 每次模型发送出去后，下一次发送的时间间隔 -------------------------
            self.model_update_interval = self.config_dict['model_update_intervel']
            self.next_model_transmit_time = time.time()
            # ---------------- 定义最新模型的保存到learner下的路径 ---------------------
            # 注释，这个model_pool_path假设是~/Desktop/Model，然后模型存放在这个地方，发送出去的时候再加上一个前缀名称，其实不要也可以 
            self.model_pool_path = self.policy_config['model_pool_path']
            # ---------------- 定义最新模型发布到哪一个网站上 --------------------------
            self.model_url = self.policy_config['model_url']
            self.logger.info("--------------- 讲初始化模型发送给configserver --------------")
            self._send_model(self.total_training_steps)
        # ------------- 由于一个plasma对应多个data_server，因此需要循环的从这个plasma id列表中选择 -------------
        self.plasma_id_queue = queue.Queue(maxsize=self.policy_config['server_number_per_device'])
        for i in range(self.policy_config['server_number_per_device']):
            plasma_id = plasma.ObjectID(generate_plasma_id(self.machine_index, self.local_rank, i))
            self.plasma_id_queue.put(plasma_id)

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
        self.next_send_log_time = time.time()


    def construct_target_model(self):
        # --------- 这个只有那种需要创建target网络的算法，比如说MADDPG，DQN等，才需要进入 ----------
        self.target_model = dict()
        for model_type in self.policy_config['agent'].keys():
            self.target_model[model_type] = dict()
            for agent_name in self.policy_config['agent'][model_type].keys():
                model_config  = deepcopy(self.policy_config['agent'][model_type][agent_name])
                self.target_model[model_type][agent_name] = create_model(model_config)
                # ------------- target model参数copy active model -------------
                self.target_model[model_type][agent_name].load_state_dict(self.model[model_type][agent_name].state_dict())

    
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
                self.optimizer[model_type][agent_name] = torch.optim.Adam(self.model[model_type][agent_name].parameters(), lr=float(self.policy_config['agent'][model_type][agent_name]['learning_rate']))
                self.scheduler[model_type][agent_name] = CosineAnnealingWarmRestarts(self.optimizer[model_type][agent_name], self.policy_config['T_zero'])
        if self.policy_config.get('using_target_network', False):
            self.construct_target_model()
        # ----------- 训练模式, 使用DDP进行包装  --------------
        dist.init_process_group(init_method=self.policy_config["ddp_root_address"], backend="nccl",rank=self.global_rank, world_size=self.world_size)
        # ----- 把模型放入到设备上 ---------
        for model_type in self.model: 
            for sub_model in self.model[model_type]:
                self.model[model_type][sub_model].to(self.local_rank).train()
                if self.policy_config.get('using_target_network', False):
                    self.target_model[model_type][sub_model].to(self.local_rank)
                self.model[model_type][sub_model] = DDP(self.model[model_type][sub_model], device_ids=[self.local_rank])
        torch.manual_seed(194862226)
        self.logger.info('----------- 完成模型的创建 ---------------')
        # ----------- 调用更新算法 ---------------
        algo_cls = get_algorithm_cls(self.policy_config['algorithm'])
        if self.policy_config.get('using_target_network', False):
            self.algo = algo_cls(self.model, self.target_model, self.optimizer, self.scheduler, self.policy_config['training_parameters'])
        else:
            self.algo = algo_cls(self.model, self.optimizer, self.scheduler, self.policy_config['training_parameters'], self.local_rank)


    def _send_model(self, training_steps):
        if time.time() > self.next_model_transmit_time:
            '''
                url_path的样子:
                    url_path['policy_url'] : string
                如果说有critic网络在，则还有一个key:
                    url_path['critic_url']: string
            '''
            url_path = serialize_model(self.policy_config['model_pool_path'], self.policy_config['model_url'], self.model, self.config_dict['model_cache_size'], self.logger)
            model_infomation = {'policy_name': self.policy_name, 'url': url_path}
            self.next_model_transmit_time += self.model_update_interval
            self.logger.info("-------------------- 发送模型到configserver，发送的信息为: {}，当前的模型更新次数为: {}".format(model_infomation, training_steps+1))
            self.model_sender.send(pickle.dumps(model_infomation))
    
    def _training(self, training_batch):
        if time.time()>self.warm_up_time:
            torch_training_batch = convert_data_format_to_torch_training(training_batch,self.local_rank)
            info = self.algo.step(torch_training_batch)
            # self.logger.info("----------- 完成一次参数更新，更新的信息为 {} -------------".format(info))
            self.logger.info("----------- 完成一次参数更新 ----------")
            self.recursive_send(info, None, self.policy_name)
        else:
            self.logger.info("----------- 模型处于预热阶段，不更新参数 ----------")
            

    def training_and_publish_model(self):  
        start_time = time.time()
        selected_plasma_id = self.plasma_id_queue.get()
        batch_data = self.plasma_client.get(selected_plasma_id)
        if self.global_rank == 0:
            self.wait_data_times.append(time.time()-start_time)
        self._training(batch_data)
        self.training_steps_per_mins += 1
        self.total_training_steps += 1
        self._send_model(self.total_training_steps)
        # ------------ 将训练数据从plasma从移除 ------------
        self.plasma_client.delete([selected_plasma_id])
        self.plasma_id_queue.put(selected_plasma_id)
        if self.global_rank == 0:
            self.logger.info("----------------- 完成第{}次训练 --------------".format(self.total_training_steps))
            end_time = time.time()
            self.training_time_list.append(end_time-start_time)
            if end_time > self.next_send_log_time:
                # ---------- 将每分钟更新模型的次数，每次更新模型的时间发送回去 -------------
                self.send_log({"learner_server/model_update_times_per_min/{}".format(self.policy_name): self.training_steps_per_mins})
                self.send_log({"learner_server/average_model_update_time_consuming_per_mins/{}".format(self.policy_name): sum(self.training_time_list)/self.training_steps_per_mins})
                self.send_log({"learner_server/time_of_wating_data_per_mins/{}".format(self.policy_name): sum(self.wait_data_times)/self.training_steps_per_mins})
                self.next_send_log_time += 60
                self.training_steps_per_mins = 0
                self.training_time_list = []
                self.wait_data_times = []
            if self.total_training_steps % self.policy_config['model_save_interval'] == 0:
                self._save_model()
            if self.total_training_steps % self.policy_config['evaluate_model_interval'] == 0:
                self._evaluate_model()

    def _save_model(self):
        timestamp = str(time.time())
        for model_type in self.policy_config['agent'].keys():
            for agent_name in self.policy_config['agent'][model_type].keys():
                model_save_path = self.policy_config['saved_model_path'] + '/' + model_type + '_' + agent_name + '_'+ timestamp
                torch.save(self.model[model_type][agent_name].state_dict(), model_save_path)

    def _evaluate_model(self):
        # -------------- 这个函数是将模型在某一个数据集合上进行前向运算，得到评估值，然后写入到tensorboard上面 --------
        # ========= 首先载入env =========
        env_name = self.config_dict['env']['id']
        env_config = deepcopy(self.config_dict['env'])
        env_config['eval_mode'] = True
        env = importlib.import_module(env_name).Environment(env_config)
        state = env.reset(file_index = 0)
        torch_state = convert_data_format_to_torch_training(state,self.local_rank)
        with torch.no_grad():
            _, torch_action  = self.model['policy']['default_single_cell'](torch_state)
        instant_reward = env.step(torch_action.cpu().numpy())
        self.send_log({'result/evaluate_channel_value/{}'.format(self.policy_name): sum(instant_reward)/len(instant_reward)})
        
        

    def run(self):
        self.logger.info("------------------ learner: {} 开始运行 ----------------".format(self.global_rank))
        while True:
            self.training_and_publish_model()
    

    # def run(self):
    #     # 这个函数是运行的主函数
    #     self.logger.info("================ learner: {} 开始运行 =============".format(self.global_rank))
    #     while True:
    #         self.learn()
    #         if time.time() > self.next_check_time:
    #             self.next_check_time += 60
    #             if self.global_rank == 0:
    #                 self.send_log({"learner_server/training_times_per_mins/{}".format(self.policy_name): self.training_steps})
    #                 self.send_log({"learner_server/training_consuming_times_per_turn/{}".format(self.policy_name): sum(self.training_time_list)/len(self.training_time_list)})
    #                 self.send_log({"learner_server/wait_data_time_per_mins/{}".format(self.policy_name): sum(self.wait_data_times)})
    #             self.training_steps = 0
    #             self.training_time_list = []
    #             self.wait_data_times = []

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', default= 0, type=int, help="rank of current process")
    parser.add_argument('--world_size', default=1, type=int, help='total gpu card')
    parser.add_argument('--init_method', default='tcp://120.0.0.1:23456')
    parser.add_argument('--config_path', type=str, default='/Learner/configs/config_multi_cell_PF_pointer_network.yaml', help='yaml format config')
    args = parser.parse_args()
    abs_path = '/'.join(os.path.abspath(__file__).split('/')[:-2])
    concatenate_path = abs_path + args.config_path
    args.config_path = concatenate_path
    learner_server_obj = learner_server(args)
    learner_server_obj.run()