# 这个函数用来收集数据
import os
import zmq
import time
import pickle
import numpy as np
import pyarrow.plasma as plasma
import argparse

from learner.basic_server import basic_server
from utils.config_parse import parse_config
from utils import setup_logger
from utils.data_utils import TrainingSet
from utils.plasma import generate_plasma_id

class data_server(basic_server):
    def __init__(self, args):
        super(data_server, self).__init__(args.config_path)
        self.policy_config = self.config_dict['policy_config']
        self.global_rank = args.rank
        self.world_size = args.workd_size
        self.gpu_num_per_machine = self.policy_config['gpu_num_per_machine']
        self.local_rank = self.global_rank % self.gpu_num_per_machine
        self.test_mode = self.config_dict['test_mode']
        self.data_server_local_rank = args.data_server_local_rank
        self.policy_name = self.config_dict['policy_name']

        log_path = os.path.join(self.config_dict["log_dir"], "./data_log/{}_{}/{}".format(self.local_rank, self.policy_name, self.data_server_local_rank))
        self.logger_handler = setup_logger("DataServer", log_path)
        self.server_ip = self.policy_config["learner_server_ip"]
        self.server_port = self.policy_config["learner_port_start"] + self.local_rank * self.policy_config["data_server_to_learner_num"] + self.data_server_local_rank
        # 数据是通过这个套接字进行接收的
        self.receiver = self.context.socket(zmq.PULL)
        self.receiver.set_hwm(1000000)
        self.receiver.bind("tcp://%s:%d"%(self.server_ip, self.server_port))
        self.poller.register(self.receiver, zmq.POLLIN)

        root_server_ip = self.config_dict["main_server_ip"]
        root_gpu_address = "tcp://{}:{}".format(root_server_ip, self.policy_config["root_gpu_pub_start_port"])
        # 如果说这个是第一个数据进程
        if self.global_rank == 0 and self.data_server_local_rank == 0:
            self.root_publisher = self.context.socket(zmq.PUB)
            self.root_publisher.bind(root_gpu_address)
        else:
            self.root_subscriber = self.context.socket(zmq.SUB)
            self.root_subscriber.connect(root_gpu_address)
            self.root_subscriber.setsockopt_string(zmq.SUBSCRIBE, "")
            self.poller.register(self.root_subscriber, zmq.POLLIN)
        self.batch_size = self.policy_config["batch_size"]
        self.traj_len = self.policy_config["traj_len"]
        self.pool_capacity = self.policy_config["pool_capacity"]
        if self.test_mode:
            self.pool_capacity = 256
        self.traing_set = TrainingSet(self.batch_size, max_capacity=self.pool_capacity)
        self.recv_training_instance_count = 0
        self.start_training = False
        self.socket_time_list = []
        self.parse_data_time_list = []
        
        # 和plasma 相关的一些变量
        plasma_location = self.config_dict['plasma_server_location']
        plasma_id = generate_plasma_id(self.global_rank, self.data_server_local_rank)
        self.plasma_data_id = plasma.ObjectID(plasma_id)
        # 连接server, 这个需要在服务器上提前进行打开的
        self.plasma_client = plasma.connect(plasma_location, 2)
        self.data_server_sampling_interval = self.policy_config["data_server_sampling_interval"]
        self.next_sampling_time = time.time()

    def receive_data(self, socks):
        # 接收从server传过来的数据
        raw_data_list = []
        if self.reveiver in socks and socks[self.receiver] == zmq.POLLIN:
            start_receive_time = time.time()
            while True:
                try:
                    data = self.receiver.recv(zmq.NOBLOCK)
                    raw_data_list.append(data)
                except zmq.ZMQError as e:
                    if type(e) != zmq.error.Again:
                        self.logger_handler.warn("=============== 异常错误 {} 发生! =============".format(e))
                    break
            if len(raw_data_list) > 0:
                self.socket_time_list.append(time.time() - start_receive_time)
            
            cur_recv_total = 0
            start_process_time = time.time()
            for raw_data in raw_data_list:
                