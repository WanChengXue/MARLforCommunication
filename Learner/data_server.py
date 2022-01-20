# 这个函数用来收集数据
import os
import zmq
import time
import pickle
import numpy as np
import pathlib
import pyarrow.plasma as plasma
import lz4.frame as frame

import sys
import os
current_path = os.path.abspath(__file__)
root_path = '/'.join(current_path.split('/')[:-2])
sys.path.append(root_path)

from Learner.basic_server import basic_server
from Utils import setup_logger
from Utils.data_utils import TrainingSet
from Utils.plasma import generate_plasma_id

class data_server(basic_server):
    def __init__(self, args):
        super(data_server, self).__init__(args.config_path)
        self.global_rank = args.rank
        self.world_size = args.world_size
        self.policy_config = self.config_dict['learners']
        self.gpu_num_per_machine = self.policy_config['gpu_num_per_machine']
        self.local_rank = self.global_rank % self.gpu_num_per_machine
        self.eval_mode = self.config_dict['eval_mode']
        self.data_server_local_rank = args.data_server_local_rank
        self.policy_name = self.policy_config['policy_id']

        log_path = pathlib.Path(self.config_dict['log_dir'] + "/data_log/{}_{}/{}".format(self.local_rank, self.policy_name, self.data_server_local_rank))
        self.logger = setup_logger("DataServer_log", log_path)
        machine_index = self.global_rank // self.gpu_num_per_machine
        self.server_ip = self.policy_config["machines"][machine_index]
        self.server_port = self.policy_config["learner_port_start"] + self.local_rank * self.policy_config["data_server_to_learner_num"] + self.data_server_local_rank
        #--------------------- 数据是通过这个套接字进行接收的 ----------------------------------
        self.receiver = self.context.socket(zmq.PULL)
        self.receiver.set_hwm(1000000)
        self.receiver.bind("tcp://%s:%d"%(self.server_ip, self.server_port))
        self.logger.info("----------------- 这个dataserver连接打开的端口为: {} -----------------".format(self.server_port))
        self.poller.register(self.receiver, zmq.POLLIN)
        self.batch_size = self.policy_config["batch_size"]
        self.traj_len = self.policy_config["traj_len"]
        self.pool_capacity = self.policy_config["pool_capacity"]
        if self.eval_mode:
            self.pool_capacity = 256

        self.traing_set = TrainingSet(self.init_replay_buffer_config())
        #-------------------------- 定义一些相关数据指标 ------------------------------
        self.recv_training_instance_count = 0
        self.socket_time_list = []
        self.parse_data_time_list = []
        self.sampling_time_list = []
        #--------------------- 定义一个变量,表示如果当前时间大于这个时间,就要进行日志的打印 -------------------------
        self.next_print_log_time = time.time()
        # 和plasma 相关的一些变量
        plasma_location = self.policy_config['plasma_server_location']
        plasma_id = generate_plasma_id(self.global_rank, self.data_server_local_rank)
        self.plasma_data_id = plasma.ObjectID(plasma_id)
        #---------------------- 连接server, 这个需要在服务器上提前进行打开的 ----------------------------------------
        self.plasma_client = plasma.connect(plasma_location, 2)
        self.data_server_sampling_interval = self.policy_config["data_server_sampling_interval"]
        self.next_sampling_time = time.time()

    def init_replay_buffer_config(self):
        # ------------ 这个函数是用来构造replay buffer的配置参数 -----------------------------
        replay_buffer_config_dict = dict()
        replay_buffer_config_dict['batch_size'] = self.batch_size
        replay_buffer_config_dict['agent_nums'] = self.policy_config['agent_nums']
        replay_buffer_config_dict['max_decoder_time'] = self.policy_config['max_decoder_time']
        replay_buffer_config_dict['seq_len'] = self.policy_config['seq_len']
        replay_buffer_config_dict['max_capacity'] = self.pool_capacity
        replay_buffer_config_dict['bs_antenna_nums'] = self.config_dict['env']['bs_antenna_nums']
        return replay_buffer_config_dict

    def receive_data(self, socks):
        #------------------- 接收从server传过来的数据 -------------------------
        raw_data_list = []
        if self.receiver in socks and socks[self.receiver] == zmq.POLLIN:
            start_receive_time = time.time()
            while True:
                try:
                    data = self.receiver.recv(zmq.NOBLOCK)
                    raw_data_list.append(data)
                except zmq.ZMQError as e:
                    if type(e) != zmq.error.Again:
                        self.logger.warn("=============== 异常错误 {} 发生! =============".format(e))
                    break
            if len(raw_data_list) > 0:
                self.socket_time_list.append(time.time() - start_receive_time)
            
            cur_recv_total = 0
            start_process_time = time.time()
            #-------------------- 接收数据, 数据加载, 保存到训练集中 --------------------
            for raw_data in raw_data_list:
                all_data = pickle.loads(frame.decompress(raw_data))
                self.traing_set.append_instance(all_data, self.logger)
                cur_recv_total += len(all_data)
            
            self.recv_training_instance_count += cur_recv_total

            #------------------------ 考察一下这次有没有数据被接收,看一下这次解析数据消耗了多少时间 -------------------
            if len(raw_data_list) > 0:
                self.parse_data_time_list.append(time.time()-start_process_time)
            
            del raw_data_list

        #-------------------------------- 日志是每一分钟打印一次的,因此 ----------------------------
        if time.time() > self.next_print_log_time:
            self.next_print_log_time += 60
            self.send_log({"data_server/dataserver_recv_instance_per_min/{}".format(self.policy_name): self.recv_training_instance_count})
            self.send_log({"data_server/dataserver_socket_time_per_min/{}".format(self.policy_name): sum(self.socket_time_list)})

        self.parse_data_time_list = []
        self.socket_time_list = []
        self.recv_training_instance_count = 0
    
    def sampling_data(self):
        if self.global_rank == 0:
            self.logger.info("============== 开始采样 ===============")
        start_time = time.time()
        # ----------- 此处随机采样出一个batch的训练数据 ----------------
        sample_data_dict = self.traing_set.slice()
        # ----------- 数据转移到plasma client里面去 -------------------
        if self.global_rank == 0:
            self.logger.info("================= 采样时间为 {}, batch size为 {}, 目前buffer的数据为 {} =============".format(time.time()-start_time, self.batch_size, self.traing_set.cursor))
        pickle_data = pickle.dumps(sample_data_dict)
        del sample_data_dict
        self.plasma_client.put(pickle_data, self.plasma_data_id, memcopy_threads=12)
        # ------------------------------------------------------------
        

    def run(self):
        self.logger.info("================= 数据服务启动，对应的是第{}张卡，数据服务索引为{} ==============".format(self.local_rank, self.data_server_local_rank))
        self.logger.info("================= 这个数据服务启动之后，对应的plasma id为: {} =================".format(self.plasma_data_id))
        while True:
            sockets = dict(self.poller.poll(timeout=100))
            if self.eval_mode:
                pass
            else:
                self.receive_data(sockets)
            # --------------- 如果说当前这个plasma id不再plasma客户端的时候，才能再放数据，并且还要这个replaybuffer是满的，超过了采样间隙才可以 ------------------
            if self.traing_set.full_buffer and self.plasma_client.contains(self.plasma_data_id) is False and time.time() > self.next_sampling_time:
                start_time = time.time()
                self.sampling_data()
                # ----------------- 添加采样时间到tensorboard上面 ------------------
                self.sampling_time_list.append(time.time()-start_time)
                self.next_sampling_time = time.time() + self.data_server_sampling_interval


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', default=0, type=int, help='rank of current process')
    parser.add_argument('--config_path', type=str, default='/Learner/configs/config_pointer_network.yaml', help='yaml format config')
    parser.add_argument('--data_server_local_rank', default=0, type=int, help='data_server_local_rank')
    parser.add_argument('--world_size', default=1, type=int, help='world_size')
    args = parser.parse_args()
    abs_path = '/'.join(os.path.abspath(__file__).split('/')[:-2])
    concatenate_path = abs_path + args.config_path
    args.config_path = concatenate_path
    server = data_server(args)
    server.run()