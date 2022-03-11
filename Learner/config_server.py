import os
import pickle
import zmq
import time
import pathlib

import sys
import os
current_path = os.path.abspath(__file__)
root_path = '/'.join(current_path.split('/')[:-2])
sys.path.append(root_path)
from Learner.basic_server import basic_server
from Utils import setup_logger
from Utils.zmq_utils import zmq_nonblocking_multipart_recv, zmq_nonblocking_recv
from multiprocessing import Process

class config_server(basic_server):
    def __init__(self, config_path):
        super(config_server, self).__init__(config_path)
        config_server_log_path = pathlib.Path(self.config_dict['log_dir']+ "/config_server_log")
        self.logger = setup_logger('config_server',  config_server_log_path)
        # 接受模型的请求,返回这个模型的url地址
        self.model_requester = self.context.socket(zmq.ROUTER)
        self.model_requester.set_hwm(1000000)
        self.model_requester.bind("tcp://{}:{}".format(self.config_dict['config_server_address'], self.config_dict['config_server_request_model_port']))
        self.poller.register(self.model_requester, zmq.POLLIN)

        # 接受模型信息
        self.model_receiver = self.context.socket(zmq.PULL)
        self.model_receiver.set_hwm(1000000)
        self.model_receiver.bind("tcp://{}:{}".format(self.config_dict['config_server_address'], self.config_dict['config_server_model_update_port']))
        self.poller.register(self.model_receiver, zmq.POLLIN)

        # 最新的模型信息
        self.latest_model_info = dict()
        # 这个next_update_delay_model_time表示的是下一次更新模型的时间
        self.next_update_delay_model_time = dict()
        self.policy_config_dict = dict()
        self.policy_name = self.config_dict["policy_config"]['policy_name']
    
        self.policy_config_dict[self.policy_name] = self.config_dict["policy_config"]
        self.latest_model_info[self.policy_name] = {}
        self.next_update_delay_model_time[self.policy_name] = time.time()


    def start_http_server(self):
        # ----------- 这个函数用来开一个http server ------------
        def _helper(server_ip, server_port, folder_path):
            pass

        server_ip = self.policy_config_dict['config_server_address']
        server_port = self.policy_config_dict['http_server_port']
        folder_path = self.policy_config_dict['model_folder_path']
        sub_process = Process(_helper, args=(server_ip, server_port, folder_path,))
        sub_process.start()



    def process_model_request(self, raw_data_list):
        # 这个函数用来处理来自于worker端的请求,用来获取最新模型的地址
        for raw_data in raw_data_list:
            model_info = None
            request_info = pickle.loads(raw_data[-1])
            policy_name = request_info["policy_name"]
            if request_info["type"] == "latest":
                if policy_name in self.latest_model_info:
                    model_info = self.latest_model_info[policy_name]
                else:
                    self.logger.warn("============= config server 没有收到来自于learner的模型信息 =============")
            else:
                self.logger.warn("=========== 目前只支持使用最新的模型 =============")

            if model_info is not None:
                raw_data[-1] = pickle.dumps(model_info)
                self.model_requester.send_multipart(raw_data)
            
    def process_new_model(self, raw_data_list):
        # -------------------- 这个地方处理新收到的模型 --------------------
        for raw_data in raw_data_list:
            model_info = pickle.loads(raw_data)
            self.logger.info("============= 接收到了新模型, 当前的策略是 {}, 模型路径在 {} =============".format(model_info["policy_name"], model_info["url"]))
            # 更新latest model
            policy_name = model_info["policy_name"]
            assert policy_name == self.policy_name
            self.latest_model_info[policy_name] = {
                "url": model_info["url"], 
                "policy_name": policy_name,
                "time": time.time()
            }
    
    def run(self):
        while True:
            sockets = dict(self.poller.poll(timeout=100))
            for key, value in sockets.items():
                if value != zmq.POLLIN:
                    continue
                
                if key == self.model_requester:
                    raw_data_list = zmq_nonblocking_multipart_recv(key)
                    self.process_model_request(raw_data_list)
                
                elif key == self.model_receiver:
                    raw_data_list = zmq_nonblocking_recv(key)
                    self.process_new_model(raw_data_list)

                else:
                    self.logger.warn("================= 错误的套接字: {} {} ==================".format(key, value))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='/Learner/configs/config_pointer_network.yaml', help='yaml format config')
    args = parser.parse_args()
    abs_path = '/'.join(os.path.abspath(__file__).split('/')[:-2])
    concatenate_path = abs_path + args.config_path
    server = config_server(concatenate_path)
    server.run()
    
