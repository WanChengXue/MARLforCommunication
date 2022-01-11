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

class config_server(basic_server):
    def __init__(self, config_path):
        super(config_server, self).__init__(config_path)
        config_server_log_path = pathlib.Path(self.config_dict['log_dir']+ "/config_server_log")
        self.log_handler = setup_logger('config_server',  config_server_log_path)
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
        self.policy_id = self.config_dict["learners"]['policy_id']
    
        self.policy_config_dict[self.policy_id] = self.config_dict["learners"]
        self.latest_model_info[self.policy_id] = {}
        self.next_update_delay_model_time[self.policy_id] = time.time()

    def process_model_request(self, raw_data_list):
        # 这个函数用来处理来自于worker端的请求,用来获取最新模型的地址
        for raw_data in raw_data_list:
            model_info = None
            request_info = pickle.loads(raw_data[-1])
            policy_id = request_info["policy_id"]
            if request_info["type"] == "latest":
                if policy_id in self.latest_model_info:
                    model_info = self.latest_model_info[policy_id]
                else:
                    self.log_handler.warn("============= config server 没有收到来自于learner的模型信息 =============")
            else:
                self.log_handler.warn("=========== 目前只支持使用最新的模型 =============")

            if model_info is not None:
                raw_data[-1] = pickle.dumps(model_info)
                self.model_requester.send_multipar(raw_data)
            
    def process_new_model(self, raw_data_list):
        # 这个地方处理新收到的模型
        for raw_data in raw_data_list:
            model_info = pickle.loads(raw_data)
            self.log_handler.info("============= 接收到了新模型, 当前的策略是 {}, 模型路径在 {} =============".format(model_info["policy_id"], model_info["url"]))
            # 更新latest model
            policy_id = model_info["policy_id"]
            assert policy_id == self.policy_id
            self.latest_model_info[policy_id] = {
                "url": model_info["url"], 
                "policy_id": policy_id,
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
                    self.process_new_model(raw_data_list)

                else:
                    self.log_handler.warn("================= 错误的套接字: {} {} ==================".format(key, value))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='/Learner/configs/config_pointer_network.yaml', help='yaml format config')
    args = parser.parse_args()
    abs_path = '/'.join(os.path.abspath(__file__).split('\\')[:-2])
    concatenate_path = abs_path + args.config_path
    server = config_server(concatenate_path)
    server.run()
    
