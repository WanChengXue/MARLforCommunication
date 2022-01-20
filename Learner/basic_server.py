import time 
import traceback
import pickle
import zmq

from Utils.config_parse import parse_config

class basic_server:
    def __init__(self, config_path):
        self.config_dict = parse_config(config_path)
        self.context = zmq.Context()
        self.context.setsockopt(zmq.MAX_SOCKETS, 102400)
        self.poller = zmq.Poller()

        # 发送给log_server
        self.log_sender = self.context.socket(zmq.PUSH)
        self.log_sender.connect("tcp://%s:%d" %(self.config_dict["log_server_address"], self.config_dict["log_server_port"]))

        # 缓存log
        self.cached_log_list = []
    
    def send_log(self, log_dict, send_threshold=3):
        self.cached_log_list.append(log_dict)
        if len(self.cached_log_list) > send_threshold:
            p = pickle.dumps(self.cached_log_list)
            self.log_sender.send(p)
            self.cached_log_list = []

    def send_statistic(self, info, prefix, suffix=None):
        # 这个prefix表示的是字符串的前缀，suffix表示的是后缀信息
        if isinstance(info, dict):
            for key, value in info.items():
                self.send_statistic(value, "{}/{}".format(prefix, key), suffix)
            
        elif isinstance(info, (tuple, list)):
            for i, value in enumerate(info):
                self.send_statistic(value, "{}_{}".format(prefix, i), suffix)
        
        else:
            if suffix is not None:
                key = "{}/{}".format(prefix, suffix)

            else:
                key = prefix
            self.send_log({key: info})

            
        
    
