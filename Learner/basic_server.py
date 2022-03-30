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

    def recursive_send(self, log_info, prefix_string, suffix_string=None):
        # ------------ 这个传入进来的数据是一个字典，需要以递归的形式全部展开加头加尾进行发送 --------------
        if isinstance(log_info, dict):
            for key, value in log_info.items():
                if prefix_string is not None:
                    self.recursive_send(value, "{}/{}".format(prefix_string,key),suffix_string)
                else:
                    self.recursive_send(value, key, suffix_string)
        elif isinstance(log_info, (tuple,list)):
            for index, value in enumerate(log_info):
                self.recursive_send(value, "{}_{}".format(prefix_string, index), suffix_string)
        else:
            key = "{}/{}".format(prefix_string, suffix_string) if suffix_string is not None else prefix_string
            self.send_log({key: log_info})

            
        
    
