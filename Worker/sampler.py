import time
import os
import sys
import uuid
import traceback
import pickle
import zmq
import argparse
from Utils import setup_logger
from Worker.rollout import rollout_sampler
from Worker.statistics import statistic_utils

from Utils import config_parse

class sampler_worker:
    def __init__(self, args):
        self.config_dict = config_parse(args.config_path)
        self.policy_id = self.config_dict['policy_id']
        self.sampler_numbers = args.sampler_numbers
        self.uuid = str(uuid.uuid4())
        self.statistic = statistic_utils()
        self.context = zmq.Context()
        worker_log_name = self.config_dict['log_dir']+ '/worker_log/' +self.uuid
        self.log_handler = setup_logger(worker_log_name)
        if "main_server_ip" in self.config_dict.keys():
            self.config_dict['log_server_address'] = self.config_dict['main_server_ip']
            self.config_dict['config_server_address'] = self.config_dict['main_server_ip']
            self.log_sender = self.context.socket(zmq.PUSH)
            self.log_sender.connect("tcp://{}:{}".format(self.config_dict['log_server_address'], self.config_dict['log_server_port']))
        
        self.rollout = rollout_sampler(args.config_path, self.config_dict, self.statistic, self.context, self.policy_id)
        
    def run(self):
        pass



if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--config_path', type=str, help='the yaml format config')
    parser.add_argument('--sampler_numbers', type=int, default=1, help='the trajectory numbers')
    args = parser.parse_args()
    worker = rollout_sampler(args)
    worker.run()