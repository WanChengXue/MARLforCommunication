import time
import os
import sys
import uuid
import traceback
import pickle
import zmq
import argparse
import pathlib
import os
import sys
current_path = os.path.abspath(__file__)
root_path = '/'.join(current_path.split('/')[:-2])
sys.path.append(root_path)

from Utils import setup_logger
from Worker.rollout import rollout_sampler
from Worker.statistics import StatisticsUtils

from Utils import config_parse

class sampler_worker:
    def __init__(self, args):
        self.config_dict = config_parse.parse_config(args.config_path)
        self.policy_id = self.config_dict['policy_id']
        self.eval_mode = self.config_dict['eval_mode']
        self.sampler_numbers = args.sampler_numbers
        self.uuid = str(uuid.uuid4())
        # --------------- 根据这个uuid修改日志的保存路径 --------------
        self.config_dict["log_dir"] = self.config_dict["log_dir"] + '/worker_log/' + self.uuid
        self.statistic = StatisticsUtils()
        self.context = zmq.Context()
        logger_path = pathlib.Path(self.config_dict['log_dir']+ '/sampler')
        self.logger = setup_logger('Sampler_'+ self.uuid, logger_path)
        if "main_server_ip" in self.config_dict.keys():
            self.config_dict['log_server_address'] = self.config_dict['main_server_ip']
            self.config_dict['config_server_address'] = self.config_dict['main_server_ip']
            self.log_sender = self.context.socket(zmq.PUSH)
            self.log_sender.connect("tcp://{}:{}".format(self.config_dict['log_server_address'], self.config_dict['log_server_port']))
        
        self.rollout = rollout_sampler(args.config_path, self.config_dict, self.statistic, self.context)
        self.logger.info("------------------ 完成采样端的构建，此worker的id为{} -----------------".format(self.uuid))
    def run(self):
        start_time = time.time()
        # ----------- TODO 这个地方确定好run one episode的结果 ------------
        self.rollout.run_one_episode()
        if self.eval_mode:
            return
        # ------------- 将对局结果记录下来 --------------
        episode_time = time.time() - start_time
        self.statistic.append("sampler/episode_time/{}".format(self.policy_id), episode_time)
        self.statistic.append("result/Average_edge_SE/{}".format(self.policy_id), 0)
        # ---------- 将统计信息发送到log server -------------
        self.logger.info("--------------- 发送结果日志到logServer上 --------------------")
        result_info = {"container_id": self.uuid}
        for key, value in self.statistic.iter_key_avg_value():
            result_info[key] = value
        self.log_sender.send(pickle.dumps([result_info]))
        self.statistic.clear()

    def run_loop(self):
        try:
            count = 0
            while True:
                self.run()
                count += 1
                if self.eval_mode:
                    if self.sampler_numbers <= count:
                        break
        except Exception as e:
            error_str = traceback.format_exc()
            self.logger.error(e)
            self.logger.error(error_str)
            if not self.eval_mode:
                # -------------- 如果不是评估模式，将报错信息也发送到logserver处 ---------
                error_message = {"error_log": error_str}
                p = pickle.dumps([error_message])
                self.log_sender.send(p)
            exit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser() 
    parser.add_argument('--config_path', type=str, default='/Learner/configs/config_pointer_network.yaml', help='yaml format config')
    parser.add_argument('--sampler_numbers', type=int, default=1, help='the trajectory numbers')
    args = parser.parse_args()
    abs_path = '/'.join(os.path.abspath(__file__).split('/')[:-2])
    concatenate_path = abs_path + args.config_path
    args.config_path = concatenate_path
    worker = sampler_worker(args)
    worker.run()