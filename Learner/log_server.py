# 这个函数用来记录日志
import os
import pickle
import time
import zmq

from torch.utils.tensorboard import SummaryWriter
import numpy as np

from Utils import setup_logger
from Learner.basic_server import basic_server



class SummaryLog(object):
    def __init__(self, tensorboard_folder):
        self.summary_writer = SummaryWriter(tensorboard_folder)
        self.tag_values_dict = {}
        self.tag_step_dict = {}
        self.tag_output_threshold_dict = {}
        self.tag_func_dict = {}
        self.tag_total_add_count = {}
        self.tag_time_threshold = {}
        self.tag_time_data_timestamp = {}
        self.tag_time_last_print_time = {}
        self.total_tag_type = [
            "time_avg", "time_total",
            "avg", "total", "max", "min"
        ]
        self.tag_bins = {}

    def add_tag(self, tag, output_threshold, cal_type, time_threshold=0, bins=100):
        self.tag_values_dict[tag] = []
        self.tag_step_dict[tag] = 0
        self.tag_output_threshold_dict[tag] = output_threshold
        self.tag_func_dict[tag] = cal_type
        self.tag_total_add_count[tag] = 0
        if cal_type.startswith("time"):
            self.tag_time_threshold[tag] = time_threshold
            self.tag_time_data_timestamp[tag] = []
            self.tag_time_last_print_time[tag] = 0
        if cal_type.find("histogram") != -1:
            self.tag_bins[tag] = bins

    def has_tag(self, tag):
        if tag in self.tag_step_dict.keys():
            return True
        else:
            return False

    def get_tag_count(self, tag):
        return self.tag_total_add_count[tag]

    def generate_time_data_output(self):
        for tag, threshold in self.tag_time_threshold.items():
            cur_time = time.time()
            if cur_time - self.tag_time_last_print_time[tag] > threshold:
                valid_list = []
                for i in range(len(self.tag_time_data_timestamp[tag])):
                    if cur_time - self.tag_time_data_timestamp[tag][i] < threshold:
                        valid_list.append(self.tag_values_dict[tag][i])

                if len(valid_list) >= 1:
                    if self.tag_func_dict[tag] == "time_avg":
                        out = sum(valid_list) / len(valid_list)
                    elif self.tag_func_dict[tag] == "time_total":
                        out = sum(valid_list)
                    else:
                        continue
                else:
                    out = 0


                self.summary_writer.add_scalar(tag, out, step=self.tag_step_dict[tag])
                self.tag_step_dict[tag] += 1
                self.tag_values_dict[tag] = []
                self.tag_time_data_timestamp[tag] = []
                self.tag_time_last_print_time[tag] = cur_time


    def add_summary(self, tag, value, timestamp=time.time()):
        self.tag_values_dict[tag].append(value)
        self.tag_total_add_count[tag] += 1
        if self.tag_func_dict[tag].startswith("time"):
            self.tag_time_data_timestamp[tag].append(timestamp)

        if not self.tag_func_dict[tag].startswith("time") and \
                len(self.tag_values_dict[tag]) >= self.tag_output_threshold_dict[tag]:
            if self.tag_func_dict[tag].find("histogram") != -1:
                # each value is a list
                all_values = []
                for i in self.tag_values_dict[tag]:
                    all_values.extend(i)
                self.log_histogram(tag, all_values, self.tag_step_dict[tag], self.tag_bins[tag])
            else:
                if self.tag_func_dict[tag] == "avg":
                    avg_value = sum(self.tag_values_dict[tag]) / len(self.tag_values_dict[tag])
                elif self.tag_func_dict[tag] == "total":
                    avg_value = sum(self.tag_values_dict[tag])
                elif self.tag_func_dict[tag] == "max":
                    avg_value = max(self.tag_values_dict[tag])
                elif self.tag_func_dict[tag] == "min":
                    avg_value = min(self.tag_values_dict[tag])
                elif self.tag_func_dict[tag] == "sd":
                    avg_value = np.array(self.tag_values_dict[tag]).std()
                else:
                    avg_value = sum(self.tag_values_dict[tag])
                self.summary_writer.add_scalar(tag, avg_value, step=self.tag_step_dict[tag])

            self.tag_step_dict[tag] += 1
            self.tag_values_dict[tag] = []




class LogServer(basic_server):
    def __init__(self, config_path):
        super(LogServer, self).__init__(config_path)

        self.receiver = self.context.socket(zmq.PULL)
        self.receiver.bind("tcp://%s:%d" % (self.config_dict["log_server_address"],
                                            self.config_dict["log_server_port"]))
        self.poller.register(self.receiver, zmq.POLLIN)

        self.log_basic_info = setup_logger("basic", os.path.join(self.config_dict["log_dir"], "log_server_log"))
        # --------------- 定义tensorboard的文件夹 ---------------
        self.summary_logger = SummaryLog(os.path.join(self.config_dict["log_dir"], "summary_log"))

        self.active_docker_dict = {}
        self.next_cal_docker_time = time.time()

    def summary_definition(self):

        # 效果类指标
        for policy_id, policy_config in self.config_dict["learners"].items():
            self.summary_logger.add_tag("result/episode_reward/{}".format(policy_id), 100, "avg")
            self.summary_logger.add_tag("result/buildin_win_rate/{}".format(policy_id), 100, "avg")
            self.summary_logger.add_tag("result/buildin_tie_rate/{}".format(policy_id), 100, "avg")
            self.summary_logger.add_tag("result/selfplay_win_rate/{}".format(policy_id), 100, "avg")
            self.summary_logger.add_tag("result/selfplay_tie_rate/{}".format(policy_id), 100, "avg")
            self.summary_logger.add_tag("result/history_win_rate/{}".format(policy_id), 100, "avg")
            self.summary_logger.add_tag("result/history_tie_rate/{}".format(policy_id), 100, "avg")
            self.summary_logger.add_tag("result/selfplay_elo_score/{}".format(policy_id), 1, "avg")
            self.summary_logger.add_tag("result/buildin_score/{}".format(policy_id), 100, "avg")
            self.summary_logger.add_tag("result/sp_score/{}".format(policy_id), 100, "avg")

        # sampler工程指标
        for policy_id, policy_config in self.config_dict["learners"].items():
            self.summary_logger.add_tag("sampler/sampler_model_request_time/{}".format(policy_id), 100, "avg")
            self.summary_logger.add_tag("sampler/sampler_model_update_interval/{}".format(policy_id), 100, "avg")
            self.summary_logger.add_tag("sampler/p2p_download_time/{}".format(policy_id), 100, "avg")
            self.summary_logger.add_tag("sampler/trajectory_running_time/{}".format(policy_id), 100, "avg")
            self.summary_logger.add_tag("sampler/error_per_min", 0, "time_total", time_threshold=60)

        # self play server工程类指标
        for policy_id, policy_config in self.config_dict["learners"].items():
            self.summary_logger.add_tag("selfplay/eval_count/{}".format(policy_id), 1, "avg")
            self.summary_logger.add_tag("selfplay/avg_history_win_rate/{}".format(policy_id), 1, "avg")
            self.summary_logger.add_tag("selfplay/avg_history_draw_rate/{}".format(policy_id), 1, "avg")
            self.summary_logger.add_tag("selfplay/avg_history_loss_rate/{}".format(policy_id), 1, "avg")

        # data server 工程类指标
        for policy_id, policy_config in self.config_dict["learners"].items():
            total_data_server = policy_config["data_server_to_learner_num"] * \
                policy_config["gpu_num_per_machine"] * len(policy_config["machines"])
            self.summary_logger.add_tag("data_server/dataserver_recv_instance_per_min/{}".format(policy_id), total_data_server, "total")
            self.summary_logger.add_tag("data_server/dataserver_parse_time_per_minutes/{}".format(policy_id), 1, "avg")
            self.summary_logger.add_tag("data_server/dataserver_socket_time_per_minutes/{}".format(policy_id), 1, "avg")
            self.summary_logger.add_tag("data_server/dataserver_sampling_time_per_min/{}".format(policy_id), 1, "avg")
            self.summary_logger.add_tag("data_server/active_docker_count", 1, "avg")
            self.summary_logger.add_tag("data_server/sample_count/{}".format(policy_id), 1, "avg")

        # learner server 工程类指标
        for policy_id, policy_config in self.config_dict["learners"].items():
            self.summary_logger.add_tag("learner_server/sgd_round_per_min/{}".format(policy_id), 1, "avg")
            self.summary_logger.add_tag("learner_server/sgd_total_time/{}".format(policy_id), 1, "avg")
            self.summary_logger.add_tag("learner_server/wait_data_time_per_min/{}".format(policy_id), 1, "avg")

        # 模型侧指标
        for policy_id, policy_config in self.config_dict["learners"].items():
            self.summary_logger.add_tag("model/entropy/{}".format(policy_id), 10, "avg")
            self.summary_logger.add_tag("model/state_value_loss/{}".format(policy_id), 10, "avg")
            self.summary_logger.add_tag("model/avg_q_value/{}".format(policy_id), 10, "avg")
            # 从sampler传过来
            self.summary_logger.add_tag("model/state_max_prob/{}".format(policy_id), 100, "avg")
            self.summary_logger.add_tag("model/advantage_std/{}".format(policy_id), 10, "avg")

        # 游戏本身的业务指标
        for policy_id, policy_config in self.config_dict["learners"].items():
            self.summary_logger.add_tag("game/cat_hp_change/{}".format(policy_id), 100, "avg")
            self.summary_logger.add_tag("game/game_length/{}".format(policy_id), 100, "avg")

    def log_detail(self, data):
        for field_key, value in data.items():
            # TODO: 如何区分 vs buildin
            if field_key == "docker_id":
                self.active_docker_dict[value] = 1
            else:
                if not self.summary_logger.has_tag(field_key):
                    self.summary_logger.add_tag(field_key, 100, "avg")
                self.summary_logger.add_summary(field_key, value)

    def run(self):
        self.summary_definition()

        while True:
            if time.time() > self.next_cal_docker_time:
                self.next_cal_docker_time = time.time() + 60 * 3
                self.summary_logger.add_summary("data_server/active_docker_count", len(self.active_docker_dict))
                self.active_docker_dict = {}

            self.summary_logger.generate_time_data_output()
            socks = dict(self.poller.poll(timeout=100))

            if self.receiver in socks and socks[self.receiver] == zmq.POLLIN:
                raw_data_list = []
                while True:
                    try:
                        data = self.receiver.recv(zmq.NOBLOCK)
                        raw_data_list.append(data)
                    except zmq.ZMQError as e:
                        if type(e) != zmq.error.Again:
                            self.log_basic_info.warn("recv zmq {}".format(e))
                        break

                for raw_data in raw_data_list:
                    data = pickle.loads(raw_data)
                    # self.log_basic_info.info(data)
                    for log in data:
                        if "error_log" in log:
                            self.log_basic_info.error("client_error, %s"%(log["error_log"]))
                            self.summary_logger.add_summary("sampler/error_per_min", 1, timestamp=time.time())
                        elif "log_info" in log:
                            self.log_basic_info.info(data)

                        else:
                            self.log_detail(log)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="yaml format config")
    args = parser.parse_args()

    server = LogServer(args.config)
    server.run()

