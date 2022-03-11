# 这个函数用来记录日志
import os
import pickle
import time
import zmq

from torch.utils.tensorboard import SummaryWriter
import numpy as np
import pathlib
import os
import sys
current_path = os.path.abspath(__file__)
root_path = '/'.join(current_path.split('/')[:-2])
sys.path.append(root_path)

from Utils import setup_logger
from Learner.basic_server import basic_server



class summary_log:
    def __init__(self, tensorboard_folder):
        self.summary_writer = SummaryWriter(tensorboard_folder)
        # ----- 这个字典表示将tag的值 ---------
        self.tag_values_dict = {}
        # ----- 这个表示对应的tag写了多少次 ---------
        self.tag_step_dict = {}
        # ----- 这个值表示要超过多少才能写入到tensorboard上 ------
        self.tag_output_threshold_dict = {}
        # ------ 这个字典表示这个tag的计算类型 ------
        self.tag_func_dict = {}
        # ------ 这个字典表示这个tag的值被添加了多少次，和上面那个step dict还是有区别的。有些tag需要超过output_threshold_dict，对应的tag_step_dict才会+1 ------
        self.tag_total_add_count = {}
        # ------ 下面三个字典存放的tag数据必须是要求计算类型为time_mean和time_sum --------
        self.tag_time_threshold = {}
        self.tag_time_data_timestamp = {}
        self.tag_time_last_print_time = {}
        # ================= 三个字典分别表示时间的threhsold,多久统计一次，时间戳的统计，最新的打印时间 ===========
        # ------- 这个列表表示所有的tag的类型，要么是在给定时间间隔内计算平均，要么是在给定时间间隔计算总和，要么是计算mean，sum，max，min
        self.total_tag_type = ["time_mean", "time_sum","mean", "sum", "max", "min"]

    def add_tag(self, tag, output_threshold, cal_type, time_threshold=0, bins=100):
        # -------- 这个output_threshold 一般都是100，表示存了多少个点就进行一次输出 --------
        self.tag_values_dict[tag] = []
        self.tag_step_dict[tag] = 0
        self.tag_output_threshold_dict[tag] = output_threshold
        self.tag_func_dict[tag] = cal_type
        self.tag_total_add_count[tag] = 0
        if cal_type.startswith("time"):
            # ------- 如果这个tag是time_mean或者是time_sum ---------
            self.tag_time_threshold[tag] = time_threshold
            self.tag_time_data_timestamp[tag] = []
            self.tag_time_last_print_time[tag] = 0

    def has_tag(self, tag):
        # --------- 这个函数用来判断tag是不是在self.tag_step_dict这个字典中 ----------
        if tag in self.tag_step_dict.keys():
            return True
        else:
            return False

    @property
    def get_tag_count(self, tag):
        # --------- 这个函数表示这个tag被添加了多少次 ----------
        return self.tag_total_add_count[tag]


    def generate_time_data_output(self, tag):
        threshold = self.tag_time_threshold[tag]
        cur_time = time.time()
        if cur_time - self.tag_time_last_print_time[tag] > threshold:
            ############ 对于时隙统计tag而言，需要判断delta时间是不是大于threshold ##############
            valid_list = []
            for i in range(len(self.tag_time_data_timestamp[tag])):
                # ----------- 这个循环写的比较笨，比较所有列表中的时间戳，将那些合格的值放入到valid_list里面 -------------
                if cur_time - self.tag_time_data_timestamp[tag][i] < threshold:
                    valid_list.append(self.tag_values_dict[tag][i])
            if len(valid_list) >= 1:
                if self.tag_func_dict[tag] == "time_mean":
                    out = sum(valid_list) / len(valid_list)
                elif self.tag_func_dict[tag] == "time_sum":
                    out = sum(valid_list)
                else:
                    # ------------ 关于时间的tag就这两种，出现其他tag都会报错 ------------
                    raise NotImplementedError
            else:
                out = 0
            self.summary_writer.add_scalar(tag, out, self.tag_step_dict[tag])
            self.tag_step_dict[tag] += 1
            self.tag_values_dict[tag] = []
            self.tag_time_data_timestamp[tag] = []
            self.tag_time_last_print_time[tag] = cur_time


    def add_summary(self, tag, value, timestamp=time.time()):
        # ============= 这个函数传入tag，对应的值，以及时间戳 ==============
        self.tag_values_dict[tag].append(value)
        self.tag_total_add_count[tag] += 1
        if self.tag_func_dict[tag].startswith("time"):
            # -------------- 如果是time类型的tag，就需要把这个tag被添加的时间戳放进去 -------------
            self.tag_time_data_timestamp[tag].append(timestamp)
        
        if not self.tag_func_dict[tag].startswith("time"):
            if len(self.tag_values_dict[tag]) >= self.tag_output_threshold_dict[tag]:
                # ============== 这个地方是说，如果这个tag不是time类型的，并且这个tag列表中存放的值已经到了最大长度 ========================
                if self.tag_func_dict[tag] == "mean":     
                    fn_value = sum(self.tag_values_dict[tag]) / len(self.tag_values_dict[tag])
                elif self.tag_func_dict[tag] == "sum":
                    fn_value = sum(self.tag_values_dict[tag])
                elif self.tag_func_dict[tag] == "max":
                    fn_value = max(self.tag_values_dict[tag])
                elif self.tag_func_dict[tag] == "min":
                    fn_value = min(self.tag_values_dict[tag])
                else:
                    ################### 这一行其实就计算某一个变量的方差会用到 ###############
                    fn_value = np.array(self.tag_values_dict[tag]).std()
                self.summary_writer.add_scalar(tag, fn_value, self.tag_step_dict[tag])
                self.tag_step_dict[tag] += 1
                self.tag_values_dict[tag] = []
        else:
            self.generate_time_data_output(tag)

    def add_frequency(self, tag, scheduling_list):
        # ------------ 传入，比如agent_0，调度列表是是这个episode，所有用户的调度次数, 计算平均值，显示所有用户的调度分布情况 -----------
        self.summary_writer.add_scalar()
        # =======================================================================
        pass


class LogServer(basic_server):
    def __init__(self, config_path):
        super(LogServer, self).__init__(config_path)
        self.policy_name = self.config_dict['policy_name']
        self.agent_nums = self.config_dict['env']['agent_nums']
        self.total_antenna_nums = self.config_dict['env']['total_antenna_nums']
        # ------- 机器的数目 * 卡的数目 * 每张卡对应的数据进程数目 = 所有的数据服务 --------------
        self.total_data_server = self.config_dict['policy_config']['gpu_num_per_machine'] * self.config_dict['policy_config']['data_server_to_learner_num'] * len(self.config_dict['policy_config']['machines'])
        self.receiver = self.context.socket(zmq.PULL)
        self.receiver.bind("tcp://%s:%d" % (self.config_dict["log_server_address"], self.config_dict["log_server_port"]))
        self.poller.register(self.receiver, zmq.POLLIN)
        log_path = pathlib.Path(os.path.join(self.config_dict["log_dir"], "log_server_log"))
        self.logger = setup_logger("LogServer_log", log_path)
        # --------------- 定义tensorboard的文件夹 ---------------
        # "./logs/summary_log"
        self.summary_logger = summary_log(os.path.join(self.config_dict["log_dir"], "summary_log"))
        # --------- 这两个指标是说，采样端的数目，以及下一次计算的时间 ——--------
        self.active_docker_dict = {}
        self.next_cal_docker_time = time.time()
        self.logger.info("================== 完成log server的构建，配置好了tensorboard的路径为 {}".format(os.path.join(self.config_dict["log_dir"], "summary_log")))


    def summary_definition(self):
        ####################### 这个部分就是初始化一个tag到tensorboard上面，定义不同tag的计算方式 ########################
        # --------- 效果类指标, 分别是采样完毕后，所有用户平均SE的和以及边缘用户的平均SE -----------
        self.summary_logger.add_tag("result/edge_average_capacity/{}".format(self.policy_name), 100, "mean")
        self.summary_logger.add_tag("result/instant_capacity_average/{}".format(self.policy_name), 100, "mean")
        self.summary_logger.add_tag("result/average_PF_sum/{}".format(self.policy_name), 100, "mean")
        # --------- 采样端的指标：采样端请求模型的时间，更新模型的时间，从configserver下载模型需要的时间，完整采样一条trajectory的时间 ----------
        self.summary_logger.add_tag("sampler/episode_time/{}".format(self.policy_name), 100, "mean")
        self.summary_logger.add_tag("sampler/model_request_time/{}".format(self.policy_name), 100, "mean")
        self.summary_logger.add_tag("sampler/model_update_interval/{}".format(self.policy_name), 100, "mean")
        self.summary_logger.add_tag("sampler/p2p_download_time/{}".format(self.policy_name), 100, "mean")
        self.summary_logger.add_tag("sampler/trajectory_running_time/{}".format(self.policy_name), 100, "mean")
        # --------- dataserver的指标，包括每分钟接收的数据量，每分钟解析的时间，每分钟套接字的时间，从trainingSet采样放入到plasma client的时间，有多少个worker，采样的数目
        self.summary_logger.add_tag("data_server/dataserver_recv_instance_per_min/{}".format(self.policy_name), self.total_data_server, "sum")
        self.summary_logger.add_tag("data_server/dataserver_parse_time_per_minutes/{}".format(self.policy_name), 1, "sum")
        self.summary_logger.add_tag("data_server/dataserver_socket_time_per_minutes/{}".format(self.policy_name), 1, "mean")
        self.summary_logger.add_tag("data_server/dataserver_sampling_time_per_min/{}".format(self.policy_name), 1, "mean")
        self.summary_logger.add_tag("data_server/active_docker_count", 1, "mean")
        # -------- 添加策略的指标，包括entropy loss，两个head的状态值, MSELoss, PolicyLoss -----------------------
        self.summary_logger.add_tag("model/entropy/{}".format(self.policy_name), 10, "mean")
        self.summary_logger.add_tag("model/state_value_loss/{}".format(self.policy_name), 10, "mean")
        self.summary_logger.add_tag("model/policy_loss/{}".format(self.policy_name), 10, "mean")
        # ---------- 添加action的指标，主要是每一个用户的调度次数 -----------------------
        for agent_index in range(self.agent_nums):
            agent_key = 'sector_' + str(agent_index + 1)
            self.summary_logger.add_tag("action/{}/mean_scheduling_numbers/{}/{}".format(agent_key, self.policy_name, 'mean_scheduling_users_per_episode'), 1, 'mean')
            for antenna_index in range(self.total_antenna_nums):
                self.summary_logger.add_tag("action/{}/individual_scheduling_numbers/{}/{}_{}".format(agent_key, self.policy_name, 'antenna', str(antenna_index+1)), 1, 'mean')


    def log_detail(self, data):
        for field_key, value in data.items():
            # TODO: 如何区分 vs buildin
            if field_key == "container_id":
                # ---------- docker_id: uuid string ------------
                self.active_docker_dict[value] = 1
            else:
                if not self.summary_logger.has_tag(field_key):
                    # ----------- 这个地方就是说，如果这个tag不在上面预先定义的tag里面，就重新添加进去就好了，但是最好不要这样用 --------------
                    self.summary_logger.add_tag(field_key, 1, "mean")
                self.summary_logger.add_summary(field_key, value)

    def run(self):
        self.summary_definition()
        while True:
            if time.time() > self.next_cal_docker_time:
                # -------------- 每三分钟统计一下，看看有多少个worker发送了数据 --------------
                self.next_cal_docker_time = time.time() + 60 * 3
                self.summary_logger.add_summary("data_server/active_docker_count", len(self.active_docker_dict))
                self.active_docker_dict = {}

            socks = dict(self.poller.poll(timeout=100))
            if self.receiver in socks and socks[self.receiver] == zmq.POLLIN:
                raw_data_list = []
                while True:
                    try:
                        data = self.receiver.recv(zmq.NOBLOCK)
                        raw_data_list.append(data)
                    except zmq.ZMQError as e:
                        if type(e) != zmq.error.Again:
                            self.logger.warn("recv zmq {}".format(e))
                        break
                for raw_data in raw_data_list:
                    data = pickle.loads(raw_data)
                    # self.logger.info("------------ 接收到的数据为: {} ------------------".format(data))
                    for log in data:
                        if "error_log" in log:
                            self.logger.error("client_error, %s"%(log["error_log"]))
                        elif "log_info" in log:
                            self.logger.info(data)
                        else:
                            self.log_detail(log)
                        

def init_string():
    output_string  = r"""
                              _ooOoo_
                             o8888888o
                             88" . "88
                             (| -_- |)
                             O\  =  /O
                          ____/`---'\____
                        .'  \\|     |//  `.
                       /  \\|||  :  |||//  \
                      /  _||||| -:- |||||-  \
                      |   | \\\  -  /// |   |
                      | \_|  ''\---/''  |   |
                      \  .-\__  `-`  ___/-. /
                    ___`. .'  /--.--\  `. . __
                 ."" '<  `.___\_<|>_/___.'  >'"".
                | | :  `- \`.;`\ _ /`;.`/ - ` : | |
                \  \ `-.   \_ __\ /__ _/   .-` /  /
           ======`-.____`-.___\_____/___.-`____.-'======
                              `=---='
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                      佛祖保佑        永无BUG
             佛曰:
                    写字楼里写字间，写字间里程序员；
                    程序人员写程序，又拿程序换酒钱。
                    酒醒只在网上坐，酒醉还来网下眠；
                    酒醉酒醒日复日，网上网下年复年。
                    但愿老死电脑间，不愿鞠躬老板前；
                    奔驰宝马贵者趣，公交自行程序员。
                    别人笑我忒疯癫，我笑自己命太贱；
                    不见满街漂亮妹，哪个归得程序员？
            """
    print(output_string)

if __name__ == "__main__":
    import argparse
    init_string()
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='/Learner/configs/config_pointer_network.yaml', help='yaml format config')
    args = parser.parse_args()
    abs_path = '/'.join(os.path.abspath(__file__).split('/')[:-2])
    concatenate_path = abs_path + args.config_path
    server = LogServer(concatenate_path)
    server.run()

