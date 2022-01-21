import pickle
import os
import zmq
import time
import pathlib
from Utils import setup_logger
# 由于这个类只有在training的时候才会实现,因此不需要考虑

class fetcher:
    def __init__(self, context, config_dict, statistic):
        self.context = context
        self.config_dict = config_dict
        self.parameter_sharing = self.config_dict['parameter_sharing']
        self.policy_id = self.config_dict['policy_id']
        self.config_server_address = self.config_dict['config_server_address']
        self.statistic = statistic
        self.policy_config = self.config_dict['learners']
        self.policy_type = self.policy_config['policy_type']
        fetcher_log_name = pathlib.Path(self.config_dict['log_dir'] + '/fetcher_log')
        self.logger = setup_logger('Fether_log',fetcher_log_name)
        self.construct_latest_model_path()
        #------------------- 获取最新的模型 -------------------
        self.latest_model_requester = self.context.socket(zmq.REQ)
        self.latest_model_requester.connect("tcp://{}:{}".format(self.config_dict['config_server_address'], self.config_dict["config_server_request_model_port"]))
        #------------------- 定义变量,模型下一次用来更新的时间 -------------------
        self.next_model_update_time = 0
        #------------------- 最新模型的时间戳以及最新模型的url -------------------
        self.last_model_timestamp = None
        self.lastest_model_url = None
        #------------------- 定义模型每次发送获取最新模型的请求间隔 -------------------
        self.sampler_model_interval = self.config_dict['sampler_model_update_interval']
        self.logger.info("======================= 构建fetcher成功, 创建最新模型请求套接字 ===============")
        self.logger.info("======================= 最新模型保存到的路径为: {} ====================".format(self.model_path))

    def construct_latest_model_path(self):
        #------------------- 模型保存在当前的目录下home/chenliang08/Desktop/Project/Combinatorial-optimization/xxx.model -------------------
        self.model_path = dict()
        if self.parameter_sharing:
            self.model_path['policy_path'] = os.path.join(os.getcwd(), "{}.model".format('policy_' + self.policy_id))
        else:
            self.model_path['policy_path'] = dict()
            for index in range(self.config_dict['env']['agent_nums']):
                agent_key = 'agent_' + str(index)
                self.model_path['policy_path'][agent_key] = os.path.join(os.getcwd(), "{}.model".format('policy_' + agent_key + '_' + self.policy_id))
        self.model_path['critic_path'] = os.path.join(os.getcwd(), "{}.model".format('critic_' + self.policy_id))
        self.logger.info("---------------- policy fetcher将最新的模型存放的地方为: {} ----------------".format(self.model_path))

    def remove_exist_model(self):
        # ----------------- 删除已经存在的模型 ---------------
        if os.path.exists(self.model_path['critic_path']):
            os.remove(self.model_path['critic_path'])

        if self.parameter_sharing:
            if os.path.exists(self.model_path['policy_path']):
                os.remove(self.model_path['policy_path'])
        else:
            for index in range(self.config_dict['env']['agent_nums']):
                agent_key = 'agent_' + str(index)
                if os.path.exists(self.model_path['policy_path'][agent_key]):
                    os.remove(self.model_path['policy_path'][agent_key])
            

    def reset(self):
        model_info = self.step()
        return model_info

    def step(self):
        if time.time() < self.next_model_update_time:
            self.logger.info("================== 当前的时间还没有达到下一次模型更新的时间 ==================")
            return None
        # ---------- 这个_get_model函数一旦调用，就会获取最新的模型信息，并且下载到本地self.model_path路径 -----------
        model_info = self._get_model()
        if model_info is not None:
            # ---------- 模型下一次更新的时间 -----------
            self.next_model_update_time + self.sampler_model_interval
            # ---------- 统计模型的更新间隔 ----------
            if self.last_model_timestamp is not None:
                self.statistic.append("sampler/model_update_interval/{}".format(self.policy_id), model_info['time'] - self.last_model_timestamp)
            self.last_model_timestamp = model_info['time']
        return model_info

    def _get_model(self):
        # 获取模型
        self.latest_model_requester.send(pickle.dumps({"policy_id": self.policy_id, 'type':self.policy_type}))
        # 获取最新的模型信息
        start_time = time.time()
        raw_model_info = self.latest_model_requester.recv()
        self.statistic.append("sampler/model_requester_time/{}".format(self.policy_id), time.time()- start_time)
        # model_info: {'policy_id': self.policy_id, 'url': url_path, 'time':timestamp}，这个timestamp是configserver收到learner发送的新模型信息时自己添加的
        # --------------- url_path是一个字典类型数据 ---------------
        model_info = pickle.loads(raw_model_info)
        if self._download(model_info):
            # -------------- 每次获取模型的时候，都需要看看，从configserver获取过来的url是不是和上次相同，相同就不更新了 -------------
            model_info['path'] = self.model_path
            return model_info
        else:
            # self.logger.info("=================== 相同模型, 跳过更新模型 =================")
            return None
        

    def copy_model(self, url):
        critic_path = '/home/chenliang08/Desktop/' + '/'.join(url['critic_url'].split("/")[3:])
        copy_command = 'cp {} {}'.format(critic_path, self.model_path['critic_path'])
        os.system(copy_command)
        if self.parameter_sharing:
            policy_path = '/home/chenliang08/Desktop/' + '/'.join(url['critic_url'].split("/")[3:])
            copy_command = 'cp {} {}'.format(policy_path, self.model_path['policy_path'])
            os.system(copy_command)

        else:
            pass


    def _download(self, model_info):
        if self.lastest_model_url != model_info['url']:
            self.lastest_model_url = model_info['url']
            url = model_info['url']
            self.logger.info("=================== 获取最新模型的url地址为: {} ================".format(url))
            self.remove_exist_model()
            # ------------- TODO 单机测试，这个地方暂时使用cp命令 -----------------
            self.copy_model(url)
            return True
        else:
            return False
            


