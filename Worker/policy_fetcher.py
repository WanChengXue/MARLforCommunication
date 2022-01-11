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
        self.policy_id = self.config_dict['policy_id']
        self.config_server_address = self.config_dict['config_server_address']
        self.statistic = statistic
        self.policy_config = self.config_dict['learners']
        fetcher_log_name = pathlib.Path(self.config_dict['log_dir'] + '/fetcher_log')
        self.logger = setup_logger('Fether_log',fetcher_log_name)
        # 模型保存在当前的目录下
        self.model_path = os.path.join(os.getcwd(), "{}.model".format(self.policy_id))
        # 获取最新的模型
        self.latest_model_requester = self.context.socket(zmq.REQ)
        self.latest_model_requester.connect("tcp://{}:{}".format(self.config_dict['config_server_address'], self.config_dict["config_server_request_model_port"]))
        # 定义变量,模型下一次用来更新的时间
        self.next_model_update_time = 0
        # 最新模型的时间戳
        self.last_model_timestamp = None
        # 定义模型每次发送获取最新模型的请求间隔
        self.lastest_model_url = None
        self.sampler_model_interval = self.config_dict['sampler_model_update_interval']
        self.logger.info("======================= 构建fetcher成功, 创建最新模型请求套接字 ===============")
        self.logger.info("======================= 最新模型保存到的路径为: {} ====================".format(self.model_path))

    def reset(self):
        self.step()

    def step(self):
        if time.time() < self.next_model_update_time:
            self.logger.info("================== 当前的时间还没有达到下一次模型更新的时间 ==================")
            return None

        model_info = self._get_model()
        if model_info is not None:
            self.next_model_update_time + self.sampler_model_interval
            # 统计模型的更新间隔
            if self.last_model_timestamp is not None:
                # 这个值是模型的更新信息,放在tensorboard上面
                self.statistic.append("sampler/model_update_interval/{}".format(self.policy_id), model_info['time'] - self.last_model_timestamp)
            self.last_model_timestamp = model_info['time']
        return model_info

    def _get_model(self):
        # 获取模型
        self.latest_model_requester.send(pickle.dumps({"policy_id": self.policy_id}))
        # 获取最新的模型信息
        start_time = time.time()
        raw_model_info = self.latest_model_requester.recv()
        self.statistic.append("sampler/model_requester_time/{}".format(self.policy_id, time.time()- start_time))
        model_info = pickle.loads(raw_model_info)
        if self._download(model_info):
            model_info['path'] = self.model_path
            return model_info
        else:
            self.logge.info("=================== 相同模型, 跳过更新模型 =================")
            return None
        


    def _download(self, model_info):
        if self.lastest_model_url != model_info['url']:
            self.lastest_model_url = model_info['url']
            url = model_info['url']
            self.logger.info("=================== 获取最新模型的url地址为: {} ================".format(url))
            # 先删除旧模型, 以防止模型下载失败不报错
            if os.path.exists(self.model_path):
                os.remove(self.model_path)
            
            start_time = time.time()
            # 这个地方通过scp命令下载文件, 第一个{}放的是ip地址,第二个{}放的是learner模型存放的位置,第三个{}放的是这个模型下载到本地要存放的位置
            scp_command =  "sshpass -p Amax1979! scp amax@{}:~/{} {}".format(self.config_server_address, url, self.model_path)
            os.system(scp_command)
            
            return True
        else:
            return False
            


