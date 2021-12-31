import torch
import torch.nn as nn
from torch.distributions import Categorical

class PPOTrainer:
    def __init__(self, net, optimizer, policy_config):
        self.policy_config = policy_config
        self.clip_epsilon = self.policy_config["clip_epsilon"]
        self.entropy_coef = self.policy_config['entropy_coef']
        self.clip_value = self.policy_config.get("clip_value", True)
        self.grad_clip = self.policy_config.get("grad_clip", None)
        self.dual_clip = self.policy_config.get("dual_clip", None)
        self.parameter_sharing = self.policy_config.get("parameter_sharing", False)


    def step(self, training_batch):
        # 这个函数是用来对网络进行更新用的, 由于默认使用的共享网络参数，因此是CTDE方式进行的训练，当然也可以开三个智能体就是了。
        obs = training_batch['obs']
        global_state = training_batch['global_state']
        # 这个target_value表示的是使用了GAE估计出来的Target V
        target_value = training_batch['target_value']
        advantage = training_batch['advantage']
        action = training_batch['action']
        # 这个值是表示旧策略下，给定状态，执行了动作A的概率值,这个是用来计算IF的
        old_action_log_probs = training_batch['old_action_log_probs']
        if self.parameter_sharing:
            # 如果是参数共享，需要将所有智能体的数据拼接在一起，在batch维度上增加 TODO
            pass
            # 还需要仔细思考一下策略网络中v部分如何设计


        else:
            pass
            # 目前不支持智能体的策略网络是不一样的，因为这就意味这上面的net其实是一个字典，optimizer也是一个字典

        
