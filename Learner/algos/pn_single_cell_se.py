from distutils.log import info
import torch
import torch.nn as nn

def get_cls():
    return SLTrainer


class SLTrainer:
    def __init__(self, net, optimizer, scheduler ,policy_config, local_rank):
        self.policy_config = policy_config
        self.policy_net = net['policy']
        self.policy_optimizer = optimizer['policy']
        self.policy_scheduler = scheduler['policy']
        # -------- 定义交叉entropy为policy loss ---------
        self.policy_loss = nn.CrossEntropyLoss(reduction='none')
        # ------- 定义一个正则化常数，使用l2正则 -------
        self.regulization = float(self.policy_config['regulization'])
        # self.regulization = 0.0
        self.policy_name = list(self.policy_net.keys())[0]


    def step(self, training_batch):
        info_dict = dict()
        current_state = training_batch['state']
        actions = training_batch['actions']
        # -------- 只要状态和动作 --------
        label  = actions.reshape(-1)
        # -------- 需要获取log prob ------
        log_prob_marix, mask = self.policy_net[self.policy_name](current_state, actions)
        mask_vector = (1 - mask.float()).reshape(-1)
        # -------- reshape一下，得到(batch_size * seq_len) * user_nums
        embedding_dimention = log_prob_marix.shape[-1]
        # -------- reshape之后，维度为(bs * seq_len) * user_nums ------ 
        reshape_log_prob = log_prob_marix.reshape(-1, embedding_dimention)
        policy_loss_vector = self.policy_loss(reshape_log_prob, label)
        policy_loss = torch.mean(policy_loss_vector * mask_vector)
        layer_norm_list = []
        for name, value in self.policy_net[self.policy_name].named_parameters():
            if value.requires_grad:
                layer_norm_list.append(torch.sum(torch.pow(value, 2)))
        l2_norm_loss = sum(layer_norm_list)
        total_loss =policy_loss + l2_norm_loss * 0.5 * self.regulization * l2_norm_loss
        self.policy_optimizer[self.policy_name].zero_grad()
        total_loss.backward()
        # ------------ 记录梯度信息 ------------
        layer_max_grads = {}
        for name,value in self.policy_net[self.policy_name].named_parameters():
            if value.requires_grad:
                layer_max_grads[name] = torch.max(value.grad).item()

        self.policy_optimizer[self.policy_name].step()
        self.policy_scheduler[self.policy_name].step()
        info_dict['Policy_loss/policy_loss'] = policy_loss.item()
        info_dict['Policy_loss/l2_norm_loss'] = l2_norm_loss.item()
        info_dict['Policy_loss/total_loss'] = total_loss.item()
        info_dict['Policy_loss/layer_grad'] = layer_norm_list
        return info_dict








