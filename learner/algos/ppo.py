import torch
import torch.nn as nn
from torch.distributions import Categorical

class PPOTrainer:
    def __init__(self, net, optimizer, policy_config):
        self.policy_config = policy_config
        self.clip_epsilon = self.policy_config["clip_epsilon"]