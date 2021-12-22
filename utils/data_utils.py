import torch
import numpy as np

class TrainingSet:
    def __init__(self, batch_size, max_capacity=10000):
        self.batch_size = batch_size
        self.max_capacity = max_capacity
        self.data_list = []
        
    def clear(self):
        self.data_list.clear()
    
    def len(self):
        return len(self.data_list)

    def append_instance(self, instance):
        self.data_list.extend([i.data for i in instance])

    def get_batched_obs(self, obs_list):
        sample = obs_list[0]
        if isinstance(sample, dict):
            batched_obs = dict()
            for key in sample:
                batched_obs[key] = self.get_batched_obs([obs[key] for obs in obs_list])
        elif isinstance(sample, (list, tuple)):
            batched_obs = [self.get_batched_obs(o) for o in zip(*obs_list)]
        else:
            batched_obs = np.asarray(obs_list)
            # TODO: 一维向量需不需要expand_dims(x, -1)
            if len(batched_obs.shape) == 1:
                batched_obs = np.expand_dims(batched_obs, -1)
        return batched_obs

    def slice(self, index_list, remove=False):
        pass