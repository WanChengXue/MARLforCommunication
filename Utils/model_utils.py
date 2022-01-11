import torch
import importlib


def serialize_model():
    # -------------- 这个函数就是通过是将模型保存到本地 -------------
    pass

def deserialize_model(model, path, device=torch.device("cpu")):
    # ------------- 这个函数是从内存中读取模型，然后加载，推断的时候默认使用CPU进行 -----------
    pass
    # model.load_state_dict(torch.load(path, map_location=device))


def create_policy_model(policy_config):
    model_name = policy_config['model_name']
    model_fn = importlib.import_module("Model.{}".format(model_name)).init_policy_net
    return model_fn(policy_config)

def create_critic_net(policy_config):
    model_name = policy_config['model_name']
    model_fn = importlib.import_module("Model.{}".format(model_name)).init_critic_net
    return model_fn(policy_config)