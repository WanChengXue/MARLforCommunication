import torch
import importlib


def serialize_model():
    # -------------- 这个函数就是通过是将模型保存到本地 -------------
    pass

def deserialize_model(model, path, device=torch.device("cpu")):
    # ------------- 这个函数是从内存中读取模型，然后加载，推断的时候默认使用CPU进行 -----------
    model.load_state_dict(torch.load(path, map_location=device))


def create_policy_model(config_dict):
    model_name = config_dict['learners']['model_name']
    model_fn = importlib.import_module("Model.{}".format(model_name)).init_policy_net
    return model_fn()

def create_critic_net(config_dict):
    model_name = config_dict['learners']['model_name']
    model_fn = importlib.import_module("Model.{}".format(model_name)).init_critic_net
    return model_fn()