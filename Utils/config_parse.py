import yaml
import os
from yaml import Loader

def load_yaml(config_path):
    with open(config_path, "r", encoding='utf-8') as f:
        config_dict = yaml.load(f, Loader=Loader)
    return config_dict


def parse_config(config_file_path):
    config_dict = load_yaml(config_file_path)
    if "test_mode" not in config_dict:
        config_dict["test_mode"] = False

    if "main_server_ip" in config_dict:
        config_dict["log_server_address"] = config_dict["main_server_ip"]
        config_dict["config_server_address"] = config_dict["main_server_ip"]

    # 这个地方是模型保存的位置,每次运行之后,worker读取模型的url位置
    p2p_root = "/home/amax/Desktop/chenliang/distributed_swap/"
    p2p_port = 6020

    # 使用单机多卡去运行
    p2p_ip = config_dict["main_server_ip"]
    # 最新模型保存路径
    p2p_path = "p2p_plf_model"
    policy_config = config_dict["policy_config"]
    policy_config["p2p_path"] = os.path.join(p2p_root, p2p_path)
    policy_config["p2p_url"] =  "http://{}:{}/{}".format(p2p_ip, p2p_port, p2p_path)
    # ddp相关参数
    ddp_port = policy_config["ddp_port"]
    policy_config["ddp_root_address"] = "tcp://{}:{}".format(p2p_ip, ddp_port)

    return config_dict

    