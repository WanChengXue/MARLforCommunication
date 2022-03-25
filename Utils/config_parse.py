import yaml
import os
from yaml import Loader

from Utils import  create_folder


def load_yaml(config_path):
    with open(config_path, "r", encoding='utf-8') as f:
        config_dict = yaml.load(f, Loader=Loader)
    return config_dict

def parse_config(config_file_path):
    function_path = os.path.abspath(__file__)
    # ------ 这个就是到了Pretrained_model这一层路径下面 ----- ~/Desktop/pretrained_model
    root_path = '/'.join(function_path.split('/')[:-2])
    config_dict = load_yaml(config_file_path)
    if "eval_mode" not in config_dict:
        config_dict["eval_mode"] = False

    if "main_server_ip" in config_dict:
        config_dict["log_server_address"] = config_dict["main_server_ip"]
        config_dict["config_server_address"] = config_dict["main_server_ip"]
    create_folder(config_dict['log_dir'])
    # ----------------- 覆盖掉原始的值 ----------------------------------------------------------------------
    # 使用单机多卡去运行
    main_server_ip = config_dict["main_server_ip"]
    policy_config = config_dict["policy_config"]
    # ddp相关参数
    ddp_port = policy_config["ddp_port"]
    policy_config["ddp_root_address"] = "tcp://{}:{}".format(main_server_ip, ddp_port)
    # --------------- 这个地方对config_dict中的learner部分进行修改，主要是将env中的一些参数复制过来 ---------------
    # ---- 处理一下plasma的保存位置，改成绝对位置,将父文件夹创建出来，然后client连接一定是文件 ----
    policy_config['plasma_server_location'] = root_path + '/' + policy_config['plasma_server_location'] 
    create_folder(policy_config['plasma_server_location'])
    policy_config['plasma_server_location']  = policy_config['plasma_server_location'] + '/' + config_dict['policy_name']
    # --------- 构建模型发布的url --------------------
    http_server_ip = "http://{}:{}".format(config_dict['config_server_address'], config_dict['config_server_http_port'])
    policy_config['model_pool_path'] = os.path.join(policy_config['model_pool_path'], config_dict['policy_name'])
    policy_config['saved_model_path'] = os.path.join(policy_config['saved_model_path'], config_dict['policy_name'])
    # ----------- 将model_pool_path和saved_model_path直接构建成绝对路径 ----------------
    abs_model_pool_path = os.path.join(root_path, policy_config['model_pool_path'])
    create_folder(abs_model_pool_path)
    create_folder(policy_config['saved_model_path'])
    policy_config['model_url'] = http_server_ip
    config_dict['policy_config'] = policy_config
    # ------------- 修改一下tensorboard的保存路劲 ----------
    policy_config['tensorboard_folder'] = os.path.join(config_dict['log_dir'], policy_config['tensorboard_folder'])
    # ------------- 最后就是说，worker需要从configserver上面下载新模型，就在本地创建一个文件夹出来 -------------
    create_folder("./Worker/Download_model")
    # ----------- 保存yaml文件 -----------
    yaml_saved_path = os.path.join(config_dict['log_dir'], 'config.yaml')
    with open(yaml_saved_path, 'w', encoding='utf8') as f:
        yaml.dump(config_dict, f)
    return config_dict
