import yaml
import os
from yaml import Loader

def load_yaml(config_path):
    with open(config_path, "r", encoding='utf-8') as f:
        config_dict = yaml.load(f, Loader=Loader)
    return config_dict


def parse_config(config_file_path):
    function_path = os.path.abspath(__file__)
    # root_path = '/'.join(function_path.split('/')[:-3])
    root_path = '/'.join(function_path.split('\\')[:-3])
    # root_path是长这个样子的'/home/miao/Desktop/ICC/Combinatorial_optimization'

    config_dict = load_yaml(config_file_path)
    if "eval_mode" not in config_dict:
        config_dict["eval_mode"] = False

    if "main_server_ip" in config_dict:
        config_dict["log_server_address"] = config_dict["main_server_ip"]
        config_dict["config_server_address"] = config_dict["main_server_ip"]

    # --------------- 这个地方对config_dict中的元素进行修改,此处是对env中的一些元素进行update ---------------------
    env_config_dict = config_dict['env']
    source_data_folder = env_config_dict['source_data_folder']
    save_data_folder = env_config_dict['save_data_folder']
    user_nums = env_config_dict['user_nums']
    velocity = env_config_dict['velocity']
    # ---------------- 根据用户数目,移动速度,拼接出来源文件的路径以及保存文件的路劲 --------------------------------
    abs_source_data_folder = root_path + '/' + source_data_folder + '/' + str(user_nums) +'_user/' + str(velocity) + 'KM'
    abs_save_data_folder = root_path+ '/' + save_data_folder + '/' + str(user_nums) +'_user/' + str(velocity) + 'KM'
    config_dict['env']['source_data_folder'] = abs_source_data_folder
    config_dict['env']['save_data_folder'] = abs_save_data_folder
    # ----------------- 覆盖掉原始的值 ----------------------------------------------------------------------
    p2p_root = "/home/amax/Desktop/chenliang/distributed_swap/"
    p2p_port = 6020
    # 使用单机多卡去运行
    p2p_ip = config_dict["main_server_ip"]
    # 最新模型保存路径
    p2p_path = "p2p_plf_model"
    policy_config = config_dict["learners"]
    policy_config["p2p_path"] = os.path.join(p2p_root, p2p_path)
    policy_config["p2p_url"] =  "http://{}:{}/{}".format(p2p_ip, p2p_port, p2p_path)
    # ddp相关参数
    ddp_port = policy_config["ddp_port"]
    policy_config["ddp_root_address"] = "tcp://{}:{}".format(p2p_ip, ddp_port)
    # --------------- 这个地方对config_dict中的learner部分进行修改，主要是将env中的一些参数复制过来 ---------------
    policy_config['conv_channel'] = env_config_dict['agent_nums']
    policy_config['action_dim'] = env_config_dict['total_antenna_nums'] + 1
    policy_config['max_decoder_time'] = env_config_dict['max_stream_nums']
    policy_config['agent_number'] = env_config_dict['agent_nums']
    policy_config['seq_len'] = env_config_dict['total_antenna_nums'] 
    config_dict['learners'] = policy_config
    # 这个地方是模型保存的位置,每次运行之后,worker读取模型的url位置
    
    return config_dict

    