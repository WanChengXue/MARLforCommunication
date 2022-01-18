import torch
import importlib
import time
import glob
import os


def serialize_model(model_path_prefix, model_url_prefix, net_dict, cache_size, logger):
    # -------------- 这个函数就是通过是将模型保存到本地 -------------
    '''
    net = {'policy_net': policy_net, 'critic_net': global_critic}
    如果说使用了parametering sharing, policy_net就是一个网络,否则它是一个字典
    '''
    url_dict = dict()
    timestamp = str(time.time())
    critic_save_path = model_path_prefix + '_critic_' + timestamp
    critic_url_path = model_url_prefix + '_critic_' + timestamp
    url_dict['critic_url'] = critic_url_path
    torch.save(net_dict['critic_net'].state_dict(), critic_save_path)
    # --------- 将多余的critic模型移除 ---------------
    remove_old_version_model(model_url_prefix + '_critic_*', cache_size, logger)
    if isinstance(net_dict['policy_net'], dict):
        # ------------ 这个是不带参数共享的操作 --------------
        url_dict['policy_url'] = dict()
        for key in net_dict['policy_net'].keys():
            policy_save_path = model_path_prefix + '_policy_' + key + '_' + timestamp
            policy_url_path = model_url_prefix + '_policy_' + key + '_' + timestamp
            url_dict['policy_url'][key] = policy_url_path
            torch.save(net_dict['policy_net'][key].state_dict(), policy_save_path)
            # ------------ 移除多余的policy网络 -------------------
            remove_old_version_model(model_path_prefix + '_policy_' + key + '_*', cache_size, logger)
    else:
        policy_save_path = model_path_prefix + '_policy_' + timestamp
        policy_url_path = model_url_prefix + 'policy_' + timestamp
        url_dict['policy_url'] = policy_url_path
        torch.save(net_dict['policy_net'].state_dict(), policy_save_path)
        remove_old_version_model(model_path_prefix + '_policy_*', cache_size, logger)


def remove_old_version_model(model_prefix_name, cache_size, logger):
    # ------------ 这个函数是将老版本的模型移除 --------------
    model_files = glob.glob(model_prefix_name)
    if len(model_files) > cache_size:
        sorted_model_files = sorted(model_files)
        old_model = sorted_model_files[0]
        os.remove(old_model)
        logger.info("------------- 移除多余的模型: {} ---------------".format(old_model))

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