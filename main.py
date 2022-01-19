import argparse
import os
import time


def start_one_machine(config_path, log_dir, gpu_num, start_gpu_id, cur_rank, world_size, policy_id,
                      root_address, data_server_local_num):
    cur_command = "plasma_store -s /home/chenliang08/Desktop/Netease/plasma/plasma_{} -m 20000000000 &".format(
        policy_id)
    os.system(cur_command)
    time.sleep(2)
    learner_log_path = os.path.join(
        log_dir, "{}_learner_log").format(policy_id)
    data_log_path = os.path.join(log_dir, "{}_data_log").format(policy_id)

    # 设置使用的GPU
    cuda_env_setting = "CUDA_VISIBLE_DEVICES=" + \
        ",".join(map(str, range(start_gpu_id, start_gpu_id + gpu_num)))
    print(cuda_env_setting)
    for i in range(gpu_num):
        for data_i in range(data_server_local_num):
            cur_command = cuda_env_setting + " nohup python -m learner.gpu_data_server"
            cur_command += " --rank {}".format(cur_rank + i)
            cur_command += " --world_size {}".format(world_size)
            cur_command += " --data_server_local_rank {}".format(data_i)
            cur_command += " --policy_id {}".format(policy_id)
            cur_command += " --config {}".format(config_path)
            cur_command += " >>{} 2>&1 &".format(data_log_path)
            os.system(cur_command)
        cur_command = cuda_env_setting + " nohup python -m learner.gpu_learner_server"
        cur_command += " --init_method {}".format(root_address)
        cur_command += " --rank {}".format(cur_rank + i)
        cur_command += " --world_size {}".format(world_size)
        cur_command += " --policy_id {}".format(policy_id)
        cur_command += " --config {}".format(config_path)
        cur_command += " >>{} 2>&1 &".format(learner_log_path)
        os.system(cur_command)


def main():
    from config import get_config

    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_id", type=str,
                        default="mouse", help="key in learners")
    parser.add_argument("--machine_index", default=0,
                        type=int, help="machine index in machines")
    parser.add_argument("--start_gpu_id", default=0,
                        type=int, help="device id")
    parser.add_argument("--config", default="./configs/config_micro_task_mouse_rnd.yaml",
                        type=str, help="yaml format config")
    args = parser.parse_args()

    config_dict = get_config(args.config)
    log_dir = config_dict["log_dir"]
    # 需要手动删除旧 log，防止误删
    os.makedirs(log_dir, exist_ok=False)

    # rank 0上启动log server，config server，self play server
    unable_cuda_devices = "CUDA_VISIBLE_DEVICES=-1"
    is_selfplay = config_dict["selfplay"]
    policy_index = list(config_dict["learners"].keys()).index(args.policy_id)
    if policy_index == 0 and args.machine_index == 0:
        # log server
        cur_command = unable_cuda_devices
        cur_command += " nohup python -m learner.log_server"
        cur_command += " --config {}".format(args.config)
        cur_command += " > {} 2>&1 &".format(os.path.join(log_dir, "log_log"))
        os.system(cur_command)
        # config server
        cur_command = unable_cuda_devices
        cur_command += " nohup python -m learner.config_server{}".format(
            " --selfplay" if is_selfplay else "")
        cur_command += " --config {}".format(args.config)
        cur_command += " > {} 2>&1 &".format(
            os.path.join(log_dir, "config_log"))
        os.system(cur_command)
        # tensorboard
        cur_command = unable_cuda_devices
        cur_command += " host=`ip route get 1 | awk '{print $NF;exit}'`;"
        cur_command += "nohup python -m tensorboard.main --logdir=./{} --host=${{host}}".format(
            log_dir)
        cur_command += " > /dev/null 2>&1 &"
        os.system(cur_command)

    policy_config = config_dict["learners"][args.policy_id]
    gpu_num = policy_config["gpu_num_per_machine"]
    cur_rank = args.machine_index * gpu_num
    world_size = len(policy_config["machines"]) * gpu_num
    data_server_local_num = policy_config["data_server_to_learner_num"]
    root_address = policy_config["ddp_root_address"]
    start_one_machine(args.config, log_dir, gpu_num, args.start_gpu_id, cur_rank, world_size,
                      args.policy_id, root_address, data_server_local_num)


# python start_server.py --policy_id xxx
if __name__ == "__main__":
    main()
