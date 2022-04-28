from multiprocessing import Process
import pathlib
import argparse
import os
import sys
current_path = os.path.abspath(__file__)
root_path = '/'.join(current_path.split('/')[:-2])
sys.path.append(root_path)
from Utils.config_parse import parse_config
from Worker.sampler import sampler_worker



def single_process_generate_sample(config_path, port_num=None):
    parser = argparse.ArgumentParser() 
    parser.add_argument('--config_path', type=str, default='', help='yaml format config')
    parser.add_argument('--sampler_numbers', type=int, default=1, help='the trajectory numbers')
    args = parser.parse_args()
    args.config_path = config_path
    args.port_num = port_num
    worker = sampler_worker(args)
    worker.run_loop()

if __name__=='__main__':
    # ---------- 导入配置文件 ------------
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='/Learner/configs/config_single_cell_ddpg.yaml')
    args = parser.parse_args()
    abs_path = '/'.join(os.path.abspath(__file__).split('/')[:-2])
    concatenate_path = abs_path + args.config_path
    args.config_path = concatenate_path
    config_dict = parse_config(args.config_path)
    parallel_env_number = min(os.cpu_count()-10, config_dict['env']['parallel_env'])
    # print('---------- 并行化的worker数目为 {} -----------'.format(parallel_env_number))
    parallel_env_number = 4
    for i in range(parallel_env_number):
        # logger_path = pathlib.Path("./config_folder") / ("process_"+ str(i))
        # logger_name = "Process_"+ str(i)
        # logger = setup_logger(logger_name, logger_path)
        # p = Process(target=single_process_generate_sample,args=(logger,))
        # p.start()
        port_num = i % config_dict['policy_config']['server_number_per_device']
        p = Process(target=single_process_generate_sample, args=(args.config_path,port_num,))
        p.start()

# ssh -p 22112 serena@10.19.92.79
# air_lab.123