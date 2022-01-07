# 这个工具函数用来对CH3D.mat进行数据划分
from scipy.io import loadmat
import sys
import os

current_path = os.path.abspath(__file__)
root_path = '/'.join(current_path.split('/')[:-2])
sys.path.append(root_path)

import pathlib
import numpy as np
from tqdm import tqdm
from Utils import check_folder_exist, config_parse, setup_logger
import argparse


class data_preprocess:
    def __init__(self, config_path):
        self.config_dict = config_parse(config_path)
        # 由于需要根据不同的配置来产生数据,比如说是不同用户速度和不同用户数量什么的
        self.source_data_folder = self.config_dict['source_data_folder']
        self.save_data_folder = self.config_dict['save_data_folder']
        self.user_nums = self.config_dict['user_nums']
        self.velocity = self.config_dict['velocity']
        self.subcarrier_nums = self.config_dict['sub_carrier_nums']
        # 根据上面的信息组合出原始数据存放的路径,以及要保存的处理好后的文件存放的位置
        # current_file_path的路径是 ~/Desktop/ICC/code_part/Tool
        current_file_path = os.path.realpath(__path__)
        # 这个root_path的具体样子是~/Desktop/ICC
        self.root_path = pathlib.Path('/'.join(current_file_path.split('/')[:-2]))
        self.special_source_data_folder = self.root_path / self.source_data_folder / (str(self.user_nums)+'_user') / (str(self.velocity)+'KM') 
        self.target_save_data_folder = self.root_path / self.save_data_folder / (str(self.user_nums)+'_user') / (str(self.velocity)+'KM') 
        folder_exist_flag = check_folder_exist(self.target_save_data_folder)
        logger_path = pathlib(self.config_dict['log_dir']) / "data_preprocess_server_log"
        self.logger_handler = setup_logger("data_preprocess_server", logger_path)
        if folder_exist_flag:
            self.logger_handler.info("===================== 用户数目: {}, 移动速度为: {}的配置下,文件已经处理过了,直接读取就可 =====================".format(self.user_nums, self.velocity))
        else:
            self.logger_handler.info("===================== 用户数目: {}, 移动速度为: {}的配置下,文件需要从源文件进行处理 ===================")
            self.preprocess_data()

    def preprocess_data(self):
        # 首先需要对数据文件进行排序
        file_list = os.listdir(self.special_source_data_folder)
        file_number = len(file_list)
        abs_file_list = [self.special_source_data_folder/('CH3D_{}.mat'.format(i+1))for i in range(file_number)]
        eval_file = abs_file_list.pop(-1)
        
        concatenate_list = []
        # 这个地方是遍历所有的文件，然后将新到的实数部分和复数部分在天线维度进行拼接
        for single_file in tqdm(abs_file_list):
            file_result = self.preprocess_single_file(single_file)
            concatenate_list.append(file_result)
        # 将这些文件全部拼接成为一个文件
        full_TTI_data = np.concatenate(concatenate_list, -1)
        eval_data = self.preprocess_single_file(eval_file)
        # 将所有载波的数据存入到本地
        self.logger_handler.info("================== 开始讲所有的载波分开，一个载波一个数据文件，存放到本地 ==========")
        for carrier_index in range(self.subcarrier_nums):
            # 定义存放的训练数据的路径
            save_training_file_name = self.target_save_data_folder /('training_channel_file_' +str(carrier_index) + '.npy')
            np.save(save_training_file_name, full_TTI_data[carrier_index,:,:,:,:,:,:,:,:])
            # 定义测试数据的存放路径
            save_eval_file_name = self.target_save_data_folder/('eval_channel_file_' + str(carrier_index) + '.npy')
            np.save(save_eval_file_name, eval_data[carrier_index,:,:,:,:,:,:,:,:])
        self.logger_handler.info("================= 数据处理部分完成 ==============")

    def preprocess_single_file(self, active_file):
        # ==================================================
        channel_file = loadmat(active_file)
        channel_matrix = channel_file['H_DL_File']
        # 9维的信道矩阵参数说明
        # 第一个维度是RB的数量，这里就取第一个RB
        # 第二个是目的基站的数量，第三个是目的扇区的数量
        # 第四个维度是总共天线的数目，第五个维度表示的每个用户的接收天线数目
        # 第六个维度表示的是源基站 第七个维度表示的是源扇区
        # 第八个维度表示的基站的发射天线，第九个维度表示的是TTI
        real_part = np.real(channel_matrix)
        imag_part = np.imag(channel_matrix)
        # real part和 imag part在天线维度进行拼接
        concatenate_data = np.concatenate((real_part, imag_part), -2)
        return concatenate_data
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, default='../learner/configs/config_pointer_network.yaml', help='yaml format config')
    args = parser.parse_args()
    data_preprocess_server = data_preprocess(args.config_path) 
