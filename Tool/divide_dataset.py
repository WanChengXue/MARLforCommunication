# 这个工具函数用来对CH3D.mat进行数据划分
from scipy.io import loadmat
import os
import pathlib
from multiprocessing import Pool
import shutil
import numpy as np
from tqdm import tqdm
from utils import check_folder_exist, config_parse, setup_logger
import argparse


def create_data_folder(data_folder):
    # 这个函数用来判断文件夹存不存在，如果存在，则删除，然后创建，如果不存在，则直接创建
    if os.path.exists(data_folder):
        shutil.rmtree(data_folder)
    data_folder.mkdir(parents=True, exist_ok=True)

class data_preprocess:
    def __init__(self, config_path):
        self.config_dict = config_parse(config_path)
        # 由于需要根据不同的配置来产生数据,比如说是不同用户速度和不同用户数量什么的
        self.source_data_folder = self.config_dict['source_data_folder']
        self.save_data_folder = self.config_dict['save_data_folder']
        self.user_nums = self.config_dict['user_nums']
        self.velocity = self.config_dict['velocity']
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
        for single_file in tqdm(abs_file_list):
            file_result = self.preprocess_single_file(single_file)
            

    def preprocess_single_file(self, active_file):
        # ==================================================
        channel_file = loadmat(active_file)
        channel_matrix = channel_file['H_DL_File'].transpose((8,7,6,5,4,3,2,1,0))
        # 9维的信道矩阵参数说明
        # 第一个维度是RB的数量，这里就取第一个RB
        # 第二个是目的基站的数量，第三个是目的扇区的数量
        # 第四个维度是总共天线的数目，第五个维度表示的每个用户的接收天线数目
        # 第六个维度表示的是源基站 第七个维度表示的是源扇区
        # 第八个维度表示的基站的发射天线，第九个维度表示的是TTI
        real_part = np.real(channel_matrix)
        imag_part = np.imag(channel_matrix)
        

        


def preprocess_data(active_file, user_antenna_number):
    # 这个文件就是对信道文件进行处理，信道文件是.mat文件
    # 首先读取这个文件
    try:
        H_file = loadmat(active_file)
    except:
        H_file = h5py.File(active_file, 'r')
    H_matrix = H_file['H_DL_File'].transpose((8,7,6,5,4,3,2,1,0))
    # 9维的信道矩阵参数说明
    # 第一个维度是RB的数量，这里就取第一个RB
    # 第二个是目的基站的数量，第三个是目的扇区的数量
    # 第四个维度是总共天线的数目，第五个维度表示的每个用户的接收天线数目
    # 第六个维度表示的是源基站 第七个维度表示的是源扇区
    # 第八个维度表示的基站的发射天线，第九个维度表示的是TTI
    tilde_H = H_matrix[0,:,:,:,:,:,:,:,:].squeeze()
    # 现在这个tilde_H变量的形状为(3,3,10,3,3,32,50)
    # 现在将对上面这个矩阵进行拆分,一个小区一个小区这样子,那么显然是根据源扇区进行拆分
    result = np.concatenate((np.real(tilde_H), np.imag(tilde_H)), axis=3)
    # 得到一个长度为三的列表，列表中的每一个元素都是长度为user_numbers *3* 3*32*1000的矩阵
    return result

def preprocess_single_file(folder_name, data_folder, user_antenna_number=2):
    print("Start: ===========" + str(folder_name) + "===========")
    # 由于每一个文件夹里面有多个个CH开头的mat文件，直接遍历这三个文件
    file_list_len = len(sorted(os.listdir(folder_name)))
    file_list = ["CH3D_"+str(i+1) +'.mat' for i in range(file_list_len)]
    training_file = file_list[:-1]
    testing_file = file_list[-1]
    file_result = []
    print("========== 开始对训练数据集进行处理 ===========")
    for file_name in tqdm(training_file):
        if "CH3D" in file_name:
            target_file_position = folder_name / file_name
            TTI_file_result =  preprocess_data(target_file_position, user_antenna_number)
            assert TTI_file_result.shape == (3,20,3,32,1000)
            file_result.append(TTI_file_result)
    print("========== 开始对测试数据集进行处理 ===========")
    testing_episode_data = preprocess_data(folder_name / testing_file, user_antenna_number)
    training_episode_data = np.concatenate(tuple(file_result), -1)
    assert training_episode_data.shape == (3,20,3,32,10000)
    save_name_traning_file = data_folder / 'training_data_10_10.npy'
    save_name_testing_file = data_folder / 'testing_data_10_10.npy'
    np.save(save_name_traning_file, training_episode_data)
    np.save(save_name_testing_file, testing_episode_data)
    

