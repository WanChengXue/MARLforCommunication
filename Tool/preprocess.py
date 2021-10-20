# 这个文件用来对信道矩阵进行数据的预处理
import os
import pathlib
from multiprocessing import Pool
import shutil
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat
import h5py
def create_data_folder(data_folder):
    # 这个函数用来判断文件夹存不存在，如果存在，则删除，然后创建，如果不存在，则直接创建
    if os.path.exists(data_folder):
        shutil.rmtree(data_folder)
    data_folder.mkdir(parents=True, exist_ok=True)

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
    

if __name__ =='__main__':
    source_data_path = pathlib.Path("../data_part/Source_data")
    # source_data_path = '../data_part/Koopman_predict_dat/
    save_data_path = pathlib.Path("../data_part/preprocess_data")
    # save_data_path = '../data_part/predict_data/'
    # user_number = ['10_user','20_user','30_user','40_user']
    
    # user_number = ['10_user','20_user','30_user']
    user_number = ['20_user']
    # velocity = ['3KM','30KM','90KM']
    velocity = ['60KM']
    source_data_folder = []
    save_data_folder = []
    for user_index in user_number:
        for velocity_index in velocity:
            channel_position = source_data_path / user_index / velocity_index  
            channel_save_position = save_data_path / user_index / velocity_index  
            source_data_folder.append(channel_position)
            create_data_folder(channel_save_position)
            save_data_folder.append(channel_save_position)
    folder_number = len(source_data_folder)
    workers = np.minimum(os.cpu_count() - 1, folder_number)
    # workers = 1
    # pool = Pool(processes=workers)
    # for folder_index in range(folder_number): 
    #     pool.apply_async(preprocess_single_file, (source_data_folder[folder_index], save_data_folder[folder_index]))
    # pool.close()
    # pool.join()
    preprocess_single_file(source_data_folder[0],  save_data_folder[0])
    # preprocess_single_file(relative_position[0])
