# 这个文件用来对信道矩阵进行数据的预处理
import os
import multiprocessing
from multiprocessing import Pool
import shutil
import numpy as np
from tqdm import tqdm
from scipy.io import loadmat

def create_data_folder(data_folder):
    # 这个函数用来判断文件夹存不存在，如果存在，则删除，然后创建，如果不存在，则直接创建
    if os.path.exists(data_folder):
        shutil.rmtree(data_folder)
    os.makedirs(data_folder)

def preprocess_data(active_file, user_antenna_number):
    # 这个文件就是对信道文件进行处理，信道文件是.mat文件
    # 首先读取这个文件
    H_file = loadmat(active_file)
    H_matrix = H_file['H_DL_File']
    # 9维的信道矩阵参数说明
    # 第一个维度是RB的数量，这里就取第一个RB
    # 第二个是目的基站的数量，第三个是目的扇区的数量
    # 第四个维度是总共天线的数目，第五个维度表示的每个用户的接收天线数目
    # 第六个维度表示的是源基站 第七个维度表示的是源扇区
    # 第八个维度表示的基站的发射天线，第九个维度表示的是TTI
    tilde_H = H_matrix[0,:,:,:,:,:,:,:,:].squeeze()
    # 现在这个tilde_H变量的形状为(3,3,10,3,3,32,50)
    # 现在将对上面这个矩阵进行拆分,一个小区一个小区这样子,那么显然是根据源扇区进行拆分
    base_station_number = tilde_H.shape[3]
    cell_number = tilde_H.shape[4]
    antenna_number = tilde_H.shape[5]
    TTI_number = tilde_H.shape[6]
    result = []
    for bs_index in range(base_station_number):
        for cell_index in range(cell_number):
            temp_result = tilde_H[bs_index,cell_index,:,:,:,:,:].squeeze()
            # 这个temp_result的维度是user_numbers * 3*3 *32*50
            # temp_result_reshape = temp_result.reshape(real_user_number, user_antenna_number,  base_station_number, TTI_number)
            result.append(np.concatenate((np.real(temp_result), np.imag(temp_result)), axis=3)) 
            # 得到一个长度为三的列表，列表中的每一个元素都是长度为user_numbers *3* 3*32*50的矩阵
    return result

def preprocess_single_file(folder_name, data_folder, user_antenna_number=2):
    print("Start: ===========" + folder_name + "===========")
    # 由于每一个文件夹里面有多个个CH开头的mat文件，直接遍历这三个文件
    file_list_len = len(sorted(os.listdir(folder_name)))
    file_list = ["CH3D_"+str(i+1) +'.mat' for i in range(file_list_len)]
    file_result = []
    for file_name in tqdm(file_list):
        if "CH3D" in file_name:
            target_file_position = folder_name + '/' + file_name
            TTI_file_result =  preprocess_data(target_file_position, user_antenna_number)
            file_result.append(TTI_file_result)
    file_number = len(file_result)
    file_result = np.array(file_result) # (22,9)
    bs_number = file_result.shape[3]
    cell_number = file_result.shape[4]
    # 上面两个变量，一个表示的是CH3D信道文件的数量，另外一个表示的是小区的数目
    count = 0
    for bs_index in range(bs_number):
        for cell_index in range(cell_number):
            temp_dataset = tuple([file_result[CH_index , count, :, :, :, :, :] for CH_index in range(file_number)])
            episode_data = np.concatenate(temp_dataset, axis=-1)
            save_name = data_folder + '/' + '10_10' + '_' + str(bs_index) + '_' + str(cell_index)
            if os.path.exists(save_name + '.npy'):
                os.remove(save_name + '.npy')
            np.save(save_name+ '.npy', episode_data)
            count += 1
    print("End: =============" + folder_name + "===========")


if __name__ =='__main__':
    source_data_path = '../data_part/source_data/'
    # source_data_path = '../data_part/Koopman_predict_dat/
    save_data_path = '../data_part/data/'
    # save_data_path = '../data_part/predict_data/'
    user_number = ['10_user','20_user','30_user','40_user']
    # user_number = ['10_user','20_user','30_user']
    # user_number = ['40_user']
    velocity = ['3KM','30KM','90KM']
    source_data_folder = []
    save_data_folder = []
    for user_index in user_number:
        for velocity_index in velocity:
            channel_position = source_data_path + user_index + '/' + velocity_index  
            channel_save_position = save_data_path + user_index + '/' + velocity_index  
            source_data_folder.append(channel_position)
            create_data_folder(channel_save_position)
            save_data_folder.append(channel_save_position)
    folder_number = len(source_data_folder)
    workers = np.minimum(os.cpu_count() - 1, folder_number)
    # workers = 1
    pool = Pool(processes=workers)
    for folder_index in range(folder_number): 
        pool.apply_async(preprocess_single_file, (source_data_folder[folder_index], save_data_folder[folder_index]))
    pool.close()
    pool.join()
    # preprocess_single_file(source_data_folder[0],  save_data_folder[0])
    # preprocess_single_file(relative_position[0])
