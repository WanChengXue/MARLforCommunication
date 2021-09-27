import numpy as np
from scipy.io import loadmat, savemat
import os
from multiprocessing import Pool
import shutil
from tqdm import tqdm
import copy
def Koopman_Operator(data):
    test_1 = data[:,0:-1]
    test_2 = data[:,1:data.shape[1]]
    
    u2,s2,v2 = np.linalg.svd(test_1)
    s2 = np.diag(s2)
    a = s2.shape[0]
    b = s2.shape[1]
    r = min(a,b)
    
    u = u2[:,0:r]
    s = s2[0:r,0:r]
    v = v2[:,0:r]
    
    #Atil = np.linalg.multi_dot(np.transpose(u),test_2,np.transpose(v),np.linalg.inv(s))
    #til = np.dot(np.dot(u.T.round(4),test_2.round(4)),np.dot(v.T.round(4),np.linalg.inv(s).round(4)))#矩阵连乘的结果与Maltab不同
    
    
    #temp = min(abs(test_2))
    
    Atil = np.dot(np.dot(u.T.conj(),test_2),np.dot(v.T.conj(),np.linalg.inv(s)))
    
    mu,w = np.linalg.eig(Atil)
    mu_diag = np.diag(mu) 
    
    phi = np.dot(u,w)
    return mu_diag,phi

def channelPred(fileName, save_path, History_TimeSpan=5):
    Predict_Delay = History_TimeSpan
    data = loadmat(fileName)
    CH = data['H_DL_File']
    
    IMF_NUM = 2
    size_ch = CH.shape
    SC_NUM = size_ch[0]
    BS_NUM = size_ch[1]
    SECTOR_NUM = size_ch[2]
    UE_NUM = size_ch[3]
    Rx_NUM = size_ch[4]
    TX_NUM = size_ch[7]
    TTI_NUM = size_ch[8]
    Method = 2
    
    s = 1
    for i in range(len(size_ch)):
        s = s*size_ch[i]
    CH_Predict = np.zeros(s,dtype = complex)
    D_CH_Predict = np.zeros(s,dtype = complex)
    D_CH = np.zeros(s,dtype = complex)
    
    CH_Predict = CH_Predict.reshape([SC_NUM, BS_NUM, SECTOR_NUM, UE_NUM, Rx_NUM, BS_NUM, SECTOR_NUM, TX_NUM, TTI_NUM])
    D_CH_Predict = D_CH_Predict.reshape([SC_NUM, BS_NUM, SECTOR_NUM, UE_NUM, Rx_NUM, BS_NUM, SECTOR_NUM, TX_NUM, TTI_NUM])
    D_CH = D_CH.reshape([SC_NUM, BS_NUM, SECTOR_NUM, UE_NUM, Rx_NUM, BS_NUM, SECTOR_NUM, TX_NUM, TTI_NUM])
    
    PREDICT_START = History_TimeSpan
    err_M = np.zeros(s,dtype = complex)
    err_M = err_M.reshape([SC_NUM, BS_NUM, SECTOR_NUM, UE_NUM, Rx_NUM, BS_NUM, SECTOR_NUM, TX_NUM, TTI_NUM])
    
    #p = PREDICT_START
    
    for t in tqdm(range(TTI_NUM)):
        p = t
        for b in range(BS_NUM):
            for s in range(SECTOR_NUM):
                for u in range(UE_NUM):
                    for nb in range(BS_NUM):
                        for ns in range(SECTOR_NUM):
                            for c in range(SC_NUM):
                                if t < History_TimeSpan or t>= TTI_NUM-History_TimeSpan:
                                    # 如果说t< History_len则表示没有足够多的数据来预测未来的TTI信道数据，就直接放在那里
                                    CH_Predict[c,b,s,u,0,nb,ns,:,t] = CH[c,b,s,u,0,nb,ns,:,t]                                
                                else:
                                    history_data = CH[c,b,s,u,0,nb,ns,:,p - History_TimeSpan : p].reshape(TX_NUM , History_TimeSpan) # Maybe it is not necessary.
                                    mu,phi = Koopman_Operator(history_data)
                                    #mu_diag = np.diag(mu) #取mu的主对角线元素
                                    mu_diag_abs = abs(mu)
                                    if Method == 1:
                                        for tz in range(len(mu_diag_abs)):
                                            mu[tz,tz] = mu[tz,tz]/mu_diag_abs[tz]
                                    elif Method == 2:
                                        []#No Operation
                                    elif Method == 3:
                                        if IMF_NUM[b,s,u] >= 3:
                                            for tz in range(len(mu_diag_abs)):
                                                if abs(mu[tz,tz]) > 1:
                                                    mu[tz,tz] = mu[tz,tz] / mu_diag_abs[tz]
                                    for d in range(1,Predict_Delay+1):
                                        pre_temp = np.dot(np.dot(phi,np.power(mu,d)),np.dot(np.linalg.pinv(phi),history_data[:,-1]))
                                    CH_Predict[c,b,s,u,0,nb,ns,:,t] = copy.deepcopy(pre_temp)
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    koopman_predict_delay_children_folder = save_path + '/delay_TTI' + str(Predict_Delay)
    if not os.path.exists(koopman_predict_delay_children_folder):
        os.mkdir(koopman_predict_delay_children_folder)
    save_name = koopman_predict_delay_children_folder +  '/' + fileName.split('/')[-1]
    savemat(save_name, {'H_DL_File':CH_Predict})



def simulate(source_data_path, save_data_path):
    # 路径拼接
    len_folder = len(os.listdir(source_data_path))
    relative_CH_file_list = ["CH3D_"+str(i+1) +'.mat' for i in range(len_folder)]
    absolute_CH_file_list = [source_data_path+'/'+file_name for file_name in relative_CH_file_list]
    for file_name in absolute_CH_file_list:
        channelPred(file_name, save_data_path)

if __name__ == '__main__':
    raw_channel_path = "../data_part/source_data/"
    save_data_path = "../data_part/Koopman_predict_data/"
    user_number = ['10_user','20_user','30_user','40_user']
    velocity = ['3KM','30KM','90KM']
    source_data_folder = []
    save_data_folder = []
    for user_index in user_number:
        for velocity_index in velocity:
            channel_position = raw_channel_path + user_index + '/' + velocity_index  
            channel_save_position = save_data_path + user_index + '/' + velocity_index  
            source_data_folder.append(channel_position)
            save_data_folder.append(channel_save_position)
    folder_number = len(source_data_folder)
    workers = np.minimum(os.cpu_count() - 1, folder_number)
    # workers = 1
    pool = Pool(processes=workers)
    for folder_index in range(folder_number): 
        pool.apply_async(simulate, (source_data_folder[folder_index], save_data_folder[folder_index]))
    pool.close()
    pool.join()
    # simulate(source_data_folder[0], save_data_folder[0])
