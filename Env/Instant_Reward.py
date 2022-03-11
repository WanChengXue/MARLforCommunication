import numpy as np


def rebuild_channel_matrix(channel_matrix):
    bs_antenna_number = channel_matrix.shape[-1] // 2
    real_part = channel_matrix[:,:,:,:bs_antenna_number]
    imag_part = channel_matrix[:,:,:,bs_antenna_number:]
    return real_part + 1j * imag_part

def select_sub_channel_matrix(channel_matrix, bool_user_scheduling_matrix, cyclic_index_matrix):
    sector_number = bool_user_scheduling_matrix.shape[0]
    # 目的扇区 × 用户数目 × 源扇区 × 发射天线数
    # channel_matrix的维度为3*20*3*16, bool_user_scheduling_matrix的维度喂3*20*1
    # cyclic_index_matrix表示的是读取顺序，012,120,201
    selected_channel_matrix = {}
    for sector_index in range(sector_number):
        sector_selected_channel_matrix = {}
        # --------------------- 根据mask向量进行取值 --------------------
        sector_selected_channel_matrix['source_channel'] = channel_matrix[sector_index, bool_user_scheduling_matrix[sector_index,:], sector_index, :]
        # --------------------- 添加相邻小区的信道数据 ------------------
        sector_selected_channel_matrix['interfence_channel'] = []
        for interference_sector in cyclic_index_matrix[sector_index,:][1:]:
            sector_selected_channel_matrix['interfence_channel'].append(channel_matrix[sector_index, bool_user_scheduling_matrix[sector_index,:], interference_sector, :])
        selected_channel_matrix['sector_'+str(sector_index)] = sector_selected_channel_matrix
    # --------- 返回的selected channel matrix中，每一个元素都是一个k*3*16的信道矩阵
    return selected_channel_matrix

def calculate_precoding_matrix_ZF(selected_channel_matrix):
    sector_number = len(selected_channel_matrix)
    precoding_matrix = []
    for sector_index in range(sector_number):
        # 将这个扇区的信道矩阵拿出来
        sector_channel_matrix = selected_channel_matrix['sector_'+str(sector_index)]['source_channel']
        pseudo_inverse_matrix = np.linalg.pinv(sector_channel_matrix)
        two_norm_distance = np.sqrt(np.sum(abs(pseudo_inverse_matrix) ** 2, 0))
        precoding_matrix.append(pseudo_inverse_matrix/two_norm_distance)
    return precoding_matrix

def calculate_precoding_matrix_MMSE(selected_channel_matrix, noise_power, transmit_power):
    sector_number = len(selected_channel_matrix)
    precoding_matrix = []
    for sector_index in range(sector_number):
        # 将这个扇区的信道矩阵拿出来, 维度为k*16
        sector_channel_matrix = selected_channel_matrix['sector_'+str(sector_index)]['source_channel']
        bs_antennas = sector_channel_matrix.shape[-1]
        scheduling_user = sector_channel_matrix.shape[0]
        # ------------------- 计算复数矩阵的共轭转置 ---------------------
        sector_channel_matrix_conjugate = sector_channel_matrix.T.conj()
        # ------------------- 计算括号里面的 (HH^H + \lambda I_K)H^H, 维度是16 × k
        sector_precoding_matrix = np.linalg.pinv(sector_channel_matrix_conjugate.dot(sector_channel_matrix) + scheduling_user * noise_power/transmit_power * np.eye(bs_antennas)).dot(sector_channel_matrix_conjugate)
        # ----------------- 计算列和，每一列代表一个用户的预编码向量 -------------
        two_norm_distance = np.sqrt(np.sum(abs(sector_precoding_matrix) ** 2, 0))
        precoding_matrix.append(sector_precoding_matrix / two_norm_distance)
    return precoding_matrix

def calculate_single_user_SE(precoding_matrix, selected_channel_matrix, noise_power, transmit_power_list, cyclic_index_vector):
    sector_index = cyclic_index_vector[0]
    current_sector_recieve_signal = selected_channel_matrix['source_channel'].dot(precoding_matrix[sector_index]) 
    # ----------- 上面得到一个k*k的信道矩阵 ------------
    current_sector_recieve_signal_norm = abs(current_sector_recieve_signal) ** 2
    # ----------- 对角线的有效信号拿出来，维度是k*1----------
    efficiency_recieve_signal = np.diag(current_sector_recieve_signal_norm) * transmit_power_list[sector_index]
    intra_sector_interfecence_value = np.sum(current_sector_recieve_signal_norm, 1) * transmit_power_list[sector_index] - efficiency_recieve_signal
    inter_sector_interference_value = 0
    for index, inter_sector_index in enumerate(cyclic_index_vector[1:]):
        current_inter_sector_interference_signal = selected_channel_matrix['interfence_channel'][index].dot(precoding_matrix[inter_sector_index])
        current_inter_sector_interference_signal_value = np.sum(np.abs(current_inter_sector_interference_signal)**2, 1)* transmit_power_list[inter_sector_index]
        inter_sector_interference_value += current_inter_sector_interference_signal_value
    # 计算当前sector下面每一个用户的SINR
    SINR = efficiency_recieve_signal / (intra_sector_interfecence_value +  inter_sector_interference_value + noise_power)
    return np.log(1 + SINR)

def calculate_sector_SE(bool_scheduling_matrix, selected_channel_matrix, precoding_channel_matrix, noise_power, transmit_power_list, sector_number, user_number, cyclic_index_matrix):
    user_instant_SE = np.zeros((sector_number, user_number))
    for sector_index in range(sector_number):
        # 计算一下是第几个扇区，以及是第几个用户
        scheduling_user_instant_SE = calculate_single_user_SE(precoding_channel_matrix, selected_channel_matrix['sector_'+str(sector_index)], noise_power, transmit_power_list, cyclic_index_matrix[sector_index])
        user_instant_SE[sector_index,bool_scheduling_matrix[sector_index,:]] = scheduling_user_instant_SE       
    return user_instant_SE
        
def calculate_instant_reward(channel_matrix, user_scheduling_matrix, noise_power, transmit_power, cyclic_index_matrix):
    # 将这个user_sheduling_matrix变成bool矩阵
    # ---------------- 传入的channel matrix的维度为3*20*3*32 -------------
    # 传入的user_scheduling_matrix是一个维度为3*20的一个矩阵
    bool_scheduling_matrix = user_scheduling_matrix != 0
    scheduling_user_list = np.sum(user_scheduling_matrix, 1)
    transmit_power_list = transmit_power / scheduling_user_list
    sector_number, user_number = bool_scheduling_matrix.shape[0], bool_scheduling_matrix.shape[1]
    # 根据调用矩阵，判断这个调度序列是不是合法的
    user_instant_SE = np.zeros((sector_number, user_number))
    # ---------调度一定是合法的 --------------
    complex_channel_matrix = rebuild_channel_matrix(channel_matrix)
    selected_channel_matrix = select_sub_channel_matrix(complex_channel_matrix, bool_scheduling_matrix, cyclic_index_matrix)
    # sector_power = transmite_power/scheduled_user_number
    precoding_channel_matrix = calculate_precoding_matrix_MMSE(selected_channel_matrix, noise_power, transmit_power)
    # precoding_channel_matrix, unitary_matrix = calculate_precoding_matrix_ZF(selected_channel_matrix)
    user_instant_SE = calculate_sector_SE(bool_scheduling_matrix, selected_channel_matrix, precoding_channel_matrix, noise_power, transmit_power_list, sector_number, user_number, cyclic_index_matrix)
    # 返回的矩阵维度为3*20*1，表示在此次调度过程中，每一个用户的瞬时容量
    return np.expand_dims(user_instant_SE, -1)

