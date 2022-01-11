import numpy as np


def rebuild_channel_matrix(channel_matrix):
    bs_antenna_number = channel_matrix.shape[-1] // 2
    real_part = channel_matrix[:,:,:,:bs_antenna_number]
    imag_part = channel_matrix[:,:,:,bs_antenna_number:]
    return real_part + 1j * imag_part

def select_sub_channel_matrix(channel_matrix, bool_user_scheduling_matrix):
    sector_number = bool_user_scheduling_matrix.shape[0]
    # channel_matrix的维度为3*20*3*16, bool_user_scheduling_matrix的维度喂3*20*1
    selected_channel_matrix = []
    for sector_index in range(sector_number):
        # --------------------- 根据mask向量进行取值 --------------------
        selected_channel_matrix.append(channel_matrix[sector_index, bool_user_scheduling_matrix[sector_index].squeeze(), :, :])
    # --------- 返回的selected channel matrix中，每一个元素都是一个k*3*16的信道矩阵
    return selected_channel_matrix

# def calculate_precoding_matrix_ZF(selected_channel_matrix):
#     sector_number = len(selected_channel_matrix)
#     precoding_matrix = []
#     unitary_matrix = []
#     for sector_id in range(sector_number):
#         # 将这个扇区的信道矩阵拿出来
#         sector_channel_matrix = selected_channel_matrix[sector_id][:, sector_id, :]
#         pseudo_inverse_matrix = np.linalg.pinv(sector_channel_matrix)
#         F_norm_square = np.linalg.norm(pseudo_inverse_matrix, 'fro') 
#         precoding_matrix.append(pseudo_inverse_matrix/F_norm_square)
#         unitary_matrix.append(np.eye(sector_channel_matrix.shape[0]))
#     return precoding_matrix, unitary_matrix

def calculate_precoding_matrix_MMSE(selected_channel_matrix, noise_power, transmit_power):
    sector_number = len(selected_channel_matrix)
    precoding_matrix = []
    for sector_id in range(sector_number):
        # 将这个扇区的信道矩阵拿出来, 维度为k*16
        sector_channel_matrix = selected_channel_matrix[sector_id][:, sector_id, :]
        bs_antennas = sector_channel_matrix.shape[-1]
        # ------------------- 计算复数矩阵的共轭转置 ---------------------
        sector_channel_matrix_conjugate = sector_channel_matrix.T.conj()
        # ------------------- 计算括号里面的 HH^H + \lambda I_K

        pseudo_inverse_matrix = np.linalg.pinv(sector_channel_matrix_conjugate.dot(sector_channel_matrix) + noise_power/transmit_power * np.eye(bs_antennas)).dot(sector_channel_matrix_conjugate)
        F_norm_square = np.linalg.norm(pseudo_inverse_matrix, 'fro') 
        precoding_matrix.append(pseudo_inverse_matrix / F_norm_square * np.sqrt(transmit_power))
    return precoding_matrix

# def calculate_precoding_matrix(selected_channel_matrix):
#     sector_number = len(selected_channel_matrix)
#     precoding_matrix = []
#     unitary_matrix = []
#     for sector_id in range(sector_number):
#         # 将这个扇区的信道矩阵拿出来
#         sector_channel_matrix = selected_channel_matrix[sector_id][:, sector_id, :]
#         # 进行svd分解
#         u, sigma, v = np.linalg.svd(sector_channel_matrix)
#         user_number = sigma.shape[0]
#         precoding_matrix.append(v[:user_number,:].T.conj())
#         unitary_matrix.append(u.T.conj())
#     return precoding_matrix, unitary_matrix


# @ti.func
def calculate_single_user_SE(precoding_matrix, selected_channel_matrix,sector_index, noise_power):
    current_sector_recieve_signal = selected_channel_matrix[sector_index][:, sector_index, :].dot(precoding_matrix[sector_index])
    # 计算intra cell干扰
    intra_recieve_signal_energy = np.abs(current_sector_recieve_signal) ** 2
    # --------- 提取出对角线上的有效能量 ---------------
    efficiency_recieve_signal = np.diag(intra_recieve_signal_energy)
    # --------- 计算干扰部分，其实就是行和减去对角线的值 -------
    intra_cell_interfecence_value = np.sum(intra_recieve_signal_energy, 1) - efficiency_recieve_signal
    # 其中useful_signal表示的是分子，dominator表示的是分母
    sector_list = [i for i in range(len(precoding_matrix))]
    sector_list.remove(sector_index)
    inter_sector_interference_value = 0
    # ------------- 计算小区之间的干扰 ------------------
    for inter_sector_index in sector_list:
        inter_sector_recieve_signal = selected_channel_matrix[sector_index][:,inter_sector_index, :].dot(precoding_matrix[inter_sector_index])
        current_inter_sector_interference_value = np.sum(np.abs(inter_sector_recieve_signal) **2, -1)  
        inter_sector_interference_value += current_inter_sector_interference_value
    # 计算当前sector下面每一个用户的SINR
    SINR = efficiency_recieve_signal / (intra_cell_interfecence_value +  inter_sector_interference_value + noise_power)
    return np.log(1 + SINR)

# @ti.kernel
def calculate_sector_SE(bool_scheduling_matrix, selected_channel_matrix, precoding_channel_matrix, noise_power, sector_number, user_number):
    user_instant_SE = np.zeros((sector_number, user_number))
    for sector_index in range(sector_number):
        # 计算一下是第几个扇区，以及是第几个用户
        scheduling_user_instant_SE = calculate_single_user_SE(precoding_channel_matrix, selected_channel_matrix, sector_index, noise_power)
        user_instant_SE[sector_index,bool_scheduling_matrix[sector_index,:,:].squeeze()] = scheduling_user_instant_SE       
    return user_instant_SE
        
def calculate_instant_reward(channel_matrix, user_scheduling_matrix, noise_power, transmit_power):
    # 将这个user_sheduling_matrix变成bool矩阵
    bool_scheduling_matrix = user_scheduling_matrix != 0
    sector_number, user_number = bool_scheduling_matrix.shape[0], bool_scheduling_matrix.shape[1]
    # 根据调用矩阵，判断这个调度序列是不是合法的
    user_instant_SE = np.zeros((sector_number, user_number))
    # ---------调度一定是合法的 --------------
    complex_channel_matrix = rebuild_channel_matrix(channel_matrix)
    selected_channel_matrix = select_sub_channel_matrix(complex_channel_matrix, bool_scheduling_matrix)
    # sector_power = transmite_power/scheduled_user_number
    precoding_channel_matrix = calculate_precoding_matrix_MMSE(selected_channel_matrix, noise_power, transmit_power)
    # precoding_channel_matrix, unitary_matrix = calculate_precoding_matrix_ZF(selected_channel_matrix)
    user_instant_SE = calculate_sector_SE(bool_scheduling_matrix, selected_channel_matrix, precoding_channel_matrix, noise_power, sector_number, user_number)
    # 返回的矩阵维度为3*20*1，表示在此次调度过程中，每一个用户的瞬时容量
    return np.expand_dims(user_instant_SE, -1)

