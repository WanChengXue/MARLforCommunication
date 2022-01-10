import numpy as np


def scheduling_is_legel(user_scheduling_matrix, legal_range):
    # 这个legal_range表示每一个扇区调度用户时，最小的调度数目和最大的调度数目
    min_sheduling_number, max_sheduling_number = legal_range
    sector_scheduling_user_sum = np.sum(user_scheduling_matrix, -1)
    legal_flag = True
    for sector_index in range(sector_scheduling_user_sum.shape[0]):
        if sector_scheduling_user_sum[sector_index]< min_sheduling_number or sector_scheduling_user_sum[sector_index] > max_sheduling_number:
            legal_flag = False
    return legal_flag, sector_scheduling_user_sum
    

def rebuild_channel_matrix(channel_matrix):
    bs_antenna_number = channel_matrix.shape[-1] // 2
    real_part = channel_matrix[:,:,:,:bs_antenna_number]
    imag_part = channel_matrix[:,:,:,bs_antenna_number:]
    return real_part + 1j * imag_part

def select_sub_channel_matrix(channel_matrix, bool_user_scheduling_matrix):
    sector_number = bool_user_scheduling_matrix.shape[0]
    # channel_matrix的维度为3*20*3*16
    selected_channel_matrix = []
    for sector_index in range(sector_number):
        selected_channel_matrix.append(channel_matrix[sector_index, bool_user_scheduling_matrix[sector_index]])
    return selected_channel_matrix

def calculate_precoding_matrix_ZF(selected_channel_matrix):
    sector_number = len(selected_channel_matrix)
    precoding_matrix = []
    unitary_matrix = []
    for sector_id in range(sector_number):
        # 将这个扇区的信道矩阵拿出来
        sector_channel_matrix = selected_channel_matrix[sector_id][:, sector_id, :]
        pseudo_inverse_matrix = np.linalg.pinv(sector_channel_matrix)
        F_norm_square = np.linalg.norm(pseudo_inverse_matrix, 'fro') 
        precoding_matrix.append(pseudo_inverse_matrix/F_norm_square)
        unitary_matrix.append(np.eye(sector_channel_matrix.shape[0]))
    return precoding_matrix, unitary_matrix

def calculate_precoding_matrix_MMSE(selected_channel_matrix, noise_power, transmit_power):
    sector_number = len(selected_channel_matrix)
    precoding_matrix = []
    for sector_id in range(sector_number):
        # 将这个扇区的信道矩阵拿出来
        sector_channel_matrix = selected_channel_matrix[sector_id][:, sector_id, :]
        bs_antennas = sector_channel_matrix.shape[-1]
        sector_channel_matrix_conjugate = sector_channel_matrix.T.conj()
        pseudo_inverse_matrix = np.linalg.pinv(sector_channel_matrix_conjugate.dot(sector_channel_matrix) + noise_power/transmit_power[sector_id] * np.eye(bs_antennas)).dot(sector_channel_matrix_conjugate)
        F_norm_square = np.linalg.norm(pseudo_inverse_matrix, 'fro') 
        precoding_matrix.append(pseudo_inverse_matrix / F_norm_square * np.sqrt(transmit_power[sector_id]))
    return precoding_matrix

def calculate_precoding_matrix(selected_channel_matrix):
    sector_number = len(selected_channel_matrix)
    precoding_matrix = []
    unitary_matrix = []
    for sector_id in range(sector_number):
        # 将这个扇区的信道矩阵拿出来
        sector_channel_matrix = selected_channel_matrix[sector_id][:, sector_id, :]
        # 进行svd分解
        u, sigma, v = np.linalg.svd(sector_channel_matrix)
        user_number = sigma.shape[0]
        precoding_matrix.append(v[:user_number,:].T.conj())
        unitary_matrix.append(u.T.conj())
    return precoding_matrix, unitary_matrix


# @ti.func
def calculate_single_user_SE(precoding_matrix, selected_channel_matrix,sector_index, noise_power):
    current_sector_recieve_signal = selected_channel_matrix[sector_index][:,sector_index,:].dot(precoding_matrix[sector_index])
    # 计算intra cell干扰
    intra_recieve_signal_energy = np.abs(current_sector_recieve_signal) ** 2
    efficiency_recieve_signal = np.diag(intra_recieve_signal_energy)
    intra_cell_interfecence_value = np.sum(intra_recieve_signal_energy, 1)
    # 其中useful_signal表示的是分子，dominator表示的是分母
    sector_list = [i for i in range(len(precoding_matrix))]
    sector_list.remove(sector_index)
    inter_sector_interference_value = 0

    for inter_sector_index in sector_list:
        inter_sector_recieve_signal = selected_channel_matrix[sector_index][:,inter_sector_index,:].dot(precoding_matrix[inter_sector_index])
        # 计算inter_cell的干扰
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
        user_instant_SE[sector_index,bool_scheduling_matrix[sector_index]] = scheduling_user_instant_SE       
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
    return np.sum(user_instant_SE)


