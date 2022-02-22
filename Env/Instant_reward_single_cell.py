import numpy as np



def rebuild_channel_matrix(channel_matrix):
    bs_antenna_number = channel_matrix.shape[-1] // 2
    real_part = channel_matrix[:,:bs_antenna_number]
    imag_part = channel_matrix[:,bs_antenna_number:]
    return real_part + 1j * imag_part


def select_sub_channel_matrix(channel_matrix, bool_user_scheduling_matrix):
    sector_number = bool_user_scheduling_matrix.shape[0]
    # channel_matrix的维度为3*20*3*16
    selected_channel_matrix = []
    for sector_index in range(sector_number):
        selected_channel_matrix.append(channel_matrix[bool_user_scheduling_matrix[sector_index]])
    return selected_channel_matrix

def calculate_precoding_matrix_ZF(selected_channel_matrix):
    sector_number = len(selected_channel_matrix)
    precoding_matrix = []
    unitary_matrix = []
    for sector_id in range(sector_number):
        # 将这个扇区的信道矩阵拿出来
        sector_channel_matrix = selected_channel_matrix[sector_id] 
        pseudo_inverse_matrix = np.linalg.pinv(sector_channel_matrix)
        F_norm_square = np.linalg.norm(pseudo_inverse_matrix, 'fro') 
        precoding_matrix.append(pseudo_inverse_matrix/F_norm_square)
        unitary_matrix.append(np.eye(sector_channel_matrix.shape[0]))
    return precoding_matrix, unitary_matrix


def calculate_precoding_matrix(selected_channel_matrix, noise_power, transmit_power):
    sector_number = len(selected_channel_matrix)
    precoding_matrix = []
    for sector_id in range(sector_number):
        # 将这个扇区的信道矩阵拿出来
        sector_channel_matrix = selected_channel_matrix[sector_id]
        bs_antennas = sector_channel_matrix.shape[-1]
        sector_channel_matrix_conjugate = sector_channel_matrix.T.conj()
        pseudo_inverse_matrix = np.linalg.pinv(sector_channel_matrix_conjugate.dot(sector_channel_matrix) + noise_power/transmit_power * np.eye(bs_antennas)).dot(sector_channel_matrix_conjugate)
        F_norm_square = np.linalg.norm(pseudo_inverse_matrix, 'fro') 
        precoding_matrix.append(pseudo_inverse_matrix / F_norm_square * np.sqrt(transmit_power))
    return precoding_matrix


# @ti.func
def calculate_single_user_SE(precoding_matrix, selected_channel_matrix, noise_power):
    current_sector_recieve_signal = selected_channel_matrix.dot(precoding_matrix)
    recieve_signal_energy = np.abs(current_sector_recieve_signal) ** 2
    efficiency_recieve_signal = np.diag(recieve_signal_energy)
    intra_cell_interference = np.sum(recieve_signal_energy, -1) - efficiency_recieve_signal
    SINR = efficiency_recieve_signal / (noise_power + intra_cell_interference)
    return np.log(1 + SINR)

# @ti.kernel
def calculate_sector_SE(bool_scheduling_matrix, selected_channel_matrix, precoding_channel_matrix, noise_power, sector_number, user_number):
    user_instant_SE = np.zeros((sector_number, user_number))
    for sector_index in range(sector_number):
        # 计算一下是第几个扇区，以及是第几个用户
        scheduling_user_instant_SE = calculate_single_user_SE(precoding_channel_matrix[sector_index], selected_channel_matrix[sector_index], noise_power)
        user_instant_SE[sector_index,bool_scheduling_matrix[sector_index]] = scheduling_user_instant_SE
        
    return user_instant_SE
        
def calculate_instant_reward(channel_matrix, user_scheduling_matrix, legal_range, noise_power, transmite_power):
    # 将这个user_sheduling_matrix变成bool矩阵
    bool_scheduling_matrix = user_scheduling_matrix != 0
    sector_number, user_number = bool_scheduling_matrix.shape
    # 根据调用矩阵，判断这个调度序列是不是合法的
    user_instant_SE = np.zeros((sector_number, user_number))
    legal_sheduling_bool_flag, scheduled_user_number = scheduling_is_legel(bool_scheduling_matrix, legal_range)
    if legal_sheduling_bool_flag:
        complex_channel_matrix = rebuild_channel_matrix(channel_matrix)
        selected_channel_matrix = select_sub_channel_matrix(complex_channel_matrix, bool_scheduling_matrix)
        # sector_power = transmite_power/scheduled_user_number
        precoding_channel_matrix = calculate_precoding_matrix(selected_channel_matrix, noise_power, transmite_power)
        # precoding_channel_matrix, unitary_matrix = calculate_precoding_matrix_ZF(selected_channel_matrix)
        user_instant_SE = calculate_sector_SE(bool_scheduling_matrix, selected_channel_matrix, precoding_channel_matrix, noise_power, sector_number, user_number)
    return np.sum(user_instant_SE)


