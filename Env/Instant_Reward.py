from select import select
import numpy as np
class Multi_cell_instant_reward:
    def __init__(self, transmit_power, noise_power, cyclic_index_matrix, sector_nums, user_nums, bs_antenna_number):
        self.transmit_power = transmit_power
        self.noise_power = noise_power
        self.cyclic_index_matrix = cyclic_index_matrix
        self.sector_nums = sector_nums 
        self.user_nums = user_nums
        self.bs_antenna_number = bs_antenna_number

    def rebuild_channel_matrix(self, channel_matrix):
        real_part = channel_matrix[:,:,:,:self.bs_antenna_number]
        imag_part = channel_matrix[:,:,:,self.bs_antenna_number:]
        return real_part + 1j * imag_part


    def select_sub_channel_matrix(self, channel_matrix, bool_user_scheduling_matrix):
        # 目的扇区 × 用户数目 × 源扇区 × 发射天线数
        # channel_matrix的维度为3*20*3*16, bool_user_scheduling_matrix的维度喂3*20*1
        # cyclic_index_matrix表示的是读取顺序，012,120,201
        scheduling_user_list = np.sum(bool_user_scheduling_matrix, 1)
        selected_channel_matrix = {}
        for sector_index in range(self.sector_nums):
            sector_selected_channel_matrix = {}
            # --------------------- 根据mask向量进行取值 --------------------
            if scheduling_user_list[sector_index] == 0:
                # --------- 如果说这个sector没有一个用户被调度，就用None表示source channel -----------
                sector_selected_channel_matrix['source_channel'] = None
            else:
                sector_selected_channel_matrix['source_channel'] = channel_matrix[sector_index, bool_user_scheduling_matrix[sector_index,:], sector_index, :]
            # --------------------- 添加相邻小区的信道数据 ------------------
            sector_selected_channel_matrix['interference_channel'] = []
            for interference_sector in self.cyclic_index_matrix[sector_index,:][1:]:
                # ---------- 如果说相邻这个小区没有用户调度，就使用None --------
                if scheduling_user_list[interference_sector] == 0 or scheduling_user_list[sector_index] == 0:
                    sector_selected_channel_matrix['interference_channel'].append(None)
                else:
                    # ----------- 如果说scheduling_user_list[sector_index]是0，则表示这个小区不调度用户，因此，根本不会有干扰矩阵的 -----------
                    sector_selected_channel_matrix['interference_channel'].append(channel_matrix[sector_index, bool_user_scheduling_matrix[sector_index,:], interference_sector, :])
            selected_channel_matrix['sector_'+str(sector_index)] = sector_selected_channel_matrix
        # --------- 返回的selected channel matrix中，每一个元素都是一个k*3*16的信道矩阵
        return selected_channel_matrix

    def calculate_precoding_matrix_ZF(self, selected_channel_matrix):
        precoding_matrix = []
        for sector_index in range(self.sector_nums):
            # 将这个扇区的信道矩阵拿出来
            sector_channel_matrix = selected_channel_matrix['sector_'+str(sector_index)]['source_channel']
            if sector_channel_matrix is None:
                precoding_matrix.append(None)
            else:
                pseudo_inverse_matrix = np.linalg.pinv(sector_channel_matrix)
                two_norm_distance = np.sqrt(np.sum(abs(pseudo_inverse_matrix) ** 2, 0))
                precoding_matrix.append(pseudo_inverse_matrix/two_norm_distance)
        return precoding_matrix

    def calculate_precoding_matrix_MMSE(self, selected_channel_matrix):
        precoding_matrix = []
        for sector_index in range(self.sector_nums):
            # 将这个扇区的信道矩阵拿出来, 维度为k*16
            sector_channel_matrix = selected_channel_matrix['sector_'+str(sector_index)]['source_channel']
            if sector_channel_matrix is None:
                precoding_matrix.append(None)
            else:
                bs_antennas = sector_channel_matrix.shape[-1]
                scheduling_user = sector_channel_matrix.shape[0]
                # ------------------- 计算复数矩阵的共轭转置 ---------------------
                sector_channel_matrix_conjugate = sector_channel_matrix.T.conj()
                # ------------------- 计算括号里面的 (HH^H + \lambda I_K)H^H, 维度是16 × k
                sector_precoding_matrix = np.linalg.pinv(sector_channel_matrix_conjugate.dot(sector_channel_matrix) + scheduling_user * self.noise_power/self.transmit_power * np.eye(bs_antennas)).dot(sector_channel_matrix_conjugate)
                # ----------------- 计算列和，每一列代表一个用户的预编码向量 -------------
                two_norm_distance = np.sqrt(np.sum(abs(sector_precoding_matrix) ** 2, 0))
                precoding_matrix.append(sector_precoding_matrix / two_norm_distance)
        return precoding_matrix

    def calculate_single_user_SE(self, precoding_matrix, sector_index, selected_channel_matrix, transmit_power_list):
        if selected_channel_matrix['source_channel'] is None:
            # ----------- 如果说这个小区没有调度用户，ok，fine，直接返回None ---------
            return None
        assert precoding_matrix[sector_index] is not None, '------------ 如果这个sector的selected矩阵不是None, precoding矩阵也必然不是None -----------'
        current_sector_recieve_signal = selected_channel_matrix['source_channel'].dot(precoding_matrix[sector_index]) 
        # ----------- 上面得到一个k*k的信道矩阵 ------------
        current_sector_recieve_signal_norm = abs(current_sector_recieve_signal) ** 2
        # ----------- 对角线的有效信号拿出来，维度是k*1----------
        efficiency_recieve_signal = np.diag(current_sector_recieve_signal_norm) * transmit_power_list[sector_index]
        intra_sector_interfecence_value = np.sum(current_sector_recieve_signal_norm, 1) * transmit_power_list[sector_index] - efficiency_recieve_signal
        inter_sector_interference_value = 0
        for index, inter_sector_index in enumerate(self.cyclic_index_matrix[sector_index,1:]):
            # ------------------ 判断干扰矩阵是不是None,如果是None，就跳过这次计算 ---------------
            if selected_channel_matrix['interference_channel'][index] is None:
                assert precoding_matrix[inter_sector_index] is None, '------- 如果干扰小区的信道是None，则表明干扰小区没有用户进行调度，其precoding矩阵必然是None ------'
                continue
            current_inter_sector_interference_signal = selected_channel_matrix['interference_channel'][index].dot(precoding_matrix[inter_sector_index])
            current_inter_sector_interference_signal_value = np.sum(np.abs(current_inter_sector_interference_signal)**2, 1)* transmit_power_list[inter_sector_index]
            inter_sector_interference_value += current_inter_sector_interference_signal_value
        # 计算当前sector下面每一个用户的SINR
        SINR = efficiency_recieve_signal / (intra_sector_interfecence_value +  inter_sector_interference_value + self.noise_power)
        return np.log(1 + SINR)

    def calculate_sector_SE(self, bool_scheduling_matrix, selected_channel_matrix, precoding_channel_matrix, transmit_power_list):
        user_instant_SE = np.zeros((self.sector_nums, self.user_nums))
        for sector_index in range(self.sector_nums):
            # 计算一下是第几个扇区，以及是第几个用户
            scheduling_user_instant_SE = self.calculate_single_user_SE(precoding_channel_matrix, sector_index, selected_channel_matrix['sector_'+str(sector_index)], transmit_power_list)
            if scheduling_user_instant_SE is not None:
                user_instant_SE[sector_index,bool_scheduling_matrix[sector_index,:]] = scheduling_user_instant_SE       
        return user_instant_SE
            
    def calculate_instant_reward(self, channel_matrix, user_scheduling_matrix):
        # 将这个user_sheduling_matrix变成bool矩阵
        # ---------------- 传入的channel matrix的维度为3*20*3*32 -------------
        # 传入的user_scheduling_matrix是一个维度为3*20的一个矩阵
        bool_scheduling_matrix = user_scheduling_matrix != 0
        scheduling_user_list = np.sum(user_scheduling_matrix, 1)
        # ----------- 如果有的cell一个用户没有进行调度，则发送功率就给0就好了 --------
        transmit_power_list = []
        for cell_index in range(self.sector_nums):
            if scheduling_user_list[cell_index] == 0:
                transmit_power_list.append(0)
            else:
                transmit_power_list.append(self.transmit_power/scheduling_user_list[cell_index])    
        # 根据调用矩阵，判断这个调度序列是不是合法的
        user_instant_SE = np.zeros((self.sector_nums, self.user_nums))
        # ---------调度一定是合法的 --------------
        complex_channel_matrix = self.rebuild_channel_matrix(channel_matrix)
        selected_channel_matrix = self.select_sub_channel_matrix(complex_channel_matrix, bool_scheduling_matrix)
        # sector_power = transmite_power/scheduled_user_number
        precoding_channel_matrix = self.calculate_precoding_matrix_MMSE(selected_channel_matrix)
        # precoding_channel_matrix, unitary_matrix = calculate_precoding_matrix_ZF(selected_channel_matrix)
        user_instant_SE = self.calculate_sector_SE(bool_scheduling_matrix, selected_channel_matrix, precoding_channel_matrix, transmit_power_list)
        # 返回的矩阵维度为3*20*1，表示在此次调度过程中，每一个用户的瞬时容量
        return np.expand_dims(user_instant_SE, -1)

