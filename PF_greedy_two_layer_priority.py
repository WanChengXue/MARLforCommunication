# 这个地方采用贪婪策略结合PF调度计算出最优的调度序列
import enum
import numpy as np
from numpy.lib.function_base import select
from tqdm import tqdm
import os 
import shutil
import multiprocessing
from multiprocessing import Pool
import copy
import json
from ast import literal_eval
import time
# 这个地方做法是 最开始初始化所有的小区优先级为0
# 然后遍历所有的小区
# 遍历所有小区中的用户，根据优先级进项添加
# 更新小区的优先级，更新所有小区中用户的优先级
class Greedy:
    def __init__(self, channel_matrix, args, priority_user, priority_cell):
        self.args=args
        self.antenna_number = self.args.user_antennas
        self.bs_antenna_number = self.args.bs_antennas
        self.total_antennas = self.args.total_user_antennas
        self.transmit_power = self.args.transmit_power 
        self.noise_spectrum_density = self.args.noise_spectrum_density
        self.system_bandwidth = self.args.system_bandwidth
        self.subcarriers_numbers = self.args.subcarrier_numbers
        self.subcarrier_gaps = self.args.subcarrier_gaps
        # 计算单个载波的频带宽度
        self.subcarrier_bandwidth = self.system_bandwidth / self.subcarriers_numbers - self.subcarrier_gaps
        self.noise_power = self.noise_spectrum_density * self.subcarrier_bandwidth
        self.single_cell_number = self.args.cell_number
        self.sector_number = self.args.sector_number
        self.cell_number = self.single_cell_number * self.sector_number
        # 传入的矩阵不再是一个了,而是变成了九个
        self.Rebuild_channel(channel_matrix)
        self.continue_flag = False
        self.min_stream = args.min_stream
        self.priority_user = copy.deepcopy(priority_user)
        self.priority_cell = copy.deepcopy(priority_cell)
        self.Divide_group()

    def Rebuild_channel(self, channel_matrix):
        # 将channel_matrix重建,变成一个9*20*32*3*3的信道矩阵
        self.Rebuild_channel_matrix = []
        for cell_index in range(self.cell_number):
            activate_channel = channel_matrix[cell_index]
            self.Rebuild_channel_matrix.append(activate_channel[:,:,:,0:self.bs_antenna_number]+ 1j*activate_channel[:,:,:,self.bs_antenna_number:])

    def Divide_group(self):
        # 这个函数是将优先级列表分成二级列表
        group_dict = {}
        max_rank = []
        cell_priority_dict = {}
        local_priority_cell = copy.deepcopy(self.priority_cell)
        for cell_index in range(self.cell_number):
            # 第一步对小区的进行切分，在一个rank里面，切分出来
            if str(local_priority_cell[cell_index]) in cell_priority_dict.keys():
                # 如果说有一个优先级存在这个优先级字典, 就直接添加新的小区进去
                cell_priority_dict[str(local_priority_cell[cell_index])].append(cell_index)
            else:
                cell_priority_dict[str(local_priority_cell[cell_index])] = [cell_index]
        
        # 第二步，直接遍历所有的小区，然后就直接划分成若干个 分离的列表即可
        for cell_index in range(self.cell_number):
            cell_user_priority = copy.deepcopy(self.priority_user[cell_index])
            cell_dict = {}
            for user_index, user_priority in enumerate(cell_user_priority):
                if str(user_priority) in cell_dict.keys():
                    cell_dict[str(user_priority)].append(user_index)
                else:
                    # 如果这个优先级不存在，则创建
                    cell_dict[str(user_priority)] = [user_index]
            max_rank.append(np.max(cell_user_priority))
            group_dict[str(cell_index)] = cell_dict
        # 此处三个变量，分别表示的是所有用户分组得到的group dict，在user rank矩阵中最大的rank，以及所有小区基于优先级划分的rank
        self.group_dict = group_dict
        self.max_rank = np.max(max_rank)+1
        self.cell_priority_dict = cell_priority_dict
        
    def Calculate_precoding_matrix(self):
        # This function is used to calculate precoding matrix. If action of current cell is satisfied stream schedule rule
        # then this cell will have precoding matrix, otherwise, the precoding matrix is setted as None
        if self.is_reasonable:
            precoding_matrix = []
            for cell_index in range(self.cell_number):
                sector_id = cell_index % self.sector_number
                cell_id = cell_index // self.sector_number
                if self.cell_schedule_user_number[cell_index] != 0:
                    cell_channel_matrix = self.select_data[cell_index][:, cell_id, sector_id, :]
                    pseudo_inverse = np.linalg.pinv(cell_channel_matrix)
                    cell_norm = np.linalg.norm(pseudo_inverse, 2 ,0)
                    cell_precoding_matrix = pseudo_inverse / cell_norm
                    precoding_matrix.append(cell_precoding_matrix)
                else:
                    precoding_matrix.append(None)
        else:
            precoding_matrix = [None for _ in range(self.cell_number)]
        self.precoding_matrix = precoding_matrix

    def Select_channel_data(self, action):  
        if self.is_reasonable:
            selected_channel = []
            for cell_index in range(self.cell_number):
                cell_action = action[cell_index]
                selected_channel.append(self.Rebuild_channel_matrix[cell_index][np.array(cell_action).astype(bool),:,:,:])
        else:
            selected_channel = [None for _ in range(self.cell_number)]
        self.select_data = selected_channel

    def Action_reasonable(self, action):
        # Define a list, which is used to decide whether is reasonable of arbitrary cell action
        is_reasonable = True
        cell_schedule_user_number = []
        power = []
        # 此处需要遍历所有cell哦
        for cell_index in range(self.cell_number):
            cell_action = action[cell_index]
            schedule_number = np.sum(cell_action)
            cell_schedule_user_number.append(schedule_number)
            if schedule_number == 0:
                power.append(0)
            else:
                power.append(self.transmit_power/schedule_number)
            if np.sum(cell_action)>self.args.max_stream:
                is_reasonable =False
        self.is_reasonable = is_reasonable
        self.cell_schedule_user_number = cell_schedule_user_number
        self.power = power

    def Calculate_user_sum_rate(self, action):
        # users_sum_rate是一个二维列表,数目与扇区的数目是一样的
        users_sum_rate = [[] for _ in range(self.cell_number)]
        # schedule_user_number表示的是每一个扇区中调度的用户数目
        schedule_user_number = copy.deepcopy(self.cell_schedule_user_number)
        # schedule_user_set是一个二维列表,第一个维度和扇区数目是一样的,第二个维度是给调度的用户进行重新记上索引
        schedule_user_set =  [[i for i in range(schedule_user_number[cell_index])]for cell_index in range(self.cell_number)]
        # 对所有的扇区进行标号
        cell_index_list = [i for i in range(self.cell_number)]
        if self.is_reasonable:
            for cell_index in range(self.cell_number):
                cell_action = action[cell_index]
                # 取出第i个小区第j个sector的信道数据
                sector_id = cell_index % self.sector_number
                cell_id = cell_index // self.sector_number
                # 两个变量，分别表示的cell的索引以及sector的索引
                cell_select_data = self.select_data[cell_index]
                cell_precoding_matrix = self.precoding_matrix[cell_index]
                cell_schedule_user_set = schedule_user_set[cell_index]
                cell_power = self.power[cell_index]
                # 如果这个cell什么用户都没有调度,则其users_sum_rate就是一个全0的向量
                # 分成四部分计算，分别是扇区内部的有效信号，intra-sector interference， inter-sector interference , inter-cell interference
                if schedule_user_number[cell_index] == 0:
                    # 如果当前这个扇区什么用户都没有调度,则所有用户的SE都给0
                    for user_index in range(self.total_antennas):
                        users_sum_rate[cell_index].append(0)
                else:
                    # 这个count表示的是调度用户的索引
                    count = 0
                    for user in range(self.total_antennas):
                        # traverse all actions, if action has selected by the policy net, SINR will be calculated, otherwise, directly add zero
                        if cell_action[user] == 1:
                            # 先把分子部分拿出来
                            target_sector_channel = cell_select_data[:,cell_id, sector_id, :]
                            target_user_sector_channel = target_sector_channel[count,:].reshape(1, self.bs_antenna_number)
                            signal_power = np.linalg.norm(target_user_sector_channel.dot(cell_precoding_matrix[:, count])) **2 * cell_power
                            # 计算intra-sector之间的干扰
                            intra_sector_interference = []
                            # 将当前用户移出去
                            intra_sector_schedule_index = copy.deepcopy(cell_schedule_user_set)
                            intra_sector_schedule_index.remove(count)
                            # 如果这个扇区只是调度了一个用户,则intra-sector干扰就是0
                            if len(intra_sector_schedule_index) == 0:
                                intra_sector_interference.append(0)
                            else:
                                # 如果这个扇区调度了其余的用户,计算扇区内部的干扰
                                for other_user_index in intra_sector_schedule_index:
                                    user_intra_sector_interference = np.linalg.norm(target_user_sector_channel.dot(cell_precoding_matrix[:,other_user_index])) ** 2 * cell_power
                                    intra_sector_interference.append(user_intra_sector_interference)
                            # 计算inter_sector之间的干扰
                            inter_sector_interference = []
                            inter_sector_index = copy.deepcopy(cell_index_list)
                            inter_sector_index.remove(cell_index)
                            for inter_sector in inter_sector_index:
                                # 首先判断这个inter-cell的小区是不是有用户调度了
                                if self.cell_schedule_user_number[inter_sector] == 0:
                                    inter_sector_interference.append(0)
                                else:
                                    # 将这个小区的发射功率拿出来
                                    inter_sector_power = self.power[inter_sector]
                                    # 先拿出干扰小区的预编码矩阵来
                                    inter_sector_precoding_matrix = self.precoding_matrix[inter_sector]
                                    # 将inter_sector变成小区,扇区索引
                                    inter_cell_id = inter_sector // self.sector_number
                                    inter_sector_id = inter_sector % self.sector_number
                                    # 将其余扇区到本扇区的信道向量拿出来
                                    inter_sector_channel = cell_select_data[:, inter_cell_id, inter_sector_id, :]
                                    inter_user_sector_channel = inter_sector_channel[count,:].reshape(1, self.bs_antenna_number)
                                    sector_interference = np.linalg.norm(inter_user_sector_channel.dot(inter_sector_precoding_matrix),2)**2
                                    inter_sector_interference.append(inter_sector_power*sector_interference)
                            # 计算SINR
                            SINR = signal_power / (self.noise_power + np.sum(intra_sector_interference) + np.sum(inter_sector_interference))
                            # 计算这个用户的SE
                            SE = np.log(1+SINR)
                            users_sum_rate[cell_index].append(SE)
                            count += 1
                        else:
                            users_sum_rate[cell_index].append(0)
        else:
            # 这个表示的当前小区没有数据进行发送，因此也就直接将所有用户的instant reward设置为0
            for cell_index in range(self.cell_number):
                for user in range(self.total_antennas):
                    users_sum_rate[cell_index].append(0)
        return users_sum_rate

    def Calculate_reward(self, action):
        # Traversal all cell, and calculate the instant rate of individual user
        # users_instant_reward = []
        # First, calculate precodin matrix
        self.Calculate_precoding_matrix()
        reward = self.Calculate_user_sum_rate(action)
        # 计算capacity
        agent_reward = reward
        capacity = np.sum(reward)
        
        return capacity, agent_reward
    
    def Greedy_add_rank(self):
        rank_optimal_capacity = 0
        Rank_action = [[0 for i in range(self.total_antennas)] for j in range(self.cell_number)]
        # 这个函数调用一次就将一个rank的小区加入进去, 第一步就是将最低的rank拿出来
        keys_of_cell_priority = sorted(list(self.cell_priority_dict.keys()))
        for cell_rank in keys_of_cell_priority:
            avaliable_cell_list = self.cell_priority_dict[cell_rank]
            # 这个地方是将一个列表扔过, 然后在给定优先级的小区中进行搜索，返回最优的调度结果(可能是一个字典),以及最有奖励
            optimal_cell_capacity, optimal_cell_reward, Rank_action, schedule_dict = self.Greedy_add_rank_cell(avaliable_cell_list, Rank_action)
            if rank_optimal_capacity < optimal_cell_capacity:
                rank_optimal_capacity = optimal_cell_capacity
                rank_optimal_user_reawrd = copy.deepcopy(optimal_cell_reward)
                rank_optimal_action = copy.deepcopy(Rank_action)
                for cell_index_str in schedule_dict.keys():
                    schedule_list = schedule_dict[cell_index_str]
                    real_cell_index = int(cell_index_str)
                    self.priority_cell[real_cell_index] += 1
                    for user_index in schedule_list:
                        self.priority_user[real_cell_index][user_index] += 1
            else:
                break
        return rank_optimal_action, rank_optimal_capacity, rank_optimal_user_reawrd

        # 如果所有小区，或者是所有用户的优先级超过0，则所有值减去1


    def  Greedy_add_rank_cell(self, avaliable_list, last_action):
        # 传入了两个列表，其中一个是给定的rank中小区索引构成的列表，另外一个是上一次基于cell选择时，得到的动作
        schedule_dict = {}
        rank_optimal_capacity = 0
        Rank_action = copy.deepcopy(last_action)
        active_avaliable_list = copy.deepcopy(avaliable_list)
        # 这里有五个列表，分别表示的是不停添加的时候，最优的小区容量，对应的小区索引，小区调度序列，小区的动作
        while True:
            New_capacity_cell = []
            New_cell_index = []
            New_cell_reward = []
            New_cell_schedule = []
            New_cell_action = []
            if active_avaliable_list:
                for cell_index in active_avaliable_list:
                    # 应该实现，传入一个小区的索引，以及上一次的
                    last_action_copy, cell_shedule_user_list, Optimal_reward, Optimal_user_reward = self.Greedy_add_rank_new_cell(Rank_action, cell_index)
                    New_capacity_cell.append(Optimal_reward)
                    New_cell_index.append(cell_index)
                    New_cell_schedule.append(copy.deepcopy(cell_shedule_user_list))
                    New_cell_reward.append(copy.deepcopy(Optimal_user_reward))
                    New_cell_action.append(copy.deepcopy(last_action_copy))
                
                # 遍历了一遍小区，找到最优的小区
                optimal_cell_index = np.argmax(New_capacity_cell)
                optimal_cell_capacity = New_capacity_cell[optimal_cell_index]
                optimal_cell_scheduling = New_cell_schedule[optimal_cell_index]
                optimal_cell_reward = New_cell_reward[optimal_cell_index]
                optimal_cell_action = New_cell_action[optimal_cell_index]
                optimal_real_cell_index = New_cell_index[optimal_cell_index]

                if rank_optimal_capacity < optimal_cell_capacity:
                    rank_optimal_capacity = optimal_cell_capacity
                    Rank_action = copy.deepcopy(optimal_cell_action)
                    schedule_dict[str(optimal_real_cell_index)] = copy.deepcopy(optimal_cell_scheduling)
                    # 移除选择了的小区
                    active_avaliable_list.remove(optimal_real_cell_index)
                else:
                    break
            else:
                break
        return optimal_cell_capacity, optimal_cell_reward, Rank_action, schedule_dict

    def Greedy_add_rank_new_cell(self, last_action, cell_index):
        last_action_copy = copy.deepcopy(last_action)
        cell_shedule_user_list = []
        # 传入了两个列表，其中一个是上一次决策之后得到的用户调度列表，另外一个是将要调度小区的索引
        single_cell_priority_dcit = self.group_dict[str(cell_index)]
        # 获得了给定小区索引的优先级序列，然后根据这个序列进行调度
        Optimal_reward = 0
        for priority_index in range(self.max_rank):
            if str(priority_index) in single_cell_priority_dcit.keys():
                # 如果这个rank中的用户是存在的，则对这个rank中的用户添加
                rank_cell_user_list = copy.deepcopy(single_cell_priority_dcit[str(priority_index)])
                while True:
                    New_capacity_array = []
                    New_user_index = []
                    New_agent_reward = []
                    if rank_cell_user_list:
                        # 由于每次循环都要剔除一个用户，所以这个列表会不断的缩短 变成空列表
                        for real_user_index in rank_cell_user_list:
                            new_action = copy.deepcopy(last_action_copy)
                            new_action[cell_index][real_user_index] = 1
                            self.Action_reasonable(new_action)
                            self.Select_channel_data(new_action)
                            capacity, agent_reward = self.Calculate_reward(new_action)
                            New_capacity_array.append(capacity)
                            New_agent_reward.append(copy.deepcopy(agent_reward))
                            New_user_index.append(real_user_index)
                        # 遍历一次找出最优的用户
                        best_capacity_index = np.argmax(New_capacity_array)
                        best_capacity = New_capacity_array[best_capacity_index]
                        best_user = New_user_index[best_capacity_index]
                        best_user_capacity = New_agent_reward[best_capacity_index]

                        if best_capacity < Optimal_reward:
                            break
                        else:
                            Optimal_reward = best_capacity
                            Optimal_user = best_user
                            Optimal_user_reward = copy.deepcopy(best_user_capacity)
                            # 更新动作
                            last_action_copy[cell_index][Optimal_user] = 1
                            # 移除候选用户
                            rank_cell_user_list.remove(Optimal_user)
                            cell_shedule_user_list.append(Optimal_user)
                    else:
                        break
        return last_action_copy, cell_shedule_user_list, Optimal_reward, Optimal_user_reward

def Calculate_init_reward(args, channel):
    noise_spectrum_density = args.noise_spectrum_density
    system_bandwidth = args.system_bandwidth
    subcarriers_numbers =  args.subcarrier_numbers
    subcarrier_gaps = args.subcarrier_gaps
    # 计算单个载波的频带宽度
    subcarrier_bandwidth = system_bandwidth / subcarriers_numbers - subcarrier_gaps
    noise_power = noise_spectrum_density * subcarrier_bandwidth
    Init_user_average_reward = np.zeros((args.n_agents, args.total_user_antennas)) 
    Rebuild_channel_matrix = []
    for cell_index in range(args.n_agents):
        activate_channel = channel[cell_index]
        Rebuild_channel_matrix.append(activate_channel[:,:,:,0:args.bs_antennas]+ 1j*activate_channel[:,:,:,args.bs_antennas:])

    for cell_index in range(args.n_agents):
        # 将用户的信道拿出来
        cell_id = cell_index // args.sector_number
        sector_id = cell_index % args.sector_number
        user_channel = Rebuild_channel_matrix[cell_index][:,cell_id, sector_id, :]
        # 遍历所有的用户即可
        for user_index in range(args.total_user_antennas):
            single_user_channel = user_channel[user_index,:].reshape(1,args.bs_antennas)
            sum_rate = np.log(1+np.linalg.norm(single_user_channel)**2 * args.transmit_power/noise_power)
            Init_user_average_reward[cell_index, user_index] = sum_rate
    return Init_user_average_reward

def Greedy_cell(channel_file, args):
    best_sequence = {}
    best_capacity = []
    best_total_capacity = []
    filter_cofficient = 1/args.TTI_length
    # 字符串处理，得到小区数据的名称, 由于这是三个小区在一起,因此将第一个小区的名字拿出来哦
    random_seed_cell = channel_file[0].split('/')[-1].split('.')[0][:-2]
    # 文件的保存路径
    Instant_capacity_path = args.greedy_folder +  'Instant_capacity_PF_' + random_seed_cell + '.npy'
    Action_path = args.greedy_folder + 'Action_PF_' + random_seed_cell  + '.json'
    Sum_capacity_path = args.greedy_folder +  'Sum_capacity_PF_' +  random_seed_cell + '.npy'
    Edge_user_capacity_path = args.greedy_folder + 'Edge_capacity_PF_' + random_seed_cell + '.npy'
    PF_sum_path = args.greedy_folder + 'PF_sum_' + random_seed_cell + '.npy'
    channel = [np.load(channel_file[i]) for i in range(args.cell_number*args.sector_number)]
    Rank = [[0 for user_index in range(args.total_user_antennas)] for cell_index in range(args.cell_number*args.sector_number)]
    Init_channel = [channel[i][:,:,:,:,0] for i in range(args.cell_number*args.sector_number)]
    Average_user_reward = Calculate_init_reward(args, Init_channel)
    # 根据这个计算一下所有小区的初始rank
    Init_cell_reward = np.sum(Average_user_reward,-1)
    Init_cell_priority = list(reversed(np.argsort(Init_cell_reward)))
    Cell_priority = [0 for _ in range(args.cell_number * args.sector_number)]
    for priority_index, cell_index in enumerate(Init_cell_priority):
        Cell_priority[cell_index] = priority_index

    # 对这个Init矩阵进行排序，然后获取优先级
    Real_user_average_reward =  np.zeros((args.cell_number*args.sector_number, args.total_user_antennas))
    Edge_user_capacity = []
    PF_sum_list = []
    for TTI in tqdm(range(args.TTI_length)):
        # 信道矩阵的维度都是user_number * cell*number * bs_antenna_number * TII_number的哦
        TTI_channel = [channel[i][:,:,:,:,TTI] for i in range(args.cell_number*args.sector_number)]
        agent = Greedy(TTI_channel ,args, Rank, Cell_priority)
        TTI_best_sequence, TTI_best_capacity, optimal_instant_reward = agent.Greedy_add_rank()
        best_capacity.append(copy.deepcopy(optimal_instant_reward))
        # 定义最优的sequence
        cell_best_sequence_dict = {}
        for cell_index in range(args.cell_number*args.sector_number):
            cell_best_sequence_dict[str(cell_index)] = str(TTI_best_sequence[cell_index])
        best_sequence[str(TTI)] = cell_best_sequence_dict
        Real_user_average_reward = (1-filter_cofficient) * Real_user_average_reward + filter_cofficient * np.array(optimal_instant_reward)
        Edge_user_capacity.append(np.mean(np.min(Real_user_average_reward,0)))
        PF_sum = np.sum(np.array(optimal_instant_reward)/Average_user_reward)
        PF_sum_list.append(PF_sum)
        best_total_capacity.append(TTI_best_capacity)
        # print(Rank)
        # print(Cell_priority)
        # print(TTI_best_sequence)
        Average_user_reward = (1-filter_cofficient) * Average_user_reward + filter_cofficient * np.array(optimal_instant_reward)
        # 更新Rank, cell_priority
        Cell_priority = agent.priority_cell
        Rank = agent.priority_user
        if np.min(Cell_priority) > 0:
            Cell_priority = (np.array(Cell_priority)-1).tolist()
        for cell_index in range(args.cell_number * args.sector_number):
            if np.min(Rank[cell_index]) > 0:
                Rank[cell_index] = (np.array(Rank[cell_index])-1).tolist()
    json_str = json.dumps(best_sequence)
    with open(Action_path, 'w') as json_file:
        json_file.write(json_str)

    np.save(Instant_capacity_path, np.array(best_capacity))
    np.save(Sum_capacity_path, np.array(best_total_capacity))
    np.save(Edge_user_capacity_path, np.array(Edge_user_capacity))
    np.save(PF_sum_path, np.array(PF_sum_list))

def Greedy_solver(args):
    cell_number = args.cell_number
    sector_number = args.sector_number
    total_cell = cell_number * sector_number
    testing_set = []
    total_file = sorted(os.listdir(args.data_folder))
    testing_length = int(len(total_file)/total_cell)
    
    for i in range(testing_length):
        testing_set.append([args.data_folder + "/" + total_file[i*cell_number+j] for j in range(total_cell)])
    if os.path.exists(args.greedy_folder):
        shutil.rmtree(args.greedy_folder)
    os.mkdir(args.greedy_folder)
    # os.mkdir(args.greedy_folder + '/Distributed_result')
    # 这个testing set是一个3*3的列表,因此要全部丢进去
    start_time = time.time()
    Greedy_cell(testing_set[0], args)
    end_time = time.time()
    continue_time = end_time - start_time
    print('User number: %d, velocity: %d ,Running time: %s Seconds'%(args.user_numbers, args.user_velocity, continue_time))
    # Greedy_cell(testing_set[0], args)
    # PF_greedy_cell(testing_set[0], args)

def main():
    from arguments import get_common_args, get_agent_args, get_MADDPG_args
    user_number = ['10_user','20_user','30_user','40_user']
    velocity = ['3KM','30KM','90KM']
    args_list = []
    for user_index in user_number:
        for velocity_index in velocity:
            common_args = get_common_args()
            common_args.data_folder = common_args.data_folder + user_index +'/' + velocity_index
            common_args.greedy_folder = common_args.greedy_folder + user_index + '/' + velocity_index + '/'  + 'Greedy_two_layer_PF_result_new/'
            common_args.TTI_length = 300
            common_args.user_numbers = int(user_index.split('_')[0])
            common_args.user_velocity = int(velocity_index.split('K')[0])
            agent_args = get_agent_args(common_args)
            args = get_MADDPG_args(agent_args)
            args_list.append(args)
    testing_length = len(args_list)
    pool = Pool(testing_length)
    # for process_id in range(cpu_count):
    #     pool.apply_async(Single_process_greedy_cell, (testing_set[0],process_id,start_point_set[process_id],TTI_length_set[process_id],args,))
    
    for i in range(testing_length):
        pool.apply_async(Greedy_solver, (args_list[i],))
    pool.close()
    pool.join()
    # Greedy_solver(args_list[11])
        # print(args_list[i].user_numbers, args_list[i].user_velocity)
    Greedy_solver(args_list[0])
if __name__=='__main__':
    main()

