# 这个地方采用贪婪策略结合PF调度计算出最优的调度序列
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

class Greedy:
    def __init__(self, channel_matrix, args, priority_array):
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
        self.priority_array = copy.deepcopy(priority_array)
        self.Divide_group()

    def Rebuild_channel(self, channel_matrix):
        # 将channel_matrix重建,变成一个9*20*32*3*3的信道矩阵
        self.Rebuild_channel_matrix = []
        for cell_index in range(self.cell_number):
            activate_channel = channel_matrix[cell_index]
            self.Rebuild_channel_matrix.append(activate_channel[:,:,:,0:self.bs_antenna_number]+ 1j*activate_channel[:,:,:,self.bs_antenna_number:])

    def Divide_group(self):
        # 这个函数是将优先级列表分成二级列表
        rank_level = []
        group_dict = {}
        max_rank = []
        for cell_index in range(self.cell_number):
            cell_priority = copy.deepcopy(self.priority_array[cell_index])
            cell_dict = {}
            cell_level = []
            for user_index, user_priority in enumerate(cell_priority):
                if str(user_priority) in cell_dict.keys():
                    cell_dict[str(user_priority)].append(user_index)
                else:
                    # 如果这个优先级不存在，则创建
                    cell_level.append(user_priority)
                    cell_dict[str(user_priority)] = [user_index]
            max_rank.append(user_priority)
            group_dict[str(cell_index)] = cell_dict
            rank_level.append(cell_level)
        self.group_dict = group_dict
        self.rank_level = rank_level
        self.max_rank = np.max(max_rank)+1

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
        # 如果三个小区的action之后大于等于3,并且有一个小区没有用户被调度到,则直接强制为0
        # zero_schedule_flag = False
        # more_schedule_flag = False
        # for cell_index in range(self.cell_number):
        #     if self.cell_schedule_user_number[cell_index] == 0:
        #         zero_schedule_flag = True
        #     if self.cell_schedule_user_number[cell_index] > 1:
        #         more_schedule_flag = True
        #     # if self.cell_schedule_user_number[cell_index] <= self.min_stream:
        #     #     self.continue_flag = True
        # if zero_schedule_flag and more_schedule_flag:
        #     capacity = 0

        
        return capacity, agent_reward

    def  Greedy_add_rank_user(self):
        # 遍历不同的rank，然后greedy添加进去，先找出最大的rank
        rank_optimal_capacity = 0
        Rank_action = [[0 for i in range(self.total_antennas)] for j in range(self.cell_number)]
        for rank_level in range(self.max_rank):
            cell_waiting_dict = {}
            for cell_index in range(self.cell_number):
                cell_dict = self.group_dict[str(cell_index)]
                if str(rank_level) in cell_dict.keys():
                    cell_waiting_dict[str(cell_index)] = cell_dict[str(rank_level)]
                else:
                    cell_waiting_dict[str(cell_index)] = []
            while True:
                result = self.Greedy_add_rank_new_user(cell_waiting_dict, Rank_action)
                if result is None:
                    break
                else:
                    best_cell_index, best_user_index, best_capacity, best_agent_reward = result
                    Rank_action[best_cell_index][best_user_index] = 1
                    if best_capacity > rank_optimal_capacity:
                        rank_optimal_capacity = best_capacity
                        # 然后将那个用户从cell_wating_dict中排除
                        cell_waiting_dict[str(best_cell_index)].remove(best_user_index)
                        self.priority_array[best_cell_index][best_user_index] += 1
                        rank_optimal_reward = copy.deepcopy(best_agent_reward)
                    else:
                        break
        return copy.deepcopy(Rank_action), rank_optimal_capacity, copy.deepcopy(rank_optimal_reward)
                

    def Greedy_add_rank_new_user(self, rank_dict, rank_last_action):
        # 传入的两个变量，第一个是当前rank的所有小区可以调度的用户字典，第二个是上一个rank调度完成后的动作
        New_capacity_array = [[] for _ in range(self.cell_number)]
        New_user_index = [[] for _ in range(self.cell_number)]
        New_agent_reward = [[] for _ in range(self.cell_number)]

        for cell_index in range(self.cell_number):
            if rank_dict[str(cell_index)]:
                # 如果这个cell 这个rank是用户存在的, 不是一个空列表
                rank_cell_user_list = rank_dict[str(cell_index)]
                for real_user_index in rank_cell_user_list:
                    new_action = copy.deepcopy(rank_last_action)
                    new_action[cell_index][real_user_index] = 1
                    self.Action_reasonable(new_action)
                    self.Select_channel_data(new_action)
                    capacity, agent_reward = self.Calculate_reward(new_action)
                    New_capacity_array[cell_index].append(capacity)
                    New_user_index[cell_index].append(real_user_index)
                    New_agent_reward[cell_index].append(copy.deepcopy(agent_reward))

        Optimal_index = []
        Cell_optimal = []
        # 这个循环是遍历所有cell,找出每一个cell中最优的容量,以及对应的位置
        for cell_index in range(self.cell_number):
            if New_capacity_array[cell_index]:
                cell_optimal_index = np.argmax(New_capacity_array[cell_index])
                cell_optimal_capacity = New_capacity_array[cell_index][cell_optimal_index]
                Optimal_index.append([cell_index, cell_optimal_index])
                Cell_optimal.append(cell_optimal_capacity)
        
        if Optimal_index:
            best_index = np.argmax(np.array(Cell_optimal))
            # best_cell_index表示的是最优小区索引，best_antenna_nomial_index表示的名义上的用户索引
            best_cell_index, best_antenna_nominal_index = Optimal_index[best_index]
            # best_capacity是所有可能中容量最大的值
            best_capacity = New_capacity_array[best_cell_index][best_antenna_nominal_index]
            # best_user_index是实际中最优的用户索引
            best_user_index = New_user_index[best_cell_index][best_antenna_nominal_index]
            # best_agent_reward表示的是最优用户瞬时容量组合
            best_agent_reward = New_agent_reward[best_cell_index][best_antenna_nominal_index]
            # 返回最好的小区索引,最后的用户索引,最好的容量,最好的用户瞬时容量向量
            return [best_cell_index, best_user_index, best_capacity, best_agent_reward]
        else:
            return None            

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
    Real_user_average_reward =  np.zeros((args.cell_number*args.sector_number, args.total_user_antennas))
    Edge_user_capacity = []
    PF_sum_list = []
    for TTI in tqdm(range(args.TTI_length)):
        # 信道矩阵的维度都是user_number * cell*number * bs_antenna_number * TII_number的哦
        TTI_channel = [channel[i][:,:,:,:,TTI] for i in range(args.cell_number*args.sector_number)]
        agent = Greedy(TTI_channel ,args, Rank)
        TTI_best_sequence, TTI_best_capacity, optimal_instant_reward = agent.Greedy_add_rank_user()
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
        Average_user_reward = (1-filter_cofficient) * Average_user_reward + filter_cofficient * np.array(optimal_instant_reward)
        # 更新Rank
        Rank = agent.priority_array
        if np.min(Rank) >0:
            Rank = (np.array(Rank) - 1).tolist()
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
    # cpu_count = multiprocessing.cpu_count() - 10
    # total_TTI_length = args.TTI_length
    # int_part = total_TTI_length // cpu_count
    # res_part = total_TTI_length % cpu_count
    # start_point_set = [0]
    # TTI_length_set = [int_part for i in range(cpu_count)]
    # for i in range(cpu_count):
    #     if i< res_part:
    #         TTI_length_set[i] += 1
    #     start_point_set.append(start_point_set[-1]+TTI_length_set[i])
    
    for i in range(testing_length):
        testing_set.append([args.data_folder + "/" + total_file[i*cell_number+j] for j in range(total_cell)])
    if os.path.exists(args.greedy_folder):
        shutil.rmtree(args.greedy_folder)
    os.mkdir(args.greedy_folder)
    # os.mkdir(args.greedy_folder + '/Distributed_result')
    # 这个testing set是一个3*3的列表,因此要全部丢进去
    start_time = time.time()
    # # # 开进程池
    # pool = Pool(cpu_count)
    # for process_id in range(cpu_count):
    #     pool.apply_async(Single_process_greedy_cell, (testing_set[0],process_id,start_point_set[process_id],TTI_length_set[process_id],args,))
    # pool.close()
    # pool.join()
    # for process_id in range(cpu_count):
    # process_id = 0
    # Single_process_greedy_cell(testing_set[0],process_id,start_point_set[process_id],TTI_length_set[process_id],args)
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
            common_args.greedy_folder = common_args.greedy_folder + user_index + '/' + velocity_index + '/'  + 'Greedy_PF_result/'
            common_args.TTI_length = 200
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

if __name__=='__main__':
    main()

# 想法在于，传入的一个priority array是一个二维列表，长度为9 * 20
# 其中每一个值放置的都是这个用户的优先级，然后第一步操作是根据这个优先级列表将用户划分成几个小小的组
# 第二步是从每一个小小的优先级组内添加用户，然后基于优先级列表进行greedy操作
