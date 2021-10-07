from os import path
import pathlib
import matplotlib.pyplot as plt
import numpy as np
def compare_capacity(user_number, velocity):
    # 读取文件
    greedy_path = pathlib.Path("../data_part/Greedy_result")/user_number/velocity/"Global_greedy_sum_SE.npy"
    # sector_greedy_path = pathlib.Path("../data_part/Greedy_result/20_user/30KM/Sector_greedy_sum_SE.npy")
    GA_path= pathlib.Path("../data_part/Greedy_result")/user_number/velocity/"GA_sum_SE.npy"
    individual_greedy_path = pathlib.Path("../data_part/Greedy_result")/user_number/velocity/"Individual_greedy_sum_SE.npy"
    infer_path = pathlib.Path("./Pointer_network_folder/Max_SE/Sharing_result/")/(user_number+'_'+velocity)/"infer_SE.npy"

    plt.figure()
    plt.plot(np.load(greedy_path))
    # plt.plot(np.load(sector_greedy_path))
    plt.plot(np.load(individual_greedy_path))
    plt.plot(np.load(GA_path).squeeze())
    plt.plot(np.load(infer_path))
    # plt.legend(['Global_greedy', 'Sector_greedy', 'Individual_greedy', 'GA', 'RL'])
    plt.legend(['Global_greedy', 'Individual_greedy', 'GA', 'RL'])
    # plt.legend(['Global_greedy', 'Individual_greedy', 'RL'])
    plt.savefig('./Figure/' + user_number+ '_' + velocity +'_' + 'SE_compare.png')

    # 看一下各个小区的调度用户数目
    greedy_scheduling_path = pathlib.Path("../data_part/Greedy_result")/user_number/velocity/"Global_greedy_shceduling_sequence.npy"
    # sector_scheduling_path = pathlib.Path("../data_part/Greedy_result/20_user/30KM/Sector_greedy_shceduling_sequence.npy")
    individual_scheduling_path = pathlib.Path("../data_part/Greedy_result")/user_number/velocity/"Individual_greedy_shceduling_sequence.npy"
    GA_scheduling_path = pathlib.Path("../data_part/Greedy_result")/user_number/velocity/"GA_shceduling_sequence.npy"
    infer_scheduling_path = pathlib.Path("./Pointer_network_folder/Max_SE/Sharing_result")/(user_number+'_'+velocity)/"infer_sequence.npy"
    
    for sector_index in range(3):
        plt.figure()
        plt.plot(np.sum(np.load(greedy_scheduling_path)[:, sector_index,:], -1))
        # plt.plot(np.sum(np.load(sector_scheduling_path)[:, sector_index, :], -1))
        plt.plot(np.sum(np.load(individual_scheduling_path)[:, sector_index, :], -1))
        plt.plot(np.sum(np.load(GA_scheduling_path)[:, sector_index, :], -1))
        plt.plot(np.sum(np.load(infer_scheduling_path)[:, sector_index, :], -1))
        # plt.legend(['Global_greedy', 'Sector_greedy', 'Individual_greedy' ,'GA', 'RL'])
        plt.legend(['Global_greedy', 'Individual_greedy', 'GA', 'RL'])
        # plt.legend(['Global_greedy', 'Individual_greedy', 'RL'])
        plt.savefig("./Figure/sector_" + user_number+ '_' + velocity +'_' +  str(sector_index) + "_scheduling_number.png")


# user_number = ['20_user','30_user','40_user']
user_number_list = ['20_user', '30_user']
# velocity = ['3KM','30KM','90KM']
velocity_list = ['30KM']
for user_number in user_number_list:
    for velocity in velocity_list:
        compare_capacity(user_number, velocity)