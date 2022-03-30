import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from tqdm import tqdm
current_path = os.path.abspath(__file__)
root_path = '/'.join(current_path.split('/')[:-2])
sys.path.append(root_path)
from Utils import create_folder
function_path = os.path.abspath(__file__)
greedy_root_path = '/'.join(function_path.split('/')[:-3]) +'/data_part/Greedy_result/single_cell_scenario_greedy/20_user/30KM/'
rl_root_path = '/'.join(function_path.split('/')[:-2]) + '/Exp/Result/Evaluate/single_cell_max_se/'
save_path = '/'.join(function_path.split('/')[:-2]) + '/Exp/Result/Figure/single_cell_max_se/'
mean_RL_result_list = []
mean_greedy_result_list = []
create_folder(save_path)
for i in tqdm(range(50)):
    for sector_index in range(3):
        greedy_data = np.load(greedy_root_path+'{}sector_{}_se_sum_result.npy'.format(i, sector_index))
        rl_data = np.load(rl_root_path+'{}_sector_{}_se_sum_result.npy'.format(i, sector_index))
        mean_RL_result_list.append(np.mean(rl_data))
        mean_greedy_result_list.append(np.mean(greedy_data))
        plt.figure()
        plt.plot(rl_data)
        plt.plot(greedy_data)
        plt.legend(['RL','Greedy'])
        plt.savefig(save_path+'{}_sector_{}_se_sum.png'.format(i, sector_index))
        plt.close()
plt.figure()
plt.plot(np.array(mean_RL_result_list))
plt.plot(np.array(mean_greedy_result_list))
plt.legend(['RL', 'Greedy'])
plt.savefig(save_path + 'performance_comprison.png')
plt.close()