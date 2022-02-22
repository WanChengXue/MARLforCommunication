import numpy as np
import pathlib
# from Instant_reward_single_cell import calculate_instant_reward
from Instant_Reward import calculate_instant_reward
import time
channel_matrix = np.random.rand(3,20,3,16) +1j*np.random.rand(3,20,3,16)
sector_power = 0.25
legal_range = [3, 16]
user_sheduling_matrix = np.stack([np.random.choice(2,(20,1)) for i in range(3)], 0).squeeze()
cyclic_index_matrix = np.array([[0,1,2], [1,2,0],[2,0,1]])
noise_spectrum_density = 3.1623e-20
subcarrier_numbers = 50
subcarrier_gaps = 5000
system_bandwidth = 1e6
subcarrier_bandwidth = system_bandwidth / subcarrier_numbers - subcarrier_gaps
# noise_power = noise_spectrum_density * subcarrier_bandwidth
start_time = time.time()
calculate_instant_reward(channel_matrix, user_sheduling_matrix, noise_spectrum_density, sector_power, cyclic_index_matrix)
end_time = time.time()
print("计算奖励值，耗费时间为: {}".format(end_time-start_time))