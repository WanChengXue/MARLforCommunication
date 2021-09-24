import numpy as np
import pathlib
from Instant_Reward import calculate_instant_reward
import time
data_path = pathlib.Path("../data_part/preprocess_data/20_user/30KM/traning_data_10_10.npy")
# 读取数据
preprocess_data = np.load(data_path)
# channel_matrix, user_scheduling_matrix, legal_range, noise_power, sector_power
channel_matrix = preprocess_data[:,:,:,:,0]
sector_power = [0.25, 0.25, 0.25]
legal_range = [3, 16]
user_sheduling_matrix = np.random.choice(2,(3,20))
noise_spectrum_density = 3.1623e-20
subcarrier_numbers = 50
subcarrier_gaps = 5000
system_bandwidth = 1e6
subcarrier_bandwidth = system_bandwidth / subcarrier_numbers - subcarrier_gaps
noise_power = noise_spectrum_density * subcarrier_bandwidth
start_time = time.time()
calculate_instant_reward(channel_matrix, user_sheduling_matrix, legal_range, noise_power, sector_power)
end_time = time.time()
print("计算奖励值，耗费时间为: {}".format(end_time-start_time))