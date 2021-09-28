from functools import total_ordering
# from sko.GA import GA
# 这个地方采用贪婪策略结合PF调度计算出最优的调度序列
from matplotlib.pyplot import axis
import numpy as np
from tqdm import tqdm
import os 
import shutil
from multiprocessing import Pool
import copy
import json
import time
import Env.Instant_Reward
from Env.Instant_Reward import calculate_instant_reward
from Tool.arguments import get_common_args 

import pathlib
import geatpy as ea

class Greedy(ea.Problem):
    def __init__(self, args, channel_matrix):
        self.name = 'Scheduling_problem'
        self.args=args
        self.sector_number = self.args.sector_number
        self.cell_number = self.args.cell_number
        self.agent_number = self.cell_number * self.sector_number
        self.bs_antenna_number = self.args.bs_antennas
        self.user_numbers = self.args.user_numbers
        self.transmit_power = [self.args.transmit_power] * self.agent_number
        self.noise_power = self.args.noise_spectrum_density
        # self.system_bandwidth = self.args.system_bandwidth
        # self.subcarriers_numbers = self.args.subcarrier_numbers
        # self.subcarrier_gaps = self.args.subcarrier_gaps
        # # 计算单个载波的频带宽度
        # self.subcarrier_bandwidth = self.system_bandwidth / self.subcarriers_numbers - self.subcarrier_gaps
        # self.noise_power = self.noise_spectrum_density * self.subcarrier_bandwidth
        self.legal_range = [self.args.min_stream, self.args.max_stream]
        self.transmit_power = [self.args.transmit_power] * self.agent_number
        self.channel_matrix = channel_matrix
        self.Dim = self.sector_number * self.user_numbers
        self.maxormins = [-1]
        self.varTypes = [1] * self.Dim
        self.lb = [0] * self.Dim
        self.ub = [1] * self.Dim
        self.lbin = [1] * self.Dim
        self.ubin = [1] * self.Dim
        ea.Problem.__init__(self, self.name, 1, self.maxormins, self.Dim, self.varTypes, self.lb, self.ub, self.lbin, self.ubin)

    def aimFunc(self, pop):
        Vars = np.array(pop.Phen, dtype=int).reshape(-1, self.sector_number, self.user_numbers)
        constraints = []
        for i in range(self.sector_number):
            constraints.append(np.sum(Vars[:,i,:], 1)[:,np.newaxis] - self.bs_antenna_number)

        pop.CV = np.hstack(constraints)
        sample_number = Vars.shape[0]
        sample_value = []
        for index in range(sample_number):
            sample_value.append(calculate_instant_reward(self.channel_matrix, Vars[index, :, :], self.legal_range, self.noise_power, self.transmit_power))
        pop.ObjV = np.array(sample_value)[:,np.newaxis]
        # print(max(pop.ObjV))


def simulation(args):
    # 这个函数用来遍历一下所有的TTI测试信道数据，包括了将instant SE和调度序列进行存储两种功能
    file_path = args.testing_path
    channel_data = np.load(file_path)
    TTI_length = channel_data.shape[-1]
    scheduling_sequence = []
    SE = []
    for TTI in tqdm(range(TTI_length)):
        problem = Greedy(args, channel_data[:,:,:,:,TTI])
        Encoding = 'RI'  # 编码方式
        NIND = 50  # 种群规模
        Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # 创建区域描述器
        population = ea.Population(Encoding, Field, NIND)  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
        """================================算法参数设置============================="""
        myAlgorithm = ea.soea_DE_best_1_L_templet(problem, population)  # 实例化一个算法模板对象
        myAlgorithm.MAXGEN = 500  # 最大进化代数
        myAlgorithm.mutOper.F = 0.5  # 差分进化中的参数F
        myAlgorithm.recOper.XOVR = 0.7  # 重组概率
        # myAlgorithm.trappedValue = 1e-6  # “进化停滞”判断阈值
        # myAlgorithm.maxTrappedCount = 10  # 进化停滞计数器最大上限值，如果连续maxTrappedCount代被判定进化陷入停滞，则终止进化
        myAlgorithm.logTras = 0 # 设置每隔多少代记录日志，若设置成0则表示不记录日志
        myAlgorithm.verbose = False  # 设置是否打印输出日志信息
        myAlgorithm.drawing = 0  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
        """===========================调用算法模板进行种群进化======================="""
        [BestIndi, population] = myAlgorithm.run()  # 执行算法模板，得到最优个体以及最后一代种群
        # BestIndi.save()  # 把最优个体的信息保存到文件中
        """==================================输出结果=============================="""
        ga_scheduling_sequence = BestIndi.Phen.reshape(args.sector_number, args.user_numbers)
        ga_max_se = BestIndi.ObjV
        scheduling_sequence.append(ga_scheduling_sequence)
        SE.append(ga_max_se)
    # 路径格式，data_part/preprocess_data/Greedy_result/用户数目/移动速度/
    Sum_se_path = args.greedy_folder / 'GA_sum_SE'
    Scheduling_path = args.greedy_folder / 'GA_shceduling_sequence'
    np.save(Sum_se_path, np.array(SE))
    np.save(Scheduling_path, np.stack(scheduling_sequence, axis=0))


def main():
    user_number_list = ['10_user','20_user','30_user','40_user']
    velocity = ['3KM','30KM','90KM']
    # for index in range(12):
    index = 4
    user_index = user_number_list[index // 3]
    velocity_index = velocity[index % 3]
    # 修改用户的数量和用户移动速度
    user_number = int(user_index.split('_')[0])
    velocity_number = int(velocity_index.split('K')[0])
    common_args = get_common_args(user_number)
    # common_args.user_numbers = user_number
    common_args.user_velocity = velocity_number
    common_args.testing_path = pathlib.Path(common_args.training_data_path)/(str(common_args.user_numbers) + '_user')/(str(common_args.user_velocity)+'KM')/'testing_data_10_10.npy'
    common_args.greedy_folder = pathlib.Path(common_args.greedy_folder)/(str(common_args.user_numbers) + '_user')/(str(common_args.user_velocity)+'KM')
    # 如果文件不存在就创建
    common_args.greedy_folder.mkdir(parents=True, exist_ok=True)
    simulation(common_args)

if __name__=='__main__':
    main()

