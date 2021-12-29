# Combinatorial-optimization


## 配置参数
1. user_nums, 用户的数目
2. sector_nums,扇区数目
3. cell_nums,小区数目
4. agent_nums, 智能体的数目
5. bs_antenna_nums, 用户接收天线的数目
6. total_antenna_nums, 一个小区所有用户天线的数目
7. sliding_windows_length, 这个值表示一个环境的长度
8. transmit_power, 这个值表示的是小区基站的发射功率
9. noise_power, 噪声的功率
10. source_data_folder,表示matlab生成了的数据放置的文件夹, data_part/Source_data
11. save_data_folder,表示处理好之后的数据存放的文件夹路径,比如说, data_part/preprocess_data
12. velocity, 表示用户的移动速度
13. subcarrier_nums, 表示载波的数目
14. training_data_total_TTI_length表示的是训练数据样本长度
15. eval_data_total_TTI_length表示的是评估数据样本的长度
16. delay_time_window表示过去的时间窗口，用来更新t+1时刻的平均SE
17. min_user_average_se表示用户最小的频谱效率
18. max_user_pf表示用户最大的pf值


## leanrer框架的一些参数定义
1. global rank表示的是这个server是用的第几张卡
2. gpu_num_per_machine 表示的是一台机器有多少张卡
3. local rank是在多机多卡的情况下，可能global rank是10，但是在这台机器上其实就是第二张卡
4. eval mode就是说需要载入模型，然后直接进行测试操作
