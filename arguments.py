import argparse
import torch
import math

def get_common_args():
    # flag = torch.cuda.is_available()
    flag = False
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--user_numbers', type=int, default=20, help='the number of users of single cell')
    parser.add_argument('--user_antennas', type=int, default=1, help='the number of user antennas')
    parser.add_argument('--user_velocity', type=int, default=3, help='the velocity of user movement')
    parser.add_argument('--bs_antennas',type=int, default=16, help='the number of base station antennas')
    parser.add_argument('--cuda', type=bool, default=flag, help='whether to use the GPU')
    parser.add_argument('--gamma', type=float, default=0.95, help='discount ratio')
    parser.add_argument('--noise_spectrum_density', type=float, default=3.1623e-20, help='the noise power')
    parser.add_argument('--subcarrier_numbers', type=int, default=50, help='the number of subcarriers')
    parser.add_argument('--subcarrier_gaps', type=int, default=5000, help='the gap of adjancy subcarriers')
    parser.add_argument('--system_bandwidth',type=int, default=1e6, help='the system bandwidth')
    parser.add_argument('--transmit_power', type=float, default=0.25, help='the total transmit power of base station')
    parser.add_argument('--tau', type=float, default=0.995, help='the filter coefficient')
    parser.add_argument('--max_norm_grad', type=float, default=2, help='grad norm clamp')
    parser.add_argument('--drop_out', type=float, default=0.1, help='dropout rate')
    parser.add_argument('--cell_number', type=int, default=1, help='the numbers of cell')
    parser.add_argument('--sector_number', type=int, default=3, help='the numbers of sectors per cell')
    parser.add_argument('--total_TTI_length', type=int, default=1100, help='the total TTI length of channel data')
    parser.add_argument('--TTI_length',type=int,default=200, help='the TTI length')
    parser.add_argument('--training_data_path', type=str, default="../data_part/preprocess_data", help='the original data folder')
    # 定义rank要不要使用
    parser.add_argument('--PF_start', type=bool, default=False, help='whether we use PF sheduling algorithm')
    parser.add_argument('--rank_start', type=bool, default= False, help='whether start rank priority algorithm')
    parser.add_argument('--weighted_start', type=bool, default=False, help='whether weighted sum algorithm')
    parser.add_argument('--weighted_ratio', type=float, default=0.55, help='the weight factor between two goals')
    parser.add_argument('--priority_start', type=bool, default=False, help='whether priority algorithm')
    parser.add_argument('--edge_max_start', type=bool, default=False, help='whether edge max algorithm')
    parser.add_argument('--transformer_start', type=bool, default=False, help='whether transformer model adopted')
    parser.add_argument('--attention_start', type=bool, default=False, help='whether attention model start')
    parser.add_argument('--prev_policy_start', type=bool, default=False, help='whether using prev policy')
    # 定义参数共享
    parser.add_argument('--koopman_predict_start',type=bool, default=False, help='whether using koopman predictor')
    parser.add_argument('--parameter_sharing',type=bool, default=False, help="whether using parameter sharing")
    # 创建greedy 算法保存的文件夹
    parser.add_argument('--greedy_folder', type=str, default="../data_part/data/", help='the folder of greedy result')
    # 定义delay 时长
    parser.add_argument('--delay_time', type=int, default=3, help='channel estimation delay time')

    args = parser.parse_args()
    return args

def get_agent_args(args):
    # 计算能够支持最大流的数目
    total_user_antennas = args.user_antennas * args.user_numbers
    max_stream = min(total_user_antennas, args.bs_antennas)
    min_stream = 1
    args.max_stream = max_stream
    args.min_stream = min_stream
    # 定义智能体的数量
    args.n_agents = args.cell_number * args.sector_number
    # 定义观测的维度，信道矩阵的维度，即接收天线的维度乘基站天线的维度
    args.obs_dim1 = args.user_numbers * args.user_antennas
    args.obs_dim2 = args.bs_antennas * 2
    # 定义状态的维度，其实就是将所有智能体的状态进行拼接
    args.state_dim1 = args.obs_dim1
    args.state_dim2 = args.obs_dim2
    args.obs_matrix_number = args.n_agents
    args.state_matrix_number = args.n_agents * args.n_agents
    args.total_obs_matrix_number = args.obs_matrix_number + 1
    args.total_state_matrix_number = args.state_matrix_number  
    args.total_bs_antennas = args.obs_dim2
    args.total_user_antennas = total_user_antennas
    args.weight_factor_number = 32
    return args

def get_transformer_args(args):
    # 定义 head, query，key，weight的维度
    args.head_number = 8
    args.sub_query_dim = 64
    args.sub_key_dim = 64
    args.sub_value_dim = 64
    args.hidden_dim = args.head_number * args.sub_query_dim
    args.query_dim = args.head_number * args.sub_query_dim
    args.key_dim = args.head_number * args.sub_key_dim
    args.value_dim = args.head_number * args.sub_value_dim
    # 使用卷积操作，得到单通道的feature map
    args.kernel_width = 1
    args.kernel_height = 1
    args.stack_layer = 6
    return args

def get_MADDPG_args(args):
    # 如果算法采用MADDPG,则在此定义网络中的参数
    # 定义epoch的数目
    args.epoches = 1000
    # 定义episode的是数目
    args.episodes = 1
    # 定义batch size每次采样的长度
    args.batch_size = args.TTI_length
    # 定义replay buffer的大小
    args.max_buffer_size = args.episodes
    # 定义actor网络中的参数,即pointer network中的网络参数
    
    # 首先定义actor网络的lr以及lr_decay
    actor_lr = 1e-4
    actor_lr_decay = 1e-5
    args.actor_lr = actor_lr
    args.actor_lr_decay = actor_lr_decay
    # 定义一层卷积层,将输入的1*5*20*32的矩阵变成 1*20*32的特征矩阵
    args.actor_kernel_size = 3
    args.actor_stride = 2
    args.W_out = math.floor((args.obs_dim2 - (args.actor_kernel_size-1)-1) / args.actor_stride + 1)
    args.H_out = math.floor((args.obs_dim1 - (args.actor_kernel_size-1)-1) / args.actor_stride + 1)
    args.actor_maxpool_kernel_size = 2
    args.W_out = math.floor((args.W_out - (args.actor_maxpool_kernel_size-1)-1) / args.actor_stride + 1)
    args.H_out = math.floor((args.H_out - (args.actor_maxpool_kernel_size-1)-1) / args.actor_stride + 1)
    # 定义策略网络的神经元个数
    args.rnn_hidden = 64
    # 定义pointer network的权重向量的长度
    args.weight_dim = 32
    
    args.embedding_dim = 2
    # 由于是按照天线进行调度的,所以每一根天线实部和虚数部分分开,得到的特征长度为32
    args.flatten_dim = args.W_out * args.H_out
    args.flatten_dim = 28
    # 定义数据嵌入的维度
    args.embbedding_dim = 2
    # flatten之后的信道矩阵进行低维的嵌入操作之后, 以及与average user sum rate进行拼接,得到状态维度是user_num * 3, 然后升维
    args.rnn_input_dim = 32

    # 定义critic网络的相关参数
    critic_lr = 1e-2
    critic_lr_decay = 1e-5
    args.critic_lr = critic_lr
    args.critic_lr_decay = critic_lr_decay
    args.update_times = 100
    # 定义preconv layer
    args.critic_pre_stride = 1
    args.critic_pre_kernel_size = 3
    args.critic_pre_padding = 1
    # 定义critic网络的一些参数,网络采用的是卷积网络
    args.layer_number = 3
    args.kernal_number = [args.user_numbers,32,8]
    args.kernal_size = [(5,4),(3,3),(2,2)]
    args.kernal_stride = [(1,2),(1,2),(1,1)]
    args.padding_size = [(0, 0), (0,0), (0, 0)]
    args.dilation = [(1,1), (1,1),(1,1)]
    args.fc_dim = 32
    # 定义GAE因子
    args.GAE_factor = 0.1
    # 定义探索率
    args.epsilon = 0.2
    args.min_epsilon = 0.01
    args.epsilon_decay = 0.005

    return args
    


    

