import argparse
import torch

def get_common_args(user_numbers):
    flag = torch.cuda.is_available()
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--user_numbers', type=int, default=user_numbers, help='the number of users of single cell')
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
    parser.add_argument('--mode', type=str, default='train', help='training or testing')
    # 定义算法开关
    parser.add_argument('--independent_learning', type=bool, default=False, help='whether using independent learning')
    parser.add_argument('--ablation_experiment', type=bool, default=False, help='whether using ablation experiment')
    parser.add_argument('--PF_start', type=bool, default=False, help='whether we use PF sheduling algorithm')
    parser.add_argument('--rank_start', type=bool, default= False, help='whether start rank priority algorithm')
    parser.add_argument('--weighted_start', type=bool, default=False, help='whether weighted sum algorithm')
    parser.add_argument('--weighted_ratio', type=float, default=0.55, help='the weight factor between two goals')
    parser.add_argument('--priority_start', type=bool, default=False, help='whether priority algorithm')
    parser.add_argument('--edge_max_start', type=bool, default=False, help='whether edge max algorithm')
    parser.add_argument('--transformer_start', type=bool, default=False, help='whether transformer model adopted')
    parser.add_argument('--attention_start', type=bool, default=False, help='whether attention model start')
    parser.add_argument('--prev_policy_start', type=bool, default=False, help='whether using prev policy')
    parser.add_argument('--commNet_start', type=bool, default=False, help='whether using communication RL')
    parser.add_argument('--maddpg_start', type=bool, default=False, help='whether using maddpg algorithm')
    parser.add_argument('--max_SE',type=bool, default=True, help='PF scheduling or MaxSE scheduling')
    parser.add_argument('--beam_size', type= int, default=2, help='the size of beam')
    parser.add_argument('--decode_sampling_method', type=int, default=0, help='the decode method, 0: sampling, 1: greedy')
    # 定义参数共享
    parser.add_argument('--koopman_predict_start',type=bool, default=False, help='whether using koopman predictor')
    parser.add_argument('--parameter_sharing',type=bool, default=True, help="whether using parameter sharing")
    # 创建greedy 算法保存的文件夹
    parser.add_argument('--greedy_folder', type=str, default="../data_part/Greedy_result", help='the folder of greedy result')
    # 定义delay 时长
    parser.add_argument('--delay_time', type=int, default=3, help='channel estimation delay time')
    # 定义epoch的数目
    parser.add_argument('--epoches', type=int, default=1000, help='Training steps')
    parser.add_argument('--episode', type=int, default=200, help='numbers of trajectory samples')
    parser.add_argument('--batch_size', type=int, default=200, help='the numbers of training samples')
    parser.add_argument('--max_buffer_size', type=int, default=200, help='the capacity of replaybuffer')
    parser.add_argument('--warm_start', type=int, default=200, help='random action step')
    args = parser.parse_args()
    # 计算能够支持最大流的数目
    total_user_antennas = args.user_antennas * args.user_numbers
    max_stream = min(total_user_antennas, args.bs_antennas)
    min_stream = 1
    args.max_stream = max_stream
    args.min_stream = min_stream
    args.total_user_antennas = total_user_antennas
    return args

def get_agent_args(args):
    # 这个地方定义神经网络的通用参数,包括RNN的输入维度,citic的网络参数等等
    args.n_agents = args.cell_number * args.sector_number
    # 定义观测的维度，信道矩阵的维度，即接收天线的维度乘基站天线的维度
    args.obs_dim1 = args.user_numbers * args.user_antennas
    args.obs_dim2 = args.bs_antennas * 2
    # 定义状态的维度，其实就是将所有智能体的状态进行拼接
    args.state_dim1 = args.obs_dim1
    args.state_dim2 = args.obs_dim2
    # 全局状态的信道矩阵维度和观测矩阵的维度是一样的
    args.obs_matrix_number = args.n_agents
    args.state_matrix_number = args.n_agents * args.n_agents
    # 这里+1是因为需要加上那个reward向量, 但是SE最大是不需要的
    args.total_obs_matrix_number = args.obs_matrix_number if args.max_SE else args.obs_matrix_number + 1
    args.total_state_matrix_number = args.state_matrix_number  
    args.weight_factor_number = 32
    # 定义策略网络和critic网络中公共部分

    # 定义critic网络的相关参数
    critic_lr = 1e-3
    critic_lr_decay = 1e-2
    critic_min_lr = 1e-4
    args.critic_lr = critic_lr
    args.critic_lr_decay = critic_lr_decay
    args.critic_min_lr = critic_min_lr
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

    # 定义actor网络的lr以及lr_decay
    actor_lr = 1e-2
    actor_lr_decay = 1e-2
    actor_min_lr = 1e-6
    args.actor_min_lr = actor_min_lr
    args.actor_lr = actor_lr
    args.actor_lr_decay = actor_lr_decay
    # 定义一层卷积层,将输入的1*3*20*32的矩阵变成 1*20*32的特征矩阵
    args.actor_kernel_size = 3
    args.actor_stride = 2
    args.actor_maxpool_kernel_size = 2
    # 只要网络中采用了RNN结构,统一定义隐藏神经元个数
    args.rnn_hidden = 64
    # 定义rnn网络的输入维度为
    args.rnn_input_dim = 32
    # 如果采用了pointer network类似的做法,统一定义权重矩阵的特征维度
    args.weight_dim = 32
    # 定义探索率,等相关参数
    args.epsilon = 0.2
    args.min_epsilon = 0.01
    args.epsilon_decay = 0.005
    # 如果是序列化决策,定义GAE系数为
    args.GAE_factor = 0.1
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
    
    return args
    

def get_communication_net_args(args):
    args.communication_turns = 4
    return args
    

