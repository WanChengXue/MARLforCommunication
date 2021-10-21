import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.distributions import Categorical

class Actor(nn.Module):
    def __init__(self, args, input_shape):
        super(Actor, self).__init__()
        self.args = args
        # 定义原始信道矩阵的列数
        self.input_channel = self.args.total_obs_matrix_number  
        self.output_channel = self.args.obs_dim1
        self.kernel_size = self.args.actor_kernel_size
        self.stride = self.args.actor_stride
        self.maxpool_kernel_size = self.args.actor_maxpool_kernel_size


        self.Encoder_conv_layer = nn.Conv2d(self.input_channel, self.output_channel, self.kernel_size, self.stride)
        self.Encoder_maxpool_layer = nn.MaxPool2d(self.maxpool_kernel_size, self.stride)
        self.Encoder_flatten = nn.Flatten(start_dim=1)
        self.flatten_dim = self.output_dimension(input_shape)
        self.Encoder_affine_layer = nn.Linear(self.flatten_dim, args.rnn_hidden)

    def output_dimension(self, input_shape):
        test = torch.rand(*input_shape)
        Encoder_conv_channel = self.Encoder_conv_layer(test)
        Encoder_maxpool_channel = self.Encoder_maxpool_layer(Encoder_conv_channel)
        Encoder_flatten_channel = self.Encoder_flatten(Encoder_maxpool_channel)
        return Encoder_flatten_channel.shape[-1]

    def forward(self, channel_matrix):
        Encoder_conv_channel = self.Encoder_conv_layer(channel_matrix*1e6)
        Encoder_maxpool_channel = self.Encoder_maxpool_layer(Encoder_conv_channel)
        Encoder_flatten_channel = self.Encoder_flatten(Encoder_maxpool_channel)
        Encoder_linear_data =  torch.relu(self.Encoder_affine_layer(Encoder_flatten_channel))
        return Encoder_linear_data

class Pointer_Actor(nn.Module):
    def __init__(self, args):
        super(Pointer_Actor, self).__init__()
        self.args = args
        self.device = "cuda" if self.args.cuda else "cpu"
        self.drop_out = self.args.drop_out
        self.Embbeding_layer = nn.Embedding(2, args.rnn_input_dim)
        self.Encoder = nn.GRU(args.rnn_input_dim, args.rnn_hidden, batch_first=True)
        self.Decoder = nn.GRU(args.rnn_input_dim, args.rnn_hidden, batch_first=True)
        self.Encoder_init_input = Parameter(torch.randn(1,1,args.rnn_hidden))
        # define key matrix W_k and query matrix W_q
        self.W_k = Parameter(torch.randn(1, args.weight_dim, args.rnn_hidden))
        self.W_q = Parameter(torch.randn(1, args.weight_dim, args.rnn_hidden))
        # define weight vector
        self.Weight_vector = Parameter(torch.randn(1, args.weight_dim, 1))
        # define drop out rate
        self.drop_rnn_out = nn.Dropout(args.drop_out)
        self.drop_rnn_hidden = nn.Dropout(args.drop_out)
        self.eps = 1e-12

    def forward(self, state):
        # user_instant_reward的维度是用户数量乘以1 
        # 信道矩阵输入的维度是1*9*用户数目*32， user_instant_reward的维度是用户数目 *1 的一个向量
        batch_size = state.shape[0]
        seq_len = state.shape[1]
        total_len = seq_len + 1  
        bos_token_encoder = [[0]] * batch_size
        bos_token_decoder = [[1]] * batch_size
        # 维度是batch_size * 1 * flatten_dim
        bos_vector = self.Embbeding_layer(torch.LongTensor(bos_token_encoder).to(self.device))
        eos_vector = self.Embbeding_layer(torch.LongTensor(bos_token_decoder).to(self.device))
        # 先对Encoder部分进行处理 Linear_data.shape = (batch_size * antenna_number, rnn_input_dim)
        # 这个地方将user_instant_reward进行repeat操作，然后和Encoder_hidden_result进行concatenate操作
        # 两个矩阵进行拼接，变成一个两通道的信息，然后再采用一次卷积操作得到一个一通道的数据？
        Input_encoder = torch.cat([bos_vector, state], 1)
        # Init_encoder表示的是Encoder网络的初始化隐状态
        Init_encoder = self.Encoder_init_input.repeat(1, batch_size, 1)
        # Init_encoder
        # 计算key矩阵,也就是Encoder得到的隐状态矩阵,维度是batch_size * (1+antenna_numbers) * rnn_hidden
        Encoder_hidden_result, Encoder_hidden_vector = self.Encoder(Input_encoder, Init_encoder)
        # Extend key matrix, query matrix, weight vector
        Extend_key = self.W_k.repeat(batch_size, 1, 1)
        Extend_query = self.W_q.repeat(batch_size, 1, 1)
        Extend_weight_vector = self.Weight_vector.repeat(batch_size, 1, 1)
        # 计算key matrix, 通过一层affine transformation batch_size * weight_dim * total_seq 
        Key_matrix = torch.bmm(Extend_key, Encoder_hidden_result.permute(0,2,1))
        # 初始化Decoder的输入
        Input_decoder = eos_vector
        # 初始化Decoder的隐状态输入 1*batch_size * rnn_hidden
        Decoder_hidden_vector = Encoder_hidden_vector
        # 定义mask,将已经出现过的用户的概率设置为0
        mask = torch.zeros(batch_size, total_len).to(self.device)
        batch_sheduling_result = []
        batch_sheduling_result.append(-1*torch.ones(batch_size,1))
        batch_prob_result = []
        selected_mask = torch.zeros(batch_size, self.args.max_stream+1).to(self.device)
        for antenna_index in range(self.args.max_stream+1):
            # 第一步使用Decoder进行解码操作
            Decoder_output_vector, Decoder_hidden_vector = self.Decoder(Input_decoder, Decoder_hidden_vector) 
            Decoder_output_vector = self.drop_rnn_out(Decoder_output_vector)
            Decoder_hidden_vector = self.drop_rnn_hidden(Decoder_hidden_vector)
            # 第二步计算出W_q h_i batch_size * weight_dim * 1
            Query_vector = torch.bmm(Extend_query, Decoder_output_vector.permute(0,2,1))
            # 计算相似矩阵 (batch_size * 32 * 21)
            Similar_matrix = torch.tanh(Key_matrix + Query_vector)
            # 计算权重向量 batch_size * 1 * (word_num + 1)
            Weight_vector = torch.relu(torch.bmm(Extend_weight_vector.permute(0,2,1), Similar_matrix)).squeeze(1)
            # 对那些已经出现过了的用户，减去一个很大的常数，这样softmax之后概率就是无限接近0
            Weight_vector = Weight_vector - 1e7 * mask
            # 这个prob_vector的维度是batch_size * 21
            prob_vector = torch.softmax(Weight_vector, -1)
            dist = Categorical(prob_vector)
            sheduling_user = dist.sample()
            # 如果说上一次的调度结果中，有些样本出现了0，则表明这个样本就结束调度了，并且之后的调度值都一直给0就好了
            terminal_flag = batch_sheduling_result[-1] == 0
            sheduling_user[terminal_flag.squeeze(-1)] = 0
            # 如果说达到了最大的调度次数，则无论采样出来的什么，都强制结束
            if antenna_index == self.args.max_stream:
                sheduling_user[:] = 0
            # 将Mask中的已经出现过了的用户变成0
            mask.scatter_(1, sheduling_user.unsqueeze(1), 1)
            #这个地方将那些已经终止了的调度序列，后面的调度概率都设置为0
            selected_mask[:,antenna_index][torch.logical_not(terminal_flag.squeeze(-1))] = 1
            batch_sheduling_result.append(sheduling_user.unsqueeze(1))
            batch_prob_result.append(dist.log_prob(sheduling_user).unsqueeze(1))
            selected_index = torch.zeros(batch_size, total_len).to(self.device).bool()
            selected_index.scatter_(1, sheduling_user.unsqueeze(-1), True)  
            Input_decoder = Input_encoder[selected_index].unsqueeze(1)
        return torch.cat(batch_sheduling_result[1:], -1), torch.cat(batch_prob_result, -1), selected_mask, mask
        

class CommNet_Actor(nn.Module):
    def __init__(self, args, input_shape):
        super(CommNet_Actor, self).__init__()
        self.args = args
        self.precoding_agent = Actor(args, input_shape)
        self.f_comm = nn.GRUCell(args.rnn_hidden, args.rnn_hidden) 
        self.pointer_agent = Pointer_Actor(args)
        self.seq_len = input_shape[2]
        self.Pointer_network_input_dim = input_shape[2] * args.rnn_input_dim
        self.decoding_layer = nn.Linear(args.rnn_hidden, self.Pointer_network_input_dim)

    def forward(self):
        pass

class Critic(nn.Module):
    def __init__(self, args, input_shape):
        super(Critic, self).__init__()
        self.args = args
        # 定义pre conv layer的参数
        self.pre_stride = self.args.critic_pre_stride
        self.pre_kernel_size = self.args.critic_pre_kernel_size
        self.pre_padding = self.args.critic_pre_padding
        # 定义实际conv_layer的参数
        self.kernal_number = self.args.kernal_number
        self.kernal_size = self.args.kernal_size
        self.kernal_stride = self.args.kernal_stride
        self.padding_size = self.args.padding_size
        self.dilation = self.args.dilation
        self.conv_layer_number = self.args.layer_number
        in_channel = self.args.total_state_matrix_number
        # self.reduction = nn.Linear(self.args.state_dim2, self.args.embedding_dim)
        self.ascend = nn.Linear(self.args.state_dim2, self.args.state_dim2)
        # self.device = "cuda" if self.args.cuda else "cpu"
        self.pre_conv_layer = nn.Conv2d(in_channel, self.args.n_agents, self.pre_kernel_size, self.pre_stride, self.pre_padding)
        in_channel = self.args.n_agents  
        conv_layer = []
        for layer in range(self.conv_layer_number):
            conv_layer.append(nn.Conv2d(in_channel, self.kernal_number[layer], self.kernal_size[layer], self.kernal_stride[layer], self.padding_size[layer], self.dilation[layer]))
            in_channel = self.kernal_number[layer]
        self.conv_layer = nn.ModuleList(conv_layer)
        self.flatten = nn.Flatten()
        conv_output_dim = self.output_dimension(input_shape)
        self.linear_layer = nn.Linear(conv_output_dim, self.args.fc_dim)
        self.output_layer = nn.Linear(self.args.fc_dim, 1)

    def output_dimension(self,input_shape):
        test = torch.rand(*input_shape)
        pre_conv_channel = torch.relu(self.pre_conv_layer(test))
        conv_result = torch.relu(self.ascend(pre_conv_channel))
        for layer in range(self.conv_layer_number):
            conv_result = self.conv_layer[layer](conv_result)
        flatten_result = self.flatten(conv_result)
        return flatten_result.shape[-1]


    def forward(self, channel):
        # 输入的channel是一个81*10*32的信道矩阵，use_instant_reward是一个9*10*1的矩阵
        # action是一个长度为batch_size * agent numbet * user antennas * 1的一个向量
        pre_conv_channel = torch.relu(self.pre_conv_layer(1e7 *channel))
        conv_result = torch.relu(self.ascend(pre_conv_channel))
        for layer in range(self.conv_layer_number):
            conv_result = torch.relu(self.conv_layer[layer](conv_result))
        flatten_result = self.flatten(conv_result)
        fc_result = torch.relu(self.linear_layer(flatten_result))
        V_value = self.output_layer(fc_result)
        return V_value