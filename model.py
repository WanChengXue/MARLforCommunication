
# 这个函数用来建立基本的模型
import torch 
import torch.nn as nn
from torch.nn.parameter import Parameter
import numpy as np
import math
from torch.distributions import Categorical

class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.args = args
        # 定义原始信道矩阵的列数
        self.feature_number = self.args.obs_dim2
        self.drop_out = self.args.drop_out
        self.input_channel = self.args.total_obs_matrix_number
        self.output_channel = self.args.obs_dim1
        self.kernel_size = self.args.actor_kernel_size
        self.stride = self.args.actor_stride
        self.maxpool_kernel_size = self.args.actor_maxpool_kernel_size
        self.hidden_dim = self.args.rnn_hidden
        self.weight_dim = self.args.weight_dim
        self.rnn_input_dim = self.args.rnn_input_dim
        self.flatten_dim = self.args.flatten_dim
        # Eecoder layer
        self.Encoder_conv_layer = nn.Conv2d(self.input_channel, self.output_channel, self.kernel_size, self.stride)
        self.Encoder_maxpool_layer = nn.MaxPool2d(self.maxpool_kernel_size, self.stride)
        self.Encoder_flatten = nn.Flatten(start_dim=2)
        self.Encoder_affine_layer = nn.Linear(self.flatten_dim, self.rnn_input_dim)
        self.Encoder = nn.GRU(self.rnn_input_dim, self.hidden_dim, batch_first=True)
        self.Decoder = nn.GRU(self.rnn_input_dim, self.hidden_dim, batch_first=True)
        self.device = "cuda" if self.args.cuda else "cpu"
        self.Encoder_init_input = torch.zeros(1,1,self.hidden_dim).to(self.device)
        # define key matrix W_k and query matrix W_q
        self.W_k = Parameter(torch.randn(1, self.weight_dim, self.hidden_dim))
        self.W_q = Parameter(torch.randn(1, self.weight_dim, self.hidden_dim))
        # define weight vector
        self.Weight_vector = Parameter(torch.randn(1, self.weight_dim, 1))
        # define drop out rate
        self.drop_rnn_out = nn.Dropout(self.drop_out)
        self.drop_rnn_hidden = nn.Dropout(self.drop_out)
        self.eps = 1e-12

    def forward(self, channel_matrix, user_instant_reward , priority_vector=None, Action=None):
        # user_instant_reward的维度是用户数量乘以1 
        # 信道矩阵输入的维度是1*9*用户数目*32， user_instant_reward的维度是用户数目 *1 的一个向量
        batch_size = channel_matrix.shape[0]
        seq_len = channel_matrix.shape[2]
        total_len = seq_len + 1  
        # 先对Encoder部分进行处理 Linear_data.shape = (batch_size * antenna_number, rnn_input_dim)
        # 这个地方将user_instant_reward进行repeat操作，然后和Encoder_hidden_result进行concatenate操作
        argumented_information = user_instant_reward.reshape(1,1,seq_len,1).repeat(1,1,1,self.feature_number)
        preprocess_channel_matrix = torch.cat([1e6*channel_matrix, argumented_information],1)
        # 两个矩阵进行拼接，变成一个两通道的信息，然后再采用一次卷积操作得到一个一通道的数据？
        Encoder_conv_channel = self.Encoder_conv_layer(preprocess_channel_matrix)
        Encoder_maxpool_channel = self.Encoder_maxpool_layer(Encoder_conv_channel)
        Encoder_flatten_channel = self.Encoder_flatten(Encoder_maxpool_channel)
        Encoder_linear_data =  torch.relu(self.Encoder_affine_layer(Encoder_flatten_channel))

        Decoder_linear_data = Encoder_linear_data.clone()
        # Init_encoder表示的是Encoder网络的初始化隐状态
        Init_encoder = self.Encoder_init_input.repeat(1, batch_size, 1)
        # 计算key矩阵,也就是Encoder得到的隐状态矩阵,维度是batch_size * (1+antenna_numbers) * hidden_dim
        Encoder_hidden_result, Encoder_hidden_vector = self.Encoder(Encoder_linear_data, Init_encoder)
        Encoder_hidden_result = torch.cat((Init_encoder.permute(1,0,2), Encoder_hidden_result),1)
        # Extend key matrix, query matrix, weight vector
        Extend_key = self.W_k.repeat(batch_size, 1, 1)
        Extend_query = self.W_q.repeat(batch_size, 1, 1)
        Extend_weight_vector = self.Weight_vector.repeat(batch_size, 1, 1)
        # 计算key matrix, 通过一层affine transformation batch_size * weight_dim * total_seq 
        Key_matrix = torch.bmm(Extend_key, Encoder_hidden_result.permute(0,2,1))
        # 初始化Decoder的输入
        Input_decoder = torch.mean(Decoder_linear_data, 1).unsqueeze(1)
        # 初始化Decoder的隐状态输入 1*batch_size * hidden_dim
        Decoder_hidden_vector = Encoder_hidden_vector
        # 定义mask,将已经出现过的用户的概率设置为0
        Mask = torch.ones(batch_size, total_len).to(self.device)
        rank_factor = priority_vector 
        schedule_result = []
        result = []
        # 这个priority vector是一个0，1向量，
        # 位置为1的地方就是说明这个用户可以调度
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
            Weight_vector = torch.relu(torch.bmm(Extend_weight_vector.permute(0,2,1), Similar_matrix))
            if priority_vector:
                prob_matrix = torch.softmax(Weight_vector.squeeze(1) * Mask * rank_factor, -1)
            else:
                prob_matrix = torch.softmax(Weight_vector.squeeze(1) * Mask, -1)
            # prob_matrix = torch.softmax(prob_matrix +  Mask.log() + rank_factor.log(), -1)
            if torch.is_tensor(Action):
                selected_user = Action[antenna_index].unsqueeze(0)
                prob_vector = torch.log(prob_matrix[0, selected_user])
            else:
                if not self.args.Training:
                    selected_user = torch.argmax(prob_matrix, -1)
                    prob_vector = torch.log(prob_matrix[0, selected_user])
                else:
                    if antenna_index == self.args.max_stream:
                        selected_user = torch.LongTensor([0]).to(self.device)
                        prob_vector = torch.log(self.eps+prob_matrix[0, selected_user])
                    else:
                        if torch.rand(1) < self.args.epsilon:
                            random_weight_matrix = self.args.epsilon * torch.randn(batch_size, total_len).to(self.device) * Mask + prob_matrix * (1-self.args.epsilon)
                            random_prob_matrix = torch.softmax(random_weight_matrix, -1)
                            selected_user = torch.argmax(random_prob_matrix, -1)
                            prob_vector = torch.log(random_prob_matrix[0, selected_user])
                        else:
                            selected_user = torch.argmax(prob_matrix, -1)
                            prob_vector = torch.log(prob_matrix[0, selected_user])
            Mask = Mask.clone()  
            Mask.scatter_(1, selected_user.unsqueeze(1), self.eps)
            result.append(prob_vector)
            if selected_user.item() == 0:
                break
            schedule_result.append(selected_user.item()-1)
            Input_decoder = Decoder_linear_data[:,selected_user.item()-1,:].unsqueeze(1)
            

        return schedule_result, torch.sum(torch.cat(result, 0))

class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.args = args
        self.feature_number = self.args.obs_dim2
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
        H_in = self.args.state_dim1
        W_in = self.args.state_dim2
        self.device = "cuda" if self.args.cuda else "cpu"
        self.pre_conv_layer = nn.Conv2d(in_channel, self.args.cell_number, self.pre_kernel_size, self.pre_stride, self.pre_padding)
        in_channel = self.args.cell_number
        conv_layer = []
        for layer in range(self.conv_layer_number):
            conv_layer.append(nn.Conv2d(in_channel, self.kernal_number[layer], self.kernal_size[layer], self.kernal_stride[layer], self.padding_size[layer], self.dilation[layer]).to(self.device))
            H_out = math.floor((H_in+2*self.padding_size[layer][0]-self.dilation[layer][0]*(self.kernal_size[layer][0]-1)-1)/self.kernal_stride[layer][0]+1)
            W_out = math.floor((W_in+2*self.padding_size[layer][1]-self.dilation[layer][1]*(self.kernal_size[layer][1]-1)-1)/self.kernal_stride[layer][1] + 1)
            H_in = H_out
            W_in = W_out
            in_channel = self.kernal_number[layer]
        self.conv_layer = nn.ModuleList(conv_layer)
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(self.kernal_number[layer]*H_out*W_out, self.args.fc_dim)
        self.output_layer = nn.Linear(self.args.fc_dim, 1)

    def forward(self, channel, user_instant_reward):
        # 输入的channel是一个81*10*32的信道矩阵，use_instant_reward是一个9*10*1的矩阵
        argumented_information = user_instant_reward.unsqueeze(-1).repeat(1,1,1,self.feature_number)
        preprocess_input_channel = torch.cat([1e6 * channel, argumented_information],1)
        pre_conv_channel = torch.relu(self.pre_conv_layer(preprocess_input_channel))
        conv_result = torch.relu(self.ascend(pre_conv_channel))
        for layer in range(self.conv_layer_number):
            conv_result = torch.relu(self.conv_layer[layer](conv_result))
        flatten_result = self.flatten(conv_result)
        fc_result = torch.relu(self.layer1(flatten_result))
        V_value = self.output_layer(fc_result)
        return V_value
