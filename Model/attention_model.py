import torch
import torch.nn as nn
# import torch.nn.parameter as Parameter
from torch.distributions import Categorical

class Policy(nn.Module):
    def __init__(self, args, input_shape):
        super(Policy, self).__init__()
        self.args = args
        self.args = args
        self.device = "cuda" if self.args.cuda else "cpu"
        # 定义原始信道矩阵的列数
        self.feature_number = self.args.obs_dim2
        self.drop_out = self.args.drop_out
        self.input_channel = self.args.total_obs_matrix_number 
        self.output_channel = self.args.obs_dim1
        self.kernel_size = self.args.actor_kernel_size
        self.stride = self.args.actor_stride
        self.maxpool_kernel_size = self.args.actor_maxpool_kernel_size
        self.hidden_dim = self.args.rnn_hidden
        # self.rnn_input_dim = self.args.rnn_input_dim
        # 定义一下权重矩阵的行数
        self.embbeding_layer = nn.Embedding(3, self.hidden_dim)
        self.weight_factor_number = self.args.weight_dim
        self.Encoder_conv_layer = nn.Conv2d(self.input_channel, self.output_channel, self.kernel_size, self.stride)
        self.Encoder_maxpool_layer = nn.MaxPool2d(self.maxpool_kernel_size, self.stride)
        self.Encoder_flatten = nn.Flatten(start_dim=2)
        self.flatten_dim = self.output_dimension(input_shape)
        self.Encoder_affine_layer = nn.Linear(self.flatten_dim, self.hidden_dim)
        self.GRU = nn.GRUCell(self.hidden_dim, self.hidden_dim)
        # 定义几个权重向量
        self.W_a = nn.Parameter(torch.rand(1, self.weight_factor_number, self.hidden_dim*2))
        self.V_a = nn.Parameter(torch.rand(1, self.weight_factor_number, 1))
        self.W_c = nn.Parameter(torch.rand(1, self.weight_factor_number, self.hidden_dim*2))
        self.V_c = nn.Parameter(torch.rand(1, self.weight_factor_number, 1))
        self.eps = 1e-6
        # 定义layernorm
        self.total_seq_len = input_shape[2]+1
        self.layer_norm = nn.LayerNorm([self.total_seq_len, self.hidden_dim])
        self.layer_norm_1 = nn.LayerNorm([self.total_seq_len, self.hidden_dim*2])
        self.layer_norm_2 = nn.LayerNorm([self.total_seq_len, self.hidden_dim*2])

    def output_dimension(self, input_shape):
        test = torch.rand(*input_shape)
        Encoder_conv_channel = self.Encoder_conv_layer(test)
        Encoder_maxpool_channel = self.Encoder_maxpool_layer(Encoder_conv_channel)
        Encoder_flatten_channel = self.Encoder_flatten(Encoder_maxpool_channel)
        return Encoder_flatten_channel.shape[-1]

    def forward(self, input_data, Action=None):
        batch_size = input_data.shape[0]
        bos_token_decoder = [[1]] * batch_size
        eos_vector = self.embbeding_layer(torch.LongTensor(bos_token_decoder).to(self.device))
        # 初始化Decoder的输入
        Input_decoder = eos_vector.squeeze(1)
        Encoder_conv_channel = self.Encoder_conv_layer(input_data*1e6)
        Encoder_maxpool_channel = self.Encoder_maxpool_layer(Encoder_conv_channel)
        Encoder_flatten_channel = self.Encoder_flatten(Encoder_maxpool_channel)
        Embeding_data =  torch.relu(self.Encoder_affine_layer(Encoder_flatten_channel))
        End_vector = torch.zeros(batch_size, self.hidden_dim).to(self.device)
        Key_matrix = self.layer_norm(torch.cat([End_vector.unsqueeze(1), Embeding_data], 1))
        Extend_W_a = self.W_a.expand(batch_size, -1, -1)
        Extend_V_a = self.V_a.expand(batch_size, -1, -1)
        Extend_W_c = self.W_c.expand(batch_size, -1, -1)
        Extend_V_c = self.V_c.expand(batch_size, -1, -1)
        # 这个地方添加一个向量，得到终止用户的向量
        Mask = torch.zeros(batch_size, self.total_seq_len).to(self.device)
        batch_sheduling_result = []
        batch_sheduling_result.append(-1*torch.ones(batch_size,1))
        batch_prob_result = []
        selected_mask = torch.zeros(batch_size, self.args.max_stream+1).to(self.device)
        hidden_cell_data = torch.mean(Embeding_data, 1)
        for antenna_index in range(self.args.max_stream+1):
            Extend_h = hidden_cell_data.unsqueeze(1).expand(-1, self.total_seq_len, -1)
            Concatenate_embedding_and_h = self.layer_norm_1(torch.cat([Key_matrix, Extend_h], -1))
            Weight_vector_u = torch.bmm(Extend_V_a.transpose(1,2), torch.tanh(torch.bmm(Extend_W_a, Concatenate_embedding_and_h.transpose(1,2))))
            c_vector = torch.sum(Key_matrix*Weight_vector_u.transpose(1,2), 1)
            Extend_c = c_vector.unsqueeze(1).expand(-1, self.total_seq_len, -1)
            Concatenate_embeding_and_c = self.layer_norm_2(torch.cat([Key_matrix, Extend_c], -1))
            Weight_vector_u_tilde = torch.bmm(Extend_V_c.transpose(1,2), torch.tanh(torch.bmm(Extend_W_c, Concatenate_embeding_and_c.transpose(1,2)))).squeeze(1)
            prob_vector = torch.softmax(Weight_vector_u_tilde - 1e7 * Mask, -1)
            dist = Categorical(prob_vector)
            sheduling_user = dist.sample()
            terminal_flag = batch_sheduling_result[-1] == 0
            sheduling_user[terminal_flag.squeeze(-1)] = 0
            if antenna_index == self.args.max_stream:
                sheduling_user[:] = 0
            # 将Mask中的某一些值变成1
            Mask.scatter_(1, sheduling_user.unsqueeze(1), 1)
            selected_mask[:,antenna_index][torch.logical_not(terminal_flag.squeeze(-1))] = 1
            batch_sheduling_result.append(sheduling_user.unsqueeze(1))
            batch_prob_result.append(dist.log_prob(sheduling_user).unsqueeze(1))
            selected_index = torch.zeros(batch_size, self.total_seq_len).to(self.device).bool()
            selected_index.scatter_(1, sheduling_user.unsqueeze(-1), True) 
            hidden_cell_data = self.GRU(Input_decoder, hidden_cell_data)
            Input_decoder = Key_matrix[selected_index]
        return torch.cat(batch_sheduling_result[1:], -1), torch.cat(batch_prob_result, -1), selected_mask, Mask


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