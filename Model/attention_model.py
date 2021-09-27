import torch
import torch.nn as nn
# import torch.nn.parameter as Parameter
from torch.distributions import Categorical
import math

class Policy(nn.Module):
    def __init__(self, args):
        super(Policy, self).__init__()
        self.args = args
        self.input_dim_row = self.args.total_user_antennas
        self.input_dim_column = self.args.total_bs_antennas
        self.input_dim_channel = self.args.obs_matrix_number
        self.hidden_dim = self.args.rnn_hidden
        self.kernel_width = self.args.kernel_width
        self.kernel_height = self.args.kernel_height
        self.device = "cuda" if self.args.cuda else "cpu"
        self.batch_size = self.args.batch_size
        self.total_seq_len = self.input_dim_row + 1
        # 定义一下权重矩阵的行数
        self.weight_factor_number = self.args.weight_factor_number
        self.GRU = nn.GRUCell(self.input_dim_column, self.hidden_dim)
        self.pre_conv_layer = nn.Conv2d(self.input_dim_channel, 1, (self.kernel_height, self.kernel_width))
        self.affine_layer = nn.Linear(self.input_dim_column, self.hidden_dim)
        self.affine_reward = nn.Linear(self.input_dim_row, self.hidden_dim)
        # 定义几个权重向量
        self.W_a = nn.Parameter(torch.rand(1, self.weight_factor_number, self.hidden_dim*2))
        self.V_a = nn.Parameter(torch.rand(1, self.weight_factor_number, 1))
        self.W_c = nn.Parameter(torch.rand(1, self.weight_factor_number, self.hidden_dim*2))
        self.V_c = nn.Parameter(torch.rand(1, self.weight_factor_number, 1))
        self.eps = 1e-6
        # 定义layernorm
        self.layer_norm = nn.LayerNorm([self.total_seq_len, self.hidden_dim])
        self.layer_norm_1 = nn.LayerNorm([self.total_seq_len, self.hidden_dim*2])
        self.layer_norm_2 = nn.LayerNorm([self.total_seq_len, self.hidden_dim*2])

    def forward(self, input_data, average_reward, Action=None):
        hidden_cell_data = self.affine_reward(average_reward)
        Conv_data = self.pre_conv_layer(1e7*input_data).squeeze(1)
        Embeding_data = torch.sigmoid(self.affine_layer(Conv_data))
        End_vector = torch.zeros(self.batch_size, self.hidden_dim).to(self.device)
        # 得到一个batch_size * seq_lem * hidden_dim
        Key_matrix = self.layer_norm(torch.cat([End_vector.unsqueeze(1), Embeding_data], 1))
        Extend_W_a = self.W_a.expand(self.batch_size, -1, -1)
        Extend_V_a = self.V_a.expand(self.batch_size, -1, -1)
        Extend_W_c = self.W_c.expand(self.batch_size, -1, -1)
        Extend_V_c = self.V_c.expand(self.batch_size, -1, -1)
        # 这个地方添加一个向量，得到终止用户的向量
        Mask = torch.ones(self.batch_size, self.total_seq_len).to(self.device)
        result = []
        schedule_result = []
        Input_GRU = torch.mean(Embeding_data, 1)
        for antenna_index in range(self.args.max_stream+1):
            # 第一步使用Decoder进行解码操作
            # print(hidden_cell_data.shape)
            Extend_h = hidden_cell_data.unsqueeze(1).expand(-1, self.total_seq_len, -1)
            Concatenate_embedding_and_h = self.layer_norm_1(torch.cat([Key_matrix, Extend_h], -1))
            Weight_vector_u = torch.bmm(Extend_V_a.transpose(1,2), torch.tanh(torch.bmm(Extend_W_a, Concatenate_embedding_and_h.transpose(1,2))))
            c_vector = torch.sum(Key_matrix*Weight_vector_u.transpose(1,2), 1)
            Extend_c = c_vector.unsqueeze(1).expand(-1, self.total_seq_len, -1)
            Concatenate_embeding_and_c = self.layer_norm_2(torch.cat([Key_matrix, Extend_c], -1))
            Weight_vector_u_tilde = torch.bmm(Extend_V_c.transpose(1,2), torch.tanh(torch.bmm(Extend_W_c, Concatenate_embeding_and_c.transpose(1,2)))).squeeze(1)
            prob_matrix = torch.softmax(Weight_vector_u_tilde * Mask, -1)

            if torch.is_tensor(Action):
                selected_user = Action[antenna_index].unsqueeze(0)
                prob_vector = torch.log(prob_matrix[0, selected_user])
            else:
                if not self.args.Training:
                    random_weight_matrix = self.args.epsilon * torch.randn(self.batch_size, self.total_seq_len).to(self.device) * Mask + prob_matrix * (1-self.args.epsilon)
                    random_prob_matrix = torch.softmax(random_weight_matrix, -1)
                    
                    sorted_index = torch.argsort(-random_prob_matrix).squeeze()
                    for index in sorted_index:
                        selected_user = index.unsqueeze(0)
                        # print(selected_user)
                        # if selected_user.item()-1 not in schedule_result:
                        if selected_user.item() != 0 and selected_user.item() not in schedule_result:
                            # 不是调度0号玩家，并且这个调度玩家不在调度序列里面
                            break
                        else:
                            if selected_user.item() == 0 and schedule_result:
                                break
                            # dist = Categorical(prob_matrix)
                    prob_vector = torch.log(random_prob_matrix[0, selected_user])
                else:
                    if antenna_index == self.args.max_stream:
                        selected_user = torch.LongTensor([0]).to(self.device)
                        prob_vector = torch.log(self.eps+prob_matrix[0, selected_user])
                    else:
                        if torch.rand(1) < self.args.epsilon:
                            random_weight_matrix = self.args.epsilon * torch.randn(self.batch_size, self.total_seq_len).to(self.device) * Mask + prob_matrix * (1-self.args.epsilon)
                            random_prob_matrix = torch.softmax(random_weight_matrix, -1)
                            sorted_index = torch.argsort(-random_prob_matrix).squeeze()
                            for index in sorted_index:
                                selected_user = index.unsqueeze(0)
                                if selected_user.item() != 0 and selected_user.item() not in schedule_result:
                                    break
                                else:
                                    if selected_user.item() == 0 and schedule_result:
                                        break
                            prob_vector = torch.log(random_prob_matrix[0, selected_user])
                        else:
                            sorted_index = torch.argsort(-prob_matrix).squeeze()
                            for index in sorted_index:
                                selected_user = index.unsqueeze(0)
                                if selected_user.item() != 0 and selected_user.item() not in schedule_result:
                                    break
                                else:
                                    if selected_user.item() == 0 and schedule_result:
                                        break
                            prob_vector = torch.log(prob_matrix[0, selected_user])
            Mask = Mask.clone()  
            Mask.scatter_(1, selected_user.unsqueeze(1), self.eps)
            result.append(prob_vector)
            if selected_user.item() == 0:
                break
            schedule_result.append(selected_user.item()-1)
            # 这个地方调用GRUcell更新RNN网络中的参数
            hidden_cell_data = self.GRU(Input_GRU, hidden_cell_data)
            Input_GRU = Embeding_data[:, selected_user-1, :].squeeze(0)
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
        conv_layer = []
        self.pre_conv_layer = nn.Conv2d(in_channel, self.args.n_agents, self.pre_kernel_size, self.pre_stride, self.pre_padding)
        in_channel = self.args.n_agents
        for layer in range(self.conv_layer_number):
            conv_layer.append(nn.Conv2d(in_channel, self.kernal_number[layer], self.kernal_size[layer], self.kernal_stride[layer], self.padding_size[layer], self.dilation[layer]))
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
        argumented_information = user_instant_reward.unsqueeze(-1).expand(-1,-1,-1,self.feature_number)
        preprocess_input_channel = torch.cat([1e6 * channel, argumented_information],1)
        pre_conv_channel = torch.relu(self.pre_conv_layer(preprocess_input_channel))
        conv_result = torch.relu(self.ascend(pre_conv_channel))
        for layer in range(self.conv_layer_number):
            conv_result = torch.relu(self.conv_layer[layer](conv_result))
        flatten_result = self.flatten(conv_result)
        fc_result = torch.relu(self.layer1(flatten_result))
        V_value = self.output_layer(fc_result)
        return V_value


        