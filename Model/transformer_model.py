import math
from math import sqrt
import torch
import torch.nn as nn
import torchvision

nn.Transformer

class Single_head_attention(nn.Module):
    def __init__(self, args):
        super(Single_head_attention, self).__init__()
        self.args = args
        self.input_dim_column = self.args.hidden_dim
        self.sub_query_dim = self.args.sub_query_dim
        self.sub_key_dim = self.args.sub_key_dim
        self.sub_value_dim = self.args.sub_value_dim
        self.scale_factor = sqrt(self.sub_key_dim)
        # 三个affine layer，将输入进行编码
        self.sub_query_layer = nn.Linear(self.input_dim_column, self.sub_query_dim)
        self.sub_key_layer = nn.Linear(self.input_dim_column, self.sub_key_dim)
        self.sub_value_layer = nn.Linear(self.input_dim_column, self.sub_value_dim)


    def forward(self, query, key, value, mask=None):
        sub_query = self.sub_query_layer(query)
        sub_key = self.sub_key_layer(key)
        sub_value = self.sub_value_layer(value)
        matmul_value = torch.bmm(sub_query, sub_key.transpose(1,2))
        scale_value = matmul_value / self.scale_factor
        if mask:
            mask_out = scale_value * mask
        else:
            mask_out = scale_value
        softmax_value = torch.softmax(mask_out, -1)
        attention_value = torch.bmm(softmax_value, sub_value)
        return attention_value


class Multi_head_attention(nn.Module):
    def __init__(self, args):
        super(Multi_head_attention, self).__init__()
        self.args = args
        self.device = "cuda" if self.args.cuda else "cpu"
        self.head_nubmer = self.args.head_number
        self.input_dim_column = self.args.hidden_dim
        self.head_dict = {}
        for head_index in range(self.head_nubmer):
            self.head_dict['head%s'%head_index] = Single_head_attention(self.args).to(self.device)
        # 将所有head的输出结果通过一次affine layer
        self.affine_layer = nn.Linear(self.input_dim_column, self.input_dim_column)

    def forward(self, Query, Key, Value):
        sub_attention_value = []
        for head_index in range(self.head_nubmer):
            sub_attention_value.append(self.head_dict['head%s'%head_index](Query, Key, Value))
        MHA_output = torch.cat(sub_attention_value, -1)
        return MHA_output




class RFF(nn.Module):
    def __init__(self, args):
        super(RFF, self).__init__()
        self.args = args
        self.input_dim_column = self.args.hidden_dim
        self.linear_1 = nn.Linear(self.input_dim_column, self.input_dim_column)
        self.linear_2 = nn.Linear(self.input_dim_column, self.input_dim_column)
        self.linear_3 = nn.Linear(self.input_dim_column, self.input_dim_column)

    def forward(self, input_data):
        hidden_1 = torch.tanh(self.linear_1(input_data))
        hidden_2 = torch.tanh(self.linear_2(hidden_1))
        RFF_result = torch.tanh(self.linear_3(hidden_2))
        return RFF_result


class SAB_block(nn.Module):
    def __init__(self, args):
        super(SAB_block, self).__init__()
        self.args = args
        self.input_dim_row = self.args.total_user_antennas
        self.input_dim_column = self.args.hidden_dim
        self.MHA = Multi_head_attention(self.args)
        self.RFF = RFF(self.args)
        self.layer_norm_1 = nn.LayerNorm([self.input_dim_row, self.input_dim_column])
        self.layer_norm_2 = nn.LayerNorm([self.input_dim_row, self.input_dim_column])
    def forward(self, input_data):
        Query = input_data.clone()
        Key = input_data.clone()
        Value = input_data.clone()
        H = self.layer_norm_1(input_data + self.MHA(Query, Key, Value))
        MAB_result = self.layer_norm_2(H + self.RFF(H))
        return MAB_result
        

class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        self.args = args
        self.SAB_1 = SAB_block(self.args)
        self.SAB_2 = SAB_block(self.args)
        # 定义输入维度
        self.input_dim_row = self.args.total_user_antennas
        self.input_dim_column = self.args.total_bs_antennas
        self.input_dim_channel = self.args.obs_matrix_number
        self.hidden_dim = self.args.hidden_dim
        # self.query_dim = self.args.query_dim
        # self.key_dim = self.args.key_dim
        # self.value_dim = self.args.value_dim
        # 使用一个卷积核对进行2维卷积，得到一个单channel的feature map
        self.kernel_width = self.args.kernel_width
        self.kernel_height = self.args.kernel_height
        self.pre_conv_layer = nn.Conv2d(self.input_dim_channel, 1, (self.kernel_height, self.kernel_width))
        self.affline_layer = nn.Linear(self.input_dim_column, self.hidden_dim)
        # 定义三个affine layer用来计算query key value
        # self.Query_layer = nn.Linear(self.input_dim_column, self.query_dim)
        # self.Key_layer = nn.Linear(self.input_dim_column, self.key_dim)
        # self.Value_layer = nn.Linear(self.input_dim_column, self.value_dim)

    def forward(self, input_data):
        conv_data = self.pre_conv_layer(1e7*input_data).squeeze(1)
        # Query = self.Query_layer(conv_data)
        # Key = self.Key_layer(conv_data)
        # Value = self.Value_layer(conv_data)
        hidden_data = torch.tanh(self.affline_layer(conv_data))
        SAB_hidden = self.SAB_1(hidden_data)
        encoder_result = self.SAB_2(SAB_hidden)
        return encoder_result

  
class Decoder(nn.Module):
    def __init__(self, args):
        super(Decoder, self).__init__()
        self.args = args
        self.hidden_dim = self.args.hidden_dim
        self.batch_size = self.args.batch_size
        self.Encoder = Encoder(self.args)
        self.Query_layer = nn.Linear(self.hidden_dim, 2)
    
    def forward(self, input_data):
        Encoder_matrix = self.Encoder(input_data)
        weight_matrix = self.Query_layer(Encoder_matrix)
        prob_matrix = torch.softmax(weight_matrix, -1)
        return prob_matrix


class Transformer(nn.Module):
    def __init__(self, args):
        super(Transformer, self).__init__()
        self.args = args
        # 定义输入维度
        self.input_dim_row = self.args.total_user_antennas
        self.input_dim_column = self.args.total_bs_antennas
        self.input_dim_channel = self.args.obs_matrix_number
        self.hidden_dim = self.args.hidden_dim
        self.kernel_width = self.args.kernel_width
        self.kernel_height = self.args.kernel_height
        self.head_nubmer = self.args.head_number
        self.stack_layer = self.args.stack_layer
        self.pre_conv_layer = nn.Conv2d(self.input_dim_channel, 1, (self.kernel_height, self.kernel_width))
        self.affline_layer = nn.Linear(self.input_dim_column, self.hidden_dim)
        self.decoder_layer = nn.TransformerEncoderLayer(self.hidden_dim, self.head_nubmer)
        self.transformer_decoder = nn.TransformerEncoder(self.decoder_layer, self.stack_layer)
        self.Query_layer = nn.Linear(self.hidden_dim, 2)

    def forward(self, input_data):
        conv_data = self.pre_conv_layer(1e7*input_data).squeeze(1)
        hidden_data = torch.relu(self.affline_layer(conv_data)).transpose(0,1)
        Encoder_matrix = self.transformer_decoder(hidden_data).transpose(0,1)
        weight_matrix = self.Query_layer(Encoder_matrix)
        prob_matrix = torch.softmax(weight_matrix, -1)
        return prob_matrix

class Critic(nn.Module):
    def __init__(self, args):
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
        in_channel = self.args.state_matrix_number
        # self.reduction = nn.Linear(self.args.state_dim2, self.args.embedding_dim)
        self.ascend = nn.Linear(self.args.state_dim2, self.args.state_dim2)
        H_in = self.args.state_dim1
        W_in = self.args.state_dim2
        self.device = "cuda" if self.args.cuda else "cpu"
        self.conv_layer = []
        self.pre_conv_layer = nn.Conv2d(in_channel, self.args.cell_number, self.pre_kernel_size, self.pre_stride, self.pre_padding)
        in_channel = self.args.cell_number
        for layer in range(self.conv_layer_number):
            self.conv_layer.append(nn.Conv2d(in_channel, self.kernal_number[layer], self.kernal_size[layer], self.kernal_stride[layer], self.padding_size[layer], self.dilation[layer]).to(self.device))
            H_out = math.floor((H_in+2*self.padding_size[layer][0]-self.dilation[layer][0]*(self.kernal_size[layer][0]-1)-1)/self.kernal_stride[layer][0]+1)
            W_out = math.floor((W_in+2*self.padding_size[layer][1]-self.dilation[layer][1]*(self.kernal_size[layer][1]-1)-1)/self.kernal_stride[layer][1] + 1)
            H_in = H_out
            W_in = W_out
            in_channel = self.kernal_number[layer]
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(self.kernal_number[layer]*H_out*W_out, self.args.fc_dim)
        self.output_layer = nn.Linear(self.args.fc_dim, 1)

    def forward(self, channel):
        pre_conv_channel = torch.relu(self.pre_conv_layer(1e6 * channel))
        conv_result = torch.relu(self.ascend(pre_conv_channel))
        for layer in range(self.conv_layer_number):
            conv_result = torch.relu(self.conv_layer[layer](conv_result))
        flatten_result = self.flatten(conv_result)
        fc_result = torch.relu(self.layer1(flatten_result))
        V_value = self.output_layer(fc_result)
        return V_value
