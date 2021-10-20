import torch
import torch.nn as nn
import math

class Actor(nn.Module):
    def __init__(self, args, input_shape):
        super(Actor, self).__init__()
        self.args = args
        # 定义原始信道矩阵的列数
        self.input_channel = self.args.total_obs_matrix_number - 1
        self.output_channel = self.args.obs_dim1
        self.kernel_size = self.args.actor_kernel_size
        self.stride = self.args.actor_stride
        self.maxpool_kernel_size = self.args.actor_maxpool_kernel_size

        self.Encoder_conv_layer = nn.Conv2d(self.input_channel, self.output_channel, self.kernel_size, self.stride)
        self.Encoder_maxpool_layer = nn.MaxPool2d(self.maxpool_kernel_size, self.stride)
        self.Encoder_flatten = nn.Flatten(start_dim=1)
        self.flatten_dim = self.output_dimension(input_shape)
        self.Encoder_affine_layer = nn.Linear(self.flatten_dim, self.output_channel)

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
        action =  torch.sigmoid(self.Encoder_affine_layer(Encoder_flatten_channel))
        return action
        

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
        self.pre_conv_layer = nn.Conv2d(in_channel, self.args.cell_number, self.pre_kernel_size, self.pre_stride, self.pre_padding)
        in_channel = self.args.cell_number
        conv_layer = []
        for layer in range(self.conv_layer_number):
            conv_layer.append(nn.Conv2d(in_channel, self.kernal_number[layer], self.kernal_size[layer], self.kernal_stride[layer], self.padding_size[layer], self.dilation[layer]))
            in_channel = self.kernal_number[layer]
        self.conv_layer = nn.ModuleList(conv_layer)
        self.flatten = nn.Flatten()
        conv_output_dim = self.output_dimension(input_shape)
        self.layer1 = nn.Linear(conv_output_dim, self.args.fc_dim)
        self.output_layer = nn.Linear(self.args.fc_dim, 1)

    def output_dimension(self,input_shape):
        test = torch.rand(*input_shape)
        Encoder_conv_channel = self.Encoder_conv_layer(test)
        Encoder_maxpool_channel = self.Encoder_maxpool_layer(Encoder_conv_channel)
        Encoder_flatten_channel = self.Encoder_flatten(Encoder_maxpool_channel)
        return Encoder_flatten_channel.shape[-1]


    def forward(self, channel):
        # 输入的channel是一个81*10*32的信道矩阵，use_instant_reward是一个9*10*1的矩阵
        # argumented_information = user_instant_reward.unsqueeze(-1).repeat(1,1,1,self.feature_number)
        pre_conv_channel = torch.relu(self.pre_conv_layer(1e7 *channel))
        conv_result = torch.relu(self.ascend(pre_conv_channel))
        for layer in range(self.conv_layer_number):
            conv_result = torch.relu(self.conv_layer[layer](conv_result))
        flatten_result = self.flatten(conv_result)
        fc_result = torch.relu(self.layer1(flatten_result))
        V_value = self.output_layer(fc_result)
        return V_value

