# 这个函数用来建立基本的模型
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, args, input_shape):
        super(Model, self).__init__()
        self.args = args
        # self.device = "cuda" if self.args.cuda else "cpu"
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
        batch_norm_layer = []
        for layer in range(self.conv_layer_number):
            batch_norm_layer.append(nn.BatchNorm2d(in_channel))
            conv_layer.append(nn.Conv2d(in_channel, self.kernal_number[layer], self.kernal_size[layer], self.kernal_stride[layer], self.padding_size[layer], self.dilation[layer]))
            in_channel = self.kernal_number[layer]
            
        self.conv_layer = nn.ModuleList(conv_layer)
        self.BN_layer = nn.ModuleList(batch_norm_layer)
        self.dropout_layer = nn.Dropout(p=self.args.drop_out)
        self.flatten = nn.Flatten()
        conv_output_dim = self.output_dimension(input_shape)
        self.linear_layer = nn.Linear(conv_output_dim, self.args.fc_dim)
        self.action_linear_layer = nn.Linear(self.args.n_agents*self.args.obs_dim1, self.args.fc_dim)
        self.affine_middle_layer = nn.Linear(2*self.args.fc_dim, self.args.fc_dim)
        self.affine_activate_fn = nn.ELU()
        self.output_layer = nn.Linear(self.args.fc_dim, 1)
        # 需要对batch_size * agent_nums * antennas进行处理,得到一个batch_size * feature_dim
        
    def output_dimension(self,input_shape):
        test_channel = torch.rand(*input_shape)
        pre_conv_channel = torch.relu(self.pre_conv_layer(test_channel))
        conv_result = torch.relu(self.ascend(pre_conv_channel))
        for layer in range(self.conv_layer_number):
            conv_result = self.conv_layer[layer](conv_result)
        flatten_result = self.flatten(conv_result)
        return flatten_result.shape[-1]


    def forward(self, channel, action):
        # 输入的channel是一个81*10*32的信道矩阵，use_instant_reward是一个9*10*1的矩阵
        # action是一个长度为batch_size * agent numbet * user antennas * 1的一个向量
        pre_conv_channel = torch.relu(self.pre_conv_layer(1e7*channel))
        conv_result = torch.relu(self.ascend(pre_conv_channel))
        for layer in range(self.conv_layer_number):
            conv_result = self.dropout_layer(torch.tanh(self.conv_layer[layer](self.BN_layer[layer](conv_result))))
        flatten_result = self.flatten(conv_result)
        state_head = self.linear_layer(flatten_result)
        flatten_action = self.flatten(action)
        action_head = self.action_linear_layer(flatten_action)
        argumented_input = torch.cat([state_head, action_head], -1)
        fc_result = self.affine_activate_fn(self.affine_middle_layer(argumented_input))
        V_value = torch.relu(self.output_layer(fc_result))
        return V_value


