# 这个函数用来建立基本的模型
import torch
import torch.nn as nn

from torch.nn.parameter import Parameter
from torch.distributions import Categorical
from Tool import utils
from Model.resnet import resnet_34
class Actor_Critic(nn.Module):
    def __init__(self, args, input_shape):
        super(Actor_Critic, self).__init__()
        self.args = args
        self.device = "cuda" if self.args.cuda else "cpu"
        # 定义原始信道矩阵的列数
        self.feature_number = self.args.obs_dim2
        self.drop_out = self.args.drop_out
        self.rnn_input_dim = self.args.rnn_input_dim
        self.hidden_dim = self.args.rnn_hidden
        self.weight_dim = self.args.weight_dim
        self.pre_conv_layer = nn.Conv2d(self.args.obs_matrix_number, 3, kernel_size=3, stride=1, padding=1)
        self.final_conv_layer = nn.Conv2d(512, self.args.obs_dim1, kernel_size=3, stride=1, padding=1)
        self.embbeding_layer = nn.Embedding(2, self.rnn_input_dim)
        # 其中第一个向量表示的是
        self.Encoder = nn.GRU(self.rnn_input_dim, self.hidden_dim, batch_first=True)
        self.Decoder = nn.GRU(self.rnn_input_dim, self.hidden_dim, batch_first=True)
        # define key matrix W_k and query matrix W_q
        self.W_k = Parameter(torch.randn(1, self.weight_dim, self.hidden_dim))
        self.W_q = Parameter(torch.randn(1, self.weight_dim, self.hidden_dim))
        # define weight vector
        self.Weight_vector = Parameter(torch.randn(1, self.weight_dim, 1))
        self.eps = 1e-12
        self.calculate_pad(input_shape)
        self.average_pool = nn.AdaptiveAvgPool2d((1,1))
        self.resnet = resnet_34()
        self.value_head_layer = nn.Linear(512, 1)
        self.policy_head_layer = nn.Linear(49, self.rnn_input_dim)


    def calculate_pad(self, input_shape):
        hight = input_shape[2]
        width = input_shape[3]
        up_pad = 0
        down_pad = 224 - hight
        left_pad = 0
        right_Pad = 224 - width
        self.pad = nn.ZeroPad2d(padding=(left_pad,right_Pad,up_pad, down_pad))

    def forward(self, channel_matrix):
        # 九个矩阵全部拿过来进行卷积操作
        batch_size = channel_matrix.shape[0]
        seq_len = channel_matrix.shape[2]
        total_len = seq_len + 1  
        bos_token_encoder = [[0]] * batch_size
        bos_token_decoder = [[1]] * batch_size
        # 首先预卷积变成一个三通道的tensor,然后pad成为224*224的tensor
        prev_conv_data = self.pre_conv_layer(channel_matrix * 1e6)
        pad_data = self.pad(prev_conv_data)
        bos_vector = self.embbeding_layer(torch.LongTensor(bos_token_encoder).to(self.device))
        deocder_bos_vector = self.embbeding_layer(torch.LongTensor(bos_token_decoder).to(self.device))
        # 通过ResNet32,得到一个bs*512*7*7的tensor
        feature_map = self.resnet(pad_data)
        average_pool_data = self.average_pool(feature_map)
        flatten_for_value_head = torch.flatten(average_pool_data, 1)
        # 这个value的维度是batchsize * 1
        value = self.value_head_layer(flatten_for_value_head)
        # 再次通过一个卷积,将这个batch size * 512 * 7*7的数据变成一个batch size * 20 * 7*7的feature map
        input_for_policy = torch.relu(self.final_conv_layer(feature_map))
        # 通过一次flatten操作变成batch size * 20 * 49
        flatten_for_policy_head = torch.flatten(input_for_policy,2)
        # 通过一次linear transformation
        Encoder_linear_data = self.policy_head_layer(flatten_for_policy_head)
        Input_encoder = torch.cat([bos_vector, Encoder_linear_data], 1)
        Init_encoder = torch.zeros(1, batch_size, self.hidden_dim).to(self.device)
        # 计算key矩阵,也就是Encoder得到的隐状态矩阵,维度是batch_size * (1+antenna_numbers) * hidden_dim
        Encoder_hidden_result, Encoder_hidden_vector = self.Encoder(Input_encoder, Init_encoder)
        # Extend key matrix, query matrix, weight vector
        Extend_key = self.W_k.repeat(batch_size, 1, 1)
        Extend_query = self.W_q.repeat(batch_size, 1, 1)
        Extend_weight_vector = self.Weight_vector.repeat(batch_size, 1, 1)
        # 计算key matrix, 通过一层affine transformation batch_size * weight_dim * total_seq 
        Key_matrix = torch.bmm(Extend_key, Encoder_hidden_result.permute(0,2,1))
        # 初始化Decoder的输入
        Input_decoder = deocder_bos_vector
        # 初始化Decoder的隐状态输入 1*batch_size * hidden_dim
        Decoder_hidden_vector = Encoder_hidden_vector
        # 定义mask,将已经出现过的用户的概率设置为0
        mask = torch.zeros(batch_size, total_len).to(self.device)
        batch_sheduling_result = [-1*torch.ones(batch_size).to(self.device)]
        batch_prob_result = []
        selected_mask = torch.zeros(batch_size, self.args.max_stream+1).to(self.device)
        for antenna_index in range(self.args.max_stream+1):
            # 第一步使用Decoder进行解码操作
            Decoder_output_vector, Decoder_hidden_vector = self.Decoder(Input_decoder, Decoder_hidden_vector) 
            Query_vector = torch.bmm(Extend_query, Decoder_output_vector.permute(0,2,1))
            # 计算相似矩阵 (batch_size * 32 * 21)
            Similar_matrix = torch.tanh(Key_matrix + Query_vector)
            # 计算权重向量 batch_size * 1 * (word_num + 1)
            Weight_vector = torch.relu(torch.bmm(Extend_weight_vector.permute(0,2,1), Similar_matrix)).squeeze(1)
            Weight_vector = Weight_vector - 1e7 * mask
            # 这个prob_vector的维度是batch_size * 21
            prob_vector = torch.softmax(Weight_vector, -1)
            dist = Categorical(prob_vector)
            if self.args.Training:
                if self.args.random_steps < self.args.warm_start:
                    # 首先需要找出来没有被选择过的UE,然后随机挑选
                    sheduling_user = torch.LongTensor(utils.random_sample(mask)).to(self.device)
                else:
                    sheduling_user = dist.sample()
            else:
                sheduling_user = dist.sample()
            terminal_flag = batch_sheduling_result[-1] == 0
            sheduling_user[terminal_flag] = 0
            if antenna_index == self.args.max_stream:
                sheduling_user[:] = 0
            # 将Mask中的某一些值变成1
            mask.scatter_(1, sheduling_user.unsqueeze(1), 1)
            selected_mask[:,antenna_index][torch.logical_not(terminal_flag)] = 1
            batch_sheduling_result.append(sheduling_user)
            batch_prob_result.append(dist.log_prob(sheduling_user).unsqueeze(1))
            selected_index = torch.zeros(batch_size, total_len).to(self.device).bool()
            selected_index.scatter_(1, sheduling_user.unsqueeze(-1), True)  
            Input_decoder = Input_encoder[selected_index].unsqueeze(1)
        return  torch.sum(torch.cat(batch_prob_result, -1) * selected_mask, -1).unsqueeze(-1), mask, value


