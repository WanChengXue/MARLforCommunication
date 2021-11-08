
# 这个函数用来建立基本的模型
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.distributions import Categorical
from Tool import utils
class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.args = args
        self.device = "cuda" if self.args.cuda else "cpu"
        # 定义原始信道矩阵的列数
        self.feature_number = self.args.obs_dim2
        self.drop_out = self.args.drop_out
        self.rnn_input_dim = self.args.rnn_input_dim
        self.hidden_dim = self.args.rnn_hidden
        self.weight_dim = self.args.weight_dim
        self.embbeding_layer = nn.Embedding(2, self.rnn_input_dim)
        # 其中第一个向量表示的是
        self.linear_layer = nn.Linear(self.feature_number, self.rnn_input_dim)
        self.Encoder = nn.GRU(self.rnn_input_dim, self.hidden_dim, batch_first=True)
        self.Decoder = nn.GRU(self.rnn_input_dim, self.hidden_dim, batch_first=True)
        # define key matrix W_k and query matrix W_q
        self.W_k = Parameter(torch.randn(1, self.weight_dim, self.hidden_dim))
        self.W_q = Parameter(torch.randn(1, self.weight_dim, self.hidden_dim))
        # define weight vector
        self.Weight_vector = Parameter(torch.randn(1, self.weight_dim, 1))
        # define drop out rate
        self.drop_rnn_out = nn.Dropout(self.drop_out)
        self.drop_rnn_hidden = nn.Dropout(self.drop_out)
        self.eps = 1e-12

    def forward(self, channel_matrix , priority_vector=None, Action=None):
        # 输入的channel_matrix的维度是batch_size * 20 * 32
        # 信道矩阵输入的维度是1*9*用户数目*32， user_instant_reward的维度是用户数目 *1 的一个向量
        batch_size = channel_matrix.shape[0]
        seq_len = channel_matrix.shape[1]
        total_len = seq_len + 1  
        bos_token_encoder = [[0]] * batch_size
        bos_token_decoder = [[1]] * batch_size
        # 维度是batch_size * 1 * flatten_dim
        bos_vector = self.embbeding_layer(torch.LongTensor(bos_token_encoder).to(self.device))
        deocder_bos_vector = self.embbeding_layer(torch.LongTensor(bos_token_decoder).to(self.device))
        # 先过一个线性层,升维
        linear_transformation = torch.tanh(self.linear_layer(channel_matrix * 1e6))
        Encoder_linear_data =  torch.relu(linear_transformation)
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
        batch_sheduling_result = -1*torch.ones(batch_size,1).to(self.device)
        batch_prob_result = 0
        selected_mask = torch.zeros(batch_size, self.args.max_stream+1).to(self.device)
        beam = [(batch_sheduling_result, batch_prob_result, selected_mask, mask, Input_decoder, Decoder_hidden_vector)]
        for antenna_index in range(self.args.max_stream+1):
            # 第一步使用Decoder进行解码操作
            beam_candinate = []
            joint_prob_list = []
            for beam_element in beam:
                Decoder_output_vector, Decoder_hidden_vector = self.Decoder(beam_element[-2], beam_element[-1]) 
                # Decoder_output_vector = self.drop_rnn_out(Decoder_output_vector)
                # Decoder_hidden_vector = self.drop_rnn_hidden(Decoder_hidden_vector)
                Query_vector = torch.bmm(Extend_query, Decoder_output_vector.permute(0,2,1))
                # 计算相似矩阵 (batch_size * 32 * 21)
                Similar_matrix = torch.tanh(Key_matrix + Query_vector)
                # 计算权重向量 batch_size * 1 * (word_num + 1)
                Weight_vector = torch.bmm(Extend_weight_vector.permute(0,2,1), Similar_matrix).squeeze(1)
                Weight_vector = Weight_vector - 1e7 * beam_element[3]
                prob_vector = torch.softmax(Weight_vector, -1)
                dist = Categorical(prob_vector)
                if self.args.Training:
                    if self.args.random_steps < self.args.warm_start:
                        # 首先需要找出来没有被选择过的UE,然后随机挑选
                        beam_candinate_index = []
                        beam_candinate_prob = []
                        for _ in range(self.args.beam_size):
                            scheduling_user  = torch.LongTensor(utils.random_sample(mask)).to(self.device)
                            scheduling_prob = torch.exp(dist.log_prob(scheduling_user))
                            beam_candinate_index.append(scheduling_user.unsqueeze(-1))
                            beam_candinate_prob.append(scheduling_prob.unsqueeze(-1))
                        beam_candinate_index = torch.cat(beam_candinate_index, -1)
                        beam_candinate_prob = torch.cat(beam_candinate_prob, -1)
                    else:
                        if self.args.decoder_sampling == 0:
                            beam_candinate_prob, beam_candinate_index = torch.topk(prob_vector, self.args.beam_size)
                        else:
                            beam_candinate_index = torch.multinomial(prob_vector, self.args.beam_size)
                            beam_candinate_prob = torch.gather(prob_vector, dim=-1, index=beam_candinate_index)
                else:
                    beam_candinate_prob, beam_candinate_index = torch.topk(prob_vector, self.args.beam_size)
                # 如果说这个antenna_index的值达到了最大的stream,然后,就需要修改两个地方
                terminal_flag = beam_element[0][:,-1] == 0
                # 如果说,当前这个beam中,有些batch的调度结果为0,就是说上一次调度过程中,这个batch就是直接终止了,因此,不需要对这个batch进行再次的拓展
                for beam_index in range(self.args.beam_size):      
                    beam_candinate_index[:,beam_index][terminal_flag] = 0
                    # beam_candinate_prob[:,beam_index][terminal_flag.squeeze(-1)] = 1
                    if antenna_index == self.args.max_stream:
                        # 这个地方的意思是说,如果遍历到了最后一次决策,那么所有的决策结果都是0,并且将这个决策概率也对应上
                        beam_candinate_index[:,beam_index] = 0

                    mask = beam_element[3].clone()
                    mask.scatter_(1, beam_candinate_index[:,beam_index].unsqueeze(1), 1)
                    # ============================================================================================================== #
                    # 这个selected mask是要和最后prob矩阵进行乘法操作,如果说这次选择有的batch是选一个新用户,对应位置就是1,如果上个时刻这个batch是选择-
                    # 0,则表明这个batch已经终止了,因此这里就直接变成0
                    selected_mask = beam_element[2].clone()
                    # 这里使用了torch.logical_not函数,传入一个布尔向量,然后取反,由于terminal_flag表示,只有这个batch中出现了0才为True,否则为False
                    # 因此这个地方的意思是说,将那些没有终止的点,所对应的selected_mask的值设置为1,这里是因为取了索引操作,没有什么问题
                    selected_mask[:,antenna_index][torch.logical_not(terminal_flag)] = 1
                    # 接下来的几行表示,首先将这个当前beam的用户调度向量与root node的用户调度集合进行合并, 然后是修改对应的概率
                    batch_sheduling_result = torch.cat([beam_element[0], beam_candinate_index[:,beam_index].unsqueeze(1)], -1)
                    # 这样一来,对于那些已经是terminate的beam而言, 其拓展出来的第一个beam,就保留这个概率,其余的beam就将这个概率缩减1e12
                    if beam_index == 0:
                        batch_prob_result = beam_element[1] + (torch.logical_not(terminal_flag)*torch.log(beam_candinate_prob[:,beam_index])).unsqueeze(-1)
                    else:
                        batch_prob_result = beam_element[1] + torch.log(beam_candinate_prob[:,beam_index]).unsqueeze(1) - terminal_flag.unsqueeze(-1) * 1e12
                    # 接下来生成一个selected index矩阵,用来提取数据输入到下一次的PN网络中
                    selected_index = torch.zeros(batch_size, total_len).to(self.device).bool()
                    selected_index.scatter_(1, beam_candinate_index[:,beam_index].unsqueeze(-1), True)  
                    # Input_decoder是下一次送入网络的的输入值
                    Input_decoder = Input_encoder[selected_index].unsqueeze(1)
                    beam_candinate.append((batch_sheduling_result, batch_prob_result, selected_mask, mask, Input_decoder, Decoder_hidden_vector))
                    joint_prob_list.append(batch_prob_result)
                # 这个地方我需要根据这个beam_candinate列表中的数据, 进行一次排序操作
            joint_prob = torch.cat(joint_prob_list, -1)
            topk_prob, topk_index = torch.topk(joint_prob, self.args.beam_size)
            # 根据topk_index将beam_candinate进行选择,修改之后得到输入
            beam = []
            for beam_index in range(self.args.beam_size):
                single_point_scheduling_result = []
                single_prob_result = []
                single_selected_mask = []
                single_mask = []
                single_input_decoder = []
                single_decoder_hidden_vecotor = []
                for batch in range(batch_size):
                    active_single_point = beam_candinate[topk_index[batch, beam_index]]
                    single_point_scheduling_result.append(active_single_point[0][batch, :].unsqueeze(0))
                    single_prob_result.append(active_single_point[1][batch, :].unsqueeze(0))
                    single_selected_mask.append(active_single_point[2][batch, :].unsqueeze(0))
                    single_mask.append(active_single_point[3][batch, :].unsqueeze(0))
                    single_input_decoder.append(active_single_point[4][batch, :,:].unsqueeze(0))
                    single_decoder_hidden_vecotor.append(active_single_point[5][:, batch, :].unsqueeze(0))
                beam.append((torch.cat(single_point_scheduling_result, 0), torch.cat(single_prob_result, 0), torch.cat(single_selected_mask, 0), torch.cat(single_mask, 0), torch.cat(single_input_decoder, 0), torch.cat(single_decoder_hidden_vecotor, 1)))
        # 这个beam[0][0]表示的是调度结果, beam[0][1]表示的是联合概率向量,beam[0][3]其实用不太到,因为这个是batch_size * 17的一个列表, 最后是一个batch_size * 
        return beam[0][1], beam[0][3]
            

class Critic(nn.Module):
    def __init__(self, args, input_shape):
        super(Critic, self).__init__()
        self.args = args
        # 由于就是一个linear mapping,因此这里就只要叠几层神经网络就可以了
        self.linear_layer = nn.Linear(self.args.obs_dim2, self.args.fc_dim)
        self.linear_layer_2 = nn.Linear(self.args.fc_dim, self.args.fc_dim)
        self.flatten_layer = nn.Flatten()
        flatten_dim = self.output_dim(input_shape)
        self.linear_layer_3 = nn.Linear(flatten_dim, self.args.fc_dim)
        self.activate_function = nn.ELU()
        self.output_layer = nn.Linear(self.args.fc_dim, 1)

    def output_dim(self, input_shape):
        test_input = torch.zeros(input_shape)
        forward_1 = torch.relu(self.linear_layer(test_input))
        forward_2 = self.linear_layer_2(forward_1)
        flatten_data = self.flatten_layer(forward_2)
        return flatten_data.shape[1]

    def forward(self, channel):
        # 输入的是一个batch size * user_Numbers * transmit antenna的矩阵
        tensor_input = 1e6*channel
        first_linear_layer = torch.relu(self.linear_layer(tensor_input))
        second_linear_layer = torch.relu(self.linear_layer_2(first_linear_layer))
        flatten_layer = self.flatten_layer(second_linear_layer)
        third_linear_layer = self.activate_function(self.linear_layer_3(flatten_layer))
        output_layer = self.output_layer(third_linear_layer)
        return output_layer