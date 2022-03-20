import torch.nn as nn
import torch
from torch.nn.modules.normalization import LayerNorm
from torch.distributions import Categorical
import sys
import os
current_path = os.path.abspath(__file__)
root_path = '/'.join(current_path.split('/')[:-2])
sys.path.append(root_path)
from Model.popart import PopArt


class transformer_model(nn.Module):
    def __init__(self, policy_config):
        super(transformer_model, self).__init__()
        # 初始化一个encoder
        self.policy_config = policy_config
        self.d_model = policy_config.get('d_model', 512)
        self.nhead = policy_config.get('nhead', 8)
        self.dim_feedforward = policy_config.get('dim_feedforward', 2048)
        self.dropout = policy_config.get('dropout', 0.0)
        self.activation = policy_config.get('activation', torch.functional.F.relu)
        self.batch_first = True
        self.layer_norm_eps = policy_config.get('layer_norm_eps',  1e-5)
        # 定义encoder的层数
        self.num_encoder_layers = policy_config.get('num_encoder_layers', 6)
        transformer_encoder_layer = nn.TransformerEncoderLayer(self.d_model, self.nhead, self.dim_feedforward, self.dropout, self.activation, batch_first = self.batch_first)
        encoder_norm = LayerNorm(self.d_model, eps=self.layer_norm_eps)
        self.transformer_encoder = nn.TransformerEncoder(transformer_encoder_layer, self.num_encoder_layers, encoder_norm)
        # 定义解码器部分，这个地方是一个Pointer Network

    def forward(self, src):
        memory = self.transformer_encoder(src)
        return memory


class transformer_pointer_network_decoder(nn.Module):
    def __init__(self, policy_config):
        super(transformer_pointer_network_decoder, self).__init__()
        self.policy_config = policy_config
        self.d_model = policy_config.get('d_model', 512)
        self.nhead = policy_config.get('nhead', 8)
        self.dim_feedforward = policy_config.get('dim_feedforward', 2048)
        self.dropout = policy_config.get('dropout', 0.0)
        self.activation = policy_config.get('activation', torch.functional.F.relu)
        self.batch_first = True
        self.hidden_dim = policy_config['hidden_dim'] * 3
        # 这个action dim其实就是用户数目 + 1
        self.action_dim = policy_config['action_dim'] +1 
        self.layer_norm_eps = policy_config.get('layer_norm_eps',  1e-5)
        decoder_layer = nn.TransformerDecoderLayer(self.d_model, self.nhead, self.dim_feedforward, self.dropout, self.activation, self.layer_norm_eps, batch_first=self.batch_first)   
        decoder_norm = LayerNorm(self.d_model, eps=self.layer_norm_eps)
        # 定义transformer解码器的层数
        num_decoder_layers = self.policy_config.get('num_decoder_layers', 6)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)
        # 这个所谓的最大解码次数指的是说，我最多循环多少次就要结束解码过程
        self.max_decoder_time = self.policy_config['max_decoder_time']
        self.init_decoder = nn.Parameter(torch.rand(1, 1, self.d_model)-0.5)
        self.affine_layer = nn.Linear(self.d_model, self.action_dim)
        self.eps = 1e-7
        self.embedding_layer = nn.Linear(self.hidden_dim, self.d_model)


    def forward(self, tgt, memory, inference_mode= True, action_list=None, device='cpu'):
        # bs * 20 * h, bs*21*h
        if inference_mode:
            assert action_list is None
        else:
            # 只有在训练模式下才会需要计算entropy
            conditional_entropy_sum = 0
            assert action_list is not None
        batch_size = tgt.shape[0]
        # 首先对tgt这个矩阵进行升高维度操作
        embbding_data = torch.relu(self.embedding_layer(tgt))
        # 上面是说，在推断，采样阶段，是不会传入action list这个列表的，只有在训练阶段才会，并且要计算V值的。
        mask = torch.zeros(batch_size, self.action_dim).bool().to(device)
        terminate_batch = torch.zeros(batch_size, 1).bool().to(device)
        repeat_init_decoder = self.init_decoder.repeat(batch_size, 1, 1)
        # 两个矩阵进行合并
        scheduling_action_list = []
        log_joint_probs = 0
        concatenate_data = torch.cat([repeat_init_decoder, embbding_data], axis=1)
        decoder_input = repeat_init_decoder.clone()
        for i in range(self.max_decoder_time):
            # 通过网络直接输出结果
            transformer_decoder = self.decoder(decoder_input, memory)
            # 通过线性变换，降维
            linear_output = self.affine_layer(transformer_decoder)
            # 我们只取最后的那一个向量就可以了, batchsize * action dim
            attention_vector = linear_output[:,-1,:]
            attention_mask = torch.zeros_like(mask, dtype=torch.float)
            attention_mask.masked_fill_(mask, float("-inf"))
            attention_vector += attention_mask
            # 如果说是第一次挑选，一定是不要选择0出来，因此在这里进行mask操作
            if i == 0:
                attention_vector[:,0] += float("-inf") 
            # 进行softmax操作，得到概率向量
            attn = torch.softmax(attention_vector, dim=-1)
            if inference_mode:
                dist = Categorical(attn)
                scheduling_index = dist.sample().unsqueeze(-1)
                # 将那些已经结束了的batch的action变成0
                scheduling_index.masked_fill_(terminate_batch, 0)
            else:
                if i == 0:
                    last_prob = 1
                else:
                    last_prob = prob * last_prob
                scheduling_index = action_list[:, i].unsqueeze(-1)
                # 计算entropy
                conditional_entropy = Categorical(attn).entropy().unsqueeze(-1)
                conditional_entropy.masked_fill_(terminate_batch, 0)
                conditional_entropy_sum += conditional_entropy * last_prob
                # 这个action向量是一个调度序列，直接从attn中取值
            prob = attn.gather(1, scheduling_index)
            # 这里会出现log 0的情况，因此增加一个小数, 这是因为前面有的batch结束了，那么就是0，但是这里强制下一次再选出0，就会出现log 0了。
            log_prob = torch.log(prob + self.eps)
            # 将那些提前结束的batch的位置，将值变成0
            log_prob.masked_fill_(terminate_batch, 0)
            # 修改mask和is terminate mask矩阵
            mask = mask.scatter(1, scheduling_index, True)
            # 得到结束了的batch
            terminate_batch = scheduling_index == 0 
            # 构建下一次循环的输入, 通过mask选择
            selected_mask = torch.zeros(batch_size, self.action_dim).bool().to(device)
            selected_mask.scatter_(1, scheduling_index, True)
            selected_input_data = concatenate_data[selected_mask]
            decoder_input = torch.cat([decoder_input, selected_input_data.unsqueeze(1)], axis=1)
            log_joint_probs += log_prob
            scheduling_action_list.append(scheduling_index)
        if inference_mode:
            return [log_joint_probs, torch.cat(scheduling_action_list, 1)]
        else:   
            return [log_joint_probs, conditional_entropy_sum]

class model(nn.Module):
    def __init__(self, policy_config):
        super(model, self).__init__()
        # 首先需要定义卷积层，嵌入层，将数据变成batch_size * seq_len * 512的形式
        self.policy_config = policy_config
        self.embedding_dim = self.policy_config.get('d_model', 512)
        self.conv_channel = self.policy_config['conv_channel']
        self.hidden_dim = self.policy_config['hidden_dim']
        self.conv_layer = nn.Conv2d(self.conv_channel, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.linear_average_reward_head = nn.Linear(1, self.hidden_dim)
        self.linear_scheduling_count_head = nn.Linear(1, self.hidden_dim)
        self.embedding_layer = nn.Linear(3*self.hidden_dim, self.embedding_dim)
        self.transformer_encoder = transformer_model(self.policy_config)
        self.pointer_decoder = transformer_pointer_network_decoder(self.policy_config)


    def forward(self, src, action_list=None, inference_mode=True, device='cpu'):
        # ===================== 在采样阶段的时候,action list不会传入,并且inference_mode是True
        channel_matrix = src['channel_matrix'] # bs * 3* 20*32  -- bs * 20 * 96
        average_reward = src['average_reward'] # bs * 20 * 1 
        scheduling_count = src['scheduling_count'] # bs * 20 * 1 
        channel_output = torch.relu(self.conv_layer(1e6*channel_matrix).squeeze(1)) # bs*20 * h
        average_reward_output = torch.relu(self.linear_average_reward_head(average_reward)) #  bs * 20 * h
        scheduling_count_output = torch.relu(self.linear_scheduling_count_head(scheduling_count)) # bs * 20 * h
        # 拼接在一起构成backbone
        backbone = torch.cat([channel_output, average_reward_output, scheduling_count_output], -1)
        embedding_output = torch.relu(self.embedding_layer(backbone))
        # ----affine layer， softmax， bs * 20 *2, 依照概率采样，出现1就调度，0就不调度 ----- 
        # ---- affine layer, bs*20*1, --- KNN --- A, K个点，--  AI+Search ----  2^N, 
        # 送入到Transformer encoder, 可以得到一个bath size * seq len * d_model的矩阵
        transformer_encoder_output = self.transformer_encoder(embedding_output)
        res = self.pointer_decoder(backbone.clone(), transformer_encoder_output, inference_mode, action_list, device=device)
        return res[0], res[1] 

class critic(nn.Module):
    # 这个是一个critic类,传入全局的状态,返回对应的v值.因为R是一个向量,传入一个状态batch,前向得到一个v向量C: bs * 9 * 20 * 16 -> R^1
    def __init__(self, policy_config):
        super(critic, self).__init__()
        self.policy_config = policy_config
        self.embedding_dim = self.policy_config.get('d_model', 512)
        self.agent_nums = self.policy_config['agent_nums']
        self.conv_channel = self.policy_config['conv_channel'] * self.agent_nums
        self.hidden_dim = self.policy_config['hidden_dim']
        self.popart_start = self.policy_config["popart_start"]
        self.multi_objective_start = self.policy_config["multi_objective_start"]
        # ======================= 对全局状态进行卷积操作, reward需要进行线性操作, count这个状态也 =====================
        self.channel_conv_layer = nn.Conv2d(self.conv_channel, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        # 上面通过先行变换,得到的维度是batch size * agent_nums * user_number * hidden_size
        self.linear_average_reward_head = nn.Linear(1, self.hidden_dim)
        # 这个地方通过2d卷积操作,将上面这个矩阵变成一个batch size * user number * hidden size的一个矩阵
        self.reward_conv_layer = nn.Conv2d(self.agent_nums, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        # 接下来的两行的处理和上面的两行是一样的,将count信息变成batch size * user number * hidden size
        self.linear_scheduling_count_head = nn.Linear(1, self.hidden_dim)
        self.count_conv_layer = nn.Conv2d(self.agent_nums, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        # 这个地方是将上面三个矩阵进行拼接了之后,然后再过一次卷积
        self.concatenat_conv_layer = nn.Conv2d(3, out_channels= 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.embedding_layer = nn.Linear(self.hidden_dim, self.embedding_dim)
        self.transformer_encoder = transformer_model(self.policy_config)
        # 定义一个一维卷积操作
        self.feature_contraction = nn.Conv1d(self.policy_config['seq_len'], 1, kernel_size=3, stride=1, padding=1 ,bias=False) 
        # 因此我这个是一个多目标或者说是multi task问题,因此需要引入两个popArt
        if self.multi_objective_start:
            if self.popart_start:
                self.PF_head = PopArt(self.embedding_dim, 1)
                self.Edge_head = PopArt(self.embedding_dim, 1)
            else:
                self.PF_head = nn.Linear(self.embedding_dim, 1)
                self.Edge_head = nn.Linear(self.embedding_dim, 1)
        else:
            self.value_head = nn.Linear(self.embedding_dim, 1)

    def update(self, input_vector):
        self.PF_head.update(input_vector[:, 0])
        self.Edge_head.update(input_vector[:, 1])

    def forward(self, src):
        # 这个scr表示的是全局信息
        global_channel_matrix = src['global_channel_matrix']
        global_average_reward = src['global_average_reward']
        global_scheduling_count = src['global_scheduling_count']
        # 这个global channel_matrix的维度是batch size * agent_nums * channel_number * user_number * 32
        conv_global_channel_output  = torch.relu(self.channel_conv_layer(1e8*global_channel_matrix))
        # global average reward的维度是batch size * agent_nums * user_number * 1
        glboal_average_reward_output = torch.relu(self.linear_average_reward_head(global_average_reward))
        conv_global_average_reward_output = torch.relu(self.reward_conv_layer(glboal_average_reward_output))
        # global_scheduling_count的维度是batch size * agent number * user_number * 1
        global_scheduling_count_output = torch.relu(self.linear_scheduling_count_head(global_scheduling_count))
        conv_global_scheduling_count_output = torch.relu(self.count_conv_layer(global_scheduling_count_output))
        # 将上面三个矩阵进行拼接,得到的backbone的维度是batch size * 3 * channel_number*user_number * 32
        concatenate_information = torch.cat([conv_global_channel_output, conv_global_average_reward_output, conv_global_scheduling_count_output], 1)
        conv_concatenate_output = torch.relu(self.concatenat_conv_layer(concatenate_information).squeeze(1))
        # 现在这个矩阵的信息是batch size * user_number * 32, 通过嵌入操作,维度升高到dmodel的维度
        embbeding_output = self.embedding_layer(conv_concatenate_output)
        transformer_encoder_output = self.transformer_encoder(embbeding_output)
        # 通过transformer之后,我得到一个维度为batch size * user number * dmodel的一个矩阵,我现在需要将这个矩阵变成一个二维的矩阵,使用一维卷积
        backbone = self.feature_contraction(transformer_encoder_output).squeeze(1)
        # 通过popArt进行计算
        if self.multi_objective_start:
            state_value_PF = self.PF_head(backbone)
            state_value_Edge = self.Edge_head(backbone)
            return state_value_PF, state_value_Edge
        else:
            state_value = 100 * self.value_head(backbone)
            return state_value

def init_policy_net(policy_config):
    return model(policy_config)

def init_critic_net(policy_config):
    return critic(policy_config)


# test_config = {}
# test_config['conv_channel'] = 3
# test_config['hidden_dim'] = 32
# test_config['action_dim'] = 21
# test_config['max_decoder_time'] = 16
# test_config['agent_nums'] = 3
# test_config['seq_len'] = 20
# test_model = critic(test_config)
# test_input = {}
# test_input['global_channel_matrix'] = torch.rand(2,9,20,32)
# test_input['global_average_reward'] = torch.rand(2,3, 20, 1)
# test_input['global_scheduling_count'] = torch.rand(2,3, 20, 1)

# with torch.no_grad():
#     output = test_model(test_input)
# print(output)
# print(output[0].numpy())
# 测试一下actor部分的代码
# test_config = {}
# test_config['conv_channel'] = 3
# test_config['hidden_dim'] = 32
# test_config['action_dim'] = 21
# test_config['max_decoder_time'] = 16
# test_input = {}
# test_input['channel_matrix'] = torch.rand(2, 3, 20, 32)
# test_input['average_reward'] = torch.rand(2, 20, 1)
# test_input['scheduling_count'] = torch.rand(2, 20, 1)
# test_actor = model(test_config)
# output_prob, output_scheduling_list = test_actor(test_input)
# # 再测试一下，当传入一个动作列表，看能够得到对应的概率
# joint_prob, entropy = test_actor(test_input, action_list=output_scheduling_list, inference_mode=False)
# print(joint_prob - output_prob)