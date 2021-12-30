import torch.nn as nn
import torch
from torch.nn.modules.normalization import LayerNorm
from torch.distributions import Categorical


class transformer_model(nn.Module):
    def __init__(self, policy_config):
        super(transformer_model, self).__init__()
        # 初始化一个encoder
        self.policy_config = policy_config
        self.d_model = policy_config.get('d_model', 512)
        self.nhead = policy_config.get('nhead', 8)
        self.dim_feedforward = policy_config.get('dim_feedforward', 2048)
        self.dropout = policy_config.get('dropout', 0.1)
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
        self.dropout = policy_config.get('dropout', 0.1)
        self.activation = policy_config.get('activation', torch.functional.F.relu)
        self.batch_first = True
        self.device = policy_config.get('device', 'cpu')
        self.hidden_dim = policy_config['hidden_dim'] * 3
        # 这个action dim其实就是用户数目 + 1
        self.action_dim = policy_config['action_dim']
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


    def forward(self, tgt, memory, inference_mode= True, action_list=None):
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
        mask = torch.zeros(batch_size, self.action_dim).bool().to(self.device)
        terminate_batch = torch.zeros(batch_size, 1).bool().to(self.device)
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
            # 进行softmax操作，得到概率向量

            attn = torch.softmax(attention_vector, dim=-1)
            if inference_mode:
                dist = Categorical(attn)
                scheduling_index = dist.sample().unsqueeze(-1)
                # 将那些已经结束了的batch的action变成0
                scheduling_index.masked_fill_(terminate_batch, 0)
            else:
                scheduling_index = action_list[:, i]
                # 计算entropy
                conditional_entropy = Categorical(attn).entropy().unsqueeze(-1)
                conditional_entropy.masked_fill_(terminate_batch, 0)
                conditional_entropy_sum + conditional_entropy
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
            selected_mask = torch.zeros(batch_size, self.action_dim).bool()
            selected_mask.scatter_(1, scheduling_index, True)
            selected_input_data = concatenate_data[selected_mask]
            decoder_input = torch.cat([decoder_input, selected_input_data.unsqueeze(1)], axis=1)
            log_joint_probs += log_prob
            scheduling_action_list.append(scheduling_index)
            print(attn)
            print(scheduling_index)
        print(scheduling_action_list)
        if inference_mode:
            return log_joint_probs, scheduling_action_list
        else:   
            return log_joint_probs, conditional_entropy_sum

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

    def forward(self, src, action_list=None, inference_mode=True):
        # 首先这个src中包含了三个部分，信道矩阵部分，维度是batch size * user num * 32
        # average reward 部分，维度是batch size * user num * 32
        # 最后是到目前位置，各个用户调度了多少次构成的向量 batch size * user num * 1
        channel_matrix = src['channel_matrix']
        average_reward = src['average_reward']
        scheduling_count = src['scheduling_count']
        channel_output = torch.relu(self.conv_layer(channel_matrix).squeeze(1))
        average_reward_output = torch.relu(self.linear_average_reward_head(average_reward))
        scheduling_count_output = torch.relu(self.linear_scheduling_count_head(scheduling_count))
        # 拼接在一起构成backbone
        backbone = torch.cat([channel_output, average_reward_output, scheduling_count_output], -1)
        embedding_output = torch.relu(self.embedding_layer(backbone))
        # 送入到Transformer encoder, 可以得到一个bath size * seq len * d_model的矩阵
        transformer_encoder_output = self.transformer_encoder(embedding_output)
        res = self.pointer_decoder(backbone.clone(), transformer_encoder_output)

test_config = {}
test_config['conv_channel'] = 3
test_config['hidden_dim'] = 32
test_config['action_dim'] = 21
test_config['max_decoder_time'] = 16
test_model = model(test_config)
test_input = {}
test_input['channel_matrix'] = torch.rand(2,3,20,32)
test_input['average_reward'] = torch.rand(2, 20, 1)
test_input['scheduling_count'] = torch.rand(2, 20, 1)
output = test_model(test_input)