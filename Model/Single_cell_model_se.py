import torch.nn as nn
import torch
from torch.nn import Parameter
from torch.distributions import categorical
import sys
import os
current_path = os.path.abspath(__file__)
root_path = '/'.join(current_path.split('/')[:-2])
sys.path.append(root_path)
from Model.popart import PopArt
from Model.Utils import gumbel_softmax_sample

class pointer_network(nn.Module):
    def __init__(self, config):
        super(pointer_network, self).__init__()
        self.config = config
        self.max_decoder_time = self.config['max_decoder_time']
        self.hidden_size = self.config['GRU_hidden_size']
        self.input_dim = self.config['input_dim']
        self.weight_dim  = self.config['weight_dim']
        self.seq_length = self.config['seq_len'] + 1
        self.min_decoder_time = self.config['min_decoder_time']
        self.encoder = nn.GRU(self.input_dim, self.hidden_size, batch_first=True)
        self.decoder = nn.GRUCell(self.input_dim, self.hidden_size)
        self.W1 = nn.Linear(self.hidden_size, self.weight_dim, bias=False)
        self.W2 = nn.Linear(self.hidden_size, self.weight_dim, bias=False)
        self.V = nn.Linear(self.weight_dim, 1, bias=False)
        self.affine_decoder_hidden_layer = nn.Linear(2*self.hidden_size, self.hidden_size)
        self.mask = Parameter(torch.ones(1, self.seq_length), requires_grad=False)
        self.inf = Parameter(torch.FloatTensor([float('-inf')]).unsqueeze(1).expand(1, self.seq_length).clone(), requires_grad=False)
        self.decoder_input = Parameter(torch.FloatTensor(1, self.input_dim))
        # ----------- 这个向量表示feature map中需要添加一个额外的向量——---------
        self.terminate_encoder_vector = Parameter(torch.zeros(1,1,self.input_dim))
        self.index_matrix = Parameter(torch.LongTensor([[i for i in range(self.seq_length)]]), requires_grad=False)
        self.terminate_vector = Parameter(torch.ones(1,1), requires_grad=False)
        self.eps = 1e-9
        self.temperature = self.config.get('temperature', 1.0)

    def forward(self,feature_map, action=None):
        #----------- Encoding -------------
        batch_size = feature_map.size(0)
        decoder_input = self.decoder_input.repeat(batch_size, 1)
        terminate_encoder_vector = self.terminate_encoder_vector.repeat(batch_size, 1, 1)
        mask = self.mask.repeat(batch_size, 1)
        _inf = self.inf.repeat(batch_size, 1)
        selected_matrix = self.index_matrix.repeat(batch_size, 1)
        _terminate_vector = self.terminate_vector.repeat(batch_size, 1)
        Argument_feature_map = torch.cat([terminate_encoder_vector, feature_map], 1)
        Encoder_states, hc = self.encoder(Argument_feature_map) # --- Encoder_states的维度是batch_size * seq_len * self.hidden_size
        decoder_hidden = hc.clone().squeeze(0) # batch_szie * hidden_dim
        probs = []
        scheduling_user = []
        conditional_entropy_list = []
        for i in range(self.max_decoder_time):
            hidden = self.decoder(decoder_input, decoder_hidden)
            # ----------- 这个地方使用指针网络的计算公式 tanh(W1E + W2d) -----------
            affine_encoder_matrix = self.W1(Encoder_states) # bs * seq_len * weight_dim
            affine_decoder_matrix = self.W2(decoder_hidden).unsqueeze(1) # batch_size * weight_dim
            tanh_output = torch.tanh(affine_encoder_matrix + affine_decoder_matrix) # batch_size * seq_len * weight_dim
            # ----------- 和权重向量相乘 ----------
            weight_output = self.V(tanh_output).squeeze(-1) # batch_size * seq_len 
            weight_output_prime = gumbel_softmax_sample(weight_output)
            bool_mask = torch.eq(mask, 0)
            weight_output_prime[bool_mask] = _inf[bool_mask]
            prob_matrix = torch.softmax(weight_output/self.temperature, -1) # batch_size * seq_len
            masked_output = prob_matrix * mask
            # ---------- 选择最大概率的值 --------------
            if action is not None:
                dist = categorical.Categorical(masked_output)
                indices = action[:,i].unsqueeze(-1)
                index_probs = masked_output.gather(1, indices)
                conditional_entropy_list.append(dist.entropy().unsqueeze(-1)*index_probs)
            else:
                indices = torch.argmax(masked_output, -1).unsqueeze(-1)
                # ---------- 每个小区至少要调度min_decoder_time个用户出来 -----------
                if i<self.min_decoder_time:
                    terminate_batch = indices == 0
                    # --------- 将第二大的概率向量拿出来 ----------
                    _, alter_matrix = torch.topk(masked_output, 2)
                    indices[terminate_batch] = alter_matrix[:,1].unsqueeze(-1)[terminate_batch]
                # if i < self.min_decoder_time:
                    # while not torch.all(indices):
                    #     terminate_batch = indices == 0
                    #     new_sample = dist.sample().unsqueeze(-1)
                    #     indices[terminate_batch] = new_sample[terminate_batch]
                index_probs = masked_output.gather(1, indices) # batch_size * 1
                # --------- 如果说当前上一个时刻的indices是0，则表示已经结束调度了，这个时刻的indices就变成0 ---------
                indices[mask[:,0] == 0] = 0
            _terminate_vector = _terminate_vector * mask[:,0].unsqueeze(-1)
            # ----------- indices： batch_size 向量， max_probs也是一个batch_size向量 -----------
            # -------- 得到一个batch_size * seq_len的矩阵，在当前batch上面，被选中的用户的位置是1，否则是0 ------------------
            one_hot_pointers = (selected_matrix == indices.repeat(1, self.seq_length))
            # -------- 将已经出现过的用户进行mask操作 --------------
            mask = mask*(1-one_hot_pointers.float())
            # -------- 构建下一次的输入向量 batch_size * hidden_size ------------
            unmask_decoder_input = Argument_feature_map[one_hot_pointers]
            # --------- 使用prob martrix 和上面那个Endoer_states进行矩阵乘法 ----------
            lienar_combination = torch.bmm(prob_matrix.unsqueeze(1), Encoder_states).squeeze(1) # batch_size *  hidden_size
            # --------- 两个向量相互拼接 ------------
            concatenate_decoder_hidden = torch.cat([hidden, lienar_combination],  -1)
            unmask_decoder_hidden = torch.tanh(self.affine_decoder_hidden_layer(concatenate_decoder_hidden))
            # ------------ 使用terminate flag对这个decoder input和decoder hidden进行mask操作 -------------
            decoder_input = unmask_decoder_input * _terminate_vector.repeat(1, self.input_dim)
            decoder_hidden = unmask_decoder_hidden * _terminate_vector.repeat(1, self.hidden_size)
            probs.append(_terminate_vector * torch.log(index_probs+self.eps))
            scheduling_user.append(indices)
        return torch.sum(torch.cat(probs,-1), -1).unsqueeze(-1), torch.cat(scheduling_user,-1), sum(conditional_entropy_list)


class model(nn.Module):
    def __init__(self, policy_config):
        super(model, self).__init__()
        # 首先需要定义卷积层，嵌入层，将数据变成batch_size * seq_len * 512的形式
        self.policy_config = policy_config
        self.state_feature_dim = self.policy_config['state_feature_dim']
        self.hidden_dim = self.policy_config['hidden_dim']
        # ------------ 这个地方定义一些线性层，对矩阵进行线性操作,前面两个线性层是对矩阵进行特征提取，后面两个线性层是对干扰矩阵的特征提取 -------------
        self.main_head_real_part_affine_layer = nn.Linear(self.state_feature_dim, self.hidden_dim)
        self.main_head_img_partt_affine_layer = nn.Linear(self.state_feature_dim, self.hidden_dim)
        self.backbone_linear_layer = nn.Linear(2*self.hidden_dim, self.hidden_dim)
        self.PN_network = pointer_network(self.policy_config)


    def forward(self, src, action_list=None, inference_mode=True):
        # ===================== 在采样阶段的时候,action list不会传入,并且inference_mode是True
        # -------------- main head branch ------------
        real_part = src['real_part'] # bs *20 *16  -
        img_part = src['img_part']   # bs * 20 * 16
        # -------------- main head branch affine layer ------------
        main_head_real_part_affine = torch.tanh(self.main_head_real_part_affine_layer(1e7*real_part))
        main_head_img_part_affine = torch.tanh(self.main_head_img_partt_affine_layer(1e7**img_part))
        # ----------- main head 和 sum_interference 进行拼接 bs * 20 * (hidden*4) ——----------------------
        backbone = torch.cat([main_head_real_part_affine, main_head_img_part_affine], -1)
        feature_map = torch.tanh(self.backbone_linear_layer(backbone))
        # ------------- 把这个特征图送入到指针网络中 ---------------
        res = self.PN_network(feature_map, action=action_list)
        if inference_mode:
            return res[0], res[1]
        else:
            return res[0], res[2]

class critic(nn.Module):
    # 这个是一个critic类,传入全局的状态,返回对应的v值.因为R是一个向量,传入一个状态batch,前向得到一个v向量C: bs * 9 * 20 * 16 -> R^1
    def __init__(self, policy_config):
        super(critic, self).__init__()
        self.policy_config = policy_config
        self.popart_start = self.policy_config.get("popart_start", False)
        self.multi_objective_start = self.policy_config.get("multi_objective_start", False)
        self.state_feature_dim = self.policy_config['state_feature_dim']
        self.hidden_dim = self.policy_config['hidden_dim']
        # ------------------- 通过上面的卷积操作，得到两个维度为bs×20*16的矩阵 ----------
        self.real_part_affine_layer = nn.Linear(self.state_feature_dim, self.hidden_dim)
        self.img_part_affine_layer = nn.Linear(self.state_feature_dim, self.hidden_dim)
        self.affine_layer = nn.Linear(2*self.hidden_dim, self.hidden_dim)
        # 因此我这个是一个多目标或者说是multi task问题,因此需要引入两个popArt
        if self.multi_objective_start:
            if self.popart_start:
                self.PF_head = PopArt(self.hidden_dim, 1)
                self.Edge_head = PopArt(self.hidden_dim, 1)
            else:
                self.PF_head = nn.Linear(self.hidden_dim, 1)
                self.Edge_head = nn.Linear(self.hidden_dim, 1)
        else:
            self.value_head = nn.Linear(self.hidden_dim, 1)

    def update(self, input_vector):
        self.PF_head.update(input_vector[:, 0])
        self.Edge_head.update(input_vector[:, 1])

    def forward(self, src):
        # 这个scr表示的是全局信息
        channel_matrix_real_part = src['real_part']
        channel_matrix_img_part = src['img_part']
        affine_real_part = torch.tanh(self.real_part_affine_layer(1e7*channel_matrix_real_part))
        affine_img_part = torch.tanh(self.img_part_affine_layer(1e7*channel_matrix_img_part))
        # ---------- 两个矩阵进行concatenate，得到一个 bs*20*(2*hidden_size)的矩阵 ---------------
        concatenate_matrix = torch.cat([affine_real_part, affine_img_part], -1)
        # ----------- 在第二个维度进行average操作，得到一个bs*32的矩阵 ----------------
        mean_pool_matrix = torch.mean(concatenate_matrix, 1)
        backbone = torch.relu(self.affine_layer(mean_pool_matrix))
        # 通过popArt进行计算
        if self.multi_objective_start:
            state_value_PF = self.PF_head(backbone)
            state_value_Edge = self.Edge_head(backbone)
            return state_value_PF, state_value_Edge
        else:
            state_value = self.value_head(backbone)
            return state_value

def init_policy_net(policy_config):
    generate_model = model(policy_config)
    for m in generate_model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
    return generate_model

def init_critic_net(policy_config):
    generate_model = critic(policy_config)
    for m in generate_model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
    return generate_model

# ------------- 测试policy net -----------------------
# test_config = {}
# test_config['state_feature_dim'] = 16
# test_config['hidden_dim'] = 64
# test_config['max_decoder_time'] = 16
# test_config['GRU_hidden_size'] = 128
# test_config['input_dim'] = 64
# test_config['seq_len'] = 20
# test_config['weight_dim'] = 32
# test_config['min_decoder_time'] = 3
# test_matrix =dict()
# test_matrix['real_part'] = torch.rand(2,20,16).to(0)
# test_matrix['img_part'] = torch.rand(2,20,16).to(0)
# test_actor = model(test_config).to(0)
# output_prob, output_scheduling_list = test_actor(test_matrix)
# print(output_scheduling_list)
# test_action = torch.LongTensor([[16,  9, 12, 10,  8, 17,  7, 18, 15, 13, 19,  5, 11,  0,  0,  0],
#         [13,  1, 14,  5, 12,  7, 19, 20,  6, 11,  4,  9,  8,  3, 16, 15]]).to(0)
# test_actor = model(test_config).to(0)
# output_prob, output_scheduling_list = test_actor(test_matrix,action_list= test_action)
# print(output_prob)
# print(output_scheduling_list)
# print("-------------------")
# # 再测试一下，当传入一个动作列表，看能够得到对应的概率
# joint_prob, entropy = test_actor(test_input, action_list=output_scheduling_list, inference_mode=False)
# print(joint_prob - output_prob)

# --------------- 测试critic网络 -----------------
# test_config = {}
# test_config['state_feature_dim'] = 16
# test_config['hidden_dim'] = 32
# test_critic = critic(test_config)
# test_matrix = dict()
# test_matrix['real_part'] = torch.rand(2,20,16)
# test_matrix['img_part'] = torch.rand(2,20,16)
# output_value = test_critic(test_matrix)
# print(output_value)