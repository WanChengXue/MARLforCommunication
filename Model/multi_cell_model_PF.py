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
        self.inf = Parameter(torch.FloatTensor([-1e10]).unsqueeze(1).expand(1, self.seq_length).clone(), requires_grad=False)
        self.decoder_input = Parameter(torch.FloatTensor(1, self.input_dim))
        # ----------- 这个向量表示feature map中需要添加一个额外的向量——---------
        self.terminate_encoder_vector = Parameter(torch.zeros(1,1,self.input_dim))
        self.index_matrix = Parameter(torch.LongTensor([[i for i in range(self.seq_length)]]), requires_grad=False)
        self.terminate_vector = Parameter(torch.ones(1,1), requires_grad=False)
        self.eps = 1e-9
        self.temperature = self.config.get('temperature', 1.0)
        self.use_gumbel_softmax = self.config.get('user_gumbel_softmax', False)

    def forward(self,feature_map, action=None, inference_mode=True):
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
            bool_mask = torch.eq(mask, 0) # bs * sq_len
            if self.use_gumbel_softmax:
                weight_output_prime = gumbel_softmax_sample(weight_output)
                weight_output_prime[bool_mask] = _inf[bool_mask]
                prob_matrix = torch.softmax(weight_output_prime/self.temperature, -1) # batch_size * seq_len
            else:
                weight_output[bool_mask] = _inf[bool_mask]
                prob_matrix = torch.softmax(weight_output, -1)
            masked_output = prob_matrix * mask
            dist = categorical.Categorical(masked_output)
            # ---------- 选择最大概率的值 --------------
            if action is not None:
                indices = action[:,i].unsqueeze(-1)
                index_probs = masked_output.gather(1, indices)
                if not inference_mode:
                    conditional_entropy_list.append(dist.entropy().unsqueeze(-1)*index_probs)
            else:
                indices = torch.argmax(masked_output, -1).unsqueeze(-1)
                # indices = dist.sample().unsqueeze(-1)
                # ---------- 每个小区至少要调度min_decoder_time个用户出来 -----------
                if i<self.min_decoder_time:
                    terminate_batch = indices == 0
                    # --------- 将第二大的概率向量拿出来 ----------
                    _, alter_matrix = torch.topk(masked_output, 2)
                    indices[terminate_batch] = alter_matrix[:,1].unsqueeze(-1)[terminate_batch]
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
        self.policy_config = policy_config
        self.state_feature_dim = self.policy_config['state_feature_dim']
        self.hidden_dim = self.policy_config['hidden_dim']
        # ------------ 这个地方定义一些线性层，对矩阵进行线性操作,前面两个线性层是对矩阵进行特征提取，后面两个线性层是对干扰矩阵的特征提取 -------------
        self.main_head_real_part_affine_layer = nn.Linear(self.state_feature_dim, self.hidden_dim)
        self.main_head_img_partt_affine_layer = nn.Linear(self.state_feature_dim, self.hidden_dim)
        self.interference_head_real_part_affine_layer = nn.Linear(self.state_feature_dim, self.hidden_dim)
        self.interference_head_img_part_affine_layer = nn.Linear(self.state_feature_dim, self.hidden_dim)
        self.main_head_average_se_affine_layer = nn.Linear(1, self.hidden_dim)
        self.main_head_scheduling_counts_affine_layer = nn.Linear(1, self.hidden_dim)
        self.backbone_linear_layer = nn.Linear(6*self.hidden_dim, self.hidden_dim)
        self.PN_network = pointer_network(self.policy_config)


    def forward(self, src, action_list=None, inference_mode=True):
        # ===================== 在采样阶段的时候,action list不会传入,并且inference_mode是True
        # -------------- main head branch ------------
        real_part = src['channel_matrix']['main_matrix']['real_part'] # bs *20 *16  -
        img_part = src['channel_matrix']['main_matrix']['img_part']   # bs * 20 * 16
        average_user_se = src['average_user_se'] # bs * 20 * 1
        user_scheduling_counts = src['user_scheduling_counts'] # bs * 20*1
        # -------------- interference_head_brach ----------
        interference_dict = src['channel_matrix']['interference_matrix'] # 是一个字典 {'interfence_1':{'real_part': xx, 'img_part': xx}}
        # -------------- main head branch affine layer ------------
        main_head_real_part_affine = torch.tanh(self.main_head_real_part_affine_layer(1e7*real_part))
        main_head_img_part_affine = torch.tanh(self.main_head_img_partt_affine_layer(1e7**img_part))
        # -------------- img head branch affine layer ---------------
        interference_head_real_part_affine = []
        interference_head_img_part_affine = []
        for interference_cell in interference_dict.keys():
            interference_head_real_part_affine.append(torch.tanh(self.interference_head_real_part_affine_layer(1e7*interference_dict[interference_cell]['real_part'])))
            interference_head_img_part_affine.append(torch.tanh(self.interference_head_img_part_affine_layer(1e7*interference_dict[interference_cell]['img_part'])))
        # -------------- 干扰矩阵实数部分和复数部分分别拼接然后做加法 ---------------------
        sum_interference_head_real_part = sum(interference_head_real_part_affine) # bs * 20 * h
        sum_interference_head_img_part = sum(interference_head_img_part_affine) # bs * 20 * h
        # -------------- 对user_se和scheduling_counts进行affine transformation --------
        affine_user_se = torch.tanh(self.main_head_average_se_affine_layer(average_user_se))
        affine_scheduling_counts = torch.tanh(self.main_head_scheduling_counts_affine_layer(user_scheduling_counts))
        # ----------- main head 和 sum_interference 进行拼接 bs * 20 * (hidden*4) ——----------------------
        backbone = torch.cat([main_head_real_part_affine, main_head_img_part_affine, sum_interference_head_real_part, sum_interference_head_img_part, affine_user_se, affine_scheduling_counts], -1)
        feature_map = torch.relu(self.backbone_linear_layer(backbone))
        # ------------- 把这个特征图送入到指针网络中 ---------------
        res = self.PN_network(feature_map, action=action_list, inference_mode=inference_mode)
        if inference_mode:
            # ------ 在推断阶段传入了动作，表示动作直接从demonstration中获取，只返回概率 -----
            if action_list is None:
                return res[0], res[1]
            else:
                return res[0]
        else:
            return res[0], res[2]

class critic(nn.Module):
    # 这个是一个critic类,传入全局的状态,返回对应的v值.因为R是一个向量,传入一个状态batch,前向得到一个v向量C: bs * 9 * 20 * 16 -> R^1
    def __init__(self, policy_config):
        super(critic, self).__init__()
        self.policy_config = policy_config
        self.agent_nums = self.policy_config['agent_nums']
        self.popart_start = self.policy_config.get("popart_start", False)
        self.multi_objective_start = self.policy_config.get("multi_objective_start", False)
        self.state_feature_dim = self.policy_config['state_feature_dim']
        self.hidden_dim = self.policy_config['hidden_dim']
        self.conv_channel = self.policy_config['conv_channel']* self.agent_nums
        # ======================= 对全局状态进行卷积操作 =====================
        self.real_channel_conv_layer = nn.Conv2d(self.conv_channel, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.img_channel_conv_layer = nn.Conv2d(self.conv_channel, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        # ------------------- 通过上面的卷积操作，得到两个维度为bs×20*16的矩阵 ----------
        self.affine_channel_layer = nn.Linear(2*self.state_feature_dim, self.hidden_dim)
        self.affine_user_se_layer = nn.Linear(1, self.hidden_dim)
        self.affine_scheduling_layer = nn.Linear(1, self.hidden_dim)
        # ------------------- 过两个卷积层 ---------------------------
        self.user_se_conv_layer = nn.Conv2d(self.policy_config['conv_channel'], out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.scheduling_conv_layer = nn.Conv2d(self.policy_config['conv_channel'], out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone_layer = nn.Linear(3*self.hidden_dim, self.hidden_dim)
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
        global_channel_matrix_real_part = src['real_part'] # bs * 9 * 20 * 16
        global_channel_matrix_img_part = src['img_part'] # bs * 9 * 20 * 16
        average_user_se = src['average_user_se'] # bs * 3 * 20 * 1
        user_scheduling_counts = src['user_scheduling_counts'] # bs * 3 * 20 * 1
        conv_real_part_matrix = torch.tanh(self.real_channel_conv_layer(1e7*global_channel_matrix_real_part)).squeeze(1) # bs * 20 *  16
        conv_img_part_matrix = torch.tanh(self.img_channel_conv_layer(1e7*global_channel_matrix_img_part)).squeeze(1)  # bs * 20 *  16
        # ------------- 将real_part和img_part进行拼接,过线性层，得到bs*20*hidden_dim ------------
        concatenate_channel_matrix = torch.cat([conv_real_part_matrix, conv_img_part_matrix], -1)
        affine_channel_feature = torch.relu(self.affine_channel_layer(concatenate_channel_matrix))
        # ------------- 将用户的平均se，调度次数都过线性层，得到两个bs * 20 * hidden_dim的矩阵  --------
        affine_user_se_feature = torch.tanh(self.affine_user_se_layer(average_user_se))
        affine_user_scheduling_feature = torch.tanh(self.affine_scheduling_layer(user_scheduling_counts))
        conv_user_se_feature = torch.tanh(self.user_se_conv_layer(affine_user_se_feature)).squeeze(1) # bs * 20 * 16
        conv_user_scheduling_feature = torch.tanh(self.scheduling_conv_layer(affine_user_scheduling_feature)).squeeze(1) # bs * 20 * 16
        # ---------- 两个矩阵进行concatenate，得到一个 bs*20*(3*hidden_size)的矩阵 ---------------
        concatenate_feature_matrix = torch.cat([affine_channel_feature, conv_user_se_feature, conv_user_scheduling_feature], -1)
        backbone = torch.relu(torch.mean(self.backbone_layer(concatenate_feature_matrix),1))
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
# import pickle 
# open_f = open('./test.pickle', 'rb')
# test_matrix = pickle.load(open_f)

# test_matrix =dict()
# test_matrix['channel_matrix'] = dict()
# test_matrix['channel_matrix']['main_matrix'] = dict()
# test_matrix['channel_matrix']['main_matrix']['real_part'] = torch.rand(2,20,16) 
# test_matrix['channel_matrix']['main_matrix']['img_part'] = torch.rand(2,20,16) 
# test_matrix['user_scheduling_counts'] = torch.rand(2,20,1)
# test_matrix['average_user_se'] = torch.rand(2,20,1)
# test_matrix['channel_matrix']['interference_matrix'] = dict()
# test_matrix['channel_matrix']['interference_matrix']['interfence_1'] = dict()
# test_matrix['channel_matrix']['interference_matrix']['interfence_1']['real_part'] = torch.rand(2,20,16) 
# test_matrix['channel_matrix']['interference_matrix']['interfence_1']['img_part'] = torch.rand(2,20,16) 
# test_matrix['channel_matrix']['interference_matrix']['interfence_2'] = dict()
# test_matrix['channel_matrix']['interference_matrix']['interfence_2']['real_part'] = torch.rand(2,20,16) 
# test_matrix['channel_matrix']['interference_matrix']['interfence_2']['img_part'] = torch.rand(2,20,16)
# test_actor = model(test_config).to(0)
# output_prob, output_scheduling_list = test_actor(test_matrix['current_state']['agent_obs']['agent_0'])
# print(output_scheduling_list)
# res = test_actor(test_matrix['current_state']['agent_obs']['agent_0'], output_scheduling_list)
# print(output_prob - res)
# print(output_scheduling_list)
# test_action = torch.LongTensor([[16,  9, 12, 10,  8, 17,  7, 18, 15, 13, 19,  5, 11,  0,  0,  0],
#         [13,  1, 14,  5, 12,  7, 19, 20,  6, 11,  4,  9,  8,  3, 16, 15]]).to(0)
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