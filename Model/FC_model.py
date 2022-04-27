# --------------- 这个模型用来实现DDPG算法 ------------------
import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self,policy_config):
        super(Actor, self).__init__()
        # ----- 传入的是一个bs * 20 * 16 的矩阵 -----------
        # 首先需要定义卷积层，嵌入层，将数据变成batch_size * action_dim * 512的形式
        self.policy_config = policy_config
        self.state_feature_dim = self.policy_config['state_feature_dim']
        self.hidden_dim = self.policy_config['hidden_dim']
        self.feature_map_size = self.policy_config['feature_map_size']
        self.action_dim = self.policy_config['action_dim']
        self.window_size = self.policy_config['window_size']
        # ------------ 这个地方定义一些线性层，对矩阵进行线性操作,前面两个线性层是对矩阵进行特征提取，后面两个线性层是对干扰矩阵的特征提取 -------------
        self.main_head_real_part_affine_layer = nn.Linear(self.state_feature_dim, self.hidden_dim)
        self.main_head_img_partt_affine_layer = nn.Linear(self.state_feature_dim, self.hidden_dim)
        # ------------ 这个地方计算出来的是一个bs * 20 * hidden_dim 的矩阵 ------
        self.backbone_linear_layer = nn.Linear(2*self.hidden_dim, self.hidden_dim)
        # ------------ 定义四个卷积核进行特征的提取操作 ----------------
        self.convs = nn.ModuleList([
                                nn.Sequential(
                                    nn.Conv1d(in_channels=self.hidden_dim, out_channels=self.feature_map_size, kernel_size=h),
                                    nn.ReLU(),
                                    nn.MaxPool1d(kernel_size=self.action_dim-h + 1)
                                    )
                                for h in self.window_size
                            ])
        self.fc_layer = nn.Linear(self.feature_map_size*len(self.window_size), self.action_dim)
    
    def forward(self, src):
        # ===================== 在采样阶段的时候,action list不会传入,并且inference_mode是True
        # -------------- main head branch ------------
        real_part = src['real_part'] # bs *20 *16  -
        img_part = src['img_part']   # bs * 20 * 16
        # -------------- main head branch affine layer ------------
        main_head_real_part_affine = torch.relu(self.main_head_real_part_affine_layer(1e7*real_part))
        main_head_img_part_affine = torch.relu(self.main_head_img_partt_affine_layer(1e7**img_part))
        # ----------- main head 和 sum_interference 进行拼接 bs * 20 * (hidden*4) ——----------------------
        backbone = torch.cat([main_head_real_part_affine, main_head_img_part_affine], -1)
        # --------- 通过转置操作，将维度变成 batch_size * hidden_dim * seq_len
        feature_map = torch.relu(self.backbone_linear_layer(backbone)).permute(0, 2, 1)
        conv_list = [conv(feature_map).squeeze(-1) for conv in self.convs]
        # -------------- 通过concatenate函数，得到维度为batch size × (n*feature_map_size) --------
        concatenate_conv_feature = torch.cat(conv_list, -1) 
        fc_output = torch.sigmoid(self.fc_layer(concatenate_conv_feature))
        return fc_output
        

class critic(nn.Module):
    # 这个是一个critic类,传入全局的状态,返回对应的v值.因为R是一个向量,传入一个状态batch,前向得到一个v向量C: bs * 9 * 20 * 16 -> R^1
    def __init__(self, policy_config):
        super(critic, self).__init__()
        self.policy_config = policy_config
        self.popart_start = self.policy_config.get("popart_start", False)
        self.multi_objective_start = self.policy_config.get("multi_objective_start", False)
        self.state_feature_dim = self.policy_config['state_feature_dim']
        self.hidden_dim = self.policy_config['hidden_dim']
        self.window_size = self.policy_config['window_size']
        self.feature_map_size = self.policy_config['feature_map_size']
        self.action_dim = self.policy_config['action_dim']
        # ------------------- 通过上面的卷积操作，得到两个维度为bs×20*16的矩阵 ----------
        self.real_part_affine_layer = nn.Linear(self.state_feature_dim, self.hidden_dim)
        self.img_part_affine_layer = nn.Linear(self.state_feature_dim, self.hidden_dim)
        self.action_affine_layer = nn.Linear(self.action_dim, self.hidden_dim)
        self.backbone_linear_layer = nn.Linear(2*self.hidden_dim, self.hidden_dim)
        self.convs = nn.ModuleList([
                                nn.Sequential(
                                    nn.Conv1d(in_channels=self.hidden_dim, out_channels=self.feature_map_size, kernel_size=h),
                                    nn.ReLU(),
                                    nn.MaxPool1d(kernel_size=self.action_dim-h + 1)
                                    )
                                for h in self.window_size
                            ])
        self.value_head = nn.Linear(self.feature_map_size*len(self.window_size) + self.hidden_dim, 1)

    def forward(self, src, action):
        # 这个scr表示的是全局信息
        channel_matrix_real_part = src['real_part']
        channel_matrix_img_part = src['img_part']
        affine_real_part = torch.relu(self.real_part_affine_layer(1e7*channel_matrix_real_part))
        affine_img_part = torch.relu(self.img_part_affine_layer(1e7*channel_matrix_img_part))
        affine_action_part = torch.relu(self.action_affine_layer(action))
        # ---------- 两个矩阵进行concatenate，得到一个 bs*20*(2*hidden_size)的矩阵 ---------------
        backbone = torch.cat([affine_real_part, affine_img_part], -1)
        feature_map = torch.relu(self.backbone_linear_layer(backbone)).permute(0,2,1)
        conv_list = [conv(feature_map).squeeze(-1) for conv in self.convs]
        # -------------- 通过concatenate函数，得到维度为batch size × (n*feature_map_size) --------
        concatenate_conv_feature = torch.cat(conv_list, -1) 
        state_value = self.value_head(torch.cat([concatenate_conv_feature, affine_action_part], -1))
        return state_value

def init_policy_net(policy_config):
    generate_model = Actor(policy_config)
    for m in generate_model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(m.weight)
    return generate_model

def init_critic_net(policy_config):
    generate_model = critic(policy_config)
    for m in generate_model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
    return generate_model