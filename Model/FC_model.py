# --------------- 这个模型用来实现DDPG算法 ------------------
import torch
import torch.nn as nn

class Actor(nn.Module):
    def __init__(self,policy_config):
        super(Actor, self).__init__()
        # ----- 传入的是一个bs * 20 * 16 的矩阵 -----------
        # 首先需要定义卷积层，嵌入层，将数据变成batch_size * seq_len * 512的形式
        self.policy_config = policy_config
        self.state_feature_dim = self.policy_config['state_feature_dim']
        self.hidden_dim = self.policy_config['hidden_dim']
        self.feature_map_size = self.policy_config['feature_map_size']
        self.seq_len = self.policy_config['seq_len']
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
                                    nn.MaxPool1d(kernel_size=self.seq_len-h + 1)
                                    )
                                for h in self.window_size
                            ])
        self.fc_layer = nn.Linear(self.feature_map_size*len(self.window_size), self.seq_len)
    
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
        feature_map = torch.relu(self.backbone_linear_layer(backbone))
        conv_list = [conv(feature_map).squeeze(-1) for conv in self.convs]
        # -------------- 通过concatenate函数，得到维度为batch size × (n*feature_map_size) --------
        concatenate_conv_feature = torch.cat(conv_list, -1) 
        fc_output = torch.sigmoid(self.fc_layer(concatenate_conv_feature))
        return fc_output

class Critic(nn.Module):
    def __init__(self):
        pass