# ---------------- 添加gumbel 分布 --------------------
import torch
def gumbel_softmax_sample(prob_matrix, eps=1e-20):
    # ----------- 首先生成均匀分布 ------------
    uniform_matrix = torch.rand_like(prob_matrix)
    inverse_transformation = -torch.log(-torch.log(uniform_matrix + eps)+ eps )
    prob_matrix_prime = prob_matrix + inverse_transformation
    return prob_matrix_prime