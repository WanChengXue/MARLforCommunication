import numpy as np
import random
def random_sample(mask):
    # 传入一个mask矩阵
    batch_size, user_numbers = mask.shape
    user_index_matrix = np.array([i for i in range(user_numbers)])
    # 将那些没有被选择过的用户挑出来
    sample_user = []
    for i in range(batch_size):
        single_avaliable_user = user_index_matrix[mask[i,:].cpu()==0]
        sample_user.append(random.choice(single_avaliable_user))
    return sample_user
