def gae_estimator():
    '''
    这个地方使用了GAE进行估计，现在有的数据有两个:
        首先就是从时刻1到时刻T的instant reward向量， R^n
        其次就是通过采用的v网络，对所有状态的v值估计向量，R^n，以及T+1的状态的V值
        计算TD Error为， \gamma V(s_{t+1}) - V(s_t) + R_t = \delta_t
        计算时刻t的A值为 R(t) + \gamma R_{t+1} + ... + \gamma^{T-t} R_T + \gamma^{T-t+1}V(s_{T+1}) - V(s_t)
        因此按照上面的公式，Q(s_t,a_t)的估计值就是R(t) + \gamma R_{t+1} + ... + \gamma^{T-t} R_T + \gamma^{T-t+1}V(s_{T+1})
    '''
    pass
    