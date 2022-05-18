import copy

def gae_estimator(traj_data, gamma, tau, bootstrap_value, multi_agent_scenario):
    '''
    bootstrap_value就是一个(1, 2)的全零numpy类型数据
    这个地方使用了GAE进行估计，现在有的数据有两个:
        首先就是从时刻1到时刻T的instant reward向量， R^n
        其次就是通过采用的v网络，对所有状态的v值估计向量，R^n，以及T+1的状态的V值
        计算TD Error为， \gamma V(s_{t+1}) - V(s_t) + R_t = \delta_t
        计算时刻t的A值为 R(t) + \gamma R_{t+1} + ... + \gamma^{T-t} R_T + \gamma^{T-t+1}V(s_{T+1}) - V(s_t)
        因此按照上面的公式，Q(s_t,a_t)的估计值就是R(t) + \gamma R_{t+1} + ... + \gamma^{T-t} R_T + \gamma^{T-t+1}V(s_{T+1})
    这个traj_data其实是一个字典数据，traj_data[0]也是一个字典，这个字典里面有的数据为：
        reward vector，这是瞬时奖励向量
        state_value，这是使用神经网络估计出来的当前状态的v向量
        mask，这个是一个标量，用来进行mask操作的，确保terminate的贡献为0
    使用gae计算的时候，这个state value其实是已经denormalizing之后的值，因此这个计算出来的advantage可以直接乘到概率上面
    '''
    next_step_state_value = bootstrap_value
    gae = 0
    if multi_agent_scenario:
        multi_agent_gae = 0
    # 传入的数据有多少个
    point_number = len(traj_data)
    for step in reversed(range(point_number)):
        # ---------- 使用denormalize_current_state_value的条件是，不开启popart的时候，因此不使用这变量？
        current_step_state_value = traj_data[step]['current_state_value']
        delta = traj_data[step]["instant_reward"] + gamma * next_step_state_value - current_step_state_value
        # np.concatenate(traj_data[step]['denormalize_current_state_value'], 1)是1*2的一个向量
        gae = delta + gamma * tau * gae * (1-traj_data[step]['done'])
        advantages = gae
        if multi_agent_scenario:
            multi_agent_gae = sum(gae)
            target_state_value = multi_agent_gae + current_step_state_value.squeeze(0)
        else:
            target_state_value = gae + current_step_state_value
        next_step_state_value = current_step_state_value
        # advantage和target state value放进去
        if multi_agent_scenario:
            traj_data[step]['advantages'] = dict()
            for agent_index, agent_name in enumerate(traj_data[step]['actions'].keys()):
                traj_data[step]['advantages'][agent_name] = advantages[agent_index]
        else:
            traj_data[step]['advantages'] = copy.deepcopy(advantages)
        traj_data[step]['target_state_value'] = copy.deepcopy(target_state_value)
    return traj_data


