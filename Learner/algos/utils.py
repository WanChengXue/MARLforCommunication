def soft_update(current_net, target_net, tau):
    for target_param, param in zip(target_net.parameters(), current_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )