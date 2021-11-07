import numpy as np
import matplotlib.pyplot as plt
import sys
import os
current_path = os.path.abspath(__file__)
root_path = '/'.join(current_path.split('/')[:-2])
sys.path.append(root_path)
from tqdm import tqdm
import pathlib

from Env.Env import Environment
from Tool import arguments
from Tool.rewrite_and_make_folder import rewrite_and_make_folder



def multiprocessing_training(index):
    GPU_id = 0
    import torch
    from torch.utils.tensorboard import SummaryWriter
    # torch.cuda.set_device(GPU_id)
    
    # 开一个多进程

    user_number_list = ['10_user','20_user','30_user','40_user']
    velocity = ['3KM','30KM','90KM']
    user_index = user_number_list[index // 3]
    velocity_index = velocity[index % 3]
    # 修改用户的数量和用户移动速度
    user_number = int(user_index.split('_')[0])
    velocity_number = int(velocity_index.split('K')[0])
    common_args = arguments.get_common_args(user_number)
    common_args.user_velocity = velocity_number
    # common_args.maddpg_start = False
    # common_args.commNet_start = True
    # common_args.attention_start = True
    common_args.mode = 'train'
    # common_args.ablation_experiment = True
    common_args.parameter_sharing = True
    common_args.independent_learning = True
    if common_args.cuda:
        torch.cuda.manual_seed_all(22)
    else:
        torch.manual_seed(22)
    np.random.seed(22)
    # common_args.attention_start = True
    config = arguments.get_agent_args(common_args)
    if config.maddpg_start:
        from Agent.maddpg_agent import Agent
        config = arguments.get_MADDPG_args(config)

    elif config.transformer_start:
        from Agent.transformer_agent import Agent
        config = arguments.get_transformer_args(config)

    elif config.attention_start:
        from Agent.attention_agent import Agent

    elif config.commNet_start:
        config = arguments.get_communication_net_args(config)
        from Agent.CommNet_agent import Agent
    elif config.independent_learning:
        from Agent.Independent_agent import Agent

    else:
        from Agent.agent import Agent

    class Project:
        def __init__(self, args):
            self.args = args
            self.args.random_steps = 0
            self.args.Training = args.mode == 'train' 
            self.sector_number = self.args.sector_number
            self.agent_number = self.args.n_agents
            self.parameter_sharing = self.args.parameter_sharing
            self.data_folder = pathlib.Path(self.args.training_data_path)/(str(self.args.total_user_antennas) + '_user')/(str(self.args.user_velocity)+'KM')/'training_data_10_10.npy'
            self.args.data_folder = self.data_folder
            self.testing_data_folder = pathlib.Path(self.args.training_data_path)/(str(self.args.total_user_antennas) + '_user')/(str(self.args.user_velocity)+'KM')/'testing_data_10_10.npy'
            # ==== 这里通过调用函数，根据当前传入的参数，生成一些文件夹的路径 ====
            model_folder, vision_folder, result_folder, figure_folder = rewrite_and_make_folder(args)
            self.model_folder = model_folder
            self.vision_folder = vision_folder
            self.result_folder = result_folder
            self.figure_folder = figure_folder
            self.args.writer = SummaryWriter(self.vision_folder)
            # ==========================================================
            self.env = Environment(self.args)
            self.agent = Agent(self.args)


        def test_agent(self):
            scheduling_res = self.agent.Pick_action_Max_SE_batch([np.zeros((3,20,32)) for i in range(3)])
            v_value = self.agent.critic(np.zeros((3,20,32)))

        def testing_model(self):
            testing_data = np.load(self.testing_data_folder).transpose(4,0,1,2,3)
            if self.args.ablation_experiment:
                obs_list = []
                for sector_index in range(self.agent_number):
                # 每一个元素都是batch_size*20*3*32
                    sub_obs = []
                    for sub_sector_index in range(self.agent_number):
                        # if sector_index == sub_sector_index:
                        #     sub_obs.append(testing_data[:, sector_index, :, sub_sector_index, :])
                        # else:
                        sub_obs.append(np.zeros((testing_data.shape[0], self.args.user_numbers, self.args.obs_dim2)))
                            
                    obs_list.append(np.stack(sub_obs, 2))
            else:
                obs_list = [testing_data[:, agent_index, :, :] for agent_index in range(self.agent_number)]
            action_list, _ = self.agent.Pick_action_Max_SE_batch(obs_list)
            agent_infer_sequence = np.stack(action_list, axis=1)
            infer_SE = self.env.calculate_batch_instant_rewrd(testing_data, agent_infer_sequence)
            infer_SE_save_path = self.result_folder /'infer_SE.npy'
            np.save(infer_SE_save_path, np.array(infer_SE))
            infer_sequence_save_path = self.result_folder /'infer_sequence.npy'
            np.save(infer_sequence_save_path, np.array(agent_infer_sequence))



        def load_model(self):
            if os.path.exists(self.args.model_folder) and os.listdir(self.args.model_folder):
                if self.parameter_sharing:
                    policy_net_path = self.args.model_folder/  'policy_net.pkl'
                    value_net_path = self.args.model_folder / 'value_net.pkl'
                    self.agent.actor.load_state_dict(torch.load(policy_net_path))
                    self.agent.critic.load_state_dict(torch.load(value_net_path))
                else:
                    for agent_id in range(self.agent_number):
                        policy_net_path = self.args.model_folder/( 'Agent_' + str(agent_id + 1) +'_policy_net.pkl')
                        self.agent.actor[agent_id].load_state_dict(torch.load(policy_net_path))
                    value_net_path = self.args.model_folder /('value_net.pkl')
                    self.agent_list[agent_id].critic.load_state_dict(torch.load(value_net_path)) 
            else:
                os.mkdir(self.args.model_folder)


    def testing_cell(args):
        test = Project(args)
        # test.testing_model()
        # print(args.user_numbers
        test.test_agent()
    testing_cell(config)


multiprocessing_training(4)
