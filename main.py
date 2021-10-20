import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pathlib

from Env.Env import Environment
from Tool import arguments
from Tool.rewrite_and_make_folder import rewrite_and_make_folder

def multiprocessing_training(index):
    GPU_id = 0
    import torch
    from torch.utils.tensorboard import SummaryWriter
    torch.cuda.set_device(GPU_id)
    
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
    common_args.commNet_start  = True
    common_args.parameter_sharing = True
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
    if config.transformer_start:
        from Agent.transformer_agent import Agent
        config = arguments.get_transformer_args(config)
    elif config.attention_start:
        from Agent.attention_agent import Agent
    elif config.commNet_start:
        config = arguments.get_communication_net_args(config)
        from Agent.CommNet_agent import Agent
    else:
        from Agent.agent import Agent

    class Project:
        def __init__(self, args, Training=True):
            self.args = args
            self.args.Training = Training
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
            if self.args.commNet_start:
                self.agent = Agent(self.args)
            else:
                if self.parameter_sharing:
                    self.agent = Agent(self.args, 0)   
                else:
                    self.agent_list = [Agent(self.args, i) for i in range(self.agent_number)]           

        def load_model(self):
            if os.path.exists(self.args.model_folder) and os.listdir(self.args.model_folder):
                if self.parameter_sharing:
                    policy_net_path = self.args.model_folder + '/'  +  'Agent_' + str(0) +'_policy_net.pkl'
                    value_net_path = self.args.model_folder + '/' +  'Agent_' + str(0) +'_value_net.pkl'
                    self.agent.actor.load_state_dict(torch.load(policy_net_path))
                    self.agent.critic.load_state_dict(torch.load(value_net_path))
                else:
                    for agent_id in range(self.agent_number):
                        policy_net_path = self.args.model_folder +  '/' +  'Agent_' + str(agent_id + 1) +'_policy_net.pkl'
                        value_net_path = self.args.model_folder + '/' + 'Agent_' + str(agent_id + 1) +'_value_net.pkl'
                        self.agent_list[agent_id].actor.load_state_dict(torch.load(policy_net_path))
                        self.agent_list[agent_id].critic.load_state_dict(torch.load(value_net_path)) 
            else:
                os.mkdir(self.args.model_folder)

        
        def Simulation_SE_only(self):
            # 这个函数就是专门正对于Max SE进行采样操作
            batch_average_reward = []
            for ep in tqdm(range(self.args.epoches)):
                self.env.Reset_batch()
                # 将这个channel_data进行划分，得到三个智能体的观测，以及critic的状态
                obs_list = self.env.get_agent_obs_SE_batch()
                if self.args.commNet_start:
                    action_list, batch_prob_list = self.agent.Pick_action_Max_SE_batch(obs_list)
                else:
                    action_list = []
                    batch_prob_list = []
                    for sector_index in range(self.sector_number):
                        if self.parameter_sharing:
                            scheduling_users, prob = self.agent.Pick_action_Max_SE_batch(obs_list[sector_index])
                        else:
                            scheduling_users, prob = self.agent_list[sector_index].Pick_action_Max_SE_batch(obs_list[sector_index])
                        action_list.append(scheduling_users)
                        batch_prob_list.append(prob)
                action_list = np.stack(action_list,axis=1)
                Instant_reward = self.env.calculate_batch_instant_rewrd(self.env.batch_data, action_list)

                if self.args.commNet_start:
                    self.agent.training(self.env.batch_data, Instant_reward, batch_prob_list)
                else:
                    for sector_index in range(self.sector_number):
                        if self.parameter_sharing:
                            if sector_index == 0:
                                zero_grad = True
                            else:
                                zero_grad = False

                            if sector_index == (self.sector_number-1):
                                release_graph = True
                            else:
                                release_graph = False
                            self.agent.training_parameter_sharing(self.env.batch_data, Instant_reward, batch_prob_list[sector_index], release_graph, zero_grad)
                        else:
                            self.agent_list[sector_index].training(self.env.batch_data, Instant_reward, batch_prob_list[sector_index])
                
                batch_average_reward.append(np.mean(Instant_reward))

            plt.figure()
            plt.plot(batch_average_reward)
            plt.savefig(self.figure_folder /'learning_curve.png')
            # 保存数据到本地
            reward_training_path = self.result_folder /'Training_rewrad.npy'
            np.save(reward_training_path, np.array(batch_average_reward))
            # 保存模型到本地
            self.save_model(ep)
            self.testing_model()

        def testing(self):
            self.load_model()
            self.testing_model()


        def testing_model(self):
            testing_data = np.load(self.testing_data_folder).transpose(4,0,1,2,3)
            obs = []
            agent_infer_sequence = []
            for sector_index in range(self.agent_number):
                # 每一个元素都是batch_size*20*3*32
                obs.append(testing_data[:,sector_index,:,:,:])
                if self.parameter_sharing:
                    scheduling_users, _ = self.agent.Pick_action_Max_SE_batch(obs[sector_index])
                else:
                    scheduling_users, _ = self.agent_list[sector_index].Pick_action_Max_SE_batch(obs[sector_index])
                agent_infer_sequence.append(scheduling_users)
            agent_infer_sequence = np.stack(agent_infer_sequence, axis=1)
            infer_SE = self.env.calculate_batch_instant_rewrd(testing_data, agent_infer_sequence)
            infer_SE_save_path = self.result_folder /'infer_SE.npy'
            np.save(infer_SE_save_path, np.array(infer_SE))
            infer_sequence_save_path = self.result_folder /'infer_sequence.npy'
            np.save(infer_sequence_save_path, np.array(agent_infer_sequence))

        def save_model(self, ep):
            # save model parameters
            if self.parameter_sharing:
                policy_net_path = self.args.model_folder/ (str(ep) + '_Agent_' + str(0) +'_policy_net.pkl')
                value_net_path = self.args.model_folder /(str(ep) + '_Agent_' + str(0) +'_value_net.pkl')
                torch.save(self.agent.actor.state_dict(), policy_net_path)
                torch.save(self.agent.critic.state_dict(), value_net_path)
            else:
                for agent_id in range(self.agent_number):
                    policy_net_path = self.args.model_folder/(str(ep) +  '_Agent_' + str(agent_id + 1) +'_policy_net.pkl')
                    value_net_path = self.args.model_folder /(str(ep) + '_Agent_' + str(agent_id + 1) + '_value_net.pkl')
                    torch.save(self.agent_list[agent_id].actor.state_dict(), policy_net_path)
                    torch.save(self.agent_list[agent_id].critic.state_dict(), value_net_path)


    def training_cell(args):
        test = Project(args)
        test.Simulation_SE_only()
        # print(args.user_numbers

    training_cell(config)


multiprocessing_training(4)
