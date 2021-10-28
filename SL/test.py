# from Env.Instant_Reward import calculate_instant_reward
import numpy as np
import pathlib
from model import Model
import sys
import os
import shutil
import torch
current_path = os.path.abspath(__file__)
root_path = '/'.join(current_path.split('/')[:-2])
sys.path.append(root_path)
from Tool import arguments
import matplotlib.pyplot as plt
from Env.Instant_Reward import calculate_instant_reward
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
class SL_project:
    def __init__(self, args):
        # 读取文件
        self.args = args
        self.training_data_path = pathlib.Path(self.args.training_data_path)/(str(self.args.total_user_antennas) + '_user')/(str(self.args.user_velocity)+'KM')/'training_data_10_10.npy'
        self.training_data = np.load(self.training_data_path)
        self.total_training_sample = self.training_data.shape[-1]
        self.n_agents = args.cell_number * args.sector_number
        self.device = "cuda" if self.args.cuda else "cpu"
        self.model = Model(self.args, (1, args.total_state_matrix_number, args.state_dim1, args.obs_dim2)).to(self.device)
        self.legal_range = [self.args.min_stream, self.args.max_stream]
        self.transmit_power = [self.args.transmit_power] * self.n_agents
        self.noise_power = self.args.noise_spectrum_density
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.critic_lr, weight_decay=self.args.critic_lr_decay)
        self.epoch = 1000
        self.iter_per_batch = 10
        self.model_folder = pathlib.Path("./Exp/SL_folder/model")
        self.figure_folder = pathlib.Path("./Exp/SL_folder/figure")
        self.result_folder = pathlib.Path("./Exp/SL_folder/result")
        self.create_foler(self.model_folder)
        self.create_foler(self.figure_folder)
        self.create_foler(self.figure_folder)
        if self.args.mode == 'train':
            self.exp_folder = pathlib.Path("./Exp/SL_folder/exp")
        else:
            self.exp_folder = pathlib.Path("./Exp/SL_folder/exp_eval")
        self.create_foler(self.exp_folder)
        self.writer = SummaryWriter(self.exp_folder)
        self.count = 0
        self.loss_list = []

    def create_foler(self, folder_name):
        if os.path.exists(folder_name):
            shutil.rmtree(folder_name)
        folder_name.mkdir(parents=True, exist_ok=True)

    def random_batch(self):
        batch_index = np.random.choice(self.total_training_sample, self.args.TTI_length, replace=False)
        batch_data = [self.training_data[:,:,:,:,index] for index in batch_index]
        self.batch_data= np.stack(batch_data, axis=0)
        self.batch_action = np.stack([np.random.choice(2, (self.args.n_agents, self.args.obs_dim1)) for _ in range(self.args.TTI_length)], axis=0)

    def calculate_batch_instant_rewrd(self, batch_channel, batch_action):
        batch_instant_reward = []
        sample_number = batch_channel.shape[0]
        for index in range(sample_number):
            batch_instant_reward.append(calculate_instant_reward(batch_channel[index], batch_action[index], self.legal_range, self.noise_power, self.transmit_power))
        return batch_instant_reward

    def generate_data(self):
        self.random_batch()
        self.label_value = self.calculate_batch_instant_rewrd(self.batch_data, self.batch_action)

    def training(self):
        state = torch.FloatTensor(self.batch_data).to(self.device).transpose(2,3).reshape(-1, self.args.sector_number **2, self.args.obs_dim1, self.args.obs_dim2)
        action = torch.FloatTensor(self.batch_action).to(self.device)
        label_value = torch.FloatTensor(self.label_value).to(self.device).unsqueeze(-1)
        # print(label_value)
        for _ in range(self.iter_per_batch):
            predict_value = self.model(state, action)
            loss = self.loss_fn(predict_value, label_value)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.writer.add_scalar("./Loss_vale", loss.item(), self.count)
            self.count += 1
            self.loss_list.append(loss.item())
            # print(loss.item())
            
        


    def save_model(self):
        model_path = self.model_folder/'model.pkl'
        torch.save(self.model.state_dict(), model_path)

    def load_model(self):
        model_path = self.model_folder/'model.pkl'
        self.model.load_state_dict(torch.load(model_path))

    def simulation(self):
        for _ in tqdm(range(self.epoch)):
            self.generate_data()
            self.training()
        self.save_model()
        plt.figure()
        plt.plot(self.loss_list)
        plt.savefig(self.figure_folder/'learning_curve.png')
        plt.close()


def main():
    args = arguments.get_common_args(20)
    args.user_velocity = 30
    args.mode = 'train'
    config = arguments.get_agent_args(args)
    test_project = SL_project(config)
    test_project.simulation()
main()