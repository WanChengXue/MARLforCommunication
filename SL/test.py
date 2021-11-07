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
        if self.device:
            torch.cuda.manual_seed_all(22)
        else:
            torch.manual_seed(22)
        self.model = Model(self.args, (1, args.total_state_matrix_number, args.state_dim1, args.obs_dim2)).to(self.device)
        self.legal_range = [self.args.min_stream, self.args.max_stream]
        self.transmit_power = [self.args.transmit_power] * self.n_agents
        self.noise_power = self.args.noise_spectrum_density
        self.lr=self.args.critic_lr
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr)
        torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.99)
        self.epoch = 2000
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
        self.test_loss_list = []

    def create_foler(self, folder_name):
        if os.path.exists(folder_name):
            shutil.rmtree(folder_name)
        folder_name.mkdir(parents=True, exist_ok=True)

    def random_batch(self):
        batch_index = np.random.choice(self.total_training_sample, self.args.TTI_length, replace=False)
        batch_data = [self.training_data[:,:,:,:,index] for index in batch_index]
        batch_data= np.stack(batch_data, axis=0)
        batch_action = np.stack([np.random.choice(2, (self.args.n_agents, self.args.obs_dim1)) for _ in range(self.args.TTI_length)], axis=0)
        return batch_data, batch_action


    def calculate_batch_instant_rewrd(self, batch_channel, batch_action):
        batch_instant_reward = []
        sample_number = batch_channel.shape[0]
        for index in range(sample_number):
            batch_instant_reward.append(calculate_instant_reward(batch_channel[index], batch_action[index], self.legal_range, self.noise_power, self.transmit_power))
        return batch_instant_reward

    def training_set_label_value_normalization(self):
        mean_value = np.mean(self.batch_label_value)
        std_value = np.std(self.batch_label_value)
        self.mean_value = mean_value
        self.std_value = std_value

    def generate_data(self):
        # 预先生成一千零一个batch,然后将其中1000个作为训练
        self.batch_data = []
        self.batch_action = []
        self.batch_label_value = []
        print("================ 正在生成训练数据和测试数据 ===================")
        for _ in tqdm(range(self.args.data_batch_number)):
            single_batch_data, single_batch_action = self.random_batch()
            self.batch_data.append(single_batch_data)
            self.batch_action.append(single_batch_action)
            label_value = self.calculate_batch_instant_rewrd(single_batch_data, single_batch_action)
            self.batch_label_value.append(label_value)
        # self.training_set_label_value_normalization()
        print("================ 正在生成测试数据 ================")
        self.testing_batch_data = []
        self.testing_batch_action = []
        self.testing_batch_label_value = []
        for _ in tqdm(range(self.args.data_batch_number//10)):
            testing_batch_data, test_batch_action = self.random_batch()
            testing_label_value = self.calculate_batch_instant_rewrd(testing_batch_data, test_batch_action)
            self.testing_batch_data.append(testing_batch_data)
            self.testing_batch_action.append(test_batch_action)
            self.testing_batch_label_value.append(testing_label_value)

    def training(self):
        # batch_loss_list = []
        batch_mse_loss = []
        for batch_index in range(self.args.data_batch_number):
            state = torch.FloatTensor(self.batch_data[batch_index]).to(self.device).transpose(2,3).reshape(-1, self.args.sector_number **2, self.args.obs_dim1, self.args.obs_dim2)
            action = torch.FloatTensor(self.batch_action[batch_index]).to(self.device)
            label_value = torch.FloatTensor(self.batch_label_value[batch_index]).to(self.device).unsqueeze(-1)
            predict_value = self.model(state, action)
            # normalized_label_value = (label_value - self.mean_value)/self.std_value
            mse_loss = self.loss_fn(label_value, predict_value)
            # loss = torch.mean((predict_value-label_value) / torch.clamp(0.5*(predict_value+label_value), 0.05))
            self.optimizer.zero_grad()
            mse_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_norm_grad)
            self.optimizer.step()
            # batch_loss_list.append(loss.item())
            batch_mse_loss.append(mse_loss.item())
                # print(loss.item())
        # epoch_average_loss = np.mean(batch_loss_list)  
        epoch_average_MSE_loss = np.mean(batch_mse_loss)
        # self.writer.add_scalar("Training/Test_loss_value", epoch_average_loss, self.count)
        self.writer.add_scalar("Training/Normalized_mse_loss_value", epoch_average_MSE_loss, self.count)
        self.count += 1
        self.loss_list.append(epoch_average_MSE_loss)


    def save_model(self, ep):
        model_path = self.model_folder/('epoch_' + str(ep) + '_model.pkl')
        torch.save(self.model.state_dict(), model_path)

    def load_model(self, ep):
        model_path = self.model_folder/('epoch_' + str(ep) + '_model.pkl')
        self.model.load_state_dict(torch.load(model_path))

    def simulation(self):
        self.generate_data()
        print("============ 进入测试阶段 ==============")
        for ep in tqdm(range(self.epoch)):
            self.training()
            self.testing_model()
            if ep>200 and ep % 20 == 0:
                print("保存模型, 模型的路径为: {}".format(self.model_folder/('epoch_' + str(ep) + '_model.pkl')))
                self.save_model(ep)

        plt.figure()
        plt.plot(self.loss_list)
        plt.savefig(self.figure_folder/'learning_curve.png')
        plt.close()
        
        plt.figure()
        plt.plot(self.test_loss_list)
        plt.savefig(self.figure_folder/'testing_curve.png')
        plt.close()



    def testing_model(self):
        # batch_loss_list = []
        batch_mse_loss = []
        for batch_index in range(self.args.data_batch_number//10):
            state = torch.FloatTensor(self.testing_batch_data[batch_index]).to(self.device).transpose(2,3).reshape(-1, self.args.sector_number **2, self.args.obs_dim1, self.args.obs_dim2)
            action = torch.FloatTensor(self.testing_batch_action[batch_index]).to(self.device)
            label_value = torch.FloatTensor(self.testing_batch_label_value[batch_index]).to(self.device).unsqueeze(-1)
            with torch.no_grad():
                predict_value = self.model(state, action)
                # unnormalized_value = self.std_value * (predict_value + self.mean_value)
                # loss = torch.mean((predict_value-label_value) / torch.clamp(0.5*(predict_value+label_value), 0.05))
                mse_loss = self.loss_fn(predict_value, label_value)
            # batch_loss_list.append(loss.item())
            batch_mse_loss.append(mse_loss.item())
        # epoch_average_loss = np.mean(batch_loss_list)   
        epoch_average_MSE_loss = np.mean(batch_mse_loss)
        # self.writer.add_scalar("Testing/Test_loss_value", epoch_average_loss, self.count-1)
        self.writer.add_scalar("Testing/MSE_loss_value", epoch_average_MSE_loss, self.count-1)
        self.test_loss_list.append(epoch_average_MSE_loss)

    def compare_curve(self):
        real_value_list = []
        predict_value_list = []
        for batch_index in range(self.args.data_batch_number//10):
            state = torch.FloatTensor(self.testing_batch_data[batch_index]).to(self.device).transpose(2,3).reshape(-1, self.args.sector_number **2, self.args.obs_dim1, self.args.obs_dim2)
            action = torch.FloatTensor(self.testing_batch_action[batch_index]).to(self.device)
            with torch.no_grad():
                predict_value = self.model(state, action)
            real_value_list += self.testing_batch_label_value[batch_index]
            predict_value_list += predict_value.cpu().tolist()    
        plt.figure()
        plt.plot(predict_value_list, label='predict_value')
        plt.plot(real_value_list, label='real_value')
        plt.legend(loc="upper right")
        plt.savefig(self.figure_folder/'test_result.png')
        plt.close()

def main():
    args = arguments.get_common_args(20)
    args.user_velocity = 30
    args.mode = 'train'
    args.TTI_length = 2560
    args.data_batch_number = 20
    config = arguments.get_agent_args(args)
    test_project = SL_project(config)
    test_project.simulation()
    test_project.compare_curve()
    
main()