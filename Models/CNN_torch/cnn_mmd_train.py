"""
The training of CNN-FT.
Code refers to: https://github.com/fyancy/DASMN/tree/main/DASMN_revised_2020_11/Model
/Yancy F. 2021.8.30/
"""
import torch
import numpy as np
import visdom
import time
import os

from Models.CNN_torch.cnn_model import CNN_MMD
from Datasets.cwru_data import CNN_DataGenerator_torch
from my_utils.init_utils import weights_init2
from my_utils.train_utils import accuracy


vis = visdom.Visdom(env='yancy_meta')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CNN_MMD_learner(object):
    def __init__(self, ways, shots):
        h_size = 64
        layers = 4
        self.shots = shots
        self.sample_len = 1024
        self.feat_size = (self.sample_len // 2 ** layers) * h_size
        self.model = CNN_MMD(in_chn=1, hidden_chn=h_size, cb_num=layers,
                             embedding_size=self.feat_size, out_size=ways).to(device)
        # self.loss_fun = torch.nn.CrossEntropyLoss(reduction='mean')

    def model_save(self, path):
        filename = path+'(1)' if os.path.exists(path) else path
        torch.save(self.model.state_dict(), filename)
        print(f'Save model at: {filename}')

    def train_cnn(self, save_path):
        self.model.apply(weights_init2)
        opt = torch.optim.Adam(self.model.parameters())
        train_dataset = CNN_DataGenerator_torch(mode='train', shot=self.shots)
        test_dataset = CNN_DataGenerator_torch(mode='test', shot=self.shots)
        valid_dataset = CNN_DataGenerator_torch(mode='validation', shot=self.shots)

        Epochs = 1000
        Episodes = train_dataset.__len__()
        counter = 0
        train_time = 0.

        for ep in range(Epochs):
            Acc = 0.
            Loss = 0.
            t0 = time.time()
            for epi in range(Episodes):
                xx, yy = train_dataset.__getitem__(epi)
                xx, yy = torch.from_numpy(xx).to(device), torch.from_numpy(yy).long().to(device)

                idx = np.random.randint(0, test_dataset.__len__())
                xx_t, _ = test_dataset.__getitem__(idx)
                xx_t = torch.from_numpy(xx_t).to(device)

                loss, acc = self.model(xx, yy, xx_t)
                Loss += loss.item()
                Acc += acc.item()

                opt.zero_grad()
                loss.backward()
                opt.step()

            t1 = time.time()
            train_time += t1 - t0
            train_dataset.on_epoch_end()

            idx = np.random.randint(0, valid_dataset.__len__())
            xx, yy = valid_dataset.__getitem__(idx)
            xx, yy = torch.from_numpy(xx).to(device), torch.from_numpy(yy).long().to(device)

            self.model.eval()
            with torch.no_grad():
                loss_, acc_ = self.model(xx, yy)
            self.model.train()

            vis.line(Y=[[Loss/Episodes, loss_.item()]], X=[counter],
                     update=None if counter == 0 else 'append', win='Loss_CNN_MMD',
                     opts=dict(legend=['train', 'val'], title='Loss_CNN_MMD'))

            vis.line(Y=[[Acc/Episodes, acc_.item()]], X=[counter],
                     update=None if counter == 0 else 'append', win='Acc_CNN_MMD',
                     opts=dict(legend=['train', 'val'], title='Acc_CNN_MMD'))
            counter += 1

            if (ep+1) >=50 and (ep+1)%2==0:
                if input('\n== Stop training? == (y/n)\n').lower() == 'y':
                    new_save_path = save_path + rf'_ep{ep + 1}'
                    self.model_save(new_save_path)
                    print(f'*** Training time: {train_time: .4f} (s) ***')
                    break
                elif input('\n== Save model? == (y/n)\n').lower() == 'y':
                    new_save_path = save_path + rf'_ep{ep + 1}'
                    self.model_save(new_save_path)
                    print(f'*** Training time: {train_time: .4f} (s) ***')

    def test_cnn(self, load_path):
        self.model.load_state_dict(torch.load(load_path))
        print(f'Load Model successfully from [{load_path}]...')
        self.model.eval()

        test_dataset = CNN_DataGenerator_torch(mode='test', shot=self.shots)

        t0 = time.time()
        Episodes = test_dataset.__len__()
        test_time = 0.
        Acc = 0.
        Loss = 0.

        for epi in range(Episodes):
            xx, yy = test_dataset.__getitem__(epi)
            xx, yy = torch.from_numpy(xx).to(device), torch.from_numpy(yy).long().to(device)

            with torch.no_grad():
                loss, acc = self.model(xx, yy)
            Loss += loss.item()
            Acc += acc.item()
        t1 = time.time()
        test_time += t1 - t0

        print(f"Acc: {Acc / Episodes:.4f}, Loss: {Loss / Episodes:.4f}")
        print(f'*** Testing time: {test_time: .4f} (s) ***\n')


if __name__ == "__main__":
    from my_utils.init_utils import seed_torch

    seed_torch(2021)
    Net = CNN_MMD_learner(ways=10, shots=5)

    if input('Train? y/n\n').lower() == 'y':
        path = r"G:\model_save\meta_learning\CNN\cnn_mmd\5shot\cnn_mmd_C30"
        Net.train_cnn(save_path=path)
    if input('Test? y/n\n').lower() == 'y':
        load_path = r"G:\model_save\meta_learning\CNN\cnn_mmd\5shot\cnn_mmd_C30_ep50"
        Net.test_cnn(load_path)  # acc: 0.758
