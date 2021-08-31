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

from Models.CNN_torch.cnn_model import CNN
from Datasets.cwru_data import CNN_DataGenerator_torch
from my_utils.init_utils import seed_torch, weights_init2
from my_utils.train_utils import accuracy


vis = visdom.Visdom(env='yancy_meta')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CNN_FT_learner(object):
    def __init__(self, ways):
        h_size = 64
        layers = 4
        self.sample_len = 1024
        self.ways = ways
        self.feat_size = (self.sample_len // 2 ** layers) * h_size
        self.model = CNN(in_chn=1, hidden_chn=h_size, cb_num=layers,
                         embedding_size=self.feat_size, out_size=ways).to(device)
        self.loss_fun = torch.nn.CrossEntropyLoss(reduction='mean')

    def model_save(self, path):
        filename = path+'(1)' if os.path.exists(path) else path
        torch.save(self.model.state_dict(), filename)
        print(f'Save model at: {filename}')

    def fine_tune(self, x, y, ways, valid_dataset, train_steps):
        self.model.classifier = torch.nn.Linear(self.feat_size, ways).to(device)
        opt = torch.optim.Adam(self.model.classifier.parameters())
        for p in self.model.feature_net.parameters():
            p.requires_grad = False

        batch_size = 10
        counter = 0
        for _ in range(train_steps):
            index = np.random.randint(0, x.shape[1], (x.shape[0], batch_size))
            xx = torch.tensor([x[i, k] for (i, k) in enumerate(index)]).reshape(-1, 1, self.sample_len)
            yy = torch.tensor([y[i, k] for (i, k) in enumerate(index)]).reshape(-1)
            xx, yy = xx.float().to(device), yy.long().to(device)
            # print(xx.shape, yy.shape)

            yy_ = self.model(xx)
            loss = self.loss_fun(yy_, yy)
            acc = accuracy(yy_, yy)

            opt.zero_grad()
            loss.backward()
            opt.step()

            # validation:
            idx = np.random.randint(0, valid_dataset.__len__())
            xx, yy = valid_dataset.__getitem__(idx)
            xx, yy = torch.from_numpy(xx).to(device), torch.from_numpy(yy).long().to(device)

            self.model.eval()
            with torch.no_grad():
                yy_ = self.model(xx)
                loss_ = self.loss_fun(yy_, yy)
                acc_ = accuracy(yy_, yy)
            self.model.train()

            vis.line(Y=[[loss.item(), loss_.item()]], X=[counter],
                     update=None if counter == 0 else 'append', win='Loss_CNN_FT',
                     opts=dict(legend=['train', 'val'], title='Loss_CNN_FT'))

            vis.line(Y=[[acc.item(), acc_.item()]], X=[counter],
                     update=None if counter == 0 else 'append', win='Acc_CNN_FT',
                     opts=dict(legend=['train', 'val'], title='Acc_CNN_FT'))
            counter += 1

    def train_cnn(self, save_path, shots):
        self.model.apply(weights_init2)
        opt = torch.optim.Adam(self.model.parameters())
        train_dataset = CNN_DataGenerator_torch(mode='train', ways=self.ways, shot=shots)
        valid_dataset = CNN_DataGenerator_torch(mode='validation', ways=self.ways, shot=shots)

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

                yy_ = self.model(xx)
                loss = self.loss_fun(yy_, yy)
                acc = accuracy(yy_, yy)
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
                yy_ = self.model(xx)
                loss_ = self.loss_fun(yy_, yy)
                acc_ = accuracy(yy_, yy)
            self.model.train()

            vis.line(Y=[[Loss/Episodes, loss_.item()]], X=[counter],
                     update=None if counter == 0 else 'append', win='Loss_CNN',
                     opts=dict(legend=['train', 'val'], title='Loss_CNN'))

            vis.line(Y=[[Acc/Episodes, acc_.item()]], X=[counter],
                     update=None if counter == 0 else 'append', win='Acc_CNN',
                     opts=dict(legend=['train', 'val'], title='Acc_CNN'))
            counter += 1

            if (ep+1) >=50 and (ep+1)%2==0:
                if input('\n== Stop training? == (y/n)\n').lower() == 'y':
                    new_save_path = save_path + rf'_ep{ep + 1}'
                    self.model_save(new_save_path)
                    print(f'*** Training time: {train_time: .3f} (s) ***')
                    break
                elif input('\n== Save model? == (y/n)\n').lower() == 'y':
                    new_save_path = save_path + rf'_ep{ep + 1}'
                    self.model_save(new_save_path)
                    print(f'*** Training time: {train_time: .3f} (s) ***')

    def test_cnn_ft(self, load_path, shots):
        self.model.load_state_dict(torch.load(load_path))
        print(f'Load Model successfully from [{load_path}]...')

        print(f"{self.ways}-ways, {shots}-shots for testing ...")
        ft_dataset = CNN_DataGenerator_torch(mode='finetune', ways=self.ways, shot=shots)
        test_dataset = CNN_DataGenerator_torch(mode='test', ways=self.ways, shot=shots)

        ft_x, ft_y = ft_dataset.x, ft_dataset.y
        test_ways = self.ways

        t0 = time.time()
        self.fine_tune(ft_x, ft_y, test_ways, test_dataset, train_steps=200)

        Episodes = test_dataset.__len__()
        self.model.eval()
        test_time = 0.
        Acc = 0.
        Loss = 0.

        for epi in range(Episodes):
            xx, yy = test_dataset.__getitem__(epi)
            xx, yy = torch.from_numpy(xx).to(device), torch.from_numpy(yy).long().to(device)
            with torch.no_grad():
                yy_ = self.model(xx)
                loss = self.loss_fun(yy_, yy)
                acc = accuracy(yy_, yy)
            Loss += loss.item()
            Acc += acc.item()
        t1 = time.time()
        test_time += t1 - t0

        print(f"Acc: {Acc/Episodes:.4f}, Loss: {Loss/Episodes:.4f}")
        print(f'*** Testing time: {test_time: .4f} (s) ***\n')

    def test_cnn(self, load_path, shots):
        self.model.load_state_dict(torch.load(load_path))
        print(f'Load Model successfully from [{load_path}]...')

        test_dataset = CNN_DataGenerator_torch(mode='test', ways=self.ways, shot=shots)

        t0 = time.time()
        Episodes = test_dataset.__len__()
        self.model.eval()
        test_time = 0.
        Acc = 0.
        Loss = 0.

        for epi in range(Episodes):
            xx, yy = test_dataset.__getitem__(epi)
            xx, yy = torch.from_numpy(xx).to(device), torch.from_numpy(yy).long().to(device)

            with torch.no_grad():
                yy_ = self.model(xx)
                loss = self.loss_fun(yy_, yy)
                acc = accuracy(yy_, yy)
            Loss += loss.item()
            Acc += acc.item()
        t1 = time.time()
        test_time += t1 - t0

        print(f"Acc: {Acc / Episodes:.4f}, Loss: {Loss / Episodes:.4f}")
        print(f'*** Testing time: {test_time: .4f} (s) ***\n')


if __name__ == "__main__":
    from my_utils.init_utils import seed_torch

    seed_torch(2021)
    # Net = CNN_FT_learner(ways=10)  # T1
    Net = CNN_FT_learner(ways=4)  # T2

    if input('Train? y/n\n').lower() == 'y':
        # path = r"G:\model_save\meta_learning\CNN\cnn_ft\5shot\cnn_ft_C30"
        # Net.train_cnn(save_path=path, shots=5)

        path = r"G:\model_save\meta_learning\CNN\cnn_ft\5shot\cnn_ft_C30"
        Net.train_cnn(save_path=path, shots=5)

    if input('Test? y/n\n').lower() == 'y':
        # load_path = r"G:\model_save\meta_learning\CNN\cnn_ft\5shot\cnn_ft_C30_ep50"
        # Net.test_cnn(load_path, shots=5)  # 0.718
        # Net.test_cnn_ft(load_path, shots=5)  # acc: 0.758

        # load_path = r"G:\model_save\meta_learning\CNN\cnn_ft\5shot\cnn_ft_C30_ep50"
        # Net.test_cnn_ft(load_path, shots=1)  # acc: 0.758

        load_path = r"G:\model_save\meta_learning\CNN\cnn_ft\5shot\cnn_ft_C30_ep72"
        Net.test_cnn_ft(load_path, shots=5)  # acc: 0.758
        Net.test_cnn_ft(load_path, shots=1)  # acc: 0.758
