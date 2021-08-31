import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
from tensorflow.keras.utils import Sequence
# import torch
# import torchvision
from torch.utils import data

from Datasets.cwru_path import T0, T3, T4w, T6w
from Datasets.mat2csv import get_data_csv
from my_utils.init_utils import one_hot_encode, sample_label_shuffle
from my_utils.init_utils import my_normalization as normalization

N_TRAIN_EACH_CLASS = 20


class Data_CWRU:
    def __init__(self, T1=True):
        if T1:
            self.train = T3  # T3
            self.test = T0  # T0
        else:
            self.train = T6w
            self.test = T4w  # 4 new classes

    def get_data(self, train_mode=True, n_each_class=10, sample_len=1024, normalize=True):
        data_file = self.train if train_mode else self.test
        data_size = n_each_class * sample_len
        n_way = len(data_file)  # the num of categories
        data_set = []
        for i in range(n_way):
            data = get_data_csv(file_dir=data_file[i], num=data_size, header=0, shift_step=200)  # (N, len)
            data = data.reshape(-1, sample_len)
            data = normalization(data) if normalize else data
            data_set.append(data)
        data_set = np.stack(data_set, axis=0)  # (n_way, n, sample_len)
        data_set = np.asarray(data_set, dtype=np.float32)
        label = np.arange(n_way, dtype=np.int32).reshape(n_way, 1)
        label = np.repeat(label, n_each_class, axis=1)  # [n_way, examples]
        return data_set, label  # [Nc,num_each_way,1,1024], [Nc, 50]


class MANN_DataGenerator(Sequence):
    def __init__(self, mode='train', batch_size=16, num_batch=16):
        self.batch_size = batch_size
        self.sample_len = 1024
        self.num_batch = num_batch
        self.n_way = self.seq_len = 10
        self.__getdata__(mode)
        self.ID_list = np.arange(0, len(self.x))

    def __len__(self):
        return self.num_batch

    def on_epoch_end(self):
        # seed_num = np.random.randint(0, 100)
        # np.random.seed(seed_num)
        # np.random.shuffle(self.x)
        # np.random.seed(seed_num)
        # np.random.shuffle(self.y)
        pass

    def __getitem__(self, item):
        batch_x = []
        batch_y = []
        for i in range(self.batch_size):
            idx = np.random.randint(0, len(self.x[0]), self.seq_len)
            batch_x.append([self.x[j, idx[j]] for j in range(self.seq_len)])  # (10, 1024)
            batch_y.append([self.y[j, idx[j]] for j in range(self.seq_len)])   # (10, )
        batch_x = np.stack(batch_x)  # (batch_size, seq_len, sample_len)
        batch_y = np.stack(batch_y)  # (batch_size, seq_len)
        batch_y = one_hot_encode(batch_y, self.n_way)
        x_label = np.concatenate([np.zeros(shape=[self.batch_size, 1, self.n_way], dtype=np.float32),
                                          batch_y[:, :-1, :]], axis=1)
        # if (item+1) == self.num_batch:
        #     self.on_epoch_end()

        return batch_x, x_label, batch_y

    def __getdata__(self, mode):
        if mode=='train':
            data = Data_CWRU().get_data(train_mode=True, n_each_class=N_TRAIN_EACH_CLASS,
                                        sample_len=self.sample_len, normalize=True)
            self.x, self.y = data[0], data[1]
        else:
            data = Data_CWRU().get_data(train_mode=False, n_each_class=200,
                                        sample_len=self.sample_len, normalize=True)
            if mode == 'validation':
                self.x, self.y = data[0][:, :100], data[1][:, :100]
            elif mode == 'test':
                self.x, self.y = data[0], data[1]
            else:
                exit('Mode error')
        self.n_way = len(self.x)
        # x: (n_way, n, len), y: (n_way, n)
        # for MANN:
        # self.x = self.x.reshape(-1, self.sample_len)
        # self.y = self.y.reshape(-1)
        # seed_num = np.random.randint(0, 100)
        # np.random.seed(seed_num)
        # np.random.shuffle(self.x)
        # np.random.seed(seed_num)
        # np.random.shuffle(self.y)

        print(f'x shape: {self.x.shape}, y shape: {self.y.shape}')


class CNN_DataGenerator(Sequence):
    def __init__(self, mode='train', shot=10):
        self.shot = shot
        self.sample_len = 1024
        self.n_way = 10
        self.__getdata__(mode)
        self.ID_list = np.arange(0, len(self.x))

    def __len__(self):
        return int(self.x.shape[1] // self.shot)

    def on_epoch_end(self):
        for i in range(len(self.x)):
            self.x[i], self.y[i] = sample_label_shuffle(self.x[i], self.y[i])
        pass

    def __getitem__(self, item):
        batch_x = self.x[:, item * self.shot:(item + 1) * self.shot]
        batch_y = self.y[:, item * self.shot:(item + 1) * self.shot]
        batch_x, batch_y = batch_x.reshape([-1, self.sample_len, 1]), batch_y.reshape(-1)
        batch_y = one_hot_encode(batch_y, self.n_way)

        if (item+1) == self.x.shape[1] // self.shot:
            self.on_epoch_end()

        return batch_x, batch_y

    def __getdata__(self, mode):
        if mode=='train':
            data = Data_CWRU().get_data(train_mode=True, n_each_class=N_TRAIN_EACH_CLASS,
                                        sample_len=self.sample_len, normalize=True)
            self.x, self.y = data[0], data[1]
        else:
            data = Data_CWRU().get_data(train_mode=False, n_each_class=200,
                                        sample_len=self.sample_len, normalize=True)
            if mode == 'validation':
                self.x, self.y = data[0][:, :100], data[1][:, :100]
            elif mode == 'test':
                self.x, self.y = data[0], data[1]
            else:
                exit('Mode error')
        self.x = np.expand_dims(self.x, axis=-1)
        self.n_way = len(self.x)
        # x: (n_way, n, len, 1), y: (n_way, n)
        print(f'x shape: {self.x.shape}, y shape: {self.y.shape}')


class CNN_DataGenerator_torch(data.Dataset):
    def __init__(self, mode, ways, shot):
        self.shot = shot
        self.sample_len = 1024
        self.n_way = ways
        self.task_mode = True if ways == 10 else False
        self.__getdata__(mode)
        self.ID_list = np.arange(0, len(self.x))

    def __len__(self):
        return int(self.x.shape[1] // self.shot)

    def on_epoch_end(self):
        for i in range(len(self.x)):
            self.x[i], self.y[i] = sample_label_shuffle(self.x[i], self.y[i])

    def __getitem__(self, item):
        batch_x = self.x[:self.n_way, item * self.shot:(item + 1) * self.shot]
        batch_y = self.y[:self.n_way, item * self.shot:(item + 1) * self.shot]
        batch_x, batch_y = batch_x.reshape([-1, 1, self.sample_len]), batch_y.reshape(-1)
        return batch_x, batch_y

    def __getdata__(self, mode):
        if mode=='train':
            data = Data_CWRU(self.task_mode).get_data(train_mode=True, n_each_class=N_TRAIN_EACH_CLASS,
                                                      sample_len=self.sample_len, normalize=True)
            self.x, self.y = data[0], data[1]
        else:
            data = Data_CWRU(self.task_mode).get_data(train_mode=False, n_each_class=200,
                                                      sample_len=self.sample_len, normalize=True)
            if mode == 'validation':
                self.x, self.y = data[0][:, :100], data[1][:, :100]
            elif mode == 'finetune':
                self.x, self.y = data[0][:, :self.shot], data[1][:, :self.shot]
            elif mode == 'test':
                self.x, self.y = data[0], data[1]
            else:
                exit('Mode error')
        self.x = np.expand_dims(self.x, axis=-2)
        # x: (n_way, n, 1, len), y: (n_way, n)
        print(f'x shape: {self.x.shape}, y shape: {self.y.shape}')


class MAML_Dataset(data.Dataset):
    def __init__(self, mode, ways):
        super().__init__()
        self.sample_len = 1024
        self.task_mode = True if ways == 10 else False
        self.__getdata__(mode)

    def __getdata__(self, mode):
        if mode == 'train':
            data = Data_CWRU(self.task_mode).get_data(train_mode=True, n_each_class=N_TRAIN_EACH_CLASS,
                                                      sample_len=self.sample_len, normalize=True)
            self.x, self.y = data[0], data[1]
        else:
            data = Data_CWRU(self.task_mode).get_data(train_mode=False, n_each_class=200,
                                                      sample_len=self.sample_len, normalize=True)
            if mode == 'validation':
                self.x, self.y = data[0][:, :100], data[1][:, :100]
            elif mode == 'test':
                self.x, self.y = data[0], data[1]
            else:
                exit('Mode error')
        # self.x = np.expand_dims(self.x, axis=-1)  # x: (n_way, n, len, 1), y: (n_way, n)
        self.x = self.x.reshape([-1, 1, self.sample_len])  # x: (n_way*n, len, 1), y: (n_way*n)
        self.y = self.y.reshape(-1)
        self.x, self.y = sample_label_shuffle(self.x, self.y)
        print(f'x-shape: {self.x.shape}, y-shape: {self.y.shape}')

    def __getitem__(self, item):
        x = self.x[item]  # (NC, l)
        y = self.y[item]
        return x, y  # , label

    def __len__(self):
        return len(self.x)


if __name__ == "__main__":
    import learn2learn as l2l

    train_dataset = l2l.data.MetaDataset(MAML_Dataset(mode='train', ways=10))
    shots = 5  # 注意要保证: shots*2*ways >= len(self.x)
    ways = 10
    num_tasks = 100

    train_tasks = l2l.data.TaskDataset(train_dataset, task_transforms=[
        l2l.data.transforms.NWays(train_dataset, ways),
        l2l.data.transforms.KShots(train_dataset, 2 * shots),
        l2l.data.transforms.LoadData(train_dataset),
        l2l.data.transforms.RemapLabels(train_dataset),
        l2l.data.transforms.ConsecutiveLabels(train_dataset),
    ], num_tasks=num_tasks)

    for i in range(num_tasks+1):
        task = train_tasks.sample()
        data, labels = task
        print(data.shape, labels.shape)
        # print(data)
        print(labels)
        print(f'{i+1}')


    exit()
    # train_data = Data_CWRU().get_data(train_mode=True, n_each_class=10,
    #                                   sample_len=1024, normalize=True)
    # print(train_data[0].shape, train_data[1].shape)

    # gen = MANN_DataGenerator(mode='train', batch_size=16)
    gen = CNN_DataGenerator(mode='train', shot=5)
    for ep in range(2):
        print(f'ep {ep+1}')
        # print(gen.x[0, :5])
        # for i_batch, (x, x_label, y) in enumerate(gen):
        #     print(f'\t{i_batch+1} batch:', x.shape, x_label.shape, y.shape)
        #     # print(x_label[0],'\n', y[0])
        #     # exit()
        # new_g = iter(gen)
        for epi in range(10):
            x, y = gen.__getitem__(epi)
            print(f'\t{epi+1} batch:', x.shape, y.shape)
            # print(x_label[0],'\n', y[0])
            # exit()

