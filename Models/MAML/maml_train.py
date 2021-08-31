"""
Code refers to: https://github.com/learnables/learn2learn/blob/master/examples/vision/maml_omniglot.py
By Yancy F. 2021-8-29
"""

import torch
import numpy as np

import learn2learn as l2l
import visdom
import os
import time

from Models.MAML.maml_model import Net4CNN
from Datasets.cwru_data import MAML_Dataset
from my_utils.train_utils import accuracy

vis = visdom.Visdom(env='yancy_meta')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MAML_learner(object):
    def __init__(self, ways):
        h_size = 64
        layers = 4
        sample_len = 1024
        feat_size = (sample_len//2**layers)*h_size
        self.model = Net4CNN(output_size=ways, hidden_size=h_size, layers=layers,
                             channels=1, embedding_size=feat_size).to(device)
        self.ways = ways
        # print(self.model)

    def build_tasks(self, mode='train', ways=10, shots=5, num_tasks=100, filter_labels=None):
        dataset = l2l.data.MetaDataset(MAML_Dataset(mode=mode, ways=ways))
        new_ways = len(filter_labels) if filter_labels is not None else ways
        # label_shuffle_per_task = False if ways <=30 else True
        assert shots * 2 * new_ways <= dataset.__len__()//ways*new_ways, "Reduce the number of shots!"
        tasks = l2l.data.TaskDataset(dataset, task_transforms=[
            l2l.data.transforms.FusedNWaysKShots(dataset, new_ways, 2 * shots, filter_labels=filter_labels),
            l2l.data.transforms.LoadData(dataset),
            # l2l.data.transforms.RemapLabels(dataset, shuffle=label_shuffle_per_task),
            l2l.data.transforms.RemapLabels(dataset, shuffle=True),
            # do not keep the original labels, use (0 ,..., n-1);
            # if shuffle=True, to shuffle labels at each task.
            l2l.data.transforms.ConsecutiveLabels(dataset),
            # re-order samples and make their original labels as (0 ,..., n-1).
        ], num_tasks=num_tasks)
        return tasks

    @staticmethod
    def fast_adapt(batch, learner, loss, adaptation_steps, shots, ways):
        data, labels = batch
        data, labels = data.to(device), labels.to(device)

        # Separate data into adaptation/evaluation sets
        adaptation_indices = np.zeros(data.size(0), dtype=bool)
        adaptation_indices[np.arange(shots * ways) * 2] = True
        evaluation_indices = torch.from_numpy(~adaptation_indices)  # 偶数序号为True, 奇数序号为False
        adaptation_indices = torch.from_numpy(adaptation_indices)  # 偶数序号为False, 奇数序号为True
        adaptation_data, adaptation_labels = data[adaptation_indices], labels[adaptation_indices]
        evaluation_data, evaluation_labels = data[evaluation_indices], labels[evaluation_indices]

        # Adapt the model
        for step in range(adaptation_steps):
            train_error = loss(learner(adaptation_data), adaptation_labels)
            learner.adapt(train_error)

        # Evaluate the adapted model
        predictions = learner(evaluation_data)
        valid_error = loss(predictions, evaluation_labels)
        valid_accuracy = accuracy(predictions, evaluation_labels)
        return valid_error, valid_accuracy

    def model_save(self, path):
        filename = path+'(1)' if os.path.exists(path) else path
        torch.save(self.model.state_dict(), filename)
        print(f'Save model at: {filename}')

    def train(self, save_path, shots=5):
        # label_shuffle_per_task=True:
        meta_lr = 0.005 # 0.005, <0.01
        fast_lr = 0.05 # 0.01

        maml = l2l.algorithms.MAML(self.model, lr=fast_lr)
        opt = torch.optim.Adam(maml.parameters(), meta_lr)
        loss = torch.nn.CrossEntropyLoss(reduction='mean')

        train_ways = valid_ways = self.ways
        print(f"{train_ways}-ways, {shots}-shots for training ...")
        train_tasks = self.build_tasks('train', train_ways, shots, 1000, None)
        valid_tasks = self.build_tasks('validation', valid_ways, shots, 1000, None)
        # test_tasks = self.build_tasks('test', test_ways, shots, 1000, None)

        counter = 0
        Epochs = 1000
        meta_batch_size = 16
        adaptation_steps = 1 if shots==5 else 3

        for ep in range(Epochs):
            t0 = time.time()

            meta_train_error = 0.0
            meta_train_accuracy = 0.0
            meta_valid_error = 0.0
            meta_valid_accuracy = 0.0

            opt.zero_grad()
            for _ in range(meta_batch_size):
                # 1) Compute meta-training loss
                learner = maml.clone()
                task = train_tasks.sample()  # or a batch
                evaluation_error, evaluation_accuracy = self.fast_adapt(task, learner, loss,
                                                                   adaptation_steps, shots, train_ways)
                evaluation_error.backward()
                meta_train_error += evaluation_error.item()
                meta_train_accuracy += evaluation_accuracy.item()

                # 2) Compute meta-validation loss
                learner = maml.clone()
                task = valid_tasks.sample()
                evaluation_error, evaluation_accuracy = self.fast_adapt(task, learner, loss,
                                                                        adaptation_steps, shots, valid_ways)
                meta_valid_error += evaluation_error.item()
                meta_valid_accuracy += evaluation_accuracy.item()

            # Print some metrics
            t1 = time.time()
            print(f'Time /epoch: {t1-t0:.4f} s')
            print('\n')
            print('Iteration', ep+1)
            print(f'Meta Train Error: {meta_train_error / meta_batch_size: .4f}')
            print(f'Meta Train Accuracy: {meta_train_accuracy / meta_batch_size: .4f}')
            print(f'Meta Valid Error: {meta_valid_error / meta_batch_size: .4f}')
            print(f'Meta Valid Accuracy: {meta_valid_accuracy / meta_batch_size: .4f}')

            # Take the meta-learning step:
            # Average the accumulated gradients and optimize
            for p in maml.parameters():
                p.grad.data.mul_(1.0 / meta_batch_size)
            opt.step()

            vis.line(Y=[[meta_train_error / meta_batch_size, meta_valid_error / meta_batch_size]], X=[counter],
                     update=None if counter == 0 else 'append', win='Loss_MAML',
                     opts=dict(legend=['train', 'val'], title='Loss_MAML'))

            vis.line(Y=[[meta_train_accuracy / meta_batch_size, meta_valid_accuracy / meta_batch_size]], X=[counter],
                     update=None if counter == 0 else 'append', win='Acc_MAML',
                     opts=dict(legend=['train', 'val'], title='Acc_MAML'))
            counter += 1

            # if (ep + 1) >= 400 and (meta_valid_accuracy / meta_batch_size) > 0.88:
            #     new_save_path = save_path + rf'_ep{ep + 1}'
            #     self.model_save(new_save_path)
                # break
            if (ep+1) >=400 and (ep+1)%2==0:
                if input('\n== Stop training? == (y/n)\n').lower() == 'y':
                    new_save_path = save_path + rf'_ep{ep + 1}'
                    self.model_save(new_save_path)
                    break
                elif input('\n== Save model? == (y/n)\n').lower() == 'y':
                    new_save_path = save_path + rf'_ep{ep + 1}'
                    self.model_save(new_save_path)

    def test(self, load_path, inner_steps=10, shots=5):
        self.model.load_state_dict(torch.load(load_path))
        print('Load Model successfully from [%s]' % load_path)

        test_ways = self.ways
        shots = shots
        print(f"{test_ways}-ways, {shots}-shots for testing ...")
        # meta_lr = 0.005  # 0.005, <0.01
        fast_lr = 0.05  # 0.01
        test_tasks = self.build_tasks('test', test_ways, shots, 1000, None)
        maml = l2l.algorithms.MAML(self.model, lr=fast_lr)
        loss = torch.nn.CrossEntropyLoss(reduction='mean')

        meta_batch_size = 100
        adaptation_steps = inner_steps  # 1
        meta_test_error = 0.0
        meta_test_accuracy = 0.0
        t0 = time.time()

        for _ in range(meta_batch_size):
            # Compute meta-testing loss
            learner = maml.clone()
            task = test_tasks.sample()
            evaluation_error, evaluation_accuracy = self.fast_adapt(task, learner, loss,
                                                                    adaptation_steps, shots, test_ways)
            meta_test_error += evaluation_error.item()
            meta_test_accuracy += evaluation_accuracy.item()

        t1 = time.time()
        print(f"-------- Time for {meta_batch_size*shots} samples: {t1-t0:.4f} sec. ----------")
        print(f'Meta Test Error: {meta_test_error / meta_batch_size: .4f}')
        print(f'Meta Test Accuracy: {meta_test_accuracy / meta_batch_size: .4f}\n')


if __name__ == "__main__":
    from my_utils.init_utils import seed_torch

    seed_torch(2021)
    # Net = MAML_learner(ways=10)  # T1
    Net = MAML_learner(ways=4)  # T2

    if input('Train? y/n\n').lower() == 'y':
        # path = r"G:\model_save\meta_learning\MAML\5shot\MAML_C30"
        # Net.train(save_path=path, shots=5)

        # path = r"G:\model_save\meta_learning\MAML\1shot\MAML_C30"
        # Net.train(save_path=path, shots=1)

        # path = r"G:\model_save\meta_learning\MAML\5shot\MAML_T2"
        # Net.train(save_path=path, shots=5)

        path = r"G:\model_save\meta_learning\MAML\1shot\MAML_T2"
        Net.train(save_path=path, shots=1)

    if input('Test? y/n\n').lower() == 'y':
        # load_path = r"G:\model_save\meta_learning\MAML\5shot\MAML_C30_ep457"  # acc: 0.847; 0.958
        # Net.test(load_path, inner_steps=10, shots=5)

        # load_path = r"G:\model_save\meta_learning\MAML\1shot\MAML_C30_ep436"  # acc: 0.874
        # Net.test(load_path, inner_steps=20, shots=1)

        # load_path = r"G:\model_save\meta_learning\MAML\5shot\MAML_T2_ep404"  # acc:
        # Net.test(load_path, inner_steps=10, shots=5)

        load_path = r"G:\model_save\meta_learning\MAML\1shot\MAML_T2_ep414"
        Net.test(load_path, inner_steps=10, shots=1)

