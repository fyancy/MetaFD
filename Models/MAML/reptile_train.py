"""
In this implementation, we use the architecture of Reptile as the same as MAML.
"""

import torch
import visdom
import numpy as np
import learn2learn as l2l
import copy
import time
import os

from Models.MAML.maml_model import Net4CNN
from Datasets.cwru_data import MAML_Dataset
from my_utils.train_utils import accuracy

vis = visdom.Visdom(env='yancy_meta')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Reptile_learner(object):
    def __init__(self, ways):
        super().__init__()
        h_size = 64
        layers = 4
        sample_len = 1024
        feat_size = (sample_len // 2 ** layers) * h_size
        self.model = Net4CNN(output_size=ways, hidden_size=h_size, layers=layers,
                             channels=1, embedding_size=feat_size).to(device)
        self.ways = ways

    @staticmethod
    def fast_adapt(batch, learner, adapt_opt, loss, adaptation_steps, shots, ways, batch_size=10):
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
            idx = torch.randint(adaptation_data.shape[0], size=(batch_size,))
            adapt_x = adaptation_data[idx]
            adapt_y = adaptation_labels[idx]

            adapt_opt.zero_grad()
            error = loss(learner(adapt_x), adapt_y)
            error.backward()
            adapt_opt.step()

        # Evaluate the adapted model
        predictions = learner(evaluation_data)
        valid_error = loss(predictions, evaluation_labels)
        valid_accuracy = accuracy(predictions, evaluation_labels)
        return valid_error, valid_accuracy

    @staticmethod
    def build_tasks(mode='train', ways=10, shots=5, num_tasks=100, filter_labels=None):
        dataset = l2l.data.MetaDataset(MAML_Dataset(mode=mode, ways=ways))
        new_ways = len(filter_labels) if filter_labels is not None else ways
        # label_shuffle_per_task = False if ways <=30 else True
        assert shots * 2 * new_ways <= dataset.__len__() // ways * new_ways, "Reduce the number of shots!"
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

    def model_save(self, model_path):
        filename = model_path+'(1)' if os.path.exists(model_path) else model_path
        torch.save(self.model.state_dict(), filename)
        print(f'Save model at: {filename}')

    def train(self, save_path, shots):
        # 5 shot:
        # meta_lr = 1.0  # 1.0
        # fast_lr = 0.001  # 0.001-0.005

        # 1 shot:
        meta_lr = 0.1  # 0.005, <0.01
        fast_lr = 0.001  # 0.05

        opt = torch.optim.SGD(self.model.parameters(), meta_lr)
        adapt_opt = torch.optim.Adam(self.model.parameters(), lr=fast_lr, betas=(0, 0.999))  # 5 shot

        # adapt_opt_state = adapt_opt.state_dict()
        init_adapt_opt_state = adapt_opt.state_dict()
        adapt_opt_state = None
        loss = torch.nn.CrossEntropyLoss(reduction='mean')

        train_ways = valid_ways = self.ways
        print(f"{train_ways}-ways, {shots}-shots for training ...")
        train_tasks = self.build_tasks('train', train_ways, shots, 1000, None)
        valid_tasks = self.build_tasks('validation', valid_ways, shots, 1000, None)

        counter = 0
        Epochs = 10000
        meta_batch_size = 16
        train_steps = 4  # 8
        test_steps = 4
        train_bsz = 10
        test_bsz = 15

        for ep in range(Epochs):
            t0 = time.time()
            if ep == 0:
                adapt_opt_state = init_adapt_opt_state
            opt.zero_grad()

            meta_train_error = 0.0
            meta_train_accuracy = 0.0
            meta_valid_error = 0.0
            meta_valid_accuracy = 0.0

            # anneal meta-lr
            frac_done = float(ep) / 100000
            new_lr = frac_done * meta_lr + (1 - frac_done) * meta_lr
            for pg in opt.param_groups:
                pg['lr'] = new_lr

            # zero-grad the parameters
            for p in self.model.parameters():
                p.grad = torch.zeros_like(p.data)

            for task in range(meta_batch_size):
                # Compute meta-training loss
                learner = copy.deepcopy(self.model)
                adapt_opt = torch.optim.Adam(learner.parameters(), lr=fast_lr, betas=(0, 0.999))
                adapt_opt.load_state_dict(adapt_opt_state)
                batch = train_tasks.sample()
                evaluation_error, evaluation_accuracy = self.fast_adapt(batch, learner, adapt_opt, loss,
                                                                        train_steps, shots, train_ways, train_bsz)
                adapt_opt_state = adapt_opt.state_dict()
                for p, l in zip(self.model.parameters(), learner.parameters()):
                    p.grad.data.add_(l.data, alpha=-1.0)

                meta_train_error += evaluation_error.item()
                meta_train_accuracy += evaluation_accuracy.item()

                # Compute meta-validation loss
                learner = copy.deepcopy(self.model)
                adapt_opt = torch.optim.Adam(learner.parameters(), lr=fast_lr, betas=(0, 0.999))
                # adapt_opt.load_state_dict(adapt_opt_state)
                adapt_opt.load_state_dict(init_adapt_opt_state)
                batch = valid_tasks.sample()
                evaluation_error, evaluation_accuracy = self.fast_adapt(batch, learner, adapt_opt, loss,
                                                                        test_steps, shots, train_ways, test_bsz)
                meta_valid_error += evaluation_error.item()
                meta_valid_accuracy += evaluation_accuracy.item()

            # Print some metrics
            t1 = time.time()
            print(f'Time /epoch: {t1-t0:.3f} s')

            print('\n')
            print('Iteration', ep + 1)
            print(f'Meta Train Error: {meta_train_error / meta_batch_size: .4f}')
            print(f'Meta Train Accuracy: {meta_train_accuracy / meta_batch_size: .4f}')
            print(f'Meta Valid Error: {meta_valid_error / meta_batch_size: .4f}')
            print(f'Meta Valid Accuracy: {meta_valid_accuracy / meta_batch_size: .4f}')

            # Take the meta-learning step:
            # Average the accumulated gradients and optimize
            for p in self.model.parameters():
                p.grad.data.mul_(1.0 / meta_batch_size).add_(p.data)
            opt.step()

            vis.line(Y=[[meta_train_error / meta_batch_size, meta_valid_error / meta_batch_size]], X=[counter],
                     update=None if counter == 0 else 'append', win='Loss_Reptile',
                     opts=dict(legend=['train', 'val'], title='Loss_Reptile'))

            vis.line(Y=[[meta_train_accuracy / meta_batch_size, meta_valid_accuracy / meta_batch_size]], X=[counter],
                     update=None if counter == 0 else 'append', win='Acc_Reptile',
                     opts=dict(legend=['train', 'val'], title='Acc_Reptile'))
            counter += 1

            if (ep + 1) >= 700 and (ep + 1) % 2 == 0:
                if input('\n== Stop training? == (y/n)\n').lower() == 'y':
                    new_save_path = save_path + rf'_ep{ep + 1}'
                    self.model_save(new_save_path)
                    break
                elif input('\n== Save model? == (y/n)\n').lower() == 'y':
                    new_save_path = save_path + rf'_ep{ep + 1}'
                    self.model_save(new_save_path)

    def test(self, load_path, shots, inner_test_steps=10):
        self.model.load_state_dict(torch.load(load_path))
        print('Load Model successfully from [%s]' % load_path)

        # meta_lr = 1.0  # 0.005, <0.01
        fast_lr = 0.001  # 0.05
        # opt = torch.optim.SGD(self.model.parameters(), meta_lr)
        adapt_opt = torch.optim.Adam(self.model.parameters(), lr=fast_lr, betas=(0, 0.999))
        init_adapt_opt_state = adapt_opt.state_dict()
        loss = torch.nn.CrossEntropyLoss(reduction='mean')

        test_ways = self.ways
        print(f"{test_ways}-ways, {shots}-shots for testing ...")
        test_tasks = self.build_tasks('test', test_ways, shots, 1000, None)

        meta_batch_size = 100
        test_steps = inner_test_steps  # 50
        test_bsz = 15

        meta_test_error = 0.0
        meta_test_accuracy = 0.0
        t0 = time.time()

        # zero-grad the parameters
        for p in self.model.parameters():
            p.grad = torch.zeros_like(p.data)

        for task in range(meta_batch_size):
            # Compute meta-validation loss
            learner = copy.deepcopy(self.model)
            adapt_opt = torch.optim.Adam(learner.parameters(), lr=fast_lr, betas=(0, 0.999))
            # adapt_opt.load_state_dict(adapt_opt_state)
            adapt_opt.load_state_dict(init_adapt_opt_state)
            batch = test_tasks.sample()
            evaluation_error, evaluation_accuracy = self.fast_adapt(batch, learner, adapt_opt, loss,
                                                                    test_steps, shots, test_ways, test_bsz)
            meta_test_error += evaluation_error.item()
            meta_test_accuracy += evaluation_accuracy.item()

        t1 = time.time()
        print(f"-------- Time for {meta_batch_size * shots} samples: {t1 - t0:.4f} sec. ----------")
        print(f'Meta Test Error: {meta_test_error / meta_batch_size: .4f}')
        print(f'Meta Test Accuracy: {meta_test_accuracy / meta_batch_size: .4f}\n')


if __name__ == "__main__":
    from my_utils.init_utils import seed_torch

    seed_torch(2021)
    # Net = Reptile_learner(ways=10)  # T1
    Net = Reptile_learner(ways=4)  # T2

    if input('Train? y/n\n').lower() == 'y':
        # path = r"G:\model_save\meta_learning\Reptile\5shot\Reptile_C30"
        # Net.train(save_path=path, shots=5)

        # path = r"G:\model_save\meta_learning\Reptile\1shot\Reptile_C30"
        # Net.train(save_path=path, shots=1)

        # path = r"G:\model_save\meta_learning\Reptile\5shot\Reptile_T2"
        # Net.train(save_path=path, shots=5)

        path = r"G:\model_save\meta_learning\Reptile\5shot\Reptile_T2"
        Net.train(save_path=path, shots=1)

    if input('Test? y/n\n').lower() == 'y':
        # path = r"G:\model_save\meta_learning\Reptile\5shot\Reptile_C30_ep730"
        # Net.test(path, shots=5, inner_test_steps=30)

        # path = r"G:\model_save\meta_learning\Reptile\5shot\Reptile_T2_ep732"
        # Net.test(path, shots=5, inner_test_steps=30)

        path = r"G:\model_save\meta_learning\Reptile\1shot\Reptile_T2_ep702"
        Net.test(path, shots=5, inner_test_steps=30)
