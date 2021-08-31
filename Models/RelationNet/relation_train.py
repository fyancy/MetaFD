"""
Relation Networks programmed by Yancy F. (2021/8/30)
"""
import torch
import numpy as np

import learn2learn as l2l
import visdom
import os
import time

from Models.RelationNet.relation_model import encoder_net, relation_net
from Datasets.cwru_data import MAML_Dataset
from my_utils.train_utils import accuracy
from my_utils.init_utils import weights_init2

vis = visdom.Visdom(env='yancy_meta')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RelationNet_learner(object):
    def __init__(self, ways):
        self.feature = encoder_net(in_chn=1, hidden_chn=64, cb_num=4).to(device)
        embed_size = (1024//2**4)//2**2*64
        self.relation = relation_net(hidden_chn=64, embed_size=embed_size, h_size=256).to(device)
        self.ways = ways

    def fast_adapt(self, batch, loss_fun, query_num, shots, ways):
        data, labels = batch
        data, labels = data.to(device), labels.to(device)

        # Separate data into adaptation/evaluation sets
        sort = torch.sort(labels)
        data = data.squeeze(0)[sort.indices].squeeze(0)
        labels = labels.squeeze(0)[sort.indices].squeeze(0)

        support_indices = np.zeros(data.size(0), dtype=bool)
        # print(data.size(0))
        selection = np.arange(ways) * (shots + query_num)  # 0, shot+q, 2*(shot+q), 3*(), ...
        for offset in range(shots):
            support_indices[selection + offset] = True  # 0:shots, (shot+q):(shot+q+shots), ...

        query_indices = torch.from_numpy(~support_indices)  # shots:2*shots, (shot+q+shots):4*shots, ...
        support_indices = torch.from_numpy(support_indices)  # 0:shots, (shot+q):(shot+q+shots), ...

        embeddings = self.feature(data)
        support = embeddings[support_indices]  # (n_support, chn, length)
        query = embeddings[query_indices]  # (n_query, chn, length)
        labels = labels[query_indices].long()  # (n_query)

        support = support.reshape(ways, shots, *support.shape[-2:]).mean(dim=1)  # (ways, chn, length)
        support = support.unsqueeze(0).repeat(query.shape[0], 1, 1, 1)  # (n_q, ways, chn, length)
        query = query.unsqueeze(1).repeat(1, ways, 1, 1)  # (n_q, ways, chn, length)

        relation_pairs = torch.cat((support, query), 2).reshape(query.shape[0]*ways, -1, query.shape[-1])
        scores = self.relation(relation_pairs).reshape(-1, ways)  # (n_q, ways)
        # print(scores.shape)
        # exit()

        error = loss_fun(scores, labels)
        acc = accuracy(scores, labels)
        return error, acc

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

    def model_save(self, path):
        filename = path+'(1)' if os.path.exists(path) else path
        state_dict = {
            'feature': self.feature.state_dict(),
            'relation': self.relation.state_dict(),
        }
        torch.save(state_dict, filename)
        print(f'Save model at: {filename}')

    def train(self, save_path, shots):
        train_ways = valid_ways = self.ways
        query_num = shots
        print(f"{train_ways}-ways, {shots}-shots for training ...")
        train_tasks = self.build_tasks('train', train_ways, shots, 1000, None)
        valid_tasks = self.build_tasks('validation', valid_ways, shots, 50, None)
        # valid_loader = DataLoader(valid_tasks, pin_memory=True, shuffle=True)

        self.feature.apply(weights_init2)
        self.relation.apply(weights_init2)

        # optimizer_f = torch.optim.Adam(self.feature.parameters(), lr=0.0005, weight_decay=2e-5)  # 0.8568
        # optimizer_r = torch.optim.Adam(self.relation.parameters(), lr=0.005, weight_decay=2e-5)

        optimizer_f = torch.optim.Adam(self.feature.parameters(), lr=0.0001, weight_decay=2e-5)  # 0.9160
        optimizer_r = torch.optim.Adam(self.relation.parameters(), lr=0.001, weight_decay=2e-5)

        # lr_scheduler_f = torch.optim.lr_scheduler.ExponentialLR(optimizer_f, gamma=0.99)
        lr_scheduler_r = torch.optim.lr_scheduler.ExponentialLR(optimizer_r, gamma=0.99)
        loss_fun = torch.nn.CrossEntropyLoss()

        Epochs = 10000
        Episodes = 40
        counter = 0

        for ep in range(Epochs):
            # 1) training:
            t0 = time.time()
            self.feature.train(), self.relation.train()

            meta_train_error = 0.0
            meta_train_accuracy = 0.0
            for epi in range(Episodes):
                batch = train_tasks.sample()
                loss, acc = self.fast_adapt(batch, loss_fun, query_num, shots, train_ways)
                meta_train_error += loss.item()
                meta_train_accuracy += acc.item()

                optimizer_f.zero_grad()
                optimizer_r.zero_grad()

                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.feature.parameters(), 0.5)
                torch.nn.utils.clip_grad_norm_(self.relation.parameters(), 0.5)

                optimizer_f.step()
                optimizer_r.step()

            # lr_scheduler_f.step()
            lr_scheduler_r.step()

            t1 = time.time()
            print(f'*** Time /epoch {t1-t0:.3f} ***')
            print(f'epoch {ep+1}, train, loss: {meta_train_error/Episodes:.3f}, '
                  f'acc: {meta_train_accuracy/Episodes:.3f}')

            # 2) validation:
            self.feature.eval(), self.relation.eval()
            meta_valid_error = 0.0
            meta_valid_accuracy = 0.0
            for i, batch in enumerate(valid_tasks):
                with torch.no_grad():
                    loss, acc = self.fast_adapt(batch, loss_fun, query_num, shots, train_ways)
                meta_valid_error += loss.item()
                meta_valid_accuracy += acc.item()
            print(f'epoch {ep + 1}, validation, loss: {meta_valid_error / len(valid_tasks):.4f}, '
                  f'acc: {meta_valid_accuracy / len(valid_tasks):.4f}\n')

            vis.line(Y=[[meta_train_error / Episodes, meta_valid_error / len(valid_tasks)]], X=[counter],
                     update=None if counter == 0 else 'append', win='Loss_RelationNet',
                     opts=dict(legend=['train', 'val'], title='Loss_RelationNet'))

            vis.line(Y=[[meta_train_accuracy / Episodes, meta_valid_accuracy / len(valid_tasks)]], X=[counter],
                     update=None if counter == 0 else 'append', win='Acc_RelationNet',
                     opts=dict(legend=['train', 'val'], title='Acc_RelationNet'))
            counter += 1

            if (ep+1) >=200 and (ep+1)%2==0:  # generally (ep+1) >=200
                if input('\n== Stop training? == (y/n)\n').lower() == 'y':
                    new_save_path = save_path + rf'_ep{ep + 1}'
                    self.model_save(new_save_path)
                    break
                elif input('\n== Save model? == (y/n)\n').lower() == 'y':
                    new_save_path = save_path + rf'_ep{ep + 1}'
                    self.model_save(new_save_path)

    def test(self, load_path, shots):
        state_dict = torch.load(load_path)
        self.feature.load_state_dict(state_dict['feature'])
        self.relation.load_state_dict(state_dict['relation'])
        print(f'Load Model successfully from [{load_path}]...')

        test_ways = self.ways
        query_num = shots
        print(f"{test_ways}-ways, {shots}-shots for testing ...")
        test_tasks = self.build_tasks('test', test_ways, shots, 1000, None)
        loss_fun = torch.nn.CrossEntropyLoss()

        # 2) validation:
        self.feature.eval(), self.relation.eval()
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        t0 = time.time()

        for i, batch in enumerate(test_tasks):
            with torch.no_grad():
                loss, acc = self.fast_adapt(batch, loss_fun, query_num, shots, test_ways)
                meta_valid_error += loss.item()
                meta_valid_accuracy += acc.item()

        t1 = time.time()
        print(f"*** Time for {len(test_tasks)} tasks: {t1 - t0:.4f} (s)")
        print(f'Testing, loss: {meta_valid_error / len(test_tasks):.4f}, '
              f'acc: {meta_valid_accuracy / len(test_tasks):.4f}')


if __name__ == "__main__":
    from my_utils.init_utils import seed_torch

    seed_torch(2021)
    # Net = RelationNet_learner(ways=10)  # T1
    Net = RelationNet_learner(ways=4)  # T2

    if input('Train? y/n\n').lower() == 'y':
        # path = r"G:\model_save\meta_learning\RelationNet\5shot\RelationNet_C30"
        # Net.train(save_path=path, shots=5)

        # path = r"G:\model_save\meta_learning\RelationNet\1shot\RelationNet_C30"
        # Net.train(save_path=path, shots=1)

        # path = r"G:\model_save\meta_learning\RelationNet\5shot\RelationNet_T2"
        # Net.train(save_path=path, shots=5)

        path = r"G:\model_save\meta_learning\RelationNet\1shot\RelationNet_T2"
        Net.train(save_path=path, shots=1)

    if input('Test? y/n\n').lower() == 'y':
        # load_path = r"G:\model_save\meta_learning\RelationNet\5shot\RelationNet_C30_ep200"
        # Net.test(load_path, shots=5)

        # load_path = r"G:\model_save\meta_learning\RelationNet\1shot\RelationNet_C30_ep252"
        # Net.test(load_path, shots=1)

        # load_path = r"G:\model_save\meta_learning\RelationNet\5shot\RelationNet_T2_ep394"
        # Net.test(load_path, shots=5)

        load_path = r"G:\model_save\meta_learning\RelationNet\1shot\RelationNet_T2_ep284"
        Net.test(load_path, shots=1)
