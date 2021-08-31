"""
The training of MANN. Code refers to: https://github.com/snowkylin/ntm
/Yancy F. 2021.8.27/
"""
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa
import time

from Models.MANN.mann_model import OneShotMANN
from Datasets.cwru_data import MANN_DataGenerator
from my_utils.init_utils import seed_tensorflow
import visdom

vis = visdom.Visdom(env='yancy_meta')

# Hyper parameters
BATCH_SIZE = 16
VEC_DIM = 100


class MANN_learner:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.model = OneShotMANN(batch_size=BATCH_SIZE, vector_dim=VEC_DIM, num_classes=self.num_classes)
        self.model.build(input_shape=[(None, 10, 1024), (None, 10, self.num_classes),
                                      (None, 10, self.num_classes)])

    def train(self, save_path):
        # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        optimizer = tfa.optimizers.AdamW(weight_decay=2e-5, learning_rate=1e-3)

        gen_train = MANN_DataGenerator(mode='train', batch_size=BATCH_SIZE)
        gen_val = MANN_DataGenerator(mode='validation', batch_size=BATCH_SIZE)
        # self.model.trainable = True
        counter = 0

        for ep in range(1000):
            t0 = time.time()
            for i_batch, (x, x_label, y) in enumerate(gen_train):
                # y: [(x_0, y_0), (x_1, y_1), (x_2, y_2), ...], [],
                # x_label: [(x_0, 0), (x_1, y_0), (x_2, y_1), ...], [],
                # x: (batch_size, seq_len, sample_len), x_label/y: (batch_size, seq_len, n_classes)
                with tf.GradientTape() as tape:
                    loss, acc = self.model((x, x_label, y))
                optimizer.minimize(loss, self.model.trainable_variables, tape=tape)

                print(f'[ep-{ep + 1}][batch-{i_batch + 1}] acc: {acc:.4f}')

                # For validation
                idx = np.random.randint(0, gen_val.__len__())
                x_, x_label_, y_ = gen_val.__getitem__(idx)
                loss_, acc_ = self.model.call((x_, x_label_, y_), training=False)

                vis.line(Y=[[loss, loss_]], X=[counter],
                         update=None if counter == 0 else 'append', win='Loss_MANN',
                         opts=dict(legend=['train', 'val'], title='Loss_MANN'))

                vis.line(Y=[[acc, acc_]], X=[counter],
                         update=None if counter == 0 else 'append', win='Acc_MANN',
                         opts=dict(legend=['train', 'val'], title='Acc_MANN'))
                counter += 1
            t1  =time.time()
            print(f'\n*** Time/epoch: {t1-t0:.4f} (s) ***\n')
            if (ep + 1) >= 30 and (ep + 1) % 2 == 0:
                if input('\n== Stop training? == (y/n)\n').lower() == 'y':
                    new_save_path = save_path + rf'_ep{ep + 1}.h5'
                    # save_path = save_path + '(1)' if os.path.exists(save_path) else save_path
                    self.model.save_weights(new_save_path)
                    print(f'Save model at: {new_save_path}')
                    break
                elif input('\n== Save model? == (y/n)\n').lower() == 'y':
                    new_save_path = save_path + rf'_ep{ep + 1}.h5'
                    print(f'Save model at: {new_save_path}')
                    self.model.save_weights(new_save_path)

    def test(self, load_path):
        self.model.load_weights(load_path)

        num_batch = 200
        gen_test = MANN_DataGenerator(mode='test', batch_size=BATCH_SIZE, num_batch=50)
        acc_list = []

        t0 = time.time()
        for i_batch, (x, x_label, y) in enumerate(gen_test):
            loss, acc = self.model.call((x, x_label, y), training=False)
            acc_list.append(acc)
        t1 = time.time()

        print(f"*** Time for {num_batch} batches: {t1-t0: .4f} (s), "
              f"i.e. {(t1-t0)/(num_batch*BATCH_SIZE)*200} (s) for 200 samples/class. ***")
        Acc = tf.reduce_mean(acc_list)
        print("\n===================")
        print(f"Accuracy: {Acc:.4f}")


if __name__ == "__main__":
    seed_tensorflow(2021)
    learner = MANN_learner(num_classes=10)
    print(learner.model.summary())

    if input('Train? y/n\n').lower() == 'y':
        save_path = r"G:\model_save\meta_learning\MANN\C30"
        learner.train(save_path)
    if input('Test? y/n\n').lower() == 'y':
        load_path = r"G:\model_save\meta_learning\MANN\C30_ep30.h5"
        learner.test(load_path)
