"""
The training of CNN.
Code refers to: https://github.com/fyancy/DASMN/tree/main/DASMN_revised_2020_11/Model
/Yancy F. 2021.8.27/
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np

from Models.CNN.cnn_model import CNN_Model
from Datasets.cwru_data import CNN_DataGenerator
from my_utils.init_utils import seed_tensorflow

import visdom
vis = visdom.Visdom(env='yancy_meta')


class CNN_learner:
    def __init__(self, num_classes):
        self.num_classes = num_classes
        # self.model = CNN_Model(sample_len=1024, num_classes=self.num_classes)
        self.model = CNN_Model(num_classes=self.num_classes)
        self.model.build(input_shape=[(None, 1024, 1), (None, self.num_classes)])

    def train(self, save_path):
        # optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
        optimizer = tfa.optimizers.AdamW(weight_decay=2e-5, learning_rate=1e-3)

        gen_train = CNN_DataGenerator(mode='train', shot=5)
        gen_val = CNN_DataGenerator(mode='validation', shot=5)
        # self.model.trainable = True
        counter = 0

        for ep in range(100):
            for i_batch, (x, y) in enumerate(gen_train):
                with tf.GradientTape() as tape:
                    loss, acc = self.model.call((x, y), training=True)
                optimizer.minimize(loss, self.model.trainable_variables, tape=tape)
                print(f'[ep-{ep+1}][batch-{i_batch+1}] acc: {acc:.3f} loss: {loss:.3f}')

                # For validation
                idx = np.random.randint(0, gen_val.__len__())
                x_, y_ = gen_val.__getitem__(idx)
                loss_, acc_ = self.model.call((x_, y_), training=False)
                # loss_, acc_ = self.model.call((x_, y_))

                vis.line(Y=[[loss, loss_]], X=[counter],
                         update=None if counter == 0 else 'append', win='Loss_CNN',
                         opts=dict(legend=['train', 'val'], title='Loss_CNN'))

                vis.line(Y=[[acc, acc_]], X=[counter],
                         update=None if counter == 0 else 'append', win='Acc_CNN',
                         opts=dict(legend=['train', 'val'], title='Acc_CNN'))
                counter += 1

            if (ep + 1) >= 15 and (ep + 1) % 2 == 0:
                if input('\n== Stop training? == (y/n)\n').lower() == 'y':
                    new_save_path = save_path + rf'_ep{ep + 1}.h5'
                    # save_path = save_path + '(1)' if os.path.exists(save_path) else save_path
                    self.model.save_weights(new_save_path)
                    print(f'Save model at: {new_save_path}')
                    break
                elif input('\n== Save model? == (y/n)\n').lower() == 'y':
                    new_save_path = save_path + rf'_ep{ep + 1}.h5'
                    self.model.save_weights(new_save_path)

    def test(self, load_path):
        self.model.load_weights(load_path)
        gen_test = CNN_DataGenerator(mode='test', shot=5)
        acc_list = []
        for i_batch, (x, y) in enumerate(gen_test):
            loss, acc = self.model.call((x, y), training=False)
            acc_list.append(acc)
        Acc = tf.reduce_mean(acc_list)
        print("\n===================")
        print(f"Accuracy: {Acc:.3f}")


if __name__ == "__main__":
    seed_tensorflow(2021)

    learner = CNN_learner(num_classes=10)
    print(learner.model.summary())

    if input('Train? y/n\n').lower() == 'y':
        save_path = r"G:\model_save\meta_learning\CNN\C30"
        learner.train(save_path)
    if input('Test? y/n\n').lower() == 'y':
        load_path = r"G:\model_save\meta_learning\CNN\C30_ep54.h5"
        learner.test(load_path)
