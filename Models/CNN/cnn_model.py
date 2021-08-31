import tensorflow as tf
from tensorflow.keras.layers import Conv1D, BatchNormalization, MaxPooling1D, \
    Dense, ReLU, Flatten, Dropout, Input


class CNN_Model(tf.keras.Model):
    def __init__(self, num_classes: int):
        super().__init__()
        # channels_last, (N, L, C)
        # self.feature = tf.keras.Sequential([
        #     Conv1D(64, 3, 1, 'same', input_shape=(sample_len, 1)), BatchNormalization(),
        #     Dropout(0.2), ReLU(), MaxPooling1D(2),
        #     Conv1D(64, 3, 1, 'same'), BatchNormalization(), Dropout(0.2), ReLU(), MaxPooling1D(2),
        #     Conv1D(64, 3, 1, 'same'), BatchNormalization(), Dropout(0.2), ReLU(), MaxPooling1D(2),
        #     Conv1D(64, 3, 1, 'same'), BatchNormalization(), Dropout(0.2), ReLU(), MaxPooling1D(2),  # (None, 64, 64)
        #     Flatten(),  # (None, 64*64)
        # ])
        # self.classifier = tf.keras.Sequential([
        #     Dense(512), BatchNormalization(), Dropout(0.2), ReLU(), Dense(num_classes),  # (None, num_classes)
        # ])

        self.feature = tf.keras.Sequential([
            # Input(shape=(1024, 1)),
            Conv1D(64, 3, 1, 'same'), BatchNormalization(), ReLU(), MaxPooling1D(2),
            Conv1D(64, 3, 1, 'same'), BatchNormalization(), ReLU(), MaxPooling1D(2),
            Conv1D(64, 3, 1, 'same'), BatchNormalization(), ReLU(), MaxPooling1D(2),
            Conv1D(64, 3, 1, 'same'), BatchNormalization(), ReLU(), MaxPooling1D(2),  # (None, 64, 64)
            Flatten(),  # (None, 64*64)
        ])
        self.classifier = tf.keras.Sequential([
            Dense(num_classes),  # (None, num_classes)
        ])

    @staticmethod
    def acc(y_true, y_pre):
        M = tf.metrics.Accuracy()
        M.update_state(y_true, y_pre)
        # print(y_true.numpy(), '======', y_pre.numpy())
        acc = M.result()
        M.reset_states()
        return acc

    def call(self, inputs, training=None, mask=None):
        x, y = inputs
        feats = self.feature(x)  # (None, 4096)
        logits = self.classifier(feats)
        loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y, logits, from_logits=True))
        # print(y.shape, logits.shape)
        acc = self.acc(tf.argmax(y, -1), tf.argmax(logits, -1))

        return loss, acc

