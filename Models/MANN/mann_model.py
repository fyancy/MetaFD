import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from Models.MANN.mann_cell import MANNCell


class OneShotMANN(tf.keras.Model):
    def __init__(self, batch_size, vector_dim, num_classes):
        super().__init__()
        self.batch_size = batch_size
        self.vector_dim = vector_dim

        # self.eof = tf.one_hot([self.vector_dim] * batch_size, depth=self.vector_dim + 1)
        # self.eof = tf.one_hot([self.vector_dim] * batch_size, depth=self.vector_dim + num_classes)
        # self.zero = tf.zeros([batch_size, vector_dim + 1], dtype=tf.float32)

        self.cell = MANNCell(
            rnn_size=200,  # Size of hidden states of controller
            memory_size=128,  # Number of memory locations (N)
            memory_vector_dim=40,  # The vector size at each location (M), 40, default
            head_num=1,  # # of read & write head (in MANN, #(read head) = #(write head))
            gamma=0.95,  # Usage decay of the write weights (in eq 20)
            k_strategy='summary',  # 'separate', 'summary'
            # In the original MANN paper, query key vector 'k' are used in both reading (eq 17) and writing (eq 23).
            # You can set k_strategy='summary' if you want this way. However, in the NTM paper they are separated. If
            # you set k_strategy='separate', the controller will generate a new add vector 'a' to replace the query
            # vector 'k' in eq 23.
        )
        self.classifier = tf.keras.Sequential(
            [tf.keras.layers.Dense(units=100, activation='relu'),
            tf.keras.layers.Dense(units=num_classes, activation='relu')])

    @staticmethod
    def acc(y_true, y_pre):
        M = tf.metrics.Accuracy()
        # M.update_state(tf.reshape(y_true, (-1)), tf.reshape(y_pre, (-1)))
        M.update_state(y_true, y_pre)
        acc = M.result()
        M.reset_states()
        return acc

    # @tf.function
    def call(self, inputs, training=None, mask=None):
        x, x_label, y = inputs  # x: (batch_size, seq_len, vec_dim), e.g. (64, 10, 1024)

        state = self.cell.get_initial_state(batch_size=self.batch_size, dtype=tf.float32)
        out = []
        for t in range(x.shape[1]):
            output, state = self.cell(tf.concat([x[:, t, :], x_label[:, t, :]], axis=1), state)
            # output, state = self.cell(x[:, t, :], state)
            # output: (Batch_size, rnn_dim + memory_dim)
            batch_y_pre = self.classifier(output)  # (Batch_size, n_classes)
            out.append(batch_y_pre)

        # Note that, computation of y_pre_t depends on y_(t-1) instead of y_(t),
        # thus, only one previous sample provides the label information, i.e. 1-shot.

        out = tf.stack(out, axis=1)  # (batch_size, seq_len, n_classes)
        loss = tf.reduce_mean(tf.losses.categorical_crossentropy(y, out, from_logits=True))
        acc = self.acc(tf.argmax(y, axis=-1), tf.argmax(out, axis=-1))

        return loss, acc


if __name__ == "__main__":
    import numpy as np

    def one_hot_encode(x, dim):
        res = np.zeros(np.shape(x) + (dim,), dtype=np.float32)
        it = np.nditer(x, flags=['multi_index'])
        while not it.finished:
            res[it.multi_index][it[0]] = 1
            it.iternext()
        return res

    n_classes = 7
    batch_size = 32
    seq_len = 10
    dim = 1024

    model = OneShotMANN(batch_size, 40, n_classes)
    x = np.random.random(size=[batch_size, seq_len, dim]).astype(np.float32)

    seq = np.random.randint(0, n_classes, [batch_size, seq_len])  # (batch_size, seq_length)
    seq_encoded = one_hot_encode(seq, n_classes)  # (batch_size, seq_len, output_dim/n_c)
    seq_encoded_shifted = np.concatenate([np.zeros(shape=[batch_size, 1, n_classes], dtype=np.float32),
                                          seq_encoded[:, :-1, :]], axis=1)  # (batch_size, seq_len, output_dim/n_c)

    y_pred, loss, acc = model((x, seq_encoded_shifted, seq_encoded, seq_len))
