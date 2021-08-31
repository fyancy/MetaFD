import tensorflow as tf
import numpy as np


class MANNCell(tf.keras.layers.AbstractRNNCell):
    def __init__(self, rnn_size, memory_size, memory_vector_dim, head_num, gamma, k_strategy='separate', **kwargs):
        super().__init__(**kwargs)
        self.rnn_size = rnn_size
        self.memory_size = memory_size
        self.memory_vector_dim = memory_vector_dim
        self.head_num = head_num  # #(read head) == #(write head)
        self.gamma = gamma
        self.k_strategy = k_strategy
        # self.output_dim = output_dim

        # Controller RNN layer
        self.controller = tf.keras.layers.LSTMCell(units=self.rnn_size)

        # Calculate number of parameters per read/write head
        # controller_output     -> k (dim = memory_vector_dim, compared to each vector in M)
        #                       -> a (dim = memory_vector_dim, add vector, only when k_strategy='separate')
        #                       -> alpha (scalar, combination of w_r and w_lu)

        if self.k_strategy == 'summary':
            self.num_parameters_per_head = self.memory_vector_dim + 1
        elif self.k_strategy == 'separate':
            self.num_parameters_per_head = self.memory_vector_dim * 2 + 1
        else:
            raise ValueError('K strategy not supported')
        self.total_parameter_num = self.num_parameters_per_head * self.head_num

        # From controller output to parameters:
        self.controller_output_to_params = tf.keras.layers.Dense(units=self.total_parameter_num, use_bias=True)

        # From controller output to NTM output:
        # self.controller_output_to_ntm_output = tf.keras.layers.Dense(units=self.output_dim, use_bias=True)

    @property
    def state_size(self):
        return self.rnn_size

    # def build(self, inputs_shape):
    #     if not self.output_dim:
    #         self.output_dim = inputs_shape[1]

    def call(self, inputs, states):
        """

        :param inputs: (N, dim)
        :param states:
        :return:
        """
        prev_read_vector_list = states['read_vector_list']  # read vector in Sec 3.1 (the content that is
        # read out, length = memory_vector_dim)
        prev_controller_state = states['controller_state']  # state of controller (LSTM hidden state)

        # x + prev_read_vector -> controller (RNN) -> controller_output
        controller_input = tf.concat([inputs] + prev_read_vector_list, axis=1)
        controller_output, controller_state = self.controller(controller_input, prev_controller_state)

        # controller_output -> k, a and alpha
        parameters = self.controller_output_to_params(controller_output)
        head_parameter_list = tf.split(parameters, self.head_num, axis=1)
        # erase_add_list = tf.split(parameters[:, self.num_parameters_per_head * self.num_heads:],
        #                           2 * self.write_head_num, axis=1)

        prev_w_r_list = states['w_r_list']  # vector of weightings (blurred address) over locations
        prev_M = states['M']
        prev_w_u = states['w_u']
        prev_indices, prev_w_lu = self._least_used(prev_w_u)
        w_r_list = []
        w_w_list = []
        k_list = []
        a_list = []
        # p_list = []   # For debugging

        for i, head_parameter in enumerate(head_parameter_list):
            k = tf.tanh(head_parameter[:, 0:self.memory_vector_dim], name='k')
            if self.k_strategy == 'separate':
                a = tf.tanh(head_parameter[:, self.memory_vector_dim:self.memory_vector_dim * 2], name='a')
            sig_alpha = tf.sigmoid(head_parameter[:, -1:], name='sig_alpha')
            w_r = self._read_head_addressing(k, prev_M)
            w_w = self._write_head_addressing(sig_alpha, prev_w_r_list[i], prev_w_lu)

            w_r_list.append(w_r)
            w_w_list.append(w_w)
            k_list.append(k)
            if self.k_strategy == 'separate':
                a_list.append(a)
            # p_list.append({'k': k, 'sig_alpha': sig_alpha, 'a': a})   # For debugging

        w_u = self.gamma * prev_w_u + tf.add_n(w_r_list) + tf.add_n(w_w_list)  # eq (20)

        # Set least used memory location computed from w_(t-1)^u to zero
        M_ = prev_M * tf.expand_dims(1. - tf.one_hot(prev_indices[:, -1], self.memory_size), axis=2)

        # Writing
        M = M_
        for i in range(self.head_num):
            w = tf.expand_dims(w_w_list[i], axis=2)
            if self.k_strategy == 'summary':
                k = tf.expand_dims(k_list[i], axis=1)
            elif self.k_strategy == 'separate':
                k = tf.expand_dims(a_list[i], axis=1)
            else:
                raise ValueError('K strategy not supported')
            M = M + tf.matmul(w, k)

        # Reading
        read_vector_list = []
        for i in range(self.head_num):
            read_vector = tf.reduce_sum(tf.expand_dims(w_r_list[i], axis=2) * M, axis=1)
            read_vector_list.append(read_vector)

        # controller_output -> MANN output
        mann_output = tf.concat([controller_output] + read_vector_list, axis=1)

        state = {
            'controller_state': controller_state,
            'read_vector_list': read_vector_list,
            'w_r_list': w_r_list,
            'w_w_list': w_w_list,
            'w_u': w_u,
            'M': M,
        }

        return mann_output, state

    def _read_head_addressing(self, k, prev_M):
        # Cosine Similarity
        k = tf.expand_dims(k, axis=2)
        inner_product = tf.matmul(prev_M, k)
        k_norm = tf.sqrt(tf.reduce_sum(tf.square(k), axis=1, keepdims=True))
        M_norm = tf.sqrt(tf.reduce_sum(tf.square(prev_M), axis=2, keepdims=True))
        norm_product = M_norm * k_norm
        K = tf.squeeze(inner_product / (norm_product + 1e-8))  # eq (17)

        # Calculating w^c
        K_exp = tf.exp(K)
        w = K_exp / tf.reduce_sum(K_exp, axis=1, keepdims=True)  # eq (18)

        return w

    def _write_head_addressing(self, sig_alpha, prev_w_r, prev_w_lu):
        # Write to (1) the place that was read in t-1 (2) the place that was least used in t-1
        return sig_alpha * prev_w_r + (1. - sig_alpha) * prev_w_lu  # eq (22)

    def _least_used(self, w_u):
        _, indices = tf.nn.top_k(w_u, k=self.memory_size)
        w_lu = tf.reduce_sum(tf.one_hot(indices[:, -self.head_num:], depth=self.memory_size), axis=1)
        return indices, w_lu

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        one_hot_weight_vector = np.zeros([batch_size, self.memory_size])
        one_hot_weight_vector[..., 0] = 1
        one_hot_weight_vector = tf.constant(one_hot_weight_vector, dtype=tf.float32)
        # print(one_hot_weight_vector.shape)
        initial_state = {
            'controller_state': self.controller.get_initial_state(inputs, batch_size, dtype),
            'read_vector_list': [tf.zeros([batch_size, self.memory_vector_dim])
                                 for _ in range(self.head_num)],
            'w_r_list': [one_hot_weight_vector for _ in range(self.head_num)],
            'w_u': one_hot_weight_vector,
            'M': tf.constant(np.ones([batch_size, self.memory_size, self.memory_vector_dim]) * 1e-6, dtype=tf.float32)
        }
        return initial_state
