"""Demo Dynamic Recurrent Neural Network.
TensorFlow implementation of a Recurrent Neural Network (LSTM) that performs
dynamic computation over sequences with variable length.This example is using
a memlog_200_1 and data_200_200_hard dataset to classify sequences.
"""

from __future__ import print_function

import tensorflow as tf
import numpy as np
import random
import time

# ==========
#   MODEL
# ==========

# Parameters
#learning_rate = 6.2298265 * 10**-6
learning_rate = 10**-6
training_steps = 3000
batch_size = 512
display_step = 100

# Network Parameters
seq_max_len = 797  # Sequence max length
n_hidden = 64  # hidden layer num of features
n_classes = 2  # linear sequence or not

# ====================
#  DATA GENERATOR
# ====================
class Memlog_1(object):
    def __init__(self, n_samples=1000, max_seq_len=20, min_seq_len=10, max_value=1000, path=""):
        self.data = []
        self.labels = []
        self.seqlen = []
        data_raw = []
        data_processing_due_time = []
        data_start_time = []

        f_all = open('data/data_200_200_hard.txt', "r")
        j = 1
        for line_all in f_all:
            if not line_all.startswith('#'):
                list_line_all = line_all.split()
                list_line_all = list(map(int, list_line_all))
                data_processing_due_time.append(list_line_all)
                j = j + 1

        l = 0
        data_instance = []
        for l in range(200):
            data_instance.extend(data_processing_due_time[l+201])
            l = l + 1

        f = open(path, "r")
        i = 1
        k = 0

        zero = 0
        one = 0
        for line in f:
            list_line = line.split()
            list_line = list(map(int, list_line))
            # odd_numbered line
            if i % 2 == 1:
                if (list_line[0] == 0):
                    self.labels.append([0., 0.])
                    zero += 1
                else:
                    self.labels.append([0., 1.])
                    one += 1
                self.seqlen.append(list_line[1])
                data_start_time.append(list_line[2])
            # even_numbered line
            if i % 2 == 0:
                length = len(list_line)
                data_tmp = []
                for job in range(length):
                    data_tmp.extend(data_processing_due_time[list_line[job] - 1])
                data_tmp.insert(0, data_start_time[k])
                data_tmp = data_instance + data_tmp

                data_tmp += [0 for i in range(max_seq_len - 2 * length - 1 - 400)]
                #print(len(data_tmp))
                k = k + 1
                data_raw.append(data_tmp)
                # print(len(data_tmp))
                # print(data_tmp)
                # time.sleep(0.1)
            i = i + 1

        print("positive:", one, "negative:", zero)
        # print(self.labels)
        # print(data_raw)
        print(np.array(data_raw).shape)
        self.data = np.array(data_raw).reshape(-1, 198 * 2 + 1 + 400, 1)
        # self.data = data_raw
        #print(self.data)
        self.batch_id = 0

    def next(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0
        batch_data = (self.data[self.batch_id:min(self.batch_id +
                                                  batch_size, len(self.data))])
        batch_labels = (self.labels[self.batch_id:min(self.batch_id +
                                                      batch_size, len(self.data))])
        batch_seqlen = (self.seqlen[self.batch_id:min(self.batch_id +
                                                      batch_size, len(self.data))])
        self.batch_id = min(self.batch_id + batch_size, len(self.data))
        l = list(zip(batch_data, batch_labels, batch_seqlen))
        random.shuffle(l)
        batch_data, batch_labels, batch_seqlen = zip(*l)
        return batch_data, batch_labels, batch_seqlen


def dynamicRNN(x, seqlen, weights, biases):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, seq_max_len, 1)

    # Define a lstm cell with tensorflow
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)

    # Get lstm cell output, providing "sequence_length" will perform dynamic
    # calculation.
    outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32, sequence_length=seqlen)

    # When performing dynamic calculation, we must retrieve the last
    # dynamically computed output, i.e., if a sequence length is 10, we need
    # to retrieve the 10th output.
    # However TensorFlow doesn't support advanced indexing yet, so we build
    # a custom op that for each sample in batch size, get its length and
    # get the corresponding relevant output.

    # 'outputs' is a list of output at every timestep, we pack them in a Tensor
    # and change back dimension to [batch_size, n_step, n_input]
    outputs = tf.stack(outputs)
    outputs = tf.transpose(outputs, [1, 0, 2])

    # Hack to build the indexing and retrieve the right output.
    batch_size = tf.shape(outputs)[0]
    # Start indices for each sample
    index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
    # Indexing
    outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)

    # Linear activation, using outputs computed above
    return tf.matmul(outputs, weights['out']) + biases['out']

if __name__ == "__main__":
    # 25096 data in total, 3/4 to train, 1/4 to test
    #trainset = Memlog_1(n_samples=37644, max_seq_len=seq_max_len)
    #testset = Memlog_1(n_samples=12548, max_seq_len=seq_max_len)
    trainset = Memlog_1(max_seq_len=seq_max_len, path="data/memlog_2.txt")
    testset = Memlog_1(max_seq_len=seq_max_len, path="data/memog_2_test.txt")

    # tf Graph input
    x = tf.placeholder(tf.float32, [None, seq_max_len, 1])
    y = tf.placeholder(tf.float32, [None, n_classes])
    # A placeholder for indicating each sequence length
    seqlen = tf.placeholder(tf.int32, [None])

    # Define weights
    weights = {
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    pred = dynamicRNN(x, seqlen, weights, biases)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables
    init = tf.global_variables_initializer()

    # Start training
    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)

        for step in range(1, training_steps + 1):
            batch_x, batch_y, batch_seqlen = trainset.next(batch_size)
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                           seqlen: batch_seqlen})
            if step % display_step == 0 or step == 1:
                # Calculate batch accuracy & loss
                acc, loss = sess.run([accuracy, cost], feed_dict={x: batch_x, y: batch_y,
                                                                  seqlen: batch_seqlen})
                print("Step " + str(step * batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
        print("Optimization Finished!")

        # Calculate accuracy
        test_data = testset.data
        test_label = testset.labels
        test_seqlen = testset.seqlen
        l = list(zip(test_data, test_label, test_seqlen))
        random.shuffle(l)
        test_data, test_label, test_seqlen = zip(*l)
        print("Testing Accuracy:", \
              sess.run(accuracy, feed_dict={x: test_data, y: test_label,
                                            seqlen: test_seqlen}))
