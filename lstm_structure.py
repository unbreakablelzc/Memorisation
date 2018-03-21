"""A Recurrent Neural Network (LSTM) that performs dynamic computation
   over sequences with variable length.This example is using a memlog_200_1
   and data_200_200_hard dataset to classify sequences.
"""

from __future__ import print_function

import tensorflow as tf
import numpy as np
import random
import os

# ==========
#   MODEL
# ==========

# Parameters
#learning_rate = 6.2298265 * 10**-6
start_learning_rate = 10**-3
training_steps = 1000
batch_size = 512
display_step = 50
log_dir = "data/01"

# Network Parameters
seq_max_len = 200  # Sequence max length, Nbjobs
n_hidden = 128  # hidden layer num of features.  Size of weight array in a cell. The capacity of learning ability.
n_classes = 2  # appeared sequence or not


dir_instances = 'data/ins_2.txt'
dir_solutions = 'data/memlog_200_2.txt'
dir_train_data = 'data/memlog_2.txt'
dir_test_data = 'data/memlog_2_test.txt'

# ====================
#  DATA GENERATOR
# ====================
class DataBalance:
    """ This class is for data balancing

        There are two kinds of data and we need to balance them to make the learning
        process more accurate. And the data will be divided into training data and
        test data.
        If the data is already balanced, we skip it.
    """
    def __init__(self):
        """Inits numbers of training samples and test samples."""
        self.nb_samples_train = 15907
        self.nb_samples_test = 5000
    def remove_zero(self):
        """Remove some negtive samples to balance data."""
        fo = open(dir_solutions, "r")
        if os.path.exists(dir_train_data) == False and os.path.exists(dir_test_data) == False:
            f_train = open(dir_train_data, "w+")
            f_test = open(dir_test_data, 'w+')
            i = 1
            ling = 0 #
            yi = 0

            test_ling = 0
            test_yi = 0

            for line in fo:
                list_line = line.split()
                list_line = list(map(int, list_line))
                # odd_numbered line
                if i % 2 == 1:
                    flag_train = 0
                    flag_test = 0
                    if (list_line[0] == 0):
                        if ling <= self.nb_samples_train:
                            f_train.write(line)
                            flag_train = 1

                        else:
                            if test_ling <= self.nb_samples_test:
                                f_test.write(line)
                                flag_test = 1
                                test_ling += 1

                        ling += 1
                    else:
                        if yi <= self.nb_samples_train:
                            f_train.write(line)
                            flag_train = 1
                        else:
                            if test_yi <= self.nb_samples_test:
                                f_test.write(line)
                                flag_test = 1
                                test_yi += 1
                        yi += 1

                # even_numbered line
                if i % 2 == 0:
                    if (flag_train == 1):
                        f_train.write(line)
                    if (flag_test == 1):
                        f_test.write(line)
                    length = len(list_line)

                i = i + 1
        else:
            print("Training and testing data already banlanced.")

class DataGenerator:
    def __init__(self, max_seq_len, path=""):
        self.data = []
        self.labels = []
        self.seqlen = []
        data_raw = []
        data_processing_due_time = []

        # instance data
        f_all = open(dir_instances, "r")
        for line_all in f_all:
            list_line_all = line_all.split()
            list_line_all = list(map(int, list_line_all))
            data_processing_due_time.append(list_line_all)

        # solution data
        f = open(path, "r")
        i = 1
        k = 0
        nb_zero = 0
        nb_one = 0
        for line in f:
            list_line = line.split()
            list_line = list(map(int, list_line))
            # odd_numbered line. nbUsed, nbJobs, startT
            if i % 2 == 1:
                if (list_line[0] == 0):
                    self.labels.append([0., 1.])
                    nb_zero += 1
                else:
                    self.labels.append([1., 0.])
                    nb_one += 1
                self.seqlen.append(list_line[1])
                data_start_time = list_line[2]
            # even_numbered line
            if i % 2 == 0:
                length = len(list_line)
                data_tmp = []  # The data of this example, in 1D vector
                tj = data_start_time # the starting time of each job in the sequence
                # [ [tj, pj, dj ],...]
                for job in range(length):
                    data_tmp.append(tj)
                    data_tmp.extend(data_processing_due_time[list_line[job] - 1])
                    tj += data_processing_due_time[list_line[job] - 1][0]
                data_tmp.extend([0] * (max_seq_len - length) * 3)
                #print(len(data_tmp))
                k = k + 1
                data_raw.append(data_tmp)
            i = i + 1

        print("positive(one):", nb_one, "negative(zero):", nb_zero)
        print("data_raw_shape:", np.array(data_raw).shape)
        self.data = np.array(data_raw).reshape(len(data_raw), seq_max_len, 3)
        l = list(zip(self.data, self.labels, self.seqlen))
        random.shuffle(l)
        self.data, self.labels, self.seqlen = zip(*l)
        #print(self.data)
        self.batch_id = 0

    def next_batch(self, batch_size):
        """ Return a batch of data. When dataset end is reached, start over.
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0
        end_batch = min(self.batch_id + batch_size, len(self.data))
        batch_data = (self.data[self.batch_id:end_batch])
        batch_labels = (self.labels[self.batch_id:end_batch])
        batch_seqlen = (self.seqlen[self.batch_id:end_batch])
        self.batch_id = end_batch
        return batch_data, batch_labels, batch_seqlen

class ModelRNN:
    def dynamic_rnn(self, x, seqlen, weights, biases):
        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, n_steps/lenJobs, n_input/3values)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

        # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.unstack(value=x, num=seq_max_len, axis=1)

        # Define a lstm cell with tensorflow
        #
        with tf.name_scope("LSTM_layers"):
            with tf.name_scope("LSTM_cell"):
                # ! try more advanced models
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden)
                #lstm_cell = tf.contrib.rnn.LSTMCell(n_hidden, use_peepholes=True)
                # Get lstm cell output, providing "sequence_length" will perform dynamic calculation.
                # To check the structure of states, check LSTMStateTuple.
                # Stores two elements: `(c, h)`, in that order. Where `c` is the hidden state and `h`
                # is the output.
                # The names are misleading
                # Hidden state: the output of EACH cell, used to decide how to modify the memory
                # Cell state: the memory. The final output.

            # args[0]: An instance of RNNCell

            # inputs: A length T list of inputs, each a 'Tensor' of shape '[batch_size, input_size]'
            # outputs is a length T list of outputs(one for each input) or a nested tuple of such
            # elements
            # state is the final state
            outputs, state = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32, sequence_length=seqlen)
        # When performing dynamic calculation, we must retrieve the last
        # dynamically computed output, i.e., if a sequence length is 10, we need
        # to retrieve the 10th output.
        # However TensorFlow doesn't support advanced indexing yet, so we build
        # a custom op that for each sample in batch size, get its length and
        # get the corresponding relevant output.

        # 'outputs' is a list of output at every timestep, we pack them in a Tensor
        # and change back dimension to [batch_size, n_step, n_input]
        #outputs = tf.stack(outputs)
        #outputs = tf.transpose(outputs, [1, 0, 2])

        # Hack to build the indexing and retrieve the right output.
        #batch_size = tf.shape(outputs)[0]
        # Start indices for each sample
        #index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
        # Indexing
        # 可以看出index和gather操作是为了得到这一批数据中，每个list在最后一次有效循环（list长度）结束时的输出值。
        #outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)
        # Linear activation, using outputs computed above
        state = tf.stack(state[1])
        pred_state = tf.matmul(tf.stack(state), weights['out']) + biases['out']
        #return tf.matmul(outputs, weights['out']) + biases['out']
        return pred_state

class Te():
    def __init__(self, sess):
        testset = DataGenerator(max_seq_len=seq_max_len, path=dir_test_data)
        test_data = testset.data
        test_labels = testset.labels
        test_seqlen = testset.seqlen
        l = list(zip(test_data, test_labels, test_seqlen))
        random.shuffle(l)
        test_data, test_labels, test_seqlen = zip(*l)
        print("Testing Accuracy:", \
              sess.run(accuracy, feed_dict={x: test_data, y: test_labels,
                                            seqlen: test_seqlen}))

if __name__ == "__main__":
    # 41816 data in total, 3/4 to train, 1/4 to test
    DataBalance().remove_zero()
    print("-----------------Reading training data------------------")
    trainset = DataGenerator(max_seq_len=seq_max_len, path=dir_train_data)
    print("-----------------Reading testing data-------------------")
    testset = DataGenerator(max_seq_len=seq_max_len, path=dir_test_data)

    with tf.name_scope('input'):
        # tf Graph input
        x = tf.placeholder(tf.float32, [None, seq_max_len, 3], name="x_input")
        y = tf.placeholder(tf.float32, [None, n_classes], name="y_input")
        # A placeholder for indicating each sequence length
        seqlen = tf.placeholder(tf.int32, [None], name="sequence_len")

    with tf.name_scope('weights'):
    # Define weights
        weights = {
            'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
        }

    with tf.name_scope('biases'):
        biases = {
            'out': tf.Variable(tf.random_normal([n_classes]))
        }

    pred = ModelRNN().dynamic_rnn(x, seqlen, weights, biases)
    tf.summary.histogram('predict', pred)

    with tf.name_scope('cost'):
        # Define loss and optimizer
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
        #tf.summary.histogram('predict',tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)[0])

    with tf.name_scope('learning_rate'):
        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 100, 0.96, staircase=True)

    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost, global_step=global_step)

    with tf.name_scope('accuracy'):
        # Evaluate model
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('cost', cost)
    tf.summary.scalar('accuracy', accuracy)

    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(log_dir)
    # Initialize the variables
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    # Start training
    print("------------------------Start training------------------------")
    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)

        writer.add_graph(sess.graph)
        for step in range(1, training_steps + 1):
            batch_x, batch_y, batch_seqlen = trainset.next_batch(batch_size)
            # Run optimization op (backprop)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                           seqlen: batch_seqlen})
            if step % display_step == 0 or step == 1:
                # Calculate batch accuracy & loss
                lr = sess.run(learning_rate)
                #pred_outputs = sess.run(pred, feed_dict={x: batch_x, y: batch_y,
                #                                                  seqlen: batch_seqlen})

                #print("outputs:", pred_outputs)
                #pred_states = sess.run(pred_state, feed_dict={x: batch_x, y: batch_y,
                #                                                  seqlen: batch_seqlen})
                #print("state:", pred_states)

                acc, loss = sess.run([accuracy, cost], feed_dict={x: batch_x, y: batch_y,
                                                                  seqlen: batch_seqlen})
                summary = sess.run(merged_summary, feed_dict={x: batch_x, y: batch_y,
                                                                  seqlen: batch_seqlen})
                writer.add_summary(summary, step)
                print("Step " + str(step * batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
        print("Optimization Finished!")

        # Test accuracy
        #Test().test(sess)

        print("------------------------Start testing------------------------")
        test_data = testset.data
        test_labels = testset.labels
        test_seqlen = testset.seqlen
        l = list(zip(test_data, test_labels, test_seqlen))
        random.shuffle(l)
        test_data, test_labels, test_seqlen = zip(*l)
        print("Testing Accuracy:", \
              sess.run(accuracy, feed_dict={x: test_data, y: test_labels,
                                            seqlen: test_seqlen}))

        saver.save(sess,"D:/PRD/Code/Memorisation/model/model.ckpt")