"""A Recurrent Neural Network (LSTM) that performs dynamic computation
   over sequences with variable length.This example is using a memlog_200_1
   and data_200_200_hard dataset to classify sequences.
"""
import tensorflow as tf
import numpy as np
import random
import os

# Parameters
start_learning_rate = 10**-5
training_steps = 2000 # Number of training steps
batch_size = 512 # Number of data used to train each step
display_step = 50 # Print the loss and accuracy each 50 steps
log_dir = "data/batchsize" # Directory to save the TensorFlow events files
model_dir = "D:/PRD/Code/Memorisation/MODEL/batchsize/model.ckpt" # Directory to save the TensorFlow model

# Network Parameters
seq_max_len = 200  # Sequence max length, Nbjobs
n_hidden = 256  # Hidden layer num of features. Size of weight array in a cell. The capacity of learning ability.
n_classes = 2  # Appeared sequence or not

dir_instances = 'data/ins_2.txt' # Directory of instance data
dir_solutions = 'data/memlog_200_2.txt' # Directory of solution data
dir_train_data = 'data/memlog_2.txt' # Directory of training data
dir_test_data = 'data/memlog_2_test.txt' # Directory of test data

# ====================
#  DATA GENERATOR
# ====================
class DataGenerator:
    """ Generate data, labels, length of sequence.
    """
    def __init__(self):
        """Inits data, labels, length of sequences.
        """
        #self.nb_samples_train = 15907
        #self.nb_samples_test = 5000
        self.nb_samples_train = 0
        self.nb_samples_test = 0
        self.data = []
        self.labels = []
        self.seqlen = []

    def data_count(self, path):
        """ Count the data number of two classes.
        :param path: Path of data file.
        :return: Number of negtive data.
                  Number of positive data.

        """
        # solution data
        f = open(path, "r")
        i = 1
        nb_zero = 0
        nb_one = 0
        for line in f:
            list_line = line.split()
            list_line = list(map(int, list_line))
            # odd_numbered line.
            if i % 2 == 1:
                if (list_line[0] == 0):
                    nb_zero += 1
                else:
                    nb_one += 1
            i += 1
        #print(nb_zero, nb_one)
        return  nb_zero, nb_one

    def data_balance(self, path_solutions, path_train, path_test):
        """ Remove some negtive samples to balance data.

        There are two kinds of data and we need to balance them to make the learning
        process more accurate. And the data will be divided into training data and
        test data.
        If the data is already balanced, we skip it.

        """
        nb_zero, nb_one = DataGenerator().data_count(path_solutions)
        if(nb_one%2 == 0):
            self.nb_samples_test = nb_one/4
            self.nb_samples_train = nb_one*3/4
        else:
            self.nb_samples_test = int(nb_one/4)
            self.nb_samples_train = nb_one - self.nb_samples_test
        #print(self.nb_samples_train, self.nb_samples_test)
        fo = open(dir_solutions, "r")
        if os.path.exists(path_train) == False and os.path.exists(path_test) == False:
            f_train = open(path_train, "w+")
            f_test = open(path_test, 'w+')
            i = 1
            ling = 0
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
                        if ling < self.nb_samples_train:
                            f_train.write(line)
                            flag_train = 1

                        else:
                            if test_ling < self.nb_samples_test:
                                f_test.write(line)
                                flag_test = 1
                                test_ling += 1

                        ling += 1
                    else:
                        if yi < self.nb_samples_train:
                            f_train.write(line)
                            flag_train = 1
                        else:
                            if test_yi < self.nb_samples_test:
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
                i = i + 1
        else:
            print("Training and testing data already banlanced.")

    def data_extract(self,  max_seq_len, path=""):
        """Extracts useful data from data file and reshapes it.

        :param max_seq_len: An int which represents maximum length of jobs in one solution.
        :param path: Path of file used to generate training data or test data.
        :return: null

        """
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
                data_tmp = []  # The data of this sample, in 1D vector
                tj = data_start_time # The starting time of each job in the sequence
                # [ [tj, pj, dj ],...]
                for job in range(length):
                    data_tmp.append(tj)
                    data_tmp.extend(data_processing_due_time[list_line[job] - 1])
                    tj += data_processing_due_time[list_line[job] - 1][0]
                data_tmp.extend([0] * (max_seq_len - length) * 3)
                k = k + 1
                data_raw.append(data_tmp)
            i = i + 1

        print("positive(one):", nb_one, "negative(zero):", nb_zero)
        print("data_raw_shape:", np.array(data_raw).shape)
        self.data = np.array(data_raw).reshape(len(data_raw), seq_max_len, 3)
        #l = list(zip(self.data, self.labels, self.seqlen))
        #random.shuffle(l)
        #self.data, self.labels, self.seqlen = zip(*l)
        self.batch_id = 0

    def data_shuffle(self):
        """ Shuffle the data."""
        l = list(zip(self.data, self.labels, self.seqlen))
        random.shuffle(l)
        self.data, self.labels, self.seqlen = zip(*l)

    def next_batch(self, batch_size):
        """ Form a batch of data. When dataset end is reached, start over.

        :param batch_size: Size of each batch, the number of samples used to train each step.
        :return: A batch of data.
                  A batch of labels.
                  A batch of sequence length.
        """
        if self.batch_id == len(self.data):
            self.batch_id = 0
        end_batch = min(self.batch_id + batch_size, len(self.data))
        batch_data = (self.data[self.batch_id:end_batch])
        batch_labels = (self.labels[self.batch_id:end_batch])
        batch_seqlen = (self.seqlen[self.batch_id:end_batch])
        self.batch_id = end_batch
        return batch_data, batch_labels, batch_seqlen

# ==========
#   MODEL
# ==========
class ModelRNN:
    """ Define a dynamic rnn structure."""
    def dynamic_rnn(self, x, seqlen, weights, biases):
        """ Prepare data shape to match `rnn` function requirements
        Current data input shape: (batch_size, n_steps/lenJobs, n_input/3values)
        Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

        :param x: A tensor which holds inputs value, shape:(?, 200, 3), dtype:float32.
        :param seqlen: A tensor which holds sequence length value, dtype:int32.
        :param weights: A tf.Variable of shape(128, 2).
        :param biases: A tf.Variable of shape(2, ).
        :return: The output of lstm cell.
        """
        x = tf.unstack(value=x, num=seq_max_len, axis=1) # Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        with tf.name_scope("LSTM_layers"):
            with tf.name_scope("LSTM_cell"):
                lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden) # Define a lstm cell with tensorflow
                #lstm_cell = tf.contrib.rnn.LSTMCell(n_hidden, use_peepholes=True) # Try more lstm cell

                # To check the structure of states, check LSTMStateTuple.
                # Stores two elements: `(c, h)`, in that order. Where `c` is the hidden state and `h`
                # is the output.
                # The names are misleading
                # Hidden state: the output of EACH cell, used to decide how to modify the memory
                # Cell state: the memory. The final output.

            """ Get lstm cell output, providing "sequence_length" will perform dynamic calculation.
            Args:
                cell: An instance of RNNCell.
                inputs: A length T list of inputs, each a Tensor of shape [batch_size, input_size].
                dtype: The data type for the initial state and expected output.
                sequence_length: Specifies the length of each sequence in inputs. If the sequence_length 
                                 vector is provided, dynamic calculation is performed. This method of 
                                 calculation does not compute the RNN steps past the maximum sequence length
                                 of the minibatch (thus saving computational time), and properly propagates 
                                 the state at an example's sequence length to the final state output.
            
            Returns:
                'outputs' is a list of output at every timestep.
                'state' is the final state.
            """
            outputs, state = tf.contrib.rnn.static_rnn(cell=lstm_cell, inputs=x, dtype=tf.float32, sequence_length=seqlen)

        # Old version
        # 'outputs' is a list of output at every timestep, we pack them in a Tensor
        # and change back dimension to [batch_size, n_step, n_input]
        # outputs = tf.stack(outputs)
        # outputs = tf.transpose(outputs, [1, 0, 2])
        # Hack to build the indexing and retrieve the right output.
        # batch_size = tf.shape(outputs)[0]
        # Start indices for each sample
        # index = tf.range(0, batch_size) * seq_max_len + (seqlen - 1)
        # Indexing
        # outputs = tf.gather(tf.reshape(outputs, [-1, n_hidden]), index)
        # Linear activation, using outputs computed above
        # return tf.matmul(outputs, weights['out']) + biases['out']

        state = tf.stack(state[1])
        pred_state = tf.matmul(tf.stack(state), weights['out']) + biases['out']
        return pred_state

# ==========
#   TEST
# ==========
class AccuarcyTest():
    """Use test data to test accuracy."""
    def test_accuarcy(self, sess):
        """Generate test data and test.

        :param sess: An instance of tf.Session for running Tensorflow test operation.
        """
        print("-----------------Reading test data------------------")
        test = DataGenerator()
        test.data_extract(max_seq_len=seq_max_len, path=dir_test_data)
        test_data = test.data
        test_labels = test.labels
        test_seqlen = test.seqlen
        #l = list(zip(test_data, test_labels, test_seqlen))
        #random.shuffle(l)
        #test_data, test_labels, test_seqlen = zip(*l)
        test.data_shuffle()
        print("------------------------Start testing------------------------")
        print("Testing Accuracy:", \
              sess.run(accuracy, feed_dict={x: test_data, y: test_labels,
                                            seqlen: test_seqlen}))
        """acc, loss = sess.run([accuracy, cost], feed_dict={x: test_data, y: test_labels,
                                                          seqlen: test_seqlen})
        summary = sess.run(merged_summary, feed_dict={x: test_data, y: test_labels,
                                                      seqlen: test_seqlen})
        writer.add_summary(summary)
        print(" Loss= " + "{:.6f}".format(loss) + ", Test Accuracy= " + "{:.5f}".format(acc))"""

if __name__ == "__main__":
    train = DataGenerator()
    print("--------------------Data balancing----------------------")
    train.data_balance(dir_solutions, dir_train_data, dir_test_data)
    print("-----------------Reading training data------------------")
    trainset = train.data_extract(max_seq_len=seq_max_len, path=dir_train_data)
    train.data_shuffle()

    # Create a graph of Tensorflow
    print("---------------------Creating graph---------------------")
    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, seq_max_len, 3], name="x_input") # tf Graph x_input
        y = tf.placeholder(tf.float32, [None, n_classes], name="y_input") # tf Graph y_input
        seqlen = tf.placeholder(tf.int32, [None], name="sequence_len") # A placeholder for indicating each sequence length

    with tf.name_scope('weights'):
        weights = {
            'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
        } # Define weights

    with tf.name_scope('biases'):
        biases = {
            'out': tf.Variable(tf.random_normal([n_classes]))
        } # Define biases

    pred = ModelRNN().dynamic_rnn(x, seqlen, weights, biases) # Create an instance of dynamic rnn model
    tf.summary.histogram('predict', pred) # Adding a histogram summary for visualizing data in TensorBoard

    with tf.name_scope('loss'):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=pred, labels=y)) # Define loss

    #with tf.name_scope('learning_rate'):
        # global_step = tf.Variable(0)
        # Apply exponential decay to the learning rate.
        #learning_rate = tf.train.exponential_decay(start_learning_rate, global_step, 200, 0.48, staircase=True)

    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=start_learning_rate).minimize(cost)
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=start_learning_rate).minimize(cost, global_step=global_step)
        #optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(cost, global_step) # Define optimizer
    with tf.name_scope('accuracy'):
        # Evaluate model
        correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    #tf.summary.scalar('learning_rate', learning_rate)
    tf.summary.scalar('loss', cost)
    tf.summary.scalar('accuracy', accuracy)

    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(log_dir)

    init = tf.global_variables_initializer() # Initialize the variables
    saver = tf.train.Saver() # Save the model

    # Start training
    print("------------------------Start training------------------------")
    with tf.Session() as sess:
        sess.run(init)  # Run the initializer

        writer.add_graph(sess.graph)
        for step in range(1, training_steps + 1):
            batch_x, batch_y, batch_seqlen = train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
                                           seqlen: batch_seqlen})  # Run optimization op (backprop)
            if step % display_step == 0 or step == 1:
                #lr = sess.run(learning_rate)
                # Verify the relation of outputs and state
                # pred_outputs = sess.run(pred, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
                # print("outputs:", pred_outputs)
                # pred_states = sess.run(pred_state, feed_dict={x: batch_x, y: batch_y, seqlen: batch_seqlen})
                # print("state:", pred_states)
                acc, loss = sess.run([accuracy, cost], feed_dict={x: batch_x, y: batch_y,
                                                                  seqlen: batch_seqlen}) # Calculate batch accuracy & loss
                summary = sess.run(merged_summary, feed_dict={x: batch_x, y: batch_y,
                                                                  seqlen: batch_seqlen})
                writer.add_summary(summary, step)
                print("Step " + str(step * batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
        print("Optimization Finished!")

        AccuarcyTest().test_accuarcy(sess) # Test accuracy
        saver.save(sess, model_dir) # Save the model
        print("Save model success!")