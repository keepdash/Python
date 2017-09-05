import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops

# Class DNN_MC : Multi-classes deep neural network
class DNN_MC:
    def __init__(self, X_train, Y_train, X_test, Y_test, learning_rate, hidden_layer_units, hidden_layer_dropout,
                 minibatch_size, num_epochs, print_cost=True):
        assert (len(hidden_layer_units) == len(hidden_layer_dropout))
        self.parameters = {}
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        self.learning_rate = learning_rate
        self.minibatch_size = minibatch_size
        self.num_epochs = num_epochs
        self.print_cost = print_cost
        (self.m, self.n_x) = X_train.shape
        self.n_y = Y_train.shape[1]
        self.layers_dims = [self.n_x] + hidden_layer_units + [self.n_y]
        self.kp_train = hidden_layer_dropout
        self.kp_eval = np.ones(len(hidden_layer_dropout), dtype=np.float32)
        self.L = len(hidden_layer_units) + 1

    def create_placeholders(self):
        X = tf.placeholder(tf.float32, [None, self.n_x])
        Y = tf.placeholder(tf.float32, [None, self.n_y])
        keep_prob = tf.placeholder(tf.float32, [self.L - 1])
        return X, Y, keep_prob


    def initialize_parameters(self):
        tf.set_random_seed(1)
        for l in range(1, self.L + 1):
            self.parameters['W' + str(l)] = tf.get_variable('W'+str(l), [self.layers_dims[l-1], self.layers_dims[l]], initializer=tf.contrib.layers.xavier_initializer(seed=1))
            self.parameters['b' + str(l)] = tf.get_variable('b'+str(l), [1, self.layers_dims[l]], initializer=tf.zeros_initializer())


    def forward_propagation(self, X, keep_prob):
        A = X
        for l in range(1, self.L):
            A_prev = A
            A_relu = tf.nn.relu(tf.add(tf.matmul(A_prev, self.parameters['W' + str(l)]), self.parameters['b' + str(l)]))
            A = tf.nn.dropout(A_relu, keep_prob[l-1])

        logits = tf.add(tf.matmul(A, self.parameters['W' + str(self.L)]), self.parameters['b' + str(self.L)])
        return logits


    def compute_cost(sel, logits, Y):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
        return cost


    def random_mini_batches(self, seed):
        np.random.seed(seed=seed)
        indices = np.arange(self.X_train.shape[0])
        np.random.shuffle(indices)

        for start_idx in range(0, self.X_train.shape[0] - self.minibatch_size + 1, self.minibatch_size):
            excerpt = indices[start_idx:start_idx + self.minibatch_size]
            yield (self.X_train[excerpt, :], self.Y_train[excerpt, :])

    def train(self):
        ops.reset_default_graph()
        tf.set_random_seed(1)
        seed = 3
        costs = []

        X, Y, keep_prob = self.create_placeholders()
        self.initialize_parameters()
        logits = self.forward_propagation(X, keep_prob)
        cost = self.compute_cost(logits, Y)

        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(cost)
        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(self.num_epochs):
                epoch_cost = 0.
                num_minibatches = int(self.m / self.minibatch_size)
                seed = seed + 1
                minibatches = self.random_mini_batches(seed)

                for minibatch in minibatches:
                    (minibatch_X, minibatch_Y) = minibatch
                    _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y, keep_prob : self.kp_train})
                    epoch_cost += minibatch_cost / num_minibatches

                if self.print_cost == True and epoch % 10 == 0:
                    print("Cost after epoch %i: %f" % (epoch, epoch_cost))
                if self.print_cost == True and epoch % 2 == 0:
                    costs.append(epoch_cost)

            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iter    ations (per tens)')
            plt.title("Learning rate =" + str(self.learning_rate))
            plt.show()

            parameters = sess.run(self.parameters)
            print("Parameters have been trained!")

            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("Train Accuracy:", accuracy.eval({X: self.X_train, Y: self.Y_train, keep_prob : self.kp_eval}))
            print("Test Accuracy:", accuracy.eval({X: self.X_test, Y: self.Y_test, keep_prob : self.kp_eval}))

        return parameters
# Class DNN_MC
