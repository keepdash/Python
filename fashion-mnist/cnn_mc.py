import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops

# Class CNN_MC : Multi-classes deep neural network
class CNN_MC:
    def __init__(self, X_train, Y_train, X_test, Y_test, learning_rate, minibatch_size, num_epochs, print_cost=True):
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


    def create_placeholders(self):
        X = tf.placeholder(tf.float32, [None, self.n_x])
        Y = tf.placeholder(tf.float32, [None, self.n_y])
        is_training = tf.placeholder(tf.bool)
        return X, Y, is_training


    def forward_propagation(self, X, is_training):
        # Input Layer
        input_layer = tf.reshape(X, [-1, 28, 28, 1])

        # Convolutional Layer #1 and Pooling Layer #1 (28->26->13, 1->32)
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[3, 3],
            padding="valid",
            activation=tf.nn.relu)
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)
        dp_pool1 = tf.layers.dropout(inputs=pool1, rate=0.25, training=is_training)

        # Convolutional Layer #2 and Pooling Layer #2 (13->11->5, 32->64)
        conv2 = tf.layers.conv2d(
            inputs=dp_pool1,
            filters=64,
            kernel_size=[3, 3],
            padding="valid",
            activation=tf.nn.relu)
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)
        dp_pool2 = tf.layers.dropout(inputs=pool2, rate=0.25, training=is_training)

        # Convolutional Layer #3 (5->3, 64->128)
        conv3 = tf.layers.conv2d(
            inputs=dp_pool2,
            filters=128,
            kernel_size=[3, 3],
            padding="valid",
            activation=tf.nn.relu)
        dp_conv3 = tf.layers.dropout(inputs=conv3, rate=0.4, training=is_training)

        # Dense Layer (3*3*128->128)
        conv3_flat = tf.reshape(dp_conv3, [-1, 3 * 3 * 128])
        dense1 = tf.layers.dense(inputs=conv3_flat, units=128, activation=tf.nn.relu)
        dp_dense1 = tf.layers.dropout(inputs=dense1, rate=0.3, training=is_training)

        # Logits Layer (128->10)
        logits = tf.layers.dense(inputs=dp_dense1, units=self.n_y)
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

        X, Y, is_training = self.create_placeholders()
        logits = self.forward_propagation(X, is_training)
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
                    _, minibatch_cost = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y, is_training: True})
                    epoch_cost += minibatch_cost / num_minibatches

                if self.print_cost == True and epoch % 5 == 0:
                    print("Cost after epoch %i: %f" % (epoch, epoch_cost))
                if self.print_cost == True: #and epoch % 2 == 0:
                    costs.append(epoch_cost)

            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iter    ations (per tens)')
            plt.title("Learning rate =" + str(self.learning_rate))
            plt.show()

            correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            train_acc = 0.0
            for i in range(60):
                train_acc += accuracy.eval({X: self.X_train[i*1000:(i+1)*1000, :], Y: self.Y_train[i*1000:(i+1)*1000, :], is_training : False})
            train_acc /= 60.0
            test_acc = 0.0
            for i in range(10):
                test_acc += accuracy.eval({X: self.X_test[i*1000:(i+1)*1000, :], Y: self.Y_test[i*1000:(i+1)*1000, :], is_training : False})
            test_acc /= 10.0
            print("Train Accuracy:", train_acc)
            print("Test Accuracy:", test_acc)

# Class CNN_MC
