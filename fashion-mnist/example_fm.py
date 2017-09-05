import numpy as np
import tensorflow as tf
import pandas as pd
from dnn_mc import DNN_MC
from cnn_mc import CNN_MC

def load_dataset():
    train_data = pd.read_csv('../input/fashion-mnist_train.csv', dtype=np.uint8, low_memory=False)
    test_data = pd.read_csv('../input/fashion-mnist_test.csv', dtype=np.uint8, low_memory=False)

    X_train = train_data.drop('label', axis=1).as_matrix()
    Y_train = train_data['label']
    X_test = test_data.drop('label', axis=1).as_matrix()
    Y_test = test_data['label']
    classes = np.max(Y_train) + 1

    return X_train, Y_train, X_test, Y_test, classes


def one_hot_matrix(Y, classes):
    C = tf.constant(classes, name='C')
    one_hot_matrix = tf.one_hot(Y, C, 1.0, 0.0, -1)
    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()
    return one_hot


if __name__ == '__main__':
    print("load dataset")
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

    print("pre-process dataset")
    X_train = X_train_orig / 255.
    X_test = X_test_orig / 255.
    Y_train = one_hot_matrix(Y_train_orig, classes)
    Y_test = one_hot_matrix(Y_test_orig, classes)

    print("training")
    '''
    learning_rate = 0.0003
    hidden_layer_units = [64, 20]
    hidden_layer_dropout = [0.5, 0.6]
    minibatch_size = 256
    num_epochs = 100

    model = DNN_MC(X_train, Y_train, X_test, Y_test, learning_rate, hidden_layer_units, hidden_layer_dropout, minibatch_size, num_epochs)
    model.train()
    '''

    learning_rate = 0.0003
    minibatch_size = 256
    num_epochs = 50

    model = CNN_MC(X_train, Y_train, X_test, Y_test, learning_rate, minibatch_size, num_epochs)
    model.train()