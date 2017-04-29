# Projects: single- and multi-layer perceptrons
#
# :Authors: Jaco du Toit <jacowp357@gmail.com>
# :Description: This is linear and logistic regression classes used in
#               single- and multi-layer perceptrons using Tensorflow.
#
import tensorflow as tf
import numpy as np


class LinearRegression(object):
    """ Linear Regression class.
    """
    def __init__(self, input, n_in, n_out, W=None, b=None):
        if W is None:
            self.W = tf.Variable(tf.zeros([n_in, n_out]), name="W", dtype=tf.float32)
        else:
            self.W = W
        if b is None:
            self.b = tf.Variable(tf.zeros([1, n_out]), name="b", dtype=tf.float32)
        else:
            self.b = b
        self.y_pred = tf.matmul(input, self.W) + self.b
        self.params = [self.W, self.b]
        self.input = input

    def mean_squared_errors(self, y):
        return tf.reduce_mean(tf.pow(self.y_pred - y, 2))


class HiddenLayer(object):
    """ Hidden layer class for a Multi-layer Perceptron.
    """
    def __init__(self, input, n_in, n_out, W=None, b=None, activation=tf.nn.sigmoid, seed=None):
        self.input = input
        if W is None:
            W_values = tf.random_normal([n_in, n_out], mean=0, stddev=(np.sqrt(6 / n_in + n_out + 1)), seed=seed)
            if activation == tf.nn.sigmoid:
                W_values *= 4
            W = tf.Variable(W_values, name="h_W")
        if b is None:
            b_values = tf.random_normal([1, n_out], mean=0, stddev=(np.sqrt(6 / n_in + n_out + 1)), seed=seed)
            b = tf.Variable(b_values, name="h_b")
        self.W = W
        self.b = b
        lin_output = tf.matmul(input, self.W) + self.b
        self.output = (lin_output if activation is None else activation(lin_output))
        self.params = [self.W, self.b]


class MLPRegression(object):
    """ Multi-Layer perceptron consisting of a hidden layer and a fully connected
        linear regression layer.
    """
    def __init__(self, input, n_in, n_hidden, n_out, seed=None):
        self.hiddenLayer = HiddenLayer(input=input, n_in=n_in, n_out=n_hidden, activation=tf.nn.sigmoid, seed=seed)
        self.linRegressionLayer = LinearRegression(input=self.hiddenLayer.output, n_in=n_hidden, n_out=n_out)
        self.L1 = tf.reduce_sum(abs(self.linRegressionLayer.W)) + tf.reduce_sum(abs(self.hiddenLayer.W))
        self.L2 = tf.reduce_sum(self.linRegressionLayer.W ** 2) + tf.reduce_sum(self.hiddenLayer.W ** 2)
        self.mean_squared_errors = self.linRegressionLayer.mean_squared_errors
        self.y_pred = self.linRegressionLayer.y_pred
        self.params = self.linRegressionLayer.params
        self.input = input


# def train_mlp_model(learning_rate=0, momentum=0, L1_reg=0, L2_reg=0, max_iter=0):
#     train_x = np.asarray([[0.0, 0.1, 0.2], [1.0, 1.1, 1.3], [2.0, 2.1, 1.9], [3.0, 3.1, 3.11], [4.0, 4.1, 13.9]])
#     train_y = np.asarray([[1.2, 2.2], [5.3, 5.1], [2.9, 2.1], [1.9, 9.1], [4.2, 3.2]])
#     n_dims = train_x.shape[1]
#     n_out = train_y.shape[1]
#     X = tf.placeholder(tf.float32, [None, n_dims])
#     Y = tf.placeholder(tf.float32, [None, n_out])
#     regressor = MLPRegression(input=X, n_in=n_dims, n_hidden=1, n_out=n_out, seed=1)
#     init = tf.initialize_all_variables()
#     cost = (regressor.mean_squared_errors(Y) + L1_reg * regressor.L1 + L2_reg * regressor.L2)
#     training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#     sess = tf.Session()
#     sess.run(init)
#     iterations = 0

#     while iterations < max_iter:
#         iterations += 1
#         sess.run(training_step, feed_dict={X: train_x, Y: train_y})
#         print("iteration: {} cost: {}".format(iterations, sess.run(cost, feed_dict={X: train_x, Y: train_y})))

# train_mlp_model(learning_rate=0.01, L1_reg=0.001, L2_reg=0.001, max_iter=1000)
