# Projects: facial_keypoint_detection
#
# :Author: Jaco du Toit <jacowp357@gmail.com>
# :Date: 01/11/2016
# :Description: Attempt at modelling the facial keypoint detection
#               problem presented in a Kaggle challenge.
# :URL: <https://www.kaggle.com/c/facial-keypoints-detection>
#
import os
import sys
from PIL import Image
import timeit
import numpy as np
import theano
from pandas.io.parsers import read_csv
import theano.tensor as T
from six.moves import cPickle
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
import models.mlp_regression as mlp
import models.gradient_optimisation as gradopt
plt.style.use('ggplot')


def load(fname, test=None):
    # adapted from Daniel Nouri <http://danielnouri.org> #
    df = read_csv(os.path.expanduser(fname))
    print(df.shape)
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))
    # drop nans for now #
    df = df.dropna()
    # scale pixel values to [0, 1] #
    df2 = pd.DataFrame((np.vstack(df['Image'].values) / 255.).astype(np.float64))
    df2.columns = ['img_' + str(i) for i in range(df2.shape[1])]
    df.reset_index(drop=True, inplace=True)
    df2.reset_index(drop=True, inplace=True)
    df = pd.concat([df, df2], axis=1)
    X = df.filter(regex='img_').as_matrix()
    if not test:
        for col in df.columns[0:30]:
            # scale target coordinates to [-1, 1] #
            df[col] = ((df[col] - 48) / 48).astype(np.float64)
        Y = df[df.columns[0:30]].as_matrix()
        print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(X.shape, X.min(), X.max()))
        print("Y.shape == {}; Y.min == {:.3f}; Y.max == {:.3f}".format(Y.shape, Y.min(), Y.max()))
        return X, Y
    else:
        print("X.shape == {}; X.min == {:.3f}; X.max == {:.3f}".format(X.shape, X.min(), X.max()))
        return X


def convert_data_theano(dataset):
    # train_set, valid_set, test_set = dataset[0], dataset[1], dataset[2]
    train_set, valid_set = dataset[0], dataset[1]
    assert (train_set[0].shape)[1] == (valid_set[0].shape)[1], "Number of features for train, val do not match: {} and {}.".format(train_set.shape[1], valid_set.shape[1])
    num_features, num_output = (train_set[0].shape)[1], (train_set[1].shape)[1]

    def shared_dataset(data_xy, borrow=True):
        # function that loads the dataset into shared variables #
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x, dtype=theano.config.floatX), borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y, dtype=theano.config.floatX), borrow=borrow)
        return shared_x, shared_y

    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    # test_set_x, test_set_y = shared_dataset(test_set)
    # rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y)]
    return rval, num_features, num_output


class MLP(object):
    # Multi-Layer Perceptron consisting of a hidden layer and a fully connected #
    # linear regression layer #
    def __init__(self, rng, model, input, n_in, n_out, n_hidden, n_hidden2):
        # one hidden layer with sigmoid activations, connected to the final linear regression layer #
        if model:
            self.hiddenLayer = mlp.HiddenLayer(rng=rng, W=model.params[0], b=model.params[1], input=input, n_in=n_in, n_out=n_hidden, activation=T.nnet.softplus)  # T.tanh, T.nnet.sigmoid, T.nnet.relu, T.nnet.softplus
            # self.hiddenLayer2 = mlp.HiddenLayer(rng=rng, input=self.hiddenLayer.output, n_in=n_hidden, n_out=n_hidden2, activation=T.tanh)
            self.linRegressionLayer = mlp.LinearRegression(W=model.params[2], b=model.params[3], input=self.hiddenLayer.output, n_in=n_hidden, n_out=n_out)
        else:
            self.hiddenLayer = mlp.HiddenLayer(rng=rng, W=None, b=None, input=input, n_in=n_in, n_out=n_hidden, activation=T.nnet.softplus)  # T.tanh, T.nnet.sigmoid, T.nnet.relu, T.nnet.softplus
            # self.hiddenLayer2 = mlp.HiddenLayer(rng=rng, input=self.hiddenLayer.output, n_in=n_hidden, n_out=n_hidden2, activation=T.nnet.softplus)
            self.linRegressionLayer = mlp.LinearRegression(W=None, b=None, input=self.hiddenLayer.output, n_in=n_hidden, n_out=n_out)
        # self.linRegressionLayer = mlp.LinearRegression(input=input, n_in=n_in, n_out=n_out)
        # two norms along with sum of squares loss function (output of linear regression layer) #
        # self.L1 = abs(self.linRegressionLayer.W).sum()
        # self.L2 = (self.linRegressionLayer.W ** 2).sum()
        self.L1 = abs(self.hiddenLayer.W).sum() + abs(self.linRegressionLayer.W).sum()
        self.L2 = (self.hiddenLayer.W ** 2).sum() + (self.linRegressionLayer.W ** 2).sum()
        self.mean_squared_errors = self.linRegressionLayer.mean_squared_errors
        self.y_pred = self.linRegressionLayer.y_pred
        self.params = self.hiddenLayer.params + self.linRegressionLayer.params
        # self.params = self.linRegressionLayer.params
        self.input = input


def train_mlp_model(dataset, model_name, model=None, learning_rate=0, momentum=0, L1_reg=0, L2_reg=0, n_hidden=0, n_hidden2=0, max_iter=0):
    # shared Theano format for data #
    datasets, num_features, num_outputs = convert_data_theano(dataset)
    train_set_x, train_set_y = datasets[0]
    cros_set_x, cros_set_y = datasets[1]
    # test_img = np.asmatrix(load(file_test, test=True))
    # test_set_x, test_set_y = datasets[2]

    print('build the model...')
    # mini-batch index #
    index = T.lscalar()
    # input matrix #
    x = T.matrix('x')
    # output matrix #
    y = T.matrix('y')
    # random state for weight initialisation #
    rng = np.random.RandomState(1234)
    # initialise multi-layer perceptron #
    regressor = MLP(rng=rng, model=model, input=x, n_in=num_features, n_out=num_outputs, n_hidden=n_hidden, n_hidden2=n_hidden2)
    # compute the cost function #
    cost = (regressor.mean_squared_errors(y) + L1_reg * regressor.L1 + L2_reg * regressor.L2)
    # theano function that computes the MSE #
    validate_train_model = theano.function(inputs=[index], outputs=regressor.mean_squared_errors(y), givens={x: train_set_x, y: train_set_y}, on_unused_input='ignore')
    validate_cross_model = theano.function(inputs=[index], outputs=regressor.mean_squared_errors(y), givens={x: cros_set_x, y: cros_set_y}, on_unused_input='ignore')
    # validate_test_model = theano.function(inputs=[index], outputs=regressor.mean_squared_errors(y), givens={x: test_set_x, y: test_set_y}, on_unused_input='ignore')
    # compute the gradient of the cost function #
    # gparams = [T.grad(cost, param) for param in regressor.params]
    # specify update of the model parameters as list of (variable, update_expression) pairs #
    # updates = [(param, param - learning_rate * gparam) for param, gparam in zip(regressor.params, gparams)]
    # updates = gradopt.momentum(cost, regressor.params, learning_rate, momentum)
    updates = gradopt.nesterov_accelerated_gradient(cost, regressor.params, learning_rate, momentum)
    # updates = gradopt.adadelta(regressor.params, gparams)
    # updates = gradopt.rms_prop(cost, regressor.params, learning_rate, momentum)
    # compiling a Theano function `train_model` that returns the cost and updates parameters #
    # train_model = theano.function(inputs=[index], outputs=cost, updates=updates, givens={x: train_set_x, y: train_set_y}, on_unused_input='ignore')
    train_model = theano.function(inputs=[index], outputs=cost, updates=updates, givens={x: train_set_x, y: train_set_y}, on_unused_input='ignore')
    print('training the model...')

    start_time = timeit.default_timer()
    iterations = 0
    tr_error = []
    cv_error = []
    # ts_error = []

    while iterations < max_iter:
        iterations += 1
        train_model(0)
        train_losses = np.sqrt(validate_train_model(0)) * 48
        validation_losses = np.sqrt(validate_cross_model(0)) * 48
        # test_losses = np.sqrt(validate_test_model(0)) * 48
        tr_error.append(train_losses)
        cv_error.append(validation_losses)
        # ts_error.append(test_losses)
        print("Iteration {}: MSE training {:.5f}, validation {:.5f}".format(iterations, train_losses, validation_losses))
        # save the model #
        with open(model_name, "wb") as f:
            for obj in [regressor, tr_error, cv_error]:
                cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
        # plot_predictions(test_img, pickle.load(open(model_name, "rb"))[0], 4, run_time=True)
    end_time = timeit.default_timer()
    print(('The code ran for %.2fm' % ((end_time - start_time) / 60.)))


def plot_predictions(data, model, img_num, run_time=None):
    # compile a predictor function #
    predict_model_n = theano.function(inputs=[model.input], outputs=model.y_pred, allow_input_downcast=True)
    if run_time:
        plt.ion()
    fig = plt.figure(figsize=(7, 7))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.02, wspace=0.02)
    for i in range(img_num):
        ax = fig.add_subplot(np.sqrt(img_num), np.sqrt(img_num), i + 1, xticks=[], yticks=[])
        img = data[i].reshape(96, 96)
        ax.imshow(img, cmap='gray')
        pred_y = predict_model_n(np.asmatrix(data[i]))[0]
        ax.scatter(pred_y[0::2] * 48 + 48, pred_y[1::2] * 48 + 48, linewidth=2, marker='+', color='magenta', s=20)
        plt.xlim([0, 96])
        plt.ylim([96, 0])
    if run_time:
        plt.pause(0.001)
    else:
        plt.show()


def plot_performance(tr_error, cv_error, method):
    plt.plot(tr_error, linewidth=2, label="Training loss (%s)" % (method))
    plt.plot(cv_error, linewidth=2, label="Validation loss (%s)" % (method))
    plt.legend()
    plt.xlabel("Iterations")
    plt.ylabel("MSE Loss")
    plt.title("MSE loss per iteration")
    plt.yscale("log")
    print("Method %s : min train %.3f, valid %.3f)" % (method, np.min(tr_error), np.min(cv_error)))
    plt.show()


if __name__ == '__main__':
    file_test = 'data/test.csv'
    file_train = 'data/training.csv'
    # X, Y = load(file_train)
    X, Y = pickle.load(open("data/data.p", "rb"))

    # keep random seed for now #
    # train_set_x, test_set_x, train_set_y, test_set_y = train_test_split(X, Y, test_size=0.2, random_state=0)
    # test_set_x, cros_set_x, test_set_y, cros_set_y = train_test_split(test_set_x, test_set_y, test_size=0.5, random_state=0)
    train_set_x, cros_set_x, train_set_y, cros_set_y = train_test_split(X, Y, test_size=0.45, random_state=None)

    print('Train set: X dim %s, Y dim %s' % (str(train_set_x.shape), str(train_set_y.shape)))
    print('Cross set: X dim %s, Y dim %s' % (str(cros_set_x.shape), str(cros_set_y.shape)))
    # print('Test set : X dim %s, Y dim %s' % (str(test_set_x.shape), str(test_set_y.shape)))

    # data = [[train_set_x, train_set_y], [cros_set_x, cros_set_y], [test_set_x, test_set_y]]
    data = [[train_set_x, train_set_y], [cros_set_x, cros_set_y]]

    model_name = "l0.01_m0.9_h200.p"

    # # prev_model = pickle.load(open("test2.p", "rb"))

    train_mlp_model(dataset=data, model=None, model_name=model_name, learning_rate=0.01, momentum=0.9, L1_reg=0.0, L2_reg=0.0, max_iter=10000, n_hidden=200, n_hidden2=None)

    # plot performance #
    # loaded_obj = []
    # with open(model_name, "rb") as f:
    #     for i in range(3):
    #         loaded_obj.append(cPickle.load(f))

    # plot_performance(loaded_obj[1], loaded_obj[2], 'test')

    # X = load(file_test, test=True)
    # X = X[np.random.permutation(len(X))]
    # plot_predictions(X, loaded_obj[0], 4, run_time=False)
