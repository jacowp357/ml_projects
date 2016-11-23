# Projects: facial_keypoint_detection
#
# :Authors: Jaco du Toit <jacowp357@gmail.com>
# :Description: Attempt at modelling the facial keypoint detection
#               problem presented in a Kaggle challenge.
#
import os
import sys
from PIL import Image
import timeit
import numpy as np
import theano
from pandas.io.parsers import read_csv
import theano.tensor as T
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cross_validation import train_test_split
import models.mlp_regression as mlp


def load(fname, test=None):
    df = read_csv(os.path.expanduser(fname))
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
    train_set, valid_set, test_set = dataset[0], dataset[1], dataset[2]
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
    test_set_x, test_set_y = shared_dataset(test_set)
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
    return rval, num_features, num_output


def plot_sample(x, pred_y, act_y, axis):
    img = x.reshape(96, 96)
    plt.ion()
    plt.imshow(img, cmap='gray')
    plt.scatter(pred_y[0::2] * 48 + 48, pred_y[1::2] * 48 + 48, marker='x', color='red', s=20)
    plt.scatter(act_y[0::2] * 48 + 48, act_y[1::2] * 48 + 48, marker='o', color='blue', s=10)
    plt.pause(0.005)


class MLP(object):
    # Multi-Layer Perceptron consisting of a hidden layer and a fully connected #
    # linear regression layer #
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        # one hidden layer with sigmoid activations, connected to the final linear regression layer #
        self.hiddenLayer = mlp.HiddenLayer(rng=rng, input=input, n_in=n_in, n_out=n_hidden, activation=T.tanh)
        # the logistic regression layer gets as input the hidden units of the linear regression layer #
        self.linRegressionLayer = mlp.LinearRegression(input=self.hiddenLayer.output, n_in=n_hidden, n_out=n_out)
        # two norms along with sum of squares loss function (output of linear regression layer) #
        self.L1 = abs(self.hiddenLayer.W).sum() + abs(self.linRegressionLayer.W).sum()
        self.L2_sqr = (self.hiddenLayer.W ** 2).sum() + (self.linRegressionLayer.W ** 2).sum()
        self.mean_squared_errors = self.linRegressionLayer.mean_squared_errors
        self.y_pred = self.linRegressionLayer.y_pred
        self.params = self.hiddenLayer.params + self.linRegressionLayer.params
        self.input = input


def train_mlp_model(dataset, learning_rate=None, L1_reg=None, L2_reg=None, n_epochs=None, batch_size=None, n_hidden=None):
    # shared Theano format for data #
    datasets, num_features, num_outputs = convert_data_theano(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    # compute number of mini-batches #
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

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
    regressor = MLP(rng=rng, input=x, n_in=num_features, n_hidden=n_hidden, n_out=num_outputs)
    # compute the cost function #
    cost = regressor.mean_squared_errors(y) + L1_reg * regressor.L1 + L2_reg * regressor.L2_sqr
    # Theano function that computes the MSE on a minibatch #
    validate_model = theano.function(inputs=[index], outputs=regressor.mean_squared_errors(y), givens={x: valid_set_x[index * batch_size:(index + 1) * batch_size], y: valid_set_y[index * batch_size:(index + 1) * batch_size]})
    test_model = theano.function(inputs=[index], outputs=regressor.mean_squared_errors(y), givens={x: test_set_x[index * batch_size:(index + 1) * batch_size], y: test_set_y[index * batch_size:(index + 1) * batch_size]})
    # compute the gradient of the cost function #
    gparams = [T.grad(cost, param) for param in regressor.params]
    # specify update of the model parameters as list of (variable, update_expression) pairs #
    updates = [(param, param - learning_rate * gparam) for param, gparam in zip(regressor.params, gparams)]
    # compiling a Theano function `train_model` that returns the cost and updates parameters #
    train_model = theano.function(inputs=[index], outputs=cost, updates=updates, givens={x: train_set_x[index * batch_size:(index + 1) * batch_size], y: train_set_y[index * batch_size:(index + 1) * batch_size]})

    print('training the model...')
    # early stopping parameters #
    patience = 10                  # look at this many examples regardless #
    patience_increase = 2          # wait this much longer when a new best is found #
    improvement_threshold = 0.995  # a relative improvement of this much is considered significant #
    validation_frequency = min(n_train_batches, patience // 2)
    # go through this many minibatche before checking the network #
    # on the validation set; in this case we check every epoch #
    best_validation_loss = np.inf
    best_iter = 0
    test_score = 0.
    start_time = timeit.default_timer()
    epoch = 0
    done_looping = False
    while (epoch < n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            train_model(minibatch_index)
            # iteration number #
            iter = (epoch - 1) * n_train_batches + minibatch_index
            # evaluate on validation set #
            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set #
                validation_losses = [validate_model(i) for i in range(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)
                print("epoch {}, minibatch {}/{}, validation MSE {:.5f}".format(epoch, minibatch_index + 1, n_train_batches, np.sqrt(this_validation_loss) * 48))
                # save the best model #
                # pickle.dump(regressor, open("model.p", "wb"))
                # predict(dataset)
                # if best valid so far, improve patience and update the 'best' variables #
                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)
                    best_validation_loss = this_validation_loss
                    best_iter = iter
                    test_losses = [test_model(i) for i in range(n_test_batches)]
                    test_score = np.mean(test_losses)
                    print(('epoch %i, minibatch %i/%i, test MSE of best model %f') % (epoch, minibatch_index + 1, n_train_batches, np.sqrt(test_score) * 48))
                    # print(('epoch %i, minibatch %i/%i, test error of best model %f %%, learning rate: %f') % (epoch, minibatch_index + 1, n_train_batches, test_score * 100., learning_rate.get_value(borrow=True)))
            if patience <= iter:
                done_looping = True
                break
        # decay_learning_rate()
    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f obtained at iteration %i') % (np.sqrt(best_validation_loss) * 48, best_iter + 1))
    print(('The code for file ran for %.2fm' % ((end_time - start_time) / 60.)))


def predict(data):
    # load the saved model
    classifier = pickle.load(open("model.p", "rb"))
    # compile a predictor function
    predict_model_n = theano.function(inputs=[classifier.input], outputs=classifier.y_pred)

    fig = plt.figure(figsize=(6, 6))
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05)
    # ax = fig.add_subplot(1, 1, 1, xticks=[], yticks=[])
    for i in range(16):
        ax = fig.add_subplot(4, 4, i + 1, xticks=[], yticks=[])
        plot_sample(data[0][0][i:i + 1][0], predict_model_n(data[0][0][i:i + 1])[0], data[0][1][i], ax)
        # plot_sample(data[2][0][i:i + 1][0], predict_model_n(data[2][0][i:i + 1])[0], 0, ax)
    # plot_sample(data[2][0][1:2][0], predict_model_n(data[2][0][1:2])[0], ax)
    # plt.show()


def make_visual(layer_weights):
    max_scale = layer_weights.max(axis=-1).max(axis=-1)[np.newaxis, np.newaxis]
    min_scale = layer_weights.min(axis=-1).min(axis=-1)[np.newaxis, np.newaxis]
    return (255 * (layer_weights - min_scale) / (max_scale - min_scale)).astype('uint8')


def make_mosaic(layer_weights):
    # Dirty hack (TM)
    lw_shape = layer_weights.shape
    lw = make_visual(layer_weights).reshape(8, 12, *lw_shape[1:])
    lw = lw.transpose(0, 3, 1, 4, 2)
    lw = lw.reshape(8 * lw_shape[-1], 12 * lw_shape[-2], lw_shape[1])
    return lw


def plot_filters(layer_weights, title=None, show=False):
    mosaic = make_mosaic(layer_weights)
    plt.imshow(mosaic, interpolation='nearest')
    ax = plt.gca()
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    if title is not None:
        plt.title(title)
    if show:
        plt.show()


if __name__ == '__main__':
    file_train = './data/facepoints_train.csv'
    file_test = './data/facepoints_test.csv'

    X, Y = load(file_train)
    # keep random seed for now #
    train_set_x, test_set_x, train_set_y, test_set_y = train_test_split(X, Y, test_size=0.3, random_state=0)
    test_set_x, cros_set_x, test_set_y, cros_set_y = train_test_split(test_set_x, test_set_y, test_size=0.5, random_state=0)

    print('Train set: X - %s, Y - %s' % (str(train_set_x.shape), str(train_set_y.shape)))
    print('Cross set: X - %s, Y - %s' % (str(cros_set_x.shape), str(cros_set_y.shape)))
    print('Test set : X - %s, Y - %s' % (str(test_set_x.shape), str(test_set_y.shape)))

    data = [[train_set_x, train_set_y], [cros_set_x, cros_set_y], [test_set_x, test_set_y]]

    train_mlp_model(dataset=data, learning_rate=0.1, L1_reg=0.0, L2_reg=0.001, n_epochs=1000, batch_size=50, n_hidden=0)
