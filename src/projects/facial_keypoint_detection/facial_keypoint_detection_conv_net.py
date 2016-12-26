# Projects: facial_keypoint_detection_conv_net
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


def train_conv_net(dataset, model_name=None, learning_rate=0, n_epochs=0, nkerns=[32, 64, 128], batch_size=0):
    rng = np.random.RandomState(None)
    datasets, num_features, num_outputs = convert_data_theano(dataset)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    # test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    # n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches //= batch_size
    n_valid_batches //= batch_size
    # n_test_batches //= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch

    # start-snippet-1
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.matrix('y')  # the labels are presented as 1D vector of

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # Reshape matrix of rasterized images of shape (batch_size, 96 * 96)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    # (96, 96) is the size of MNIST images.
    layer0_input = x.reshape((batch_size, 1, 96, 96))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (96-3+1 , 96-3+1) = (94, 94)
    # maxpooling reduces this further to (94/2, 94/2) = (47, 47)
    # 4D output tensor is thus of shape (batch_size, nkerns[0], 47, 47)
    layer0 = mlp.LeNetConvPoolLayer(
        rng,
        input=layer0_input,
        image_shape=(batch_size, 1, 96, 96),
        filter_shape=(nkerns[0], 1, 3, 3),
        poolsize=(2, 2),
        activation=T.nnet.relu
    )

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (47-2+1, 47-2+1) = (46, 46)
    # maxpooling reduces this further to (46/2, 46/2) = (23, 23)
    # 4D output tensor is thus of shape (batch_size, nkerns[1], 23, 23)
    layer1 = mlp.LeNetConvPoolLayer(
        rng,
        input=layer0.output,
        image_shape=(batch_size, nkerns[0], 47, 47),
        filter_shape=(nkerns[1], nkerns[0], 2, 2),
        poolsize=(2, 2),
        activation=T.nnet.relu
    )

    layer2 = mlp.LeNetConvPoolLayer(
        rng,
        input=layer1.output,
        image_shape=(batch_size, nkerns[1], 23, 23),
        filter_shape=(nkerns[2], nkerns[1], 2, 2),
        poolsize=(2, 2),
        activation=T.nnet.relu
    )

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size, num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (batch_size, nkerns[1] * 11 * 11),
    # or (500, 64 * 11 * 11) = (500, 15488) with the default values.
    layer3_input = layer2.output.flatten(2)

    # construct a fully-connected sigmoidal layer
    layer3 = mlp.HiddenLayer(
        rng,
        input=layer3_input,
        n_in=nkerns[2] * 11 * 11,
        n_out=500,
        activation=None
    )

    layer4 = mlp.HiddenLayer(
        rng,
        input=layer3.output,
        n_in=500,
        n_out=500,
        activation=None
    )

    regressor = mlp.LinearRegression(input=layer4.output, n_in=500, n_out=30)
    cost = regressor.mean_squared_errors(y)

    validate_cross_model = theano.function(inputs=[index], outputs=regressor.mean_squared_errors(y), givens={x: valid_set_x[index * batch_size: (index + 1) * batch_size], y: valid_set_y[index * batch_size: (index + 1) * batch_size]})
    validate_train_model = theano.function(inputs=[index], outputs=regressor.mean_squared_errors(y), givens={x: train_set_x[index * batch_size: (index + 1) * batch_size], y: train_set_y[index * batch_size: (index + 1) * batch_size]})

    params = regressor.params + layer4.params + layer3.params + layer2.params + layer1.params + layer0.params
    # grads = T.grad(cost, params)

    # updates = [(param_i, param_i - learning_rate * grad_i) for param_i, grad_i in zip(params, grads)]
    momentum = 0.9
    updates = gradopt.nesterov_accelerated_gradient(cost, params, learning_rate, momentum)

    train_model = theano.function(inputs=[index], outputs=cost, updates=updates, givens={x: train_set_x[index * batch_size: (index + 1) * batch_size], y: train_set_y[index * batch_size: (index + 1) * batch_size]})

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')

    start_time = timeit.default_timer()
    epoch = 0
    tr_error = []
    cv_error = []

    while epoch < n_epochs:
        epoch = epoch + 1
        for minibatch_index in range(n_train_batches):
            iter = (epoch - 1) * n_train_batches + minibatch_index

            train_model(minibatch_index)

            train_losses = np.mean([validate_train_model(i) for i in range(n_train_batches)])
            train_losses = np.sqrt(train_losses) * 48
            validation_losses = np.mean([validate_cross_model(i) for i in range(n_valid_batches)])
            validation_losses = np.sqrt(validation_losses) * 48

            tr_error.append(train_losses)
            cv_error.append(validation_losses)

            print("Iteration: {}, MSE train: {:.5f}, MSE val: {:.5f}, MSE train/val: {:.5f}".format(iter, train_losses, validation_losses, train_losses / validation_losses))
        # save the model #
        with open(model_name, "wb") as f:
            for obj in [regressor, tr_error, cv_error]:
                cPickle.dump(obj, f, protocol=cPickle.HIGHEST_PROTOCOL)
        # plot performance #
        # loaded_obj = []
        # with open(model_name, "rb") as f:
        #     for i in range(3):
        #         loaded_obj.append(cPickle.load(f))
        # plot_predictions(test_img, loaded_obj[0], 4, run_time=True)
        print("Epoch {}, minibatch {}/{}...".format(epoch, minibatch_index + 1, n_train_batches))

    end_time = timeit.default_timer()
    print(('The code ran for %.2fm' % ((end_time - start_time) / 60.)))


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


def transformed(Xb, yb):
    flip_indices = [(0, 2), (1, 3), (4, 8), (5, 9),
                    (6, 10), (7, 11), (12, 16), (13, 17),
                    (14, 18), (15, 19), (22, 24), (23, 25)]
    # Flip half of the images in this batch at random:
    Xb = Xb.reshape(-1, 96, 96)
    bs = Xb.shape[0]
    indices = np.random.choice(bs, bs, replace=False)
    Xb[indices] = Xb[indices, :, ::-1]

    if yb is not None:
        # Horizontal flip of all x coordinates:
        yb[indices, ::2] = yb[indices, ::2] * -1

        # Swap places, e.g. left_eye_center_x -> right_eye_center_x
        for a, b in flip_indices:
            yb[indices, a], yb[indices, b] = (yb[indices, b], yb[indices, a])
    Xb = Xb.reshape(-1, 9216)
    return Xb, yb


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
    # plt.yscale("log")
    print("Method %s : min train %.3f, valid %.3f)" % (method, np.min(tr_error), np.min(cv_error)))
    plt.show()


def plot_sample(x, y, axis):
    img = x.reshape(96, 96)
    axis.imshow(img, cmap='gray')
    axis.scatter(y[0::2] * 48 + 48, y[1::2] * 48 + 48, marker='x', s=10)


if __name__ == '__main__':
    file_test = 'data/test.csv'
    file_train = 'data/training.csv'
    # X, Y = load(file_train)
    X, Y = pickle.load(open("data/data.p", "rb"))
    X_flipped, Y_flipped = transformed(X.copy(), Y.copy())

    X = np.concatenate((X, X_flipped), axis=0)
    Y = np.concatenate((Y, Y_flipped), axis=0)

    # keep random seed for now #
    # train_set_x, test_set_x, train_set_y, test_set_y = train_test_split(X, Y, test_size=0.3, random_state=0)
    # test_set_x, cros_set_x, test_set_y, cros_set_y = train_test_split(test_set_x, test_set_y, test_size=0.5, random_state=0)
    train_set_x, cros_set_x, train_set_y, cros_set_y = train_test_split(X, Y, test_size=0.30, random_state=None)

    print('Train set: X dim %s, Y dim %s' % (str(train_set_x.shape), str(train_set_y.shape)))
    print('Cross set: X dim %s, Y dim %s' % (str(cros_set_x.shape), str(cros_set_y.shape)))
    # print('Test set : X dim %s, Y dim %s' % (str(test_set_x.shape), str(test_set_y.shape)))

    # data = [[train_set_x, train_set_y], [cros_set_x, cros_set_y], [test_set_x, test_set_y]]
    data = [[train_set_x, train_set_y], [cros_set_x, cros_set_y]]

    model_name = "conv.p"

    # prev_model = pickle.load(open("test2.p", "rb"))

    train_conv_net(dataset=data, model_name=model_name, learning_rate=0.01, n_epochs=1000, nkerns=[16, 32, 64], batch_size=128)

    # # plot performance #
    # loaded_obj = []
    # with open(model_name, "rb") as f:
    #     for i in range(3):
    #         loaded_obj.append(cPickle.load(f))

    # plot_performance(loaded_obj[1], loaded_obj[2], 'test')

    # X = load(file_test, test=True)[6:7]
    # # X = X[np.random.permutation(len(X))]
    # plot_predictions(X, loaded_obj[0], 1, run_time=False)

    # # plot weights of hidden layer #
    # # print(loaded_obj[0].params[0].get_value().shape)
    # # val_W = loaded_obj[0].params[0].get_value().T
    # # activations = [val_W[i, :].reshape((96, 96)) for i in range(val_W.shape[0])]

    # # fig = plt.figure(figsize=(24, 24))
    # # fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.001, wspace=0.001)
    # # for i, w in enumerate(activations[:20]):
    # #     ax = fig.add_subplot(5, 4, i + 1, xticks=[], yticks=[])
    # #     ax.imshow(w, cmap='gray')
    # #     plt.xlim([0, 96])
    # #     plt.ylim([96, 0])
    # #     ax.set_aspect('equal')
    # # plt.show()
