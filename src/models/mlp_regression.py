# Projects: single- and multi-layer perceptrons
#
# :Authors: Jaco du Toit <jacowp357@gmail.com>
# :Description: This is linear and logistic regression classes used in
#               single- and multi-layer perceptrons using Theano.
#
import numpy as np
import theano
import theano.tensor as T


class LinearRegression(object):
    """ Linear Regression class.
    """
    def __init__(self, input, n_in, n_out):
        self.W = theano.shared(value=np.zeros((n_in, n_out), dtype=theano.config.floatX), name='W', borrow=True)
        self.b = theano.shared(value=np.zeros((n_out,), dtype=theano.config.floatX), name='b', borrow=True)
        self.y_pred = T.dot(input, self.W) + self.b
        self.params = [self.W, self.b]
        self.input = input

    def mean_squared_errors(self, y):
        return T.mean((self.y_pred - y) ** 2)


class LogisticRegression(object):
    """ Multi-class Logistic Regression class.
    """
    def __init__(self, input, n_in, n_out):
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out) #
        self.W = theano.shared(value=np.zeros((n_in, n_out), dtype=theano.config.floatX), name='W', borrow=True)
        # initialize the biases b as a vector of n_out 0s #
        self.b = theano.shared(value=np.zeros((n_out,), dtype=theano.config.floatX), name='b', borrow=True)
        # x is a matrix where row-j  represents input training sample-j #
        # b is a vector where element-k represent the free parameter of hyperplane-k #
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        # symbolic description of how to compute prediction as class whose probability is maximal #
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)
        # parameters of the model #
        self.params = [self.W, self.b]
        # keep track of model input #
        self.input = input

    def negative_log_likelihood(self, y):
        """ Return the mean of the negative log-likelihood of the prediction
            of this model under a given target distribution.

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

    def errors(self, y):
        """ Return a float representing the number of errors in the minibatch
            over the total number of examples of the minibatch; zero one loss over
            the size of the minibatch.
        """
        # check if y has same dimension of y_pred #
        if y.ndim != self.y_pred.ndim:
            raise TypeError('y should have the same shape as self.y_pred', ('y', y.type, 'y_pred', self.y_pred.type))
        # check if y is of the correct datatype #
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1 represents a mistake in prediction #
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()


class HiddenLayer(object):
    """ Hidden layer class for a Multi-layer Perceptron.
    """
    def __init__(self, rng, input, n_in, n_out, W=None, b=None, activation=T.nnet.sigmoid):
        """
        This is a hidden layer class for a Multi-layer Perceptron.
        The units are fully-connected and have a sigmoidal activation function.
        The w eight matrix W is of shape (n_in, n_out) and the bias vector b is of shape (n_out,).
        The nonlinearity is user defined. Hidden unit activation is given by: tanh(dot(input,W) + b) or
        sigmoid (T.tanh or T.nnet.sigmoid) We will be using tanh in this tutorial because it typically
        yields to faster training (and sometimes also to better local minima).
        """
        self.input = input
        # optimal initialization of weights is dependent on the #
        # activation function used (among other things). #
        # For example: results presented in [Xavier10] suggest that you #
        # should use 4 times larger initial weights for sigmoid #
        # compared to tanh. #
        if W is None:
            W_values = np.asarray(rng.uniform(low=-np.sqrt(6. / (n_in + n_out)), high=np.sqrt(6. / (n_in + n_out)), size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value=W_values, name='W', borrow=True)
        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
        self.W = W
        self.b = b
        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None else activation(lin_output))
        # parameters of the model #
        self.params = [self.W, self.b]


class MLPRegression(object):
    """ Multi-Layer perceptron consisting of a hidden layer and a fully connected
        linear regression layer.
    """
    def __init__(self, rng, input, n_in, n_hidden, n_hidden2, n_out):
        # one hidden layer with tanh activation, connected to the final linear regression layer #
        self.hiddenLayer = HiddenLayer(rng=rng, input=input, n_in=n_in, n_out=n_hidden, activation=T.tanh)
        self.hiddenLayer2 = HiddenLayer(rng=rng, input=self.hiddenLayer.output, n_in=n_hidden, n_out=n_hidden2, activation=T.tanh)
        # The logistic regression layer gets as input the hidden units of the linear reg. layer #
        self.linRegressionLayer = LinearRegression(input=self.hiddenLayer2.output, n_in=n_hidden2, n_out=n_out)
        # Two norms, along with sum of squares loss function (output of LinearRegression layer) #
        self.L1 = abs(self.hiddenLayer.W).sum() + abs(self.hiddenLayer2.W).sum() + abs(self.linRegressionLayer.W).sum()
        self.L2 = (self.hiddenLayer2.W ** 2).sum() + (self.hiddenLayer.W ** 2).sum() + (self.linRegressionLayer.W ** 2).sum()
        self.mean_squared_errors = self.linRegressionLayer.mean_squared_errors
        self.y_pred = self.linRegressionLayer.y_pred
        # the parameters of the model are the parameters of the layers it is made out of #
        # (remember to add more hidden layers if used) #
        self.params = self.hiddenLayer.params + self.linRegressionLayer.params
        # keep track of model input #
        self.input = input


class MLPLogisticRegression(object):
    """ Multi-layer perceptron consisting of a hidden layer and a fully connected
        logistic regression layer.
    """
    def __init__(self, rng, input, n_in, n_hidden, n_out):
        # one hidden layer with tanh activation, connected to the final linear regression layer #
        self.hiddenLayer = HiddenLayer(rng=rng, input=input, n_in=n_in, n_out=n_hidden, activation=T.tanh)
        # the logistic regression layer gets as input the hidden units #
        self.logRegressionLayer = LogisticRegression(input=self.hiddenLayer.output, n_in=n_hidden, n_out=n_out)
        # L1 norm regularization (remember to add more hidden layers if used) #
        self.L1 = (abs(self.hiddenLayer.W).sum() + abs(self.logRegressionLayer.W).sum())
        # L2 norm regularization (remember to add more hidden layers if used) #
        self.L2 = ((self.hiddenLayer.W ** 2).sum() + (self.logRegressionLayer.W ** 2).sum())
        # negative log likelihood of the MLP is given by the negative #
        # log likelihood of the output of the model, computed in the logistic regression layer #
        self.negative_log_likelihood = (self.logRegressionLayer.negative_log_likelihood)
        # same holds for the function computing the number of errors #
        self.errors = self.logRegressionLayer.errors
        self.p_y_given_x = self.logRegressionLayer.p_y_given_x
        # symbolic description of how to compute prediction as class whose probability is maximal #
        self.y_pred = self.logRegressionLayer.y_pred
        # the parameters of the model are the parameters of the layers it is made out of #
        # (remember to add more hidden layers if used) #
        self.params = self.hiddenLayer.params + self.logRegressionLayer.params
        # keep track of model input #
        self.input = input
