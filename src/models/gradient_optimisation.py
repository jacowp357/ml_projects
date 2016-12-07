# Projects: gradient_optimisation
#
# :Authors: Jaco du Toit <jacowp357@gmail.com>
# :Description: This is gradient decent optimisation methods.
#
#
import numpy as np
import theano
import theano.tensor as T
from collections import OrderedDict


def nesterov_accelerated_gradient(cost, params, learning_rate, momentum):
    # nesterov’s accelerated gradient #
    assert momentum < 1 and momentum >= 0.0

    grads = T.grad(cost, params)
    updates = OrderedDict()

    for param, grad in zip(params, grads):
        value = param.get_value(borrow=True)
        velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype), name="v", broadcastable=param.broadcastable)
        updates[param] = param - learning_rate * grad
        updates[velocity] = momentum * velocity + updates[param] - param
        updates[param] = momentum * updates[velocity] + updates[param]
    return updates


def momentum(cost, params, learning_rate, momentum):
    # momentum #
    assert momentum < 1 and momentum >= 0

    updates = []

    for param in params:
        # For each parameter, we'll create a previous_step shared variable.
        # This variable will keep track of the parameter's update step across iterations.
        # We initialize it to 0
        previous_step = theano.shared(param.get_value() * 0., broadcastable=param.broadcastable)
        # Each parameter is updated by taking a step in the direction of the gradient.
        # However, we also "mix in" the previous step according to the given momentum value.
        # Note that we don't need to derive backpropagation to compute updates - just use T.grad!
        step = momentum * previous_step - learning_rate * T.grad(cost, param)
        # Add an update to store the previous step value
        updates.append((previous_step, step))
        # Add an update to apply the gradient descent step to the parameter itself
        updates.append((param, param + step))
    return updates


def ada_delta(params, gparams):
    # http://deeplearning.net/tutorial/code/lstm.py #
    zipped_grads = [theano.shared(p.get_value() * np.asarray(0., dtype=theano.config.floatX)) for p in params]
    running_up2 = [theano.shared(p.get_value() * np.asarray(0., dtype=theano.config.floatX)) for p in params]
    running_grads2 = [theano.shared(p.get_value() * np.asarray(0., dtype=theano.config.floatX)) for p in params]

    zgup = [(zg, g) for zg, g in zip(zipped_grads, gparams)]
    rg2up = [(rg2, 0.95 * rg2 + 0.05 * (g ** 2)) for rg2, g in zip(running_grads2, gparams)]

    updir = [-T.sqrt(ru2 + 1e-6) / T.sqrt(rg2 + 1e-6) * zg for zg, ru2, rg2 in zip(zipped_grads, running_up2, running_grads2)]
    ru2up = [(ru2, 0.95 * ru2 + 0.05 * (ud ** 2)) for ru2, ud in zip(running_up2, updir)]
    param_up = [(p, p + ud) for p, ud in zip(params, updir)]

    updates = zgup + rg2up + ru2up + param_up

    return updates


def rms_prop(cost, params, lr=0.01, rho=0.9, epsilon=1e-6):
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates
