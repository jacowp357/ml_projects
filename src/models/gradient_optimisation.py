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

