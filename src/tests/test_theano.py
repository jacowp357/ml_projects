import theano
import numpy as np
import theano.tensor as T

# test LR for multivariate output #
train_x = np.asarray([[0.0, 0.1, 0.2], [1.0, 1.1, 1.3], [2.0, 2.1, 1.9], [3.0, 3.1, 3.11], [4.0, 4.1, 13.9]])
train_y = np.asarray([[1.2, 2.2], [5.3, 5.1], [2.9, 2.1], [1.9, 9.1], [4.2, 3.2]])

learning_rate = 0.001
n_dims = train_x.shape[1]
n_steps = 100
n_out = 2

x = T.matrix(name='x')
w = theano.shared(value=np.zeros((n_dims, n_out), dtype=theano.config.floatX), name='w', borrow=True)
b = theano.shared(value=np.zeros((n_out, ), dtype=theano.config.floatX), name='b', borrow=True)
f = theano.function([x], T.dot(x, w) + b)

y = T.matrix(name='y')

loss = T.mean((T.dot(x, w) + b - y) ** 2)

g_loss = T.grad(loss, wrt=w)
b_loss = T.grad(loss, wrt=b)

train_model = theano.function(inputs=[], outputs=loss, updates=[(w, w - learning_rate * g_loss), (b, b - learning_rate * b_loss)], givens={x: train_x, y: train_y})

for i in range(n_steps):
    print("cost", train_model())

print(f(np.array([[4, 3, 3]])))
