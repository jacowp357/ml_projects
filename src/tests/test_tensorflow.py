import tensorflow as tf
import numpy as np

# test LR for multivariate output #
train_x = np.asarray([[0.0, 0.1, 0.2], [1.0, 1.1, 1.3], [2.0, 2.1, 1.9], [3.0, 3.1, 3.11], [4.0, 4.1, 13.9]])
train_y = np.asarray([[1.2, 2.2], [5.3, 5.1], [2.9, 2.1], [1.9, 9.1], [4.2, 3.2]])

learning_rate = 0.001
n_dims = train_x.shape[1]
n_steps = 100
n_out = 2

X = tf.placeholder(tf.float32, [None, n_dims])
Y = tf.placeholder(tf.float32, [None, n_out])
W = tf.Variable(tf.zeros([n_dims, n_out]))
b = tf.Variable(tf.zeros([n_out]))

init = tf.initialize_all_variables()

y_ = tf.matmul(X, W) + b
loss = tf.reduce_mean(tf.pow(y_ - Y, 2))
training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

sess = tf.Session()
sess.run(init)

for i in range(n_steps):
    sess.run(training_step, feed_dict={X: train_x, Y: train_y})
    print("cost", sess.run(loss, feed_dict={X: train_x, Y: train_y}))

print(sess.run(y_, feed_dict={X: np.array([[4, 3, 3]])}))
