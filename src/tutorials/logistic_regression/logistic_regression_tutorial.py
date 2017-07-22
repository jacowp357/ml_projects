# Tutorial: logistic_regression_tutorial
#
# :Author: Jaco du Toit <jacowp357@gmail.com>
# :Date: 21/07/2017
# :Description: This is an illustration of a simple binary
#               classifier in Tensorflow with decision boundary
#               visualisation.
#
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


if __name__ == "__main__":

    visualise = True

    ###################
    # create data set #
    ###################

    df1 = pd.DataFrame()
    df1['x1'], df1['x2'] = np.random.multivariate_normal([3, 5], [[0.8, -1],
                                                                  [-1, 1.5]], 5000).T
    df1['class'] = 0
    df2 = pd.DataFrame()
    df2['x1'], df2['x2'] = np.random.multivariate_normal([5, 9], [[0.6, 0],
                                                                  [0, 0.6]], 5000).T
    df2['class'] = 1
    df = df1.append(df2, ignore_index=True)

    # plt.scatter(df['x1'].values, df['x2'].values, c=df['class'].values, alpha=0.4)
    # plt.show()

    train_set = df.sample(frac=0.8, random_state=1024)
    val_set = df.drop(train_set.index).sample(frac=0.5, random_state=1024)
    test_set = df.drop(val_set.index.values.tolist() + train_set.index.values.tolist())

    train_x, train_y = train_set.ix[:, ['x1', 'x2']].as_matrix(), pd.get_dummies(train_set.ix[:, 'class'].values).as_matrix()
    val_x, val_y = val_set.ix[:, ['x1', 'x2']].as_matrix(), pd.get_dummies(val_set.ix[:, 'class'].values).as_matrix()
    test_x, test_y = test_set.ix[:, ['x1', 'x2']].as_matrix(), pd.get_dummies(test_set.ix[:, 'class'].values).as_matrix()

    x_dim, n_classes = train_x.shape[1], train_y.shape[1]

    ####################
    # define the model #
    ####################

    # None means that a dimension can be of any length #
    x = tf.placeholder(tf.float32, [None, x_dim], name='input')    # input placeholder holds data #
    W = tf.Variable(tf.zeros([x_dim, n_classes]), name='weights')  # model parameters/weights are variables as they will change during training #
    b = tf.Variable(tf.zeros([n_classes]), name='biases')          # model bias term #
    y = tf.placeholder(tf.float32, [None, n_classes])              # this is our output #

    #############################
    # define loss/cost function #
    #############################

    y_ = tf.nn.sigmoid(tf.add(tf.matmul(x, W), b))  # sigmoid output prediction/logits #
    cross_entropy_loss = tf.reduce_mean(tf.reduce_sum((-y * tf.log(y_)) - ((1 - y) * tf.log(1 - y_)), reduction_indices=[1]))

    # define backpropgagation algorithm/optimiser #
    learning_rate = 0.1
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy_loss)

    ###########################
    # define model evaluation #
    ###########################

    pred_class, act_class = tf.argmax(y, 1), tf.argmax(y_, 1)
    correct_prediction = tf.equal(pred_class, act_class)
    classification_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    #########################
    # define training cycle #
    #########################

    epochs = 100
    batch_size = 50
    train_examples = train_x.shape[0]
    iterations = train_examples // batch_size

    ######################
    # visualise training #
    ######################

    if visualise:
        f, axarr = plt.subplots(2, sharex=False)
        axarr[0].axis([train_x[:, 0].min(), train_x[:, 0].max(), train_x[:, 1].min(), train_x[:, 1].max()])
        plt.grid()
        plt.ion()
        plt.show()

    # initializing the variables #
    init = tf.global_variables_initializer()

    with tf.Session() as sess:

        sess.run(init)

        for epoch in range(epochs):

            idx = np.random.permutation(train_x.shape[0])
            X, Y = train_x[idx], train_y[idx]

            for i in range(iterations):
                batch_xs, batch_ys = X[i * batch_size:(i + 1) * batch_size], Y[i * batch_size:(i + 1) * batch_size]
                _, train_loss = sess.run([train_step, cross_entropy_loss], feed_dict={x: batch_xs, y: batch_ys})

            val_loss = sess.run(cross_entropy_loss, feed_dict={x: val_x, y: val_y})
            print('Epoch %d: training loss: %.5f, validation loss: %.5f' % (epoch, train_loss, val_loss))

            if visualise:
                axarr[0].cla()
                axarr[0].axis([train_x[:, 0].min(), train_x[:, 0].max(), train_x[:, 1].min(), train_x[:, 1].max()])
                axarr[0].scatter(train_x[:, 0], train_x[:, 1], c=train_y[:, 0], alpha=0.4)
                axarr[0].scatter(batch_xs[:, 0], batch_xs[:, 1], color='black', marker='^')
                axarr[0].set_xlabel('x1')
                axarr[0].set_ylabel('x2')
                axarr[0].set_title('Decision boundary')
                axarr[1].plot(epoch, train_loss, marker='o', linewidth=1.5, color='blue')
                axarr[1].plot(epoch, val_loss, marker='*', linewidth=1.5, color='red')
                axarr[1].set_xlabel('# Epochs')
                axarr[1].set_ylabel('Cross entropy')
                axarr[1].set_title('Training and validation loss')
                W_val, b_val = sess.run([W, b])
                W_val = W_val[:, 0]
                b_val = b_val[0]
                # w0*x1 + w1*x2 + b0 = 0 #
                x_sep = np.linspace(train_x[:, 0].min(), train_x[:, 0].max())
                y_sep = (-b_val - W_val[0] * x_sep) / W_val[1]
                axarr[0].plot(x_sep, y_sep, color='black', linewidth=2, alpha=0.7)
                plt.tight_layout()
                plt.pause(0.001)

        ###############################
        # test and evaluate the model #
        ###############################

        p_class, a_class, c_acc = sess.run([pred_class, act_class, classification_accuracy], feed_dict={x: test_x, y: test_y})
        print('\nClassification accuracy: {}'.format(c_acc))
        print('\n', classification_report(a_class, p_class, target_names=['class_0', 'class_1']))
        print('\n', confusion_matrix(a_class, p_class))

        ######################
        # predict new values #
        ######################

        print(sess.run(y_, {x: [[2, 4]]}))
        print(sess.run(y_, {x: [[8, 8]]}))
