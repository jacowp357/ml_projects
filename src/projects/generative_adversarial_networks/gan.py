# Projects: gan
#
# :Author: Jaco du Toit <jacowp357@gmail.com>
# :Date: 28/04/2017
# :Description: This code explores a generative adversarial network
#               on facial images.
#
import tensorflow as tf
import os
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from skimage.io import imsave
from pandas.io.parsers import read_csv
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd


def generator(z_prior, z_size, img_size):
    """The generator takes z-dimensional vector and returns img-dimensional vector,
       which is the MNIST image size (28 x 28) in this case. The prior for the generator is z.
       It learns a mapping between the prior space to the actual data.
    """
    h1_size = 100
    h2_size = 100
    w1 = tf.Variable(tf.truncated_normal([z_size, h1_size], stddev=0.1), name="gen_w1", dtype=tf.float32)
    b1 = tf.Variable(tf.zeros(shape=[h1_size]), name="gen_b1", dtype=tf.float32)
    h1 = tf.nn.relu(tf.matmul(z_prior, w1) + b1)
    w2 = tf.Variable(tf.truncated_normal([h1_size, h2_size], stddev=0.1), name="gen_w2", dtype=tf.float32)
    b2 = tf.Variable(tf.zeros(shape=[h2_size]), name="gen_b2", dtype=tf.float32)
    h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
    w3 = tf.Variable(tf.truncated_normal([h2_size, img_size], stddev=0.1), name="gen_w3", dtype=tf.float32)
    b3 = tf.Variable(tf.zeros([img_size]), name="gen_b3", dtype=tf.float32)
    h3 = tf.matmul(h2, w3) + b3
    x_generate = tf.nn.tanh(h3)
    g_params = [w1, b1, w2, b2, w3, b3]
    return x_generate, g_params


def discriminator(x_data, x_generated, img_size, keep_prob):
    """The discriminator takes the MNIST images and returns a scalar representing
       a probability of the real MNIST image.
    """
    h1_size = 80
    h2_size = 60
    x_in = tf.concat([x_data, x_generated], 0)
    w1 = tf.Variable(tf.truncated_normal([img_size, h2_size], stddev=0.1), name="disc_w1", dtype=tf.float32)
    b1 = tf.Variable(tf.zeros(shape=[h2_size]), name="disc_b1", dtype=tf.float32)
    h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(x_in, w1) + b1), keep_prob)
    w2 = tf.Variable(tf.truncated_normal([h2_size, h1_size], stddev=0.1), name='disc_w2')
    b2 = tf.Variable(tf.zeros(shape=[h1_size]), name='disc_b2')
    h2 = tf.nn.dropout(tf.nn.relu(tf.matmul(h1, w2) + b2), keep_prob)
    w3 = tf.Variable(tf.truncated_normal([h1_size, 1], stddev=0.1), name="disc_w3", dtype=tf.float32)
    b3 = tf.Variable(tf.zeros([1]), name="disc_b3", dtype=tf.float32)
    h3 = tf.matmul(h2, w3) + b3
    y_data = tf.nn.sigmoid(tf.slice(h3, [0, 0], [batch_size, -1], name=None))
    y_generated = tf.nn.sigmoid(tf.slice(h3, [batch_size, 0], [-1, -1], name=None))
    d_params = [w1, b1, w2, b2, w3, b3]
    return y_data, y_generated, d_params


def show_result(batch_res, fname, grid_size=(4, 4), grid_pad=2):
    batch_res = 0.5 * batch_res.reshape((batch_res.shape[0], img_height, img_width)) + 0.5
    img_h, img_w = batch_res.shape[1], batch_res.shape[2]
    grid_h = img_h * grid_size[0] + grid_pad * (grid_size[0] - 1)
    grid_w = img_w * grid_size[1] + grid_pad * (grid_size[1] - 1)
    img_grid = np.zeros((grid_h, grid_w), dtype=np.uint8)
    for i, res in enumerate(batch_res):
        if i >= grid_size[0] * grid_size[1]:
            break
        img = (res) * 255
        img = img.astype(np.uint8)
        row = (i // grid_size[0]) * (img_h + grid_pad)
        col = (i % grid_size[1]) * (img_w + grid_pad)
        img_grid[row:row + img_h, col:col + img_w] = img
    imsave(fname, img_grid)
    # cdict = {'red': [(0.0, 1.0, 1.0), (0.25, 1.0, 1.0), (0.5, 0.0, 0.0), (1.0, 0.0, 0.0)],
    #          'green': [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
    #          'blue': [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0), (0.75, 1.0, 1.0), (1.0, 1.0, 1.0)]}
    # redblue = LinearSegmentedColormap('red_black_blue', cdict, 256)
    # fig = plt.imshow(img_grid, cmap=redblue, clim=(-1.0, 1.0))
    # fig = plt.imshow(img_grid, cmap='gray')
    # fig = fig.get_figure()
    # fig.savefig(fname)
    # plt.show()


def train(batch_size, img_size, z_size):
    """Adversarial process for training.
    """
    # mnist = input_data.read_data_sets('data', one_hot=True)
    df = read_csv('data/training.csv')
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep=' '))
    df = pd.DataFrame((np.vstack(df['Image'].values) / 255.).astype(np.float64))
    df = df.as_matrix()

    # n_samples = mnist.train.images.shape[0]
    n_samples = df.shape[0]
    x_data = tf.placeholder(tf.float32, [batch_size, img_size], name="x_data")
    z_prior = tf.placeholder(tf.float32, [batch_size, z_size], name="z_prior")
    keep_prob = tf.placeholder(tf.float32, name="keep_prob")
    global_step = tf.Variable(0, name="global_step", trainable=False)

    x_generated, g_params = generator(z_prior, z_size, img_size)
    y_data, y_generated, d_params = discriminator(x_data, x_generated, img_size, keep_prob)

    # It’s better to maximize log(y_generated) #
    # instead of minimizing (1 - log(y_generated)) #
    d_loss = -(tf.log(y_data) + tf.log(1. - y_generated))
    g_loss = -tf.log(y_generated)

    d_trainer = tf.train.AdamOptimizer(0.0001).minimize(d_loss, var_list=d_params)
    g_trainer = tf.train.AdamOptimizer(0.0001).minimize(g_loss, var_list=g_params)

    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    sess = tf.Session()
    sess.run(init)

    z_sample_val = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)
    losses = []
    for i in range(max_epoch):
        for j in range(int(n_samples / batch_size)):
            # sample mini-batch from data generating distribution #
            # x_value, _ = mnist.train.next_batch(batch_size)
            x_value = df[(j * batch_size):((j * batch_size) + batch_size), :]
            x_value = 2 * x_value.astype(np.float32) - 1
            # sample mini-batch noise samples from noise prior z #
            z_value = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)
            # update discriminator by ascending its stochastic gradient #
            _, d_losses = sess.run([d_trainer, d_loss], feed_dict={x_data: x_value, z_prior: z_value, keep_prob: np.sum(0.6).astype(np.float32)})
            # generator training intervals #
            if j % 1 == 0:
                # sample mini-batch noise samples from noise prior z #
                # z_value = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)
                # update the generator by descending its stochastic gradient #
                _, g_losses = sess.run([g_trainer, g_loss], feed_dict={x_data: x_value, z_prior: z_value, keep_prob: np.sum(0.7).astype(np.float32)})
        losses.append([g_losses.mean(), d_losses.mean()])
        print("Epoch: {}, d_loss: {}, g_loss: {}".format(i, np.mean(d_losses), np.mean(g_losses)))
        x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_sample_val})
        show_result(x_gen_val, "output/sample{0}.jpg".format(i))
        # z_random_sample_val = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)
        # x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_random_sample_val})
        # show_result(x_gen_val, "output/random_sample{0}.jpg".format(i))
        sess.run(tf.assign(global_step, i + 1))
        saver.save(sess, os.path.join("output", "model"), global_step=global_step)
    plt.plot(range(max_epoch), [i[0] for i in losses], label='gen')
    plt.plot(range(max_epoch), [i[1] for i in losses], label='disc')
    plt.legend()
    plt.show()

# def test():
#     z_prior = tf.placeholder(tf.float32, [batch_size, z_size], name="z_prior")
#     x_generated, _ = build_generator(z_prior)
#     chkpt_fname = tf.train.latest_checkpoint(output_path)
#     init = tf.global_variables_initializer()
#     sess = tf.Session()
#     saver = tf.train.Saver()
#     sess.run(init)
#     saver.restore(sess, chkpt_fname)
#     z_test_value = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)
#     x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_test_value})
#     show_result(x_gen_val, "output/test_result.jpg")


if __name__ == '__main__':
    # plt.imshow(df.ix[0, :].values.reshape(96, 96), cmap='gray')
    # plt.imshow(df.ix[0, :].values)
    # plt.show()
    img_height = 96
    img_width = 96
    img_size = img_height * img_width
    max_epoch = 500
    z_size = 100
    batch_size = 256
    train(batch_size, img_size, z_size)
