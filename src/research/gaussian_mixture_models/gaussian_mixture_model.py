# Research: gaussian_mixture_model
#
# :Author: Jaco du Toit <jacowp357@gmail.com>
# :Date: 24/03/2017
# :Description: This code explores the GMM package in sklearn for 1D.
#
import matplotlib.pyplot as plt
import numpy as np
from sklearn import mixture
import matplotlib.mlab as mlab
plt.style.use('ggplot')


def gen_data(mu_list, sigma_list, samples):
    # mean and standard deviation for normal distribution #
    data = []
    for m, s in zip(mu_list, sigma_list):
        data.append(np.random.normal(m, s, samples))
    return np.concatenate(data).ravel()


if __name__ == '__main__':
    np.set_printoptions(suppress=True)
    mu = [-1, 2.5, 5.5]
    sigma = [0.89, 0.5, 0.5]
    samples = 1000

    data = gen_data(mu, sigma, samples)

    # plot actual data (vertical lines) #
    for i in data:
        plt.axvline(x=i, color='k', alpha=0.05)

    # fit a GMM with N components #
    N = 5
    gmix = mixture.GaussianMixture(n_components=N, init_params='kmeans', covariance_type='full', random_state=None, verbose=1, verbose_interval=1)
    gmix.fit(data.reshape(-1, 1))

    # Model information #
    print('EM algorithm converged: {} in {} EM iterations.'.format(gmix.converged_, gmix.n_iter_))
    print('Means: {}, covariances: {}'.format(gmix.means_.reshape(1, N)[0], gmix.covariances_.reshape(1, N)[0]))
    print('Precisions: {}'.format(gmix.precisions_.reshape(1, N)[0]))
    print('Log-likelihood of the best fit of EM: {}'.format(gmix.lower_bound_))
    print('The weights of each GMM component: {}'.format(gmix.weights_.reshape(1, N)[0]))

    # # generate and plot data from the GMM model #
    # samples = 1000
    # for i in gmix.sample(samples)[0]:
    #     plt.axvline(i[0], color='r', alpha=0.05)
    # plt.show()

    # plot sample distribution #
    x = np.linspace(-5, 10, 1000)
    for m, s in zip(mu, sigma):
        plt.plot(x, mlab.normpdf(x, m, s), '-.g')
    # plt.show()

    # predict probability and label of new data point #
    new_data = 5.5
    print('Predict the labels for the data sample: {}'.format(gmix.predict(new_data)[0]))
    print('Predict the posterior probability for the data sample: {}'.format(gmix.predict_proba(new_data)[0]))

    # compute the per-sample average log-likelihood and probability of the given data #
    print('Average log-likelihood for the data sample: {}'.format(gmix.score(new_data)))
    print('Probability for the data sample: {}'.format(np.exp(gmix.score(new_data))))

    # plot GMM pdf #
    # because gmm.score_samples gives log probability use exp instead #
    plt.plot(x, np.exp(gmix.score_samples(x.reshape(-1, 1))), linewidth=2, color='b')

    # plot Gaussian fits #
    for i in range(N):
        plt.plot(x, mlab.normpdf(x, gmix.means_[i][0], np.sqrt((gmix.covariances_[i][0][0]))), '--r')
    plt.show()
