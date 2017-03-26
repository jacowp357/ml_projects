# Research: data_transformations
#
# :Author: Jaco du Toit <jacowp357@gmail.com>
# :Date: 23/03/2017
# :Description: This code is used to discretise continuous data and
#               calculate a probability distribution over the composition.
#
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import mixture
import matplotlib.mlab as mlab
plt.style.use('ggplot')


if __name__ == '__main__':
    # remove scientific notation #
    np.set_printoptions(suppress=True)

    # read data #
    df = pd.read_csv('../data/kaggle/cs-training.csv', sep=',', engine='python', header='infer')

    # filter and remove nans from data #
    data = df[~df[['DebtRatio']].astype(float).isnull()][['DebtRatio']].values
    print('Max: {}, Min: {}'.format(np.max(data), np.min(data)))

    # # plot actual data (vertical lines) #
    # for i in data:
    #     plt.axvline(x=i, color='k', alpha=0.02)

    # fit a Gaussian Mixture Model with k components #
    k = 6
    gmix = mixture.GaussianMixture(n_components=k, init_params='kmeans', covariance_type='full', random_state=None, verbose=1, verbose_interval=1)
    gmix.fit(data.reshape(-1, 1))

    # print model information #
    print('EM algorithm converged: {} in {} EM iterations.'.format(gmix.converged_, gmix.n_iter_))
    print('Means: {}, covariances: {}'.format(gmix.means_.reshape(1, k)[0], gmix.covariances_.reshape(1, k)[0]))
    print('Precisions: {}'.format(gmix.precisions_.reshape(1, k)[0]))
    print('Log-likelihood of the best fit of EM: {}'.format(gmix.lower_bound_))
    print('The weights of each GMM component: {}'.format(gmix.weights_.reshape(1, k)[0]))

    # print(gmix.predict_proba(np.array([0]).reshape(-1, 1)))
    # print(np.exp(gmix.score(np.array([0]).reshape(-1, 1))))

    # plot Gaussian fits #
    x = np.linspace(-50000, 330000, 3000)
    for i in range(k):
        plt.plot(x, mlab.normpdf(x, gmix.means_[i][0], np.sqrt((gmix.covariances_[i][0][0]))), label='GMM component_{}'.format(i + 1))

    # plot GMM pdf #
    # because gmm.score_samples gives log probability use exp instead #
    plt.plot(x, np.exp(gmix.score_samples(x.reshape(-1, 1))), linewidth=1.5, color='#2370ed', label='GMM model pdf')
    plt.legend()
    plt.show()
