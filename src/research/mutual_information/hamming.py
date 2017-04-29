# Test: hamming
#
# :Author: Jaco du Toit <jacowp357@gmail.com>
# :Date: 04/04/2017
# :Description: This code explores a deterministic Hamming (7, 4) code example
#               as a test for mutual information between all codeword bits.
#
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import itertools
from itertools import combinations
import matplotlib.mlab as mlab
plt.style.use('ggplot')


def gen_data(N, noisy=False, std=0.01):
    """
    Codeword encoding (graph connections):

        b5 = [b1, b2, b3]
        b6 = [b2, b3, b4]
        b7 = [b3, b4, b1]

        H = [[1 1 1 0 1 0 0],
             [0 1 1 1 0 1 0],
             [1 0 1 1 0 0 1]]
    """
    data = []
    for i in range(N):
        m = np.random.randint(0, 2, 4)
        if ((m[0] + m[1] + m[2] - 3) % 2) == 1:
            # bits are uneven #
            b5 = 1
        else:
            # bits are even #
            b5 = 0
        if ((m[1] + m[2] + m[3] - 3) % 2) == 1:
            b6 = 1
        else:
            b6 = 0
        if ((m[2] + m[3] + m[0] - 3) % 2) == 1:
            b7 = 1
        else:
            b7 = 0
        data.append(list(m) + [b5, b6, b7])

    if noisy:
        data_noisy = []
        for i in data:
            b_list = []
            for j in i:
                if j == 0:
                    b_list.append(np.random.normal(0, std, 1)[0])
                else:
                    b_list.append(np.random.normal(1, std, 1)[0])
            data_noisy.append(b_list)
        return data_noisy
    else:
        return data


def shan_entropy(c):
    """
    Shannon entropy is a way to estimate the average minimum number of bits needed
    to encode a probability distribution (i.e., the uncertainty of a random variable).
    It provides a lower bound for the compression that can be achieved by the data
    representation compression step.
    """
    c_normalised = c / float(np.sum(c))
    c_normalised = c_normalised[np.nonzero(c_normalised)]
    # we use base 2 to interpret units as Shannon bits #
    H = -sum(c_normalised * np.log2(c_normalised))
    return H


def calc_MI(X, Y, bins):
    """
    Mutual information (MI) of two random variables is a measure of dependence between
    two variables. It quantifies the "amount of information" (in units such as bits)
    obtained from one random variable, through the other random variable. It determines
    how similar the joint distribution p(X, Y) is to the product of its marginal
    distributions p(X)p(Y).
    """
    # compute the multidimensional histogram of the data #
    c_XY = np.histogramdd(np.column_stack((X, Y)), bins=bins)[0]
    c_X = np.histogramdd(X, bins)[0]
    c_Y = np.histogramdd(Y, bins)[0]
    # average number of bits needed for X #
    H_X = shan_entropy(c_X)
    # average number of bits needed for Y #
    H_Y = shan_entropy(c_Y)
    # average number of bits needed for X, Y #
    H_XY = shan_entropy(c_XY)
    # if average number of bits needed for H(X, Y) < H(X) + H(Y) then we have MI > 0 #
    MI = H_X + H_Y - H_XY
    return MI


if __name__ == '__main__':
    """
    Our goal is to understand whether x is independent of y. One way to determine this
    is to use the empirical mutual information I(x;y). If I(x;y) = 0, then x and y are
    independent. We can use the local empirical dependencies (known as the PC algorithm)
    to learn/build a network structure of interacting random variables.
    """
    np.set_printoptions(linewidth=np.nan, suppress=True)
    pd.set_option('display.max_rows', 10000)
    df = pd.DataFrame(gen_data(5000, noisy=True, std=0.01), columns=['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7'])
    A = df.as_matrix()

    ###############
    # pairwise MI #
    ###############

    bins = 10
    n = 7
    matMI = np.zeros((7, 7))

    for ix in np.arange(7):
        for jx in np.arange(ix + 1, 7):
            # we can calculate pairwise MI between each pair of nodes #
            matMI[ix, jx] = calc_MI(A[:, ix], A[:, jx], bins)
    print("Pairwise MI:")
    print(matMI)

    # plot pairwise MI #
    fig, ax = plt.subplots()
    plt.pcolor(matMI, cmap=plt.cm.coolwarm, alpha=0.7)
    ax.set_yticks(np.arange(matMI.shape[0]) + 0.5, minor=False)
    ax.set_xticks(np.arange(matMI.shape[1]) + 0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()
    labels = ['1', '2', '3', '4', '5', '6', '7']
    ax.set_xticklabels(labels, minor=False)
    ax.set_yticklabels(labels, minor=False)
    plt.colorbar()
    plt.show()

    print("\nTest b5 ⊥ [b1, b2, b3]:")
    print("Entropy b1, b2, b3:        H(X) = {}".format(shan_entropy(np.histogramdd(np.column_stack((A[:, 0], A[:, 1], A[:, 2])), bins=bins)[0])))
    print("Entropy b5:                H(Y) = {}".format(shan_entropy(np.histogramdd(A[:, 4], bins=bins)[0])))
    print("Entropy b1, b2, b3, b5: H(X, Y) = {}\n".format(shan_entropy(np.histogramdd(np.column_stack((A[:, 0], A[:, 1], A[:, 2], A[:, 4])), bins=bins)[0])))

    # calc MI for all possible pairs of nodes #
    nodes = {0: 'b1', 1: 'b2', 2: 'b3', 3: 'b4', 4: 'b5', 5: 'b6', 6: 'b7'}

    n = len(nodes)
    mi = []

    for i in range(n):
        temp_nodes = nodes.copy()
        # remove node of interest #
        del temp_nodes[i]
        # get remaining column names of interest #
        col_names = list(temp_nodes.values())
        m = n - 1
        # create temp truth table to get all combinations #
        df_temp = pd.DataFrame(list(itertools.product([False, True], repeat=m)), columns=col_names)
        # drop first row with all false cases #
        df_temp = df_temp.drop(df_temp.index[0])
        # for each row in temp_df calculate the MI #
        for j in range(len(df_temp)):
            # get column names that are true #
            b = list(df_temp.ix[j + 1, :][df_temp.ix[j + 1, :]].index.values)
            # print(nodes[i], b, calc_MI(df[[nodes[i]]].as_matrix(), df[b].as_matrix(), bins))
            mi.append(("{} ⊥ {}|".format(nodes[i], str(b)), calc_MI(df[[nodes[i]]].as_matrix(), df[b].as_matrix(), bins)))

    df_results = pd.DataFrame(mi, columns=['vars', 'mi'])
    print(df_results.sort_values(by='mi', axis=0, ascending=False))

    ##################################
    # Conditional mutual information #
    ##################################

    # I(X;Y|Z) = I(X;Y,Z) − I(X;Z) #
    # or equivalently: I(X;Y|Z) = H(X,Z) + H(Y,Z) − H(X,Y,Z) − H(Z) #
    a = shan_entropy(np.histogramdd(df[['b1', 'b5']].as_matrix(), bins=bins)[0])
    b = shan_entropy(np.histogramdd(df[['b2', 'b5']].as_matrix(), bins=bins)[0])
    c = shan_entropy(np.histogramdd(df[['b1', 'b2', 'b5']].as_matrix(), bins=bins)[0])
    d = shan_entropy(np.histogramdd(df[['b5']].as_matrix(), bins=bins)[0])

    print("\nTest b1 ⊥ b2|b5:")
    print("Entropy H(b1, b5) = {}".format(a))
    print("Entropy H(b2, b5) = {}".format(b))
    print("Entropy H(b1, b2, b5) = {}".format(c))
    print("Entropy H(b5) = {}".format(d))
    print("H(b1,b5) + H(b2,b5) − H(b1,b2,b5) − H(b5) = {} + {} - {} - {} = {}\n".format(a, b, c, d, a + b - c - d))

    e = calc_MI(df[['b1']].as_matrix(), df[['b2', 'b5']].as_matrix(), bins)
    f = calc_MI(df[['b1']].as_matrix(), df[['b5']].as_matrix(), bins)
    print("I(b1;b2,b5) - I(b1;b5) = {} - {} = {}".format(e, f, e - f))

    nodes = {0: 'b1', 1: 'b2', 2: 'b3', 3: 'b4', 4: 'b5', 5: 'b6', 6: 'b7'}

    n = len(nodes)
    mi = []

    for i in range(n):
        temp_nodes = nodes.copy()
        # remove node of interest #
        del temp_nodes[i]
        # get remaining column names of interest #
        col_names = list(temp_nodes.values())
        edges = combinations(col_names, 2)
        for j in list(edges):
            e = calc_MI(df[[nodes[i]]].as_matrix(), df[list(j)].as_matrix(), bins)
            f = calc_MI(df[[nodes[i]]].as_matrix(), df[[j[1]]].as_matrix(), bins)
            mi.append(("{} ⊥ {}|{}".format(j[0], j[1], nodes[i]), e - f))

    df_results = pd.DataFrame(mi, columns=['vars', 'mi'])
    print(df_results.sort_values(by='mi', axis=0, ascending=False))
