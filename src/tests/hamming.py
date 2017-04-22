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
plt.style.use('ggplot')


def gen_data(N):
    """Codeword encoding (graph connections):

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
    return data


def shan_entropy(c):
    c_normalized = c / float(np.sum(c))
    c_normalized = c_normalized[np.nonzero(c_normalized)]
    H = -sum(c_normalized * np.log2(c_normalized))
    return H


def calc_MI(X, Y, bins):
    if (X.shape[1] == 1) & (Y.shape[1] == 1):
        c_XY = np.histogram2d(X[:, 0], Y[:, 0], bins)[0]
        c_X = np.histogram(X[:, 0], bins)[0]
        c_Y = np.histogram(Y[:, 0], bins)[0]
    else:
        c_XY = np.histogramdd(np.column_stack((X, Y)), bins=bins)[0]
        c_X = np.histogramdd(X, bins)[0]
        c_Y = np.histogramdd(Y, bins)[0]
    H_X = shan_entropy(c_X)
    H_Y = shan_entropy(c_Y)
    H_XY = shan_entropy(c_XY)
    MI = H_X + H_Y - H_XY
    return MI


if __name__ == '__main__':
    np.set_printoptions(linewidth=2000)
    df = pd.DataFrame(gen_data(10000), columns=['b1',
                                                'b2',
                                                'b3',
                                                'b4',
                                                'b5',
                                                'b6',
                                                'b7'])
    A = df.as_matrix()
    bins = 2
    # n = 7
    # matMI = np.zeros((7, 7))
    # pairwise MI #
    # for ix in np.arange(7):
    #     for jx in np.arange(ix + 1, 7):
    #         matMI[ix, jx] = calc_MI(A[:, ix], A[:, jx], bins)
    # print(matMI)

    # fig, ax = plt.subplots()
    # plt.pcolor(matMI, cmap=plt.cm.coolwarm, alpha=0.7)
    # ax.set_yticks(np.arange(matMI.shape[0]) + 0.5, minor=False)
    # ax.set_xticks(np.arange(matMI.shape[1]) + 0.5, minor=False)
    # ax.invert_yaxis()
    # ax.xaxis.tick_top()
    # labels = ['1', '2', '3', '4', '5', '6', '7']
    # ax.set_xticklabels(labels, minor=False)
    # ax.set_yticklabels(labels, minor=False)
    # plt.colorbar()
    # plt.show()

    print("Entropy b1, b2, b3:        H(X) = {}".format(shan_entropy(np.histogramdd(np.column_stack((A[:, 0], A[:, 1], A[:, 2])), bins=bins)[0])))
    print("Entropy b5:                H(Y) = {}".format(shan_entropy(np.histogram(A[:, 4], bins=bins)[0])))
    print("Entropy b1, b2, b3, b5: H(X, Y) = {}\n".format(shan_entropy(np.histogramdd(np.column_stack((A[:, 0], A[:, 1], A[:, 2], A[:, 4])), bins=bins)[0])))

    # calc MI for all possible pairs of nodes #
    nodes = {0: 'b1', 1: 'b2', 2: 'b3', 3: 'b4', 4: 'b5', 5: 'b6', 6: 'b7'}

    n = len(nodes)

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
            print(nodes[i], b, calc_MI(df[[nodes[i]]].as_matrix(), df[b].as_matrix(), bins))
