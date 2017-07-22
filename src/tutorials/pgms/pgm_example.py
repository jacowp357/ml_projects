# Tutorial: pgm_example
#
# :Author: Jaco du Toit <jacowp357@gmail.com>
# :Date: 28/03/2017
# :Description: This code explores the pgmpy package in Python with application to
#               the car evaluation data set available from UCI:
#               https://archive.ics.uci.edu/ml/datasets/Car+Evaluation.
#
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, accuracy_score, confusion_matrix, f1_score
import graphviz as gv
import numpy as np
from random import randint, sample
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination, BeliefPropagation
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator, BaseEstimator
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.factors.discrete import TabularCPD
import matplotlib.mlab as mlab
from pgmpy.estimators import ConstraintBasedEstimator
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)

###################
# Data processing #
###################

"""
Attribute Information:

    - Class:

        - unacc, acc, good, vgood

    - Attributes:

        - buying: vhigh, high, med, low.
        - maint: vhigh, high, med, low.
        - doors: 2, 3, 4, 5more.
        - persons: 2, 4, more.
        - lug_boot: small, med, big.
        - safety: low, med, high.
"""

df = pd.read_csv('data.csv', sep=',', engine='python', header='infer')

# mapping from labels to ordinals #
class_map = {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}
buying_map = {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3}
maint_map = {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3}
doors_map = {'2': 0, '3': 1, '4': 2, '5more': 3}
persons_map = {'2': 0, '4': 1, 'more': 2}
lug_boot_map = {'small': 0, 'med': 1, 'big': 2}
safety_map = {'low': 0, 'med': 1, 'high': 2}

df = df.replace({"class": class_map,
                 "buying": buying_map,
                 "maint": maint_map,
                 "doors": doors_map,
                 "persons": persons_map,
                 "lug_boot": lug_boot_map,
                 "safety": safety_map})

df_train, df_test = train_test_split(df, test_size=0.25, random_state=1)

######################
# Structure learning #
######################

# c = ConstraintBasedEstimator(df)
# model = c.estimate(significance_level=0.05)

#######################
# Graph visualisation #
#######################

# g = gv.Digraph(format='pdf')
# column_names = df.columns.values

# var_dict = {}
# for i in range(len(column_names)):
#     var_dict.update({column_names[i]: i})
#     g.node(column_names[i])

# for item in model.edges():
#     g.edge(item[0], item[1])
# g.render(filename='pgm')

########################################################################
# Construct Naive Bayes network (without help from structure learning) #
########################################################################

model = BayesianModel([('class', 'doors'),
                       ('class', 'safety'),
                       ('class', 'maint'),
                       ('class', 'buying'),
                       ('class', 'persons'),
                       ('class', 'lug_boot')])

model.fit(df_train, estimator=MaximumLikelihoodEstimator)

# print(model.get_cpds('class').values)
# print(model.active_trail_nodes('doors', observed='class'))

# can also use Belief propagation #
inference = BeliefPropagation(model)

print("Class variable prior:\n{}\n".format(inference.query(variables=['class'])['class']))
print("Class variable posterior after certain observation:\n{}\n".format(inference.query(variables=['class'], evidence={'doors': 0})['class']))

#####################
# Predict test data #
#####################

predict_data = df_test.copy()
predict_data.drop(['class'], axis=1, inplace=True)
# y_pred = model.predict(predict_data)

pred_values = []
for index, data_point in predict_data.iterrows():
    prob_dist = inference.query(variables=['class'], evidence=data_point.to_dict())['class'].values
    pred_values.append(prob_dist)

####################
# Model evaluation #
####################

print("Classification accuracy: {0:.3f}%".format(100 * accuracy_score(df_test['class'].values, [np.argmax(i) for i in pred_values])))
print("F1 score: {0:.3f}\n".format(f1_score(df_test['class'].values, [np.argmax(i) for i in pred_values], average='weighted')))
print("Confusion matrix:\n{}\n".format(confusion_matrix(df_test['class'].values, [np.argmax(i) for i in pred_values])))

y_test = label_binarize(df_test['class'].values, classes=[0, 1, 2, 3])
pred_values = np.array(pred_values)

fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = 4
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], pred_values[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average (flattened) ROC curve and ROC area #
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), pred_values.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure(0)
plt.plot(fpr["micro"], tpr["micro"], label="micro-average ROC curve (area = {0:0.3f})".format(roc_auc["micro"]))

# for i in range(n_classes):
#     plt.plot(fpr[i], tpr[i], label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")

###############################################
# Incorporate uncertainty to random variables #
###############################################

# convert to Markov network (undirected graph) #
model = model.to_markov_model()

# can not be a factor graph since not all random variables are connected to factors #
# model = model.to_factor_graph()
# print(model.check_model())

# add the new received variable and connect #
model.add_node('doors_received')
model.add_node('persons_received')
model.add_node('lug_boot_received')
model.add_node('maint_received')
model.add_node('safety_received')
model.add_node('buying_received')
model.add_edges_from([('persons', 'persons_received'),
                      ('doors', 'doors_received'),
                      ('buying', 'buying_received'),
                      ('safety', 'safety_received'),
                      ('maint', 'maint_received'),
                      ('lug_boot', 'lug_boot_received')])

# binary symmetric channel noise #
n = 4
cpd1 = []
for x1 in range(n):
    for x2 in range(n):
        if x1 == x2:
            cpd1.append(1)
        else:
            cpd1.append(0)

n = 3
cpd2 = []
for x1 in range(n):
    for x2 in range(n):
        if x1 == x2:
            cpd2.append(1)
        else:
            cpd2.append(0)

# std = 0.25
# cpd1 = [mlab.normpdf(0, 0, std), mlab.normpdf(0, 1, std), mlab.normpdf(0, 2, std), mlab.normpdf(0, 3, std),
#         mlab.normpdf(1, 0, std), mlab.normpdf(1, 1, std), mlab.normpdf(1, 2, std), mlab.normpdf(1, 3, std),
#         mlab.normpdf(2, 0, std), mlab.normpdf(2, 1, std), mlab.normpdf(2, 2, std), mlab.normpdf(2, 3, std),
#         mlab.normpdf(3, 0, std), mlab.normpdf(3, 1, std), mlab.normpdf(3, 2, std), mlab.normpdf(3, 3, std)]

# cpd2 = [mlab.normpdf(0, 0, std), mlab.normpdf(0, 1, std), mlab.normpdf(0, 2, std),
#         mlab.normpdf(1, 0, std), mlab.normpdf(1, 1, std), mlab.normpdf(1, 2, std),
#         mlab.normpdf(2, 0, std), mlab.normpdf(2, 1, std), mlab.normpdf(2, 2, std)]

# p(doors_received|doors) - doors was original information, but we received a noisy version of it! #
# not necessary to normalise cpd for Markov network #
factor1 = DiscreteFactor(['buying', 'buying_received'], cardinality=[4, 4], values=cpd1)
factor2 = DiscreteFactor(['maint', 'maint_received'], cardinality=[4, 4], values=cpd1)
factor3 = DiscreteFactor(['doors', 'doors_received'], cardinality=[4, 4], values=cpd1)
factor4 = DiscreteFactor(['persons', 'persons_received'], cardinality=[3, 3], values=cpd2)
factor5 = DiscreteFactor(['safety', 'safety_received'], cardinality=[3, 3], values=cpd2)
factor6 = DiscreteFactor(['lug_boot', 'lug_boot_received'], cardinality=[3, 3], values=cpd2)

# for any given x (where x can also be continuous) this reduces to a table with k scaled probabilities #
# print(factor3)
# factor3.reduce([('doors_received', 1)])
# # factor.normalize()
# print(factor3)

# add the factor to the network #
model.add_factors(factor1)
model.add_factors(factor2)
model.add_factors(factor3)
model.add_factors(factor4)
model.add_factors(factor5)
model.add_factors(factor6)

# print(model.nodes())
# print(model.edges())

# can also use VE #
inference = BeliefPropagation(model)

print("Class variable prior:\n{}\n".format(inference.query(variables=['class'])['class']))
print("Class variable posterior after noisy observation:\n{}\n".format(inference.query(variables=['class'], evidence={'doors_received': 0})['class']))

#####################
# Predict test data #
#####################

predict_data = df_test.copy()
predict_data = predict_data.rename(columns={'doors': 'doors_received',
                                            'maint': 'maint_received',
                                            'safety': 'safety_received',
                                            'lug_boot': 'lug_boot_received',
                                            'persons': 'persons_received',
                                            'buying': 'buying_received'})

predict_data.drop(['class'], axis=1, inplace=True)
# y_pred = model.predict(predict_data)

pred_values = []
for index, data_point in predict_data.iterrows():
    prob_dist = inference.query(variables=['class'], evidence=data_point.to_dict())['class'].values
    pred_values.append(prob_dist)

#######################################
# Model (with uncertainty) evaluation #
#######################################

print("Classification accuracy: {0:.3f}%".format(100 * accuracy_score(df_test['class'].values, [np.argmax(i) for i in pred_values])))
print("F1 score: {0:.3f}\n".format(f1_score(df_test['class'].values, [np.argmax(i) for i in pred_values], average='weighted')))
print("Confusion matrix:\n{}\n".format(confusion_matrix(df_test['class'].values, [np.argmax(i) for i in pred_values])))

y_test = label_binarize(df_test['class'].values, classes=[0, 1, 2, 3])
pred_values = np.array(pred_values)

fpr = dict()
tpr = dict()
roc_auc = dict()
n_classes = 4
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], pred_values[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average (flattened) ROC curve and ROC area #
fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), pred_values.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure(1)
plt.plot(fpr["micro"], tpr["micro"], label="micro-average ROC curve (area = {0:0.3f})".format(roc_auc["micro"]))

# for i in range(n_classes):
#     plt.plot(fpr[i], tpr[i], label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC curve')
plt.legend(loc="lower right")
plt.show()

# A continuous Gaussian pdf can now describe the correlation with our input variables #
# when we have a hybrid factor involving both discrete and continuous random variables #
# and we observe all the continuous ones we are left with a factor consisting purely of discrete probabilities #
