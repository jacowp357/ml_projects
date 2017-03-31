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

class_map = {'unacc': 0, 'acc': 1, 'good': 2, 'vgood': 3}
buying_map = {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3}
maint_map = {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3}
doors_map = {'2': 0, '3': 1, '4': 2, '5more': 3}
persons_map = {'2': 0, '4': 1, 'more': 2}
lug_boot_map = {'small': 0, 'med': 1, 'big': 2}
safety_map = {'low': 0, 'med': 1, 'high': 2}

df = df.replace({"class": class_map})
df = df.replace({"buying": buying_map})
df = df.replace({"maint": maint_map})
df = df.replace({"doors": doors_map})
df = df.replace({"persons": persons_map})
df = df.replace({"lug_boot": lug_boot_map})
df = df.replace({"safety": safety_map})

df_train, df_test = train_test_split(df, test_size = 0.2, random_state=1)

######################
# Structure learning #
######################

# c = ConstraintBasedEstimator(df)
# model = c.estimate(significance_level=0.01)

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

###########################
# Construct Bayes network #
###########################

model1 = BayesianModel([('class', 'doors'),
                       ('class', 'safety'),
                       ('class', 'maint'),
                       ('class', 'buying'),
                       ('class', 'persons'), 
                       ('class', 'lug_boot')])

model2 = BayesianModel([('safety', 'class'),
                       ('maint', 'class'),
                       ('persons', 'class'),
                       ('buying', 'class')])

model1.fit(df_train, estimator_type=MaximumLikelihoodEstimator)
model2.fit(df_train, estimator_type=MaximumLikelihoodEstimator)

# print(model.get_cpds('class').values)
# print(model.active_trail_nodes('doors', observed='class'))

# can also use Belief propagation #
inference1 = VariableElimination(model1)
inference2 = VariableElimination(model2)

# phi_query = inference.query(variables=['class'])
# print(phi_query['class'])
# phi_query = inference.query(variables=['class'], evidence={'doors': 0})
# print(phi_query['class'])

#####################
# Predict test data #
#####################

predict_data = df_test.copy()
predict_data.drop(['class'], axis=1, inplace=True)
# y_pred = model.predict(predict_data)

pred_values = []
for index, data_point in predict_data.iterrows():
    prob_dist = inference1.query(variables=['class'], evidence=data_point.to_dict())['class'].values
    pred_values.append(prob_dist)

print('Classification accuracy: %0.3f' % accuracy_score(df_test['class'].values, [np.argmax(i) for i in pred_values]))
print(confusion_matrix(df_test['class'].values, [np.argmax(i) for i in pred_values]))
print(f1_score(df_test['class'].values, [np.argmax(i) for i in pred_values], average='micro'))

predict_data.drop(['doors', 'lug_boot'], axis=1, inplace=True)
# y_pred = model.predict(predict_data)

pred_values = []
for index, data_point in predict_data.iterrows():
    prob_dist = inference2.query(variables=['class'], evidence=data_point.to_dict())['class'].values
    pred_values.append(prob_dist)

print('Classification accuracy: %0.3f' % accuracy_score(df_test['class'].values, [np.argmax(i) for i in pred_values]))
print(confusion_matrix(df_test['class'].values, [np.argmax(i) for i in pred_values]))
print(f1_score(df_test['class'].values, [np.argmax(i) for i in pred_values], average='micro'))







# # y_test = label_binarize(df_test['class'].values, classes=[0, 1, 2, 3])
# # pred_values = np.array(pred_values)

# # fpr = dict()
# # tpr = dict()
# # roc_auc = dict()
# # for i in range(4):
# #     fpr[i], tpr[i], _ = roc_curve(y_test[:, i], pred_values[:, i])
# #     roc_auc[i] = auc(fpr[i], tpr[i])

# # # Compute micro-average ROC curve and ROC area
# # fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), pred_values.ravel())
# # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

# # for i in range(4):
# #     plt.plot(fpr[i], tpr[i], label='ROC curve %d: (area = %0.2f)' % (i, roc_auc[i]))
# # plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
# # plt.xlim([0.0, 1.0])
# # plt.ylim([0.0, 1.05])
# # plt.xlabel('False Positive Rate')
# # plt.ylabel('True Positive Rate')
# # plt.title('Receiver operating characteristic example')
# # plt.legend(loc="lower right")
# # plt.show()
































# print(model.edges())

#################################
# Incorporate uncertainty/noise #
#################################

# convert to Markov network #
model = model.to_markov_model()

# can not be a factor graph since not all random variables are connected to factors #
# model = model.to_factor_graph()
# print(model.check_model())

# add the new received variable and connect#
model.add_node('doors_received')
model.add_edge('doors', 'doors_received')

# binary symmetric channel noise #
# n = 4
# cpd = []
# for x1 in range(n):
#     for x2 in range(n):
#         if x1 == x2:
#             cpd.append(1)
#         else:
#             cpd.append(0)

# Gaussian noise #
std = 0.2
states = [mlab.normpdf(2, 2, std), mlab.normpdf(2, 3, std), mlab.normpdf(2, 4, std), mlab.normpdf(2, 5, std),
          mlab.normpdf(2, 3, std), mlab.normpdf(3, 3, std), mlab.normpdf(4, 3, std), mlab.normpdf(5, 3, std), 
          mlab.normpdf(2, 4, std), mlab.normpdf(3, 4, std), mlab.normpdf(4, 4, std), mlab.normpdf(5, 4, std), 
          mlab.normpdf(2, 5, std), mlab.normpdf(3, 5, std), mlab.normpdf(4, 5, std), mlab.normpdf(5, 5, std)]

# p(doors_received|doors) - doors was original information, but we received a noisy version of it!
factor = DiscreteFactor(['doors', 'doors_received'], cardinality=[4, 4], values=cpd)

print(factor)
# factor.reduce([('doors_received', 0)])
# factor.normalize()
# print(factor)

# add the factor to the network #
model.add_factors(factor)

# print(model.nodes())
# print(model.edges())

# can also use VE #
inference = BeliefPropagation(model)

phi_query = inference.query(variables=['class'])
print(phi_query['class'])
phi_query = inference.query(variables=['class'], evidence={'doors_received': 0})
print(phi_query['class'])






