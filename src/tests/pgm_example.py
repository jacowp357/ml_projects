# Research: pgm_example
#
# :Author: Jaco du Toit <jacowp357@gmail.com>
# :Date: 28/03/2017
# :Description: This code explores the libpgm and pgmpy packages in python for the
#               car evaluation data set in UCI: https://archive.ics.uci.edu/ml/datasets/Car+Evaluation.
#
import pandas as pd
import graphviz as gv
import numpy as np
from libpgm.nodedata import NodeData
from libpgm.graphskeleton import GraphSkeleton
from libpgm.discretebayesiannetwork import DiscreteBayesianNetwork
from libpgm.pgmlearner import PGMLearner
from random import randint, sample
from pgmpy.models import BayesianModel
from pgmpy.inference import VariableElimination, BeliefPropagation
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator, BaseEstimator
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.factors.discrete import TabularCPD
import matplotlib.mlab as mlab
from pgmpy.estimators import ConstraintBasedEstimator

###################
# Data processing #
###################

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

#######################
# Structure learning  #
#######################

c = ConstraintBasedEstimator(df)
model = c.estimate(significance_level=0.01)

#######################
# Graph visualisation #
#######################

g = gv.Digraph(format='pdf')
column_names = df.columns.values

var_dict = {}
for i in range(len(column_names)):
    var_dict.update({column_names[i]: i})
    g.node(column_names[i])

for item in model.edges():
    g.edge(item[0], item[1])
g.render(filename='pgm')

###########################
# Construct Bayes network #
###########################

model = BayesianModel([('doors', 'class'), ('safety', 'class'), ('maint', 'class'), ('buying', 'class'), ('persons', 'class'), ('lug_boot', 'class')])
model.fit(df[['safety', 'class', 'maint', 'buying', 'persons', 'lug_boot', 'doors']], estimator_type=MaximumLikelihoodEstimator)

# can also use BP #
inference = VariableElimination(model)

phi_query = inference.query(variables=['class'])
print(phi_query['class'])
phi_query = inference.query(variables=['class'], evidence={'doors': 0})
print(phi_query['class'])

# print(model.edges())

#################################
# Incorporate uncertainty/noise #
#################################

# convert to markov network #
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
std = 0.25
states = [mlab.normpdf(2, 2, std), mlab.normpdf(2, 3, std), mlab.normpdf(2, 4, std), mlab.normpdf(2, 5, std),
          mlab.normpdf(2, 3, std), mlab.normpdf(3, 3, std), mlab.normpdf(4, 3, std), mlab.normpdf(5, 3, std), 
          mlab.normpdf(2, 4, std), mlab.normpdf(3, 4, std), mlab.normpdf(4, 4, std), mlab.normpdf(5, 4, std), 
          mlab.normpdf(2, 5, std), mlab.normpdf(3, 5, std), mlab.normpdf(4, 5, std), mlab.normpdf(5, 5, std)]

# p(doors_received|doors) - doors was original information, but we received a noisy version of it!
factor = DiscreteFactor(['doors', 'doors_received'], cardinality=[4, 4], values=states)

# print(factor)
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

# # values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)), columns=['A', 'B', 'C', 'D', 'E'])
# # train_data = values[:800]
# # predict_data = values[800:]
# # model = BayesianModel([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])
# # model.fit(values)
# # predict_data = predict_data.copy()
# # predict_data.drop('E', axis=1, inplace=True)
# # y_pred = model.predict(predict_data)
# # y_pred
