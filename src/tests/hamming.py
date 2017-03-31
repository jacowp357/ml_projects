# Test: hamming
#
# :Author: Jaco du Toit <jacowp357@gmail.com>
# :Date: 31/03/2017
# :Description: This code explores a deterministic hamming code example as a 
#               test for the structure learning algorithms.
#
import numpy as np
import pandas as pd
from pgmpy.estimators import ConstraintBasedEstimator, ExhaustiveSearch, HillClimbSearch, BicScore
import graphviz as gv


def gen_data(N):
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



if __name__ == '__main__':
	"""b5 = [b1, b2, b3]
       b6 = [b2, b3, b4]
       b7 = [b3, b4, b1]
	"""
	df = pd.DataFrame(gen_data(1600), columns=['b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7'])
	

	######################
	# Structure learning #
	######################

	est = ConstraintBasedEstimator(df)
	model = est.estimate(significance_level=1.004)
	print(model.edges())

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
	g.render(filename='hamming')
