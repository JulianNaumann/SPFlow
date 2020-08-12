import unittest

from spn.structure.Base import get_nodes_by_type
from spn.algorithms.EM import EM_optimization
from spn.algorithms.Inference import log_likelihood
from spn.algorithms.LearningWrappers import learn_parametric
from spn.structure.Base import Context, Sum, Product, Sum_sharedWeights
from spn.structure.leaves.parametric.Parametric import Categorical
from spn.structure.StatisticalTypes import MetaType
import numpy as np
from sklearn.datasets.samples_generator import make_blobs
from spn.io.Graphics import plot_spn
from spn.io.Text import spn_to_str_equation
from spn.structure.leaves.parametric.Parametric import Gaussian
from spn.algorithms.Validity import is_valid
from numpy.random.mtrand import RandomState
from spn.algorithms.Sampling import sample_instances

from spn.structure.Base import assign_ids, rebuild_scopes_bottom_up


# data1 = [1.0, 5.0] * 100
# data2 = [10.0, 12.0] * 100
# data = data1 + data2
# data = np.array(data).reshape((-1,2))
# data = data.astype(np.float32)

# g0 = Gaussian(mean=0, stdev=1, scope=0)
# g1 = Gaussian(mean=0, stdev=1, scope=1)
# p0 = Product(children=[g0,g1])
# p1 = Product(children=[g0,g1])
# spn1 = Sum(weights=[0.5,0.5], children=[p0,p1])

p0 = Product(children=[Categorical(p=[0.3, 0.7], scope=1), Categorical(p=[0.4, 0.6], scope=2)])
p1 = Product(children=[Categorical(p=[0.5, 0.5], scope=1), Categorical(p=[0.6, 0.4], scope=2)])
s1 = Sum_sharedWeights(weights=[0.3, 0.7], children=[p0, p1])
p2 = Product(children=[Categorical(p=[0.2, 0.8], scope=0), s1])
p3 = Product(children=[Categorical(p=[0.2, 0.8], scope=0), Categorical(p=[0.3, 0.7], scope=1)])
p4 = Product(children=[p3, Categorical(p=[0.4, 0.6], scope=2)])
spn = Sum_sharedWeights(sibling=s1, children=[p2, p4])

assign_ids(spn)
rebuild_scopes_bottom_up(spn)
valid, err = is_valid(spn)
print(f"Is valid: {valid}")
# plot_spn(spn, '/Users/julian/Downloads/basicspn-init.png')
# print(spn_to_str_equation(spn1))

data = sample_instances(spn, np.array([np.nan, np.nan, np.nan] * 1000).reshape(-1, 3), RandomState(123))

py_ll = np.sum(log_likelihood(spn, data))
print(f'Shared weights: {spn.weights}, ll: {py_ll}')

spn.weights = [0.1, 0.9]
s1.weights = spn.weights
# plot_spn(spn, '/Users/julian/Downloads/basicspn-afterChange.png')
py_ll = np.sum(log_likelihood(spn, data))
print(f'Shared weights: {spn.weights}, ll: {py_ll}')

EM_optimization(spn, data, iterations=1000)

py_ll_opt = np.sum(log_likelihood(spn, data))
print(f'Shared weights: {spn.weights}, ll: {py_ll}')