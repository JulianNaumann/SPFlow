import unittest

from spn.structure.Base import get_nodes_by_type
from spn.algorithms.EM import EM_optimization
from spn.algorithms.Inference import log_likelihood
from spn.algorithms.LearningWrappers import learn_parametric
from spn.structure.Base import Context, Sum, Product, Sum_sharedWeights
from spn.structure.StatisticalTypes import MetaType
import numpy as np
# from sklearn.datasets.samples_generator import make_blobs
from spn.io.Graphics import plot_spn
from spn.io.Text import spn_to_str_equation
from spn.structure.leaves.parametric.Parametric import Categorical, Gaussian, Bernoulli
from spn.algorithms.Validity import is_valid
from numpy.random.mtrand import RandomState
from spn.algorithms.Sampling import sample_instances

import matplotlib.pyplot as plt
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


x = Bernoulli(p=0.9, scope=0)
y = Bernoulli(p=0.3, scope=1)
a1 = Bernoulli(p=0.5, scope=2)
a2 = Bernoulli(p=0.01, scope=2)
b1 = Bernoulli(p=0.09, scope=3)
b2 = Bernoulli(p=0.03, scope=3)

s0 = Sum_sharedWeights(weights=[0.34,0.66], children=[a1,a2])
s1 = Sum_sharedWeights(sibling=s0, children=[b1,b2])
# s1 = Sum_sharedWeights(weights=[0.1,0.9], children=[b1,b2])
spn = Product(children=[s0,s1,x,y])

assign_ids(spn)
rebuild_scopes_bottom_up(spn)
valid, err = is_valid(spn)
print(f"Model is valid: {valid}\n")
# plot_spn(spn, '/Users/julian/Downloads/basic.png')


# ## sample from model
# data = sample_instances(spn, np.array([np.nan, np.nan, np.nan] * 1000).reshape(-1, 3), RandomState(123))

# ## create artificial data
np.random.seed(1)
dataX = np.random.binomial(size=3000, n=1, p=0.9)
dataY = np.random.binomial(size=3000, n=1, p=0.3)
dataA1 = np.random.binomial(size=1000, n=1, p=0.5)
dataA2 = np.random.binomial(size=2000, n=1, p=0.01)
dataA = np.concatenate((dataA1, dataA2))
dataB1 = np.random.binomial(size=1000, n=1, p=0.09)
dataB2 = np.random.binomial(size=2000, n=1, p=0.03)
dataB = np.concatenate((dataB1, dataB2))
data = np.stack((dataX, dataY, dataA, dataB), axis = 1)

# ## Sample testing
# print(f'{"Sampled from model":60}', end='')
# sampled_data = sample_instances(spn, np.array([np.nan] * 4 * 300000).reshape(-1,4), RandomState(1))
# py_ll = np.sum(log_likelihood(spn, sampled_data))
# print(f'{py_ll,s0.weights, s1.weights}')

# EM testing
print(f'{"Eval of artifical data":60}', end='')
py_ll = np.mean(log_likelihood(spn, data))
print(f'{py_ll:.8f},[{s0.weights[0]:.4f},{s0.weights[1]:.4f}], [{s1.weights[0]:.4f},{s1.weights[1]:.4f}]')

# print(f'{"eval of artificial data, after changed weights":60}', end='')
# s0.weights[0] = s1.weights[0] = 0.1
# s0.weights[1] = s1.weights[1] = 0.9
# s0.weights[0] = s0.weights[1] = 0.5
# py_ll = np.mean(log_likelihood(spn, data))
# print(f'{py_ll:.8f},[{s0.weights[0]:.4f},{# s0.weights[1]:.4f}], [{s1.weights[0]:.4f},{s1.weights[1]:.4f}]')

# print(f'{"Eval of artifical data, after EM":60}', end='')
# EM_optimization(spn, data, iterations=1000)
# py_ll = np.mean(log_likelihood(spn, data))
# print(f'{py_ll:.8f},[{s0.weights[0]:.4f},{s0.weights[1]:.4f}], [{s1.weights[0]:.4f},{s1.weights[1]:.4f}]')

# print(f'{"Eval of artifical data, after EM":60}', end='')
# EM_optimization(spn, data, iterations=100)
# py_ll = np.mean(log_likelihood(spn, data))
# print(f'{py_ll:.8f},[{s0.weights[0]:.4f},{s0.weights[1]:.4f}], [{s1.weights[0]:.4f},{s1.weights[1]:.4f}]')

## plot range(100) EM iteration
# print(f'{"Eval of artifical data":60}', end='')
# lls = []
# for i in range(100):
#     ll = np.mean(log_likelihood(spn, data))
#     lls.append(ll)
#     EM_optimization(spn, data, iterations=1)
#
# plt.plot(range(100), lls)
# plt.show()
# py_ll = np.mean(log_likelihood(spn, data))
# print(f'{py_ll:.8f},[{s0.weights[0]:.4f},{s0.weights[1]:.4f}], [{s1.weights[0]:.4f},{s1.weights[1]:.4f}]')


## set different starting point
print("Setting weights to diff starting point.")
s0.weights[0] = s1.weights[0] = 0.1
s0.weights[1] = s1.weights[1] = 0.9


print(f'{"Eval of artifical data":60}', end='')
lls = []
weights = []
for i in range(100):
    ll = np.mean(log_likelihood(spn, data))
    lls.append(ll)
    weights.append(s0.weights[0])
    EM_optimization(spn, data, iterations=1)


fig, axs = plt.subplots(2,1)
fig.suptitle("Starting weights [0.1, 0.9]")

axs[0].plot(range(100), lls, label="LL")
axs[0].legend()
axs[0].set_ylabel("LL")
axs[0].set_xlabel("iteration")
max_y = max(lls)

max_x = lls.index(max_y)
axs[0].annotate(f'Max @ {max_y:.4f}', xy=(max_x,max_y),arrowprops=dict(facecolor='black', shrink=0.01))
axs[0].annotate(f"{py_ll:.4f} @ weights [.34,.66]", xy=(-1,py_ll),arrowprops=dict(facecolor='black', shrink=0.01))

axs[1].plot(range(100), weights, label="s0.weights[0]")
axs[1].legend()
axs[1].set_ylabel("weight")
axs[1].set_xlabel("iteration")
axs[1].annotate(f'{weights[99]:.4f}',xy=(max_x,weights[max_x]),arrowprops=dict(facecolor='black', shrink=0.01))

plt.show()

py_ll = np.mean(log_likelihood(spn, data))
print(f'{py_ll:.8f},[{s0.weights[0]:.4f},{s0.weights[1]:.4f}], [{s1.weights[0]:.4f},{s1.weights[1]:.4f}]')

# non-shared
# print(f'{"eval of artificial data, after changed weights":60}', end='')
# s0.weights = [0.5, 0.5]
# s1.weights = [0.5, 0.5]
# py_ll = np.sum(log_likelihood(spn, data))
# print(f'{py_ll,s0.weights, s1.weights}')

# lls = []
# weights = np.array(range(100)) / np.array([100.0] * 100)
# for w in weights:
#     s0.weights[0] = s1.weights[0] = w
#     s0.weights[1] = s1.weights[1] = 1 - w
#     ll = np.mean(log_likelihood(spn,data))
#     lls.append(ll)
#
#
# fig, ax = plt.subplots()
# ax.plot(weights, lls)
# max_ll = max(lls)
# max_w = weights[lls.index(max_ll)]
# print(max_ll, max_w)
# ax.annotate(f'local max @ ({max_w},{max_ll})', xy=(max_w,max_ll), xytext=(max_w,max_ll),arrowprops=dict(facecolor='black', shrink=0.05))
# plt.show()