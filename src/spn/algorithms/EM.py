"""
Created on November 09, 2018

@author: Alejandro Molina
@author: Robert Peharz
"""

from scipy.special import logsumexp

from spn.algorithms.Gradient import gradient_backward
from spn.algorithms.Inference import log_likelihood
from spn.algorithms.Validity import is_valid

from spn.structure.Base import Sum, Sum_sharedWeights, get_nodes_by_type, get_number_of_nodes
import numpy as np
import logging

logger = logging.getLogger(__name__)


def sum_em_update(node, node_gradients=None, root_lls=None, all_lls=None, **kwargs):
    RinvGrad = node_gradients - root_lls

    for i, c in enumerate(node.children):
        new_w = RinvGrad + (all_lls[:, c.id] + np.log(node.weights[i]))
        node.weights[i] = logsumexp(new_w)

    assert not np.any(np.isnan(node.weights))

    node.weights = np.exp(node.weights - logsumexp(node.weights)) + np.exp(-100)

    node.weights = node.weights / node.weights.sum()

    if node.weights.sum() > 1:
        node.weights[np.argmax(node.weights)] -= node.weights.sum() - 1

    assert not np.any(np.isnan(node.weights))
    assert np.isclose(np.sum(node.weights), 1)
    assert not np.any(node.weights < 0)
    assert node.weights.sum() <= 1, "sum: {}, node weights: {}".format(node.weights.sum(), node.weights)


def sum_em_update_shared(node, all_gradients=None, root_lls=None, all_lls=None, weights=None, **kwargs):

    def beta(w_old, node_gradients, child_lls):
        b = w_old * (np.exp(root_lls)**-1 *
                     node_gradients *
                     np.exp(child_lls)).sum()
        return b

    normalisation = 0
    for i in range(len(node.weights)):
        for qs in node.siblings:
            normalisation += beta(weights[qs.id][i],
                                  all_gradients[:, qs.id],
                                  all_lls[:, qs.children[i].id])

    for j in range(len(node.weights)):
        num = 0
        for qs in node.siblings:
            num += beta(weights[qs.id][j],
                        all_gradients[:,qs.id],
                        all_lls[:,qs.children[j].id])
        node.weights[j] = num / normalisation

_node_updates = {Sum: sum_em_update, Sum_sharedWeights: sum_em_update_shared}


def add_node_em_update(node_type, lambda_func):
    _node_updates[node_type] = lambda_func


def EM_optimization(spn, data, iterations=5, node_updates=_node_updates, skip_validation=False, **kwargs):
    if not skip_validation:
        valid, err = is_valid(spn)
        assert valid, "invalid spn: " + err

    lls_per_node = np.zeros((data.shape[0], get_number_of_nodes(spn)))

    # node_updates = {Sum_sharedWeights: sum_em_update_shared}
    for _ in range(iterations):
        # one pass bottom up evaluating the likelihoods
        log_likelihood(spn, data, lls_matrix=lls_per_node)# dtype=data.dtype

        gradients = gradient_backward(spn, lls_per_node)

        weights = [node.weights if isinstance(node, Sum_sharedWeights) else None for node in get_nodes_by_type(spn)]

        R = lls_per_node[:, 0]
        for node_type, func in node_updates.items():
            for node in get_nodes_by_type(spn, node_type):
                func(
                    node,
                    node_lls=lls_per_node[:, node.id],
                    node_gradients=gradients[:, node.id],
                    root_lls=R,
                    all_lls=lls_per_node,
                    all_gradients=gradients,
                    data=data,
                    spn=spn,
                    weights=weights,
                    **kwargs
                )
