"""
likelihoods classes
"""

import numpy as np
from pyesbm.utilities.numba_functions import update_prob_poissongamma, compute_llk_poisson


class BaseLikelihood:
    def __init__(
        self, bipartite=False, epsilon=1e-10, degree_correction=0
    ):
        
        self.bipartite = bipartite
        self.epsilon = epsilon
        self.degree_correction = degree_correction
        
        self._type_check()
        

    def compute_likelihood(self, Y, clusters_1, clusters_2):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def update_steps(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def _type_check(self):
        if not isinstance(self.bipartite, bool):
            raise TypeError(
                f"bipartite must be a boolean. You provided {type(self.bipartite)}"
            )
        if not isinstance(self.epsilon, float):
            raise TypeError(f"epsilon must be float. You provided {type(self.epsilon)}")
        
        if not isinstance(self.degree_correction, int):
            raise TypeError(
                f"degree_correction must be int. You provided {type(self.degree_correction)}"
            )

class PlaceholderLikelihood(BaseLikelihood):
    def __init__(self, bipartite=False, eps=1e-10, degree_correction=0, **kwargs):
        super().__init__(bipartite, eps, degree_correction)
        
        self.needs_mhk = False
        self.needs_yvalues = False

    def compute_llk(self, **kwargs):
        return 1

    def update_logits(self, 
                      num_clusters, 
                      mhk, 
                      frequencies, 
                      frequencies_other_side,
                      y_values,
                      **kwargs):
    
        return np.ones(num_clusters)/num_clusters
    
    
class BetaBernoulli(BaseLikelihood):
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()

        args = {k: v for k, v in locals().items() if k != "self"}
        self._type_check(**args)

        self.alpha = alpha
        self.beta = beta

    def _type_check(self, **kwargs):
        alpha = kwargs.get("alpha")
        beta = kwargs.get("beta")
        if not isinstance(alpha, (int, float)):
            raise TypeError(
                f"alpha must be int or float. You provided {type(alpha)}"
            )
        if alpha <= 0:
            raise ValueError(f"alpha must be positive. You provided {alpha}")
        if not isinstance(beta, (int, float)):
            raise TypeError(
                f"beta must be int or float. You provided {type(beta)}"
            )
        if beta <= 0:
            raise ValueError(f"beta must be positive. You provided {beta}")

    def compute_likelihood(self, Y, clusters_1, clusters_2):
        # Implement the Bernoulli likelihood computation
        pass

    def update_logits(self):
        # Implement the steps to update the Bernoulli likelihood
        pass

class PoissonGamma(BaseLikelihood):
    def __init__(self, shape=1.0, rate=1.0, **kwargs):
        super().__init__(**kwargs)

        if not isinstance(shape, (int, float)):
            raise TypeError(
                f"shape must be int or float. You provided {type(shape)}"
            )
        if shape <= 0:
            raise ValueError(f"shape must be positive. You provided {shape}")
        if not isinstance(rate, (int, float)):
            raise TypeError(
                f"rate must be int or float. You provided {type(rate)}"
            )
        if rate <= 0:
            raise ValueError(f"rate must be positive. You provided {rate}")

        self.shape = shape
        self.rate = rate
        self.needs_mhk = True
        self.needs_yvalues = True
        
    def compute_llk(self, 
                    frequencies,
                    frequencies_other_side, 
                    mhk,
                    clustering, 
                    clustering_other_side,
                    **kwargs):
        
        llk = compute_llk_poisson(
            a=self.shape,
            b=self.rate,
            nh=frequencies,
            nk=frequencies_other_side if self.bipartite else frequencies,
            eps=self.epsilon,
            mhk=mhk,
            clustering_1=clustering,
            clustering_2=clustering_other_side if self.bipartite else clustering,
            degree_corrected=False,
            degree_param_users=1,
            degree_param_items=1,
            dg_u=np.array([1]),
            dg_i=np.array([1]),
            dg_cl_u=np.array([1]),
            dg_cl_i=np.array([1]),
        )
        return llk

    def update_logits(self, 
                      num_components,
                      mhk_minus,
                      y_values,
                      node_idx,
                      frequencies_minus,
                      frequencies_other_side_minus,
                      num_clusters,
                    **kwargs
                      ):

        print(frequencies_other_side_minus)
        print(frequencies_minus)
        out = update_prob_poissongamma(
            num_components=num_components,
            mhk=mhk_minus,
            frequencies_primary=frequencies_minus,
            frequencies_secondary=frequencies_other_side_minus,
            y_values=np.ascontiguousarray(y_values[node_idx]), # numba complains otherwise
            epsilon=self.epsilon,
            a=self.shape,
            b=self.rate,
            max_clusters=num_clusters,
            degree_corrected=False,
            degree_param=1,
            degree_cluster_minus=np.array([1]),
            degree_node=1,
        )
        
        return out
