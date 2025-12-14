"""
likelihoods classes
"""

import numpy as np
from math import factorial

from pyesbm.utilities.numba_functions import update_prob_poissongamma
from pyesbm.utilities.numba_functions import update_prob_betabernoulli
from pyesbm.utilities.numba_functions import compute_llk_poisson
from pyesbm.utilities.numba_functions import compute_llk_bernoulli


class BaseLikelihood:
    def __init__(self, epsilon=1e-10):
        self.epsilon = epsilon

        self._type_check()

    def compute_likelihood(self, Y, clusters_1, clusters_2):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def update_steps(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def _type_check(self):
        if not isinstance(self.epsilon, float):
            raise TypeError(f"epsilon must be float. You provided {type(self.epsilon)}")


class PlaceholderLikelihood(BaseLikelihood):
    def __init__(self, bipartite=False, eps=1e-10, **kwargs):
        super().__init__(bipartite, eps)

        self.needs_mhk = False
        self.needs_yvalues = False

    def compute_llk(self, **kwargs):
        return 1

    def update_logits(
        self, num_clusters, mhk, frequencies, frequencies_other_side, y_values, **kwargs
    ):
        return np.ones(num_clusters) / num_clusters


class BetaBernoulli(BaseLikelihood):
    def __init__(self, alpha=1.0, beta=1.0):
        super().__init__()

        if not isinstance(alpha, (int, float)):
            raise TypeError(f"alpha must be int or float. You provided {type(alpha)}")
        if alpha <= 0:
            raise ValueError(f"alpha must be positive. You provided {alpha}")
        if not isinstance(beta, (int, float)):
            raise TypeError(f"beta must be int or float. You provided {type(beta)}")
        if beta <= 0:
            raise ValueError(f"beta must be positive. You provided {beta}")

        self.alpha = alpha
        self.beta = beta
        self.needs_mhk = True
        self.needs_yvalues = True

    def compute_llk(
        self,
        frequencies,
        frequencies_other_side,
        mhk,
        clustering,
        clustering_other_side,
        **kwargs,
    ):
        llk = compute_llk_bernoulli(
            a=self.alpha,
            b=self.beta,
            nh=frequencies,
            nk=frequencies_other_side,
            eps=self.epsilon,
            mhk=mhk,
        )
        return llk

    def update_logits(
        self,
        num_components,
        mhk_minus,
        y_values,
        node_idx,
        frequencies_minus,
        frequencies_other_side_minus,
        num_clusters,
        side,
        **kwargs,
    ):
        out = update_prob_betabernoulli(
            num_components=num_components,
            mhk=mhk_minus,
            y_values=np.ascontiguousarray(
                y_values[node_idx]
            ),  # numba complains otherwise
            frequencies_primary=frequencies_minus,
            frequencies_secondary=frequencies_other_side_minus,
            max_clusters=num_clusters,
            side=side,
            epsilon=self.epsilon,
            a=self.alpha,
            b=self.beta,
        )

        return out

    def _estimate_theta(self, mhk, frequencies_1, frequencies_2):
        """Estimate the model parameters (theta) based on the current state.
        Using the posterior mean of the Beta distribution.

        Parameters
        ----------
        mhk : np.ndarray
            Matrix of co-cluster assignments.
        frequencies_1 : np.ndarray
            Frequencies for the first dimension.
        frequencies_2 : np.ndarray
            Frequencies for the second dimension.

        Returns
        -------
        np.ndarray
            Estimated parameters (theta).
        """
        outer = np.outer(frequencies_1, frequencies_2)
        theta = (self.alpha + mhk) / (self.alpha + self.beta + outer)

        return theta

    def point_predict(
        self,
        pairs,
        mhk,
        frequencies_1,
        frequencies_2,
        clustering_1,
        clustering_2,
        rng=None,
    ):
        """Generate point predictions for the given pairs.

        Parameters
        ----------
        pairs : _type_
            _description_
        rng : _type_, optional
            _description_, by default None
        """

        n1 = len(clustering_1)
        n2 = len(clustering_2)

        if not isinstance(pairs, list):
            raise TypeError("pairs must be a list")

        for p in pairs:
            if not isinstance(p, (tuple, list, np.ndarray)) or len(p) != 2:
                raise ValueError("Each pair must be a list, tuple or array of length 2")
            u, i = p
            if not isinstance(u, int) or not isinstance(i, int):
                raise TypeError("Each element in the pair must be an integer")
            if u < 0 or u >= n1:
                raise ValueError(f"Node index {u} out of bounds for first dimension")
            if i < 0 or i >= n2:
                raise ValueError(f"Node index {i} out of bounds for second dimension")

        if rng is None:
            rng = np.random.default_rng()

        preds = []
        theta = self._estimate_theta(
            mhk=mhk,
            frequencies_1=frequencies_1,
            frequencies_2=frequencies_2,
        )
        for p in pairs:
            u, i = p
            c_u = clustering_1[u]
            c_i = clustering_2[i] if clustering_2 is not None else clustering_1[i]
            preds.append(theta[c_u, c_i])

        return np.array(preds)

    def sample_llk_edges(
        self,
        Y,
        mhk,
        frequencies_1,
        frequencies_2,
        clustering_1,
        clustering_2,
        bipartite=False,
        rng=None,
    ):
        """Compute log-likelihoods of edges based on sampled theta values.

        Returns
        -------
        np.ndarray
            Log-likelihoods for the edges.
        """

        if rng is None:
            rng = np.random.default_rng()

        # sample theta from posterior Beta
        a_n = self.alpha + mhk
        b_bar_n = self.beta + np.outer(frequencies_1, frequencies_2)
        theta = rng.beta(a_n, b_bar_n)

        if bipartite is False:
            # symmetrize theta
            theta = theta + theta.T
            theta = theta / 2

        clustering_1_onehot = np.eye(np.max(clustering_1) + 1)[clustering_1]
        clustering_2_onehot = np.eye(np.max(clustering_2) + 1)[clustering_2]

        edge_prob = clustering_1_onehot @ theta @ clustering_2_onehot.T

        y_flat = Y.flatten()
        p_flat = edge_prob.flatten()

        out = y_flat * np.log(p_flat) + (1 - y_flat) * np.log(1 - p_flat)

        return out


class PoissonGamma(BaseLikelihood):
    def __init__(self, shape=1.0, rate=1.0, **kwargs):
        super().__init__(**kwargs)

        if not isinstance(shape, (int, float)):
            raise TypeError(f"shape must be int or float. You provided {type(shape)}")
        if shape <= 0:
            raise ValueError(f"shape must be positive. You provided {shape}")
        if not isinstance(rate, (int, float)):
            raise TypeError(f"rate must be int or float. You provided {type(rate)}")
        if rate <= 0:
            raise ValueError(f"rate must be positive. You provided {rate}")

        self.shape = shape
        self.rate = rate
        self.needs_mhk = True
        self.needs_yvalues = True

    def compute_llk(
        self,
        frequencies,
        frequencies_other_side,
        mhk,
        clustering,
        clustering_other_side,
        **kwargs,
    ):
        llk = compute_llk_poisson(
            a=self.shape,
            b=self.rate,
            nh=frequencies,
            nk=frequencies_other_side,
            eps=self.epsilon,
            mhk=mhk,
        )
        return llk

    def update_logits(
        self,
        num_components,
        mhk_minus,
        y_values,
        node_idx,
        frequencies_minus,
        frequencies_other_side_minus,
        num_clusters,
        side,
        bipartite,
        **kwargs,
    ):
        out = update_prob_poissongamma(
            num_components=num_components,
            mhk=mhk_minus,
            frequencies_primary=frequencies_minus,
            frequencies_secondary=frequencies_other_side_minus,
            y_values=np.ascontiguousarray(
                y_values[node_idx]
            ),  # numba complains otherwise
            epsilon=self.epsilon,
            a=self.shape,
            b=self.rate,
            max_clusters=num_clusters,
            side=side,
        )

        return out

    def _estimate_theta(self, mhk, frequencies_1, frequencies_2):
        """
        Estimate the model parameters (theta) based on the current state.

        Parameters
        ----------
        mhk : np.ndarray
            Matrix of co-cluster assignments.
        frequencies_1 : np.ndarray
            Frequencies for the first dimension.
        frequencies_2 : np.ndarray
            Frequencies for the second dimension.

        Returns
        -------
        np.ndarray
            Estimated parameters (theta).
        """

        outer = np.outer(frequencies_1, frequencies_2)
        theta = (self.shape + mhk) / (self.rate + outer)

        return theta

    def point_predict(
        self,
        pairs,
        mhk,
        frequencies_1,
        frequencies_2,
        clustering_1,
        clustering_2,
        rng=None,
    ):
        """Generate point predictions for the given pairs.

        Parameters
        ----------
        pairs : _type_
            _description_
        rng : _type_, optional
            _description_, by default None
        """

        n1 = len(clustering_1)
        n2 = len(clustering_2)

        if not isinstance(pairs, list):
            raise TypeError("pairs must be a list")

        for p in pairs:
            if not isinstance(p, (tuple, list, np.ndarray)) or len(p) != 2:
                raise ValueError("Each pair must be a list, tuple or array of length 2")
            u, i = p
            if not isinstance(u, int) or not isinstance(i, int):
                raise TypeError("Each element in the pair must be an integer")
            if u < 0 or u >= n1:
                raise ValueError(f"Node index {u} out of bounds for first dimension")
            if i < 0 or i >= n2:
                raise ValueError(f"Node index {i} out of bounds for second dimension")

        if rng is None:
            rng = np.random.default_rng()

        preds = []
        theta = self._estimate_theta(
            mhk=mhk,
            frequencies_1=frequencies_1,
            frequencies_2=frequencies_2,
        )
        for p in pairs:
            u, i = p
            c_u = clustering_1[u]
            c_i = clustering_2[i] if clustering_2 is not None else clustering_1[i]
            preds.append(theta[c_u, c_i])

        return np.array(preds)

    def sample_llk_edges(
        self,
        Y,
        mhk,
        frequencies_1,
        frequencies_2,
        clustering_1,
        clustering_2,
        bipartite=False,
        rng=None,
    ):
        """Compute log-likelihoods of edges based on sampled theta values.

        Returns
        -------
        np.ndarray
            Log-likelihoods for the edges.
        """

        if rng is None:
            rng = np.random.default_rng()

        # sample theta from posterior Gamma
        a_n = self.shape + mhk
        b_n = self.rate + np.outer(frequencies_1, frequencies_2)
        theta = rng.gamma(a_n, 1 / b_n)  # numpy uses shape, scale=1/rate

        if bipartite is False:
            # symmetrize theta
            theta = theta + theta.T
            theta = theta / 2

        clustering_1_onehot = np.eye(np.max(clustering_1) + 1)[clustering_1]
        clustering_2_onehot = np.eye(np.max(clustering_2) + 1)[clustering_2]

        edge_rate = clustering_1_onehot @ theta @ clustering_2_onehot.T

        y_flat = Y.flatten()
        r_flat = edge_rate.flatten()

        out = (
            y_flat * np.log(r_flat)
            - r_flat
            - np.log(np.array([factorial(int(y)) for y in y_flat]))
        )

        return out
