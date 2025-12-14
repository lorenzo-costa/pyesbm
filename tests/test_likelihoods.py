import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
import numpy as np
import pytest
from scipy.special import gammaln
from pyesbm.likelihoods import BetaBernoulli, PoissonGamma


class TestBetaBernoulliLikelihood:
    def test_placeholder(self):
        assert True
        # Placeholder test to ensure the test suite runs without errors.

    def test_log_likelihood(self):
        likelihood = BetaBernoulli(alpha=1.0, beta=1.0)

        # Simple test case
        mhk = np.array([[5, 3], [2, 4]])
        frequencies = np.array([2, 1])
        clustering = np.array([0, 0, 1])
        log_lik = likelihood.compute_llk(
            mhk=mhk,
            frequencies=frequencies,
            frequencies_other_side=frequencies,
            clustering=clustering,
            clustering_other_side=clustering,
        )
        assert isinstance(log_lik, float)

    def test_point_predict(self):
        likelihood = BetaBernoulli(alpha=1.0, beta=1.0)

        mhk = np.array([[5, 3], [2, 4]])
        frequencies = np.array([2, 1])
        clustering = np.array([0, 0, 1])
        pairs = [(0, 1), (1, 2)]

        prob_pairs = likelihood.point_predict(
            mhk=mhk,
            frequencies_1=frequencies,
            frequencies_2=frequencies,
            clustering_1=clustering,
            clustering_2=clustering,
            pairs=pairs,
        )
        assert len(prob_pairs) == len(pairs)
        assert all(0.0 <= p <= 1.0 for p in prob_pairs)

        for i in range(len(pairs)):
            p = pairs[i]
            c = (clustering[p[0]], clustering[p[1]])
            expected_prob = (mhk[c] + likelihood.alpha) / (
                frequencies[c[0]] * frequencies[c[1]]
                + likelihood.alpha
                + likelihood.beta
            )

            assert np.isclose((prob_pairs[i]), expected_prob)

    def test_sample_llk_edges(self):
        likelihood = BetaBernoulli(alpha=1.0, beta=1.0)

        Y = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0], [0, 0, 1]])
        clustering_1 = np.array([0, 0, 1])
        clustering_2 = np.array([0, 1, 1, 1])
        mhk = np.array([[2, 1], [1, 0]])
        frequencies_1 = np.array([2, 1])
        frequencies_2 = np.array([1, 3])

        llk_edges = likelihood.sample_llk_edges(
            Y=Y,
            mhk=mhk,
            frequencies_1=frequencies_1,
            frequencies_2=frequencies_2,
            clustering_1=clustering_1,
            clustering_2=clustering_2,
            bipartite=True,
            rng=np.random.default_rng(42),
        )

        assert len(llk_edges) == Y.shape[0] * Y.shape[1]
        assert all(isinstance(llk, float) for llk in llk_edges)


class TestPoissonGammaLikelihood:
    def test_placeholder(self):
        assert True
        # Placeholder test to ensure the test suite runs without errors.

    def test_log_likelihood(self):
        likelihood = PoissonGamma(shape=2.0, rate=1.0)

        # Simple test case
        mhk = np.array([[5, 3], [2, 4]])
        frequencies = np.array([2, 1])
        clustering = np.array([0, 0, 1])
        log_lik = likelihood.compute_llk(
            mhk=mhk,
            frequencies=frequencies,
            frequencies_other_side=frequencies,
            clustering=clustering,
            clustering_other_side=clustering,
        )
        assert isinstance(log_lik, float)

    def test_point_predict(self):
        likelihood = PoissonGamma(shape=2.0, rate=1.0)

        mhk = np.array([[5, 3], [2, 4]])
        frequencies = np.array([2, 1])
        clustering = np.array([0, 0, 1])
        pairs = [(0, 1), (1, 2)]

        rate_pairs = likelihood.point_predict(
            mhk=mhk,
            frequencies_1=frequencies,
            frequencies_2=frequencies,
            clustering_1=clustering,
            clustering_2=clustering,
            pairs=pairs,
        )
        assert len(rate_pairs) == len(pairs)
        assert all(r >= 0.0 for r in rate_pairs)

        for i in range(len(pairs)):
            p = pairs[i]
            c = (clustering[p[0]], clustering[p[1]])
            expected_rate = (mhk[c] + likelihood.shape) / (
                frequencies[c[0]] * frequencies[c[1]] + likelihood.rate
            )

            assert np.isclose((rate_pairs[i]), expected_rate)
