# bunch of tests to make sure sampling scheme returns expected results
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from pyesbm.priors import GibbsTypePrior
import numpy as np
import pytest
from scipy.special import gammaln


class TestSamplingSchemes:
    # base set
    params_dm = {
        "bar_h": 10,
        "scheme_type": "DM",
        "scheme_param": 4,
        "sigma": -1,
        "gamma": 1,
    }

    params_dp = {
        "bar_h": 10,
        "scheme_type": "DP",
        "scheme_param": 1,
        "sigma": 0,
        "gamma": 1,
    }

    params_py = {
        "bar_h": 10,
        "scheme_type": "PY",
        "scheme_param": 1,
        "sigma": 0.5,
        "gamma": 1,
    }

    params_gn = {
        "bar_h": 10,
        "scheme_type": "GN",
        "scheme_param": 0.5,
        "sigma": -1,
        "gamma": 0.5,
    }

    eps = 1e-6

    # saturated multinomial should not make new clusters
    @pytest.mark.parametrize(
        "V, frequencies, H",
        [
            (10, np.array([5, 5]), 2),
            (10, np.array([2, 4, 4]), 3),
            (1, np.array([1]), 1),
            (5, np.array([1, 1, 1, 1, 1]), 5),
        ],
    )
    def test_DM_scheme_no_new_cluster(self, V, frequencies, H):
        """Test that DM scheme does not create new cluster when bar_h=H."""
        params_updated = self.params_dm.copy()
        params_updated["scheme_type"] = "DM"
        params_updated["num_nodes_1"] = V
        params_updated["bar_h"] = H

        prior = GibbsTypePrior(**params_updated)

        out = prior.compute_probs(
            num_nodes=V, num_clusters=H, frequencies_minus=frequencies
        )
        assert np.all(out == (frequencies - params_updated["sigma"]))
        assert (sum(out / sum(out)) - 1) < self.eps

    # test dm behaviour
    @pytest.mark.parametrize(
        "V, frequencies, H, bar_h",
        [
            (10, np.array([5, 5]), 2, 10),
            (10, np.array([2, 4, 4]), 3, 5),
        ],
    )
    def test_DM_scheme_new_cluster(self, V, frequencies, H, bar_h):
        """Test that DM scheme creates new cluster when bar_h>H."""
        params_updated = self.params_dm.copy()
        params_updated["scheme_type"] = "DM"
        params_updated["num_nodes_1"] = V
        params_updated["bar_h"] = bar_h

        prior = GibbsTypePrior(**params_updated)

        out = prior.compute_probs(
            num_nodes=V, num_clusters=H, frequencies_minus=frequencies
        )

        expected = np.append(
            frequencies - params_updated["sigma"],
            -params_updated["sigma"] * (bar_h - H),
        )
        assert np.allclose(out, expected)
        assert (sum(out / sum(out)) - 1) < self.eps

    # test dp behaviour
    @pytest.mark.parametrize(
        "V, frequencies, H",
        [
            (10, np.array([5, 5]), 2),
            (10, np.array([2, 4, 4]), 3),
            (10, np.array([1, 1, 1, 1, 6]), 5),
            (1, np.array([1]), 1),
            (5, np.array([1, 1, 1, 1, 1]), 5),
        ],
    )
    def test_dp(self, V, frequencies, H):
        params_updated = self.params_dp.copy()
        params_updated["scheme_type"] = "DP"
        prior = GibbsTypePrior(**params_updated)

        out = prior.compute_probs(
            num_nodes=V, num_clusters=H, frequencies_minus=frequencies
        )
        expected = np.append(frequencies, params_updated["scheme_param"])

        assert np.all(out == expected)
        assert (sum(out / sum(out)) - 1) < self.eps

    # test py behaviour
    @pytest.mark.parametrize(
        "V, frequencies, H",
        [
            (10, np.array([5, 5]), 2),
            (10, np.array([2, 4, 4]), 3),
            (10, np.array([1, 1, 1, 1, 6]), 5),
            (1, np.array([1]), 1),
            (5, np.array([1, 1, 1, 1, 1]), 5),
        ],
    )
    def test_py(self, V, frequencies, H):
        params_updated = self.params_py.copy()

        prior = GibbsTypePrior(**params_updated)

        out = prior.compute_probs(
            num_nodes=V, num_clusters=H, frequencies_minus=frequencies
        )
        expected = np.append(
            frequencies - params_updated["sigma"],
            params_updated["scheme_param"] + params_updated["sigma"] * H,
        )
        assert np.allclose(out, expected)
        assert (sum(out / sum(out)) - 1) < self.eps

    # test gn behaviour
    @pytest.mark.parametrize(
        "V, frequencies, H",
        [
            (10, np.array([5, 5]), 2),
            (10, np.array([2, 4, 4]), 3),
            (10, np.array([1, 1, 1, 1, 6]), 5),
            (1, np.array([1]), 1),
            (5, np.array([1, 1, 1, 1, 1]), 5),
        ],
    )
    def test_gn(self, V, frequencies, H):
        params_updated = self.params_gn.copy()
        prior = GibbsTypePrior(**params_updated)

        out = prior.compute_probs(
            num_nodes=V, num_clusters=H, frequencies_minus=frequencies
        )

        expected = np.append(
            (frequencies + 1) * (V - H + params_updated["gamma"]),
            H * (H - params_updated["gamma"]),
        )
        assert np.all(out == expected)
        assert (sum(out / sum(out)) - 1) < self.eps

    # test py with sigma=0 is equivalent to dp
    @pytest.mark.parametrize(
        "V, frequencies, H",
        [
            (10, np.array([5, 5]), 2),
            (10, np.array([2, 4, 4]), 3),
            (10, np.array([1, 1, 1, 1, 6]), 5),
            (1, np.array([1]), 1),
            (5, np.array([1, 1, 1, 1, 1]), 5),
        ],
    )
    def test_py_equal_dp(self, V, frequencies, H):
        """Test that PY with sigma=0 is equivalent to DP"""
        params_py = self.params_py.copy()
        params_py["sigma"] = 0
        params_py["scheme_type"] = "PY"

        params_dp = self.params_dp.copy()
        params_dp["scheme_type"] = "DP"

        prior_py = GibbsTypePrior(**params_py)
        prior_dp = GibbsTypePrior(**params_dp)

        out_py = prior_py.compute_probs(
            num_nodes=V, num_clusters=H, frequencies_minus=frequencies
        )
        out_dp = prior_dp.compute_probs(
            num_nodes=V, num_clusters=H, frequencies_minus=frequencies
        )

        assert np.all(out_py == out_dp)

class TestExpectedValue:
    # base set
    params_dm = {
        "bar_h": 10,
        "scheme_type": "DM",
        "scheme_param": 4,
        "sigma": -1,
        "gamma": 1,
    }

    params_dp = {
        "bar_h": 10,
        "scheme_type": "DP",
        "scheme_param": 1,
        "sigma": 0,
        "gamma": 1,
    }

    params_py = {
        "bar_h": 10,
        "scheme_type": "PY",
        "scheme_param": 1,
        "sigma": 0.5,
        "gamma": 1,
    }

    params_gn = {
        "bar_h": 10,
        "scheme_type": "GN",
        "scheme_param": 0.5,
        "sigma": -1,
        "gamma": 0.5,
    }

    eps = 1e-6
    
    @pytest.mark.parametrize(
        "n",
        [10, 50, 100, 200],
    )
    def test_expected_num_clusters_dm(self, n):
        params_updated = self.params_dm.copy()
        params_updated["num_nodes_1"] = n
        H = params_updated["bar_h"]
        prior = GibbsTypePrior(**params_updated)
        expected = prior.expected_num_clusters(n)
        
        b = params_updated["scheme_param"]
        a = b * (1-1/H)

        log_prod = (gammaln(a + n) - gammaln(a)) - (gammaln(b + n) - gammaln(b))

        true_value = H - H * np.exp(log_prod)
        
        assert abs(expected - true_value) < self.eps

    @pytest.mark.parametrize(
        "n",
        [10, 50, 100, 200],
    )
    def test_expected_num_clusters_dp(self, n):
        params_updated = self.params_dp.copy()
        prior = GibbsTypePrior(**params_updated)
        expected = prior.expected_num_clusters(n)
        
        alpha = params_updated["scheme_param"]

        true_value = (alpha / (alpha + np.arange(1, n+1) -1)).sum()

        assert abs(expected - true_value) < self.eps
    
    @pytest.mark.parametrize(
        "n",
        [10, 50, 100, 200],
    )
    def test_expected_num_clusters_py(self, n):
        params_updated = self.params_py.copy()
        prior = GibbsTypePrior(**params_updated)
        expected = prior.expected_num_clusters(n)
        
        sigma = params_updated["sigma"]
        alpha = params_updated["scheme_param"]

        log_term = (gammaln(alpha + sigma + n) - gammaln(alpha + sigma) - 
                        gammaln(alpha + n) + gammaln(alpha + 1))
        
        true_value = 2 * np.exp(log_term) - 2
        
        assert abs(expected - true_value) < self.eps
    
    @pytest.mark.parametrize(
        "n",
        [10, 50, 100, 200],
    )
    def test_expected_num_clusters_gn(self, n):
        params_updated = self.params_gn.copy()
        prior = GibbsTypePrior(**params_updated)
        expected = prior.expected_num_clusters(n)
        
        gamma = params_updated["gamma"]

        lgamma_V_plus_1 = gammaln(n + 1)
        lgamma_1_minus_gamma = gammaln(1 - gamma)
        lgamma_V_plus_gamma = gammaln(n + gamma)

        h = np.arange(1, n + 1)

        log_terms = (lgamma_V_plus_1 - gammaln(h + 1) - gammaln(n - h + 1) +
                     gammaln(h - gamma) - lgamma_1_minus_gamma +
                     np.log(gamma) +
                     gammaln(n + gamma - h) - lgamma_V_plus_gamma)

        probs = np.exp(log_terms)
        true_value = np.sum(h * probs)

        assert abs(expected - true_value) < self.eps    