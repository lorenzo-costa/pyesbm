import pytest
import sys
from pathlib import Path
import numpy as np
from pyesbm.utilities.data_generation import generate_poisson_data, generate_bernoulli_data
# sys.path.append(str(Path(__file__).parent.parent))

# from pyesbm.baseline import Baseline
# # from src.analysis.models.esbm_rec import Esbm
# # from src.analysis.models.dc_esbm_rec import Dcesbm
# # from src.analysis.utilities.numba_functions import compute_log_likelihood
# # from src.analysis.utilities.valid_functs import generate_val_set


class TestIndividualFunctions:
    def test_placeholder(self):
        assert True
        # Placeholder test to ensure the test suite runs without errors.
    
    @pytest.mark.parametrize(
        "n1,n2,bipartite",
        [
            (10, 15, True),
            (10, 15, False),
            (20, 20, True),
            (20, 20, False),
            
        ],
    )
    def test_poisson_generation(self,
                                n1,
                                n2, 
                                bipartite,
                                ):
        """Test Poisson data generation for both bipartite and unipartite cases."""
        rng = np.random.default_rng(42)
        prior_shape = 2.0
        prior_rate = 1.0
        clustering_1 = rng.integers(0, 3, size=n1)
        clustering_2 = rng.integers(0, 3, size=n2)
        if bipartite:
            Y = generate_poisson_data(
                prior_shape,
                prior_rate,
                clustering_1,
                clustering_2,
                bipartite=True,
                rng=rng,
            )
            assert Y.shape == (n1, n2)
        else:
            Y = generate_poisson_data(
                prior_shape,
                prior_rate,
                clustering_1,
                bipartite=False,
                rng=rng,
            )
            assert Y.shape == (n1, n1)
            assert np.allclose(Y, Y.T), "Adjacency matrix should be symmetric for unipartite graphs."

    @pytest.mark.parametrize(
        "n1,n2,bipartite",
        [
            (10, 15, True),
            (10, 15, False),
            (20, 20, True),
            (20, 20, False),
        ],
    )
    def test_bernoulli_generation(self,
                                   n1,
                                   n2,
                                   bipartite):
        """Test Bernoulli data generation for both bipartite and unipartite cases."""
        rng = np.random.default_rng(42)
        prior_alpha = 2.0
        prior_beta = 5.0
        clustering_1 = rng.integers(0, 3, size=n1)
        clustering_2 = rng.integers(0, 3, size=n2)
        if bipartite:
            Y = generate_bernoulli_data(
                prior_alpha,
                prior_beta,
                clustering_1,
                clustering_2,
                bipartite=True,
                rng=rng,
            )
            assert Y.shape == (n1, n2)
        else:
            Y = generate_bernoulli_data(
                prior_alpha,
                prior_beta,
                clustering_1,
                bipartite=False,
                rng=rng,
            )
            assert Y.shape == (n1, n1)
            assert np.allclose(Y, Y.T), "Adjacency matrix should be symmetric for unipartite graphs."