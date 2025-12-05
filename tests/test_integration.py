import pytest
import sys
from pathlib import Path
import numpy as np
import warnings

sys.path.append(str(Path(__file__).parent.parent))

from pyesbm.baseline import BaseESBM
from pyesbm.utilities.misc_functs import generate_poisson_data, generate_bernoulli_data
from pyesbm.priors import GibbsTypePrior
from pyesbm.likelihoods import PoissonGamma, BetaBernoulli

class TestBaseESBM:
    
    @classmethod
    def setup_class(cls):
        cls.n_1 = 50
        cls.n_2 = 50

        # manually define clustering structure
        cls.sizes_1 = [cls.n_1//5, cls.n_1//10, cls.n_1//3, cls.n_1//4, cls.n_1 - (cls.n_1//3)-(cls.n_1//4)-(cls.n_1//5)-(cls.n_1//10)]
        cls.clustering_1 = np.array(
            [0 for _ in range(cls.sizes_1[0])] + 
            [1 for _ in range(cls.sizes_1[1])] + 
            [2 for _ in range(cls.sizes_1[2])] + 
            [3 for _ in range(cls.sizes_1[3])] +
            [4 for _ in range(cls.sizes_1[4])])

        cls.sizes_2 = [cls.n_2//4, cls.n_2//4, cls.n_2//5, cls.n_2 - (cls.n_2//4)-(cls.n_2//4)-(cls.n_2//5)]
        cls.clustering_2 = np.array(
            [0 for _ in range(cls.sizes_2[0])] + 
            [1 for _ in range(cls.sizes_2[1])] + 
            [2 for _ in range(cls.sizes_2[2])] + 
            [3 for _ in range(cls.sizes_2[3])])
        
        cls.t3 = np.array([1 if cls.clustering_1[i]%2==0 else 0 for i in range(cls.n_1)])
        cls.t4 = np.array([1 if cls.clustering_1[i]%2==0 else 0 for i in range(cls.n_1)])
        
        cls.t5 = np.array([1 if cls.clustering_2[i]%2==0 else 0 for i in range(cls.n_2)])
        cls.t6 = np.array([1 if cls.clustering_2[i]%2==0 else 0 for i in range(cls.n_2)])

        cls.t3[np.random.randint(0, len(cls.t3), size=25)] = 2
        cls.t4[np.random.randint(0, len(cls.t4), size=25)] = 0
        cls.t5[np.random.randint(0, len(cls.t5), size=20)] = 2
        cls.t6[np.random.randint(0, len(cls.t6), size=20)] = 0

        cls.covs_1 = [('genre_categorical', cls.t3.copy()), 
                ('genre2_categorical', cls.t4.copy())]
        
        cls.covs_2 = [('type_categorical', cls.t5.copy()), 
                ('type2_categorical', cls.t6.copy())]

        cls.rng_gen = np.random.default_rng(42)

        cls.rng_model = np.random.default_rng(1)

    @pytest.mark.parametrize("prior_type", ["DP", "PY", "GN"])
    def test_bipartite_poisson_no_cov(self, prior_type):
        
        Y = generate_poisson_data(0.5,
                                0.5,
                                self.clustering_1, 
                                clustering_2=self.clustering_2,
                                bipartite=True, 
                                rng=self.rng_gen)
        
        a = np.random.permutation(self.n_1)
        b = np.random.permutation(self.n_2)
        Y = Y[a][:, b]

        prior = GibbsTypePrior(scheme_type=prior_type,
                               scheme_param=1.5,
                               sigma=-0.5 if prior_type=="DM" else 0.5,
                               gamma=0.5,
                               num_nodes_1=self.n_1,
                               num_nodes_2=self.n_2)
        
        likelihood = PoissonGamma(shape=1, rate=1)
        
        model = BaseESBM(Y, 
            prior=prior, 
            likelihood=likelihood, 
            epsilon=1e-10, 
            bipartite=True, 
            verbose=False, 
            clustering='Random',
            rng=self.rng_model)

        model.fit(n_iters=10, verbose=False)
        
        assert True
    
    @pytest.mark.parametrize("prior_type", ["DP", "PY", "GN"])
    def test_bipartite_bernoulli_no_cov(self, prior_type):
        
        Y = generate_bernoulli_data(0.5,
                                0.5,
                                self.clustering_1, 
                                clustering_2=self.clustering_2,
                                bipartite=True, 
                                rng=self.rng_gen)
        
        a = np.random.permutation(self.n_1)
        b = np.random.permutation(self.n_2)
        Y = Y[a][:, b]

        prior = GibbsTypePrior(scheme_type=prior_type,
                               scheme_param=1.5,
                               sigma=-0.5 if prior_type=="DM" else 0.5,
                               gamma=0.5,
                               num_nodes_1=self.n_1,
                               num_nodes_2=self.n_2)
        
        likelihood = BetaBernoulli(alpha=1, beta=1)
        
        model = BaseESBM(Y, 
            prior=prior, 
            likelihood=likelihood, 
            epsilon=1e-10, 
            bipartite=True, 
            verbose=False, 
            clustering='Random',
            rng=self.rng_model)

        model.fit(n_iters=10, verbose=False)
        
        assert True
    
    @pytest.mark.parametrize("prior_type", ["DP", "PY", "GN"])
    def test_unipartite_poisson_no_cov(self, prior_type):
        
        Y = generate_poisson_data(0.5,
                                0.5,
                                self.clustering_1, 
                                bipartite=False, 
                                rng=self.rng_gen)
        
        a = np.random.permutation(self.n_1)
        Y = Y[a][:, a]

        prior = GibbsTypePrior(scheme_type=prior_type,
                               scheme_param=1.5,
                               sigma=-0.5 if prior_type=="DM" else 0.5,
                               gamma=0.5,
                               num_nodes_1=self.n_1)
        
        likelihood = PoissonGamma(shape=1, rate=1)
        
        model = BaseESBM(Y, 
            prior=prior, 
            likelihood=likelihood, 
            epsilon=1e-10, 
            bipartite=False, 
            verbose=False, 
            clustering='Random',
            rng=self.rng_model)

        model.fit(n_iters=10, verbose=False)
        
        assert True
    
    @pytest.mark.parametrize("prior_type", ["DP", "PY", "GN"])
    def test_unipartite_bernoulli_no_cov(self, prior_type):
        
        Y = generate_bernoulli_data(0.5,
                                0.5,
                                self.clustering_1, 
                                bipartite=False, 
                                rng=self.rng_gen)
        
        a = np.random.permutation(self.n_1)
        Y = Y[a][:, a]

        prior = GibbsTypePrior(scheme_type=prior_type,
                               scheme_param=1.5,
                               sigma=-0.5 if prior_type=="DM" else 0.5,
                               gamma=0.5,
                               num_nodes_1=self.n_1)
        
        likelihood = BetaBernoulli(alpha=1, beta=1)
        
        model = BaseESBM(Y, 
            prior=prior, 
            likelihood=likelihood, 
            epsilon=1e-10, 
            bipartite=False, 
            verbose=False, 
            clustering='Random',
            rng=self.rng_model)

        model.fit(n_iters=10, verbose=False)
        
        assert True
    
    @pytest.mark.parametrize("prior_type", ["DP", "PY", "GN"])
    def test_bipartite_poisson_with_cov(self, prior_type):
        
        Y = generate_poisson_data(0.5,
                                0.5,
                                self.clustering_1, 
                                clustering_2=self.clustering_2,
                                bipartite=True, 
                                rng=self.rng_gen)
        
        a = np.random.permutation(self.n_1)
        b = np.random.permutation(self.n_2)
        Y = Y[a][:, b]

        prior = GibbsTypePrior(scheme_type=prior_type,
                               scheme_param=1.5,
                               sigma=-0.5 if prior_type=="DM" else 0.5,
                               gamma=0.5,
                               num_nodes_1=self.n_1,
                               num_nodes_2=self.n_2)
        
        likelihood = PoissonGamma(shape=1, rate=1)
        
        model = BaseESBM(Y, 
            prior=prior, 
            likelihood=likelihood, 
            covariates_1=self.covs_1,
            covariates_2=self.covs_2,
            epsilon=1e-10, 
            bipartite=True, 
            verbose=False, 
            clustering='Random',
            rng=self.rng_model)

        model.fit(n_iters=10, verbose=False)
        
        assert True
    
    @pytest.mark.parametrize("prior_type", ["DP", "PY", "GN"])
    def test_bipartite_bernoulli_with_cov(self, prior_type):
        
        Y = generate_bernoulli_data(0.5,
                                0.5,
                                self.clustering_1, 
                                clustering_2=self.clustering_2,
                                bipartite=True, 
                                rng=self.rng_gen)
        
        a = np.random.permutation(self.n_1)
        b = np.random.permutation(self.n_2)
        Y = Y[a][:, b]

        prior = GibbsTypePrior(scheme_type=prior_type,
                               scheme_param=1.5,
                               sigma=-0.5 if prior_type=="DM" else 0.5,
                               gamma=0.5,
                               num_nodes_1=self.n_1,
                               num_nodes_2=self.n_2)
        
        likelihood = BetaBernoulli(alpha=1, beta=1)

        model = BaseESBM(Y,
            prior=prior,
            likelihood=likelihood,
            covariates_1=self.covs_1,
            covariates_2=self.covs_2,
            epsilon=1e-10, 
            bipartite=True, 
            verbose=False, 
            clustering='Random',
            rng=self.rng_model)

        model.fit(n_iters=10, verbose=False)
        
        assert True
    
    @pytest.mark.parametrize("prior_type", ["DP", "PY", "GN"])
    def test_unipartite_poisson_with_cov(self, prior_type):
        
        Y = generate_poisson_data(0.5,
                                0.5,
                                self.clustering_1, 
                                bipartite=False, 
                                rng=self.rng_gen)
        
        a = np.random.permutation(self.n_1)
        Y = Y[a][:, a]

        prior = GibbsTypePrior(scheme_type=prior_type,
                               scheme_param=1.5,
                               sigma=-0.5 if prior_type=="DM" else 0.5,
                               gamma=0.5,
                               num_nodes_1=self.n_1)
        
        likelihood = PoissonGamma(shape=1, rate=1)
        
        model = BaseESBM(Y, 
            prior=prior, 
            likelihood=likelihood, 
            covariates_1=self.covs_1,
            epsilon=1e-10, 
            bipartite=False, 
            verbose=False, 
            clustering='Random',
            rng=self.rng_model)

        model.fit(n_iters=10, verbose=False)
        
        assert True
    
    @pytest.mark.parametrize("prior_type", ["DP", "PY", "GN"])
    def test_unipartite_bernoulli_with_cov(self, prior_type):
        
        Y = generate_bernoulli_data(0.5,
                                0.5,
                                self.clustering_1, 
                                bipartite=False, 
                                rng=self.rng_gen)
        
        a = np.random.permutation(self.n_1)
        Y = Y[a][:, a]

        prior = GibbsTypePrior(scheme_type=prior_type,
                               scheme_param=1.5,
                               sigma=-0.5 if prior_type=="DM" else 0.5,
                               gamma=0.5,
                               num_nodes_1=self.n_1)

        likelihood = BetaBernoulli(alpha=1, beta=1)

        model = BaseESBM(Y,
            prior=prior, 
            likelihood=likelihood, 
            covariates_1=self.covs_1,
            epsilon=1e-10, 
            bipartite=False, 
            verbose=False, 
            clustering='Random',
            rng=self.rng_model)

        model.fit(n_iters=10, verbose=False)
        
        assert True
