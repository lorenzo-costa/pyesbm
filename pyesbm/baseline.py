import time
import numpy as np
from scipy import sparse
from scipy.stats import mode
import sys
from pathlib import Path
import warnings


from pyesbm.utilities import (sampling_scheme, compute_log_probs_cov, compute_log_likelihood,
                    compute_co_clustering_matrix, minVI)

from pyesbm.likelihoods import BaseLikelihood, Bernoulli

likelihood_dict = {
    'bernoulli': Bernoulli
}

#########################################
# baseline class
########################################
class BaseESBM:
    """Baseline ESBM model

    Parameters
    ----------
    Y : 2D array
        adjacency matrix (if None automatically generated), by default None
    bipartite : bool
        whether the network is bipartite, by default False
    likelihood : str
        likelihood type, by default 'bernoulli'
    prior_a : float
        shape parameter for gamma prior, by default 1
    prior_b : float
        rate parameter for gamma prior, by default 1
    scheme_type : str
        prior type. Possible choices are DP (dirichlet process), PY (pitman-yor process), 
        GN (gnedin process), DM (dirichlet-multinomial model)
    scheme_param : float
        additional parameter for cluster prior, by default 1
    sigma : float
        sigma parameter for Gibbs-type prior, by default 1
    gamma : float
        additional parameter for GN model, by default None
    bar_h : int
        maximum number of clusters for DM model, if None set to num_users
    degree_correction : float
        degree-correction parameter for users (relevant only for DC model), by default 1
    alpha_c : float or list
        additional parameter for categorical covariate model (if int defaults to 
        vector of equal numbers), by default 1
    covariates_1 : list
        list of tuples (covname_covtype, covvalues), by default None
    covariates_2 : list
        list of tuples (covname_covtype, covvalues), by default None
    verbose : bool
        whether to print verbose output for user-related computations, by default False
    
    Attributes
    ----------
    #TODO
    """
    
    def __init__(self, 
                 Y,
                 bipartite=True,
                 likelihood='bernoulli',
                 *args,
                 clustering=None,
                 prior_a=1, 
                 prior_b=1, 
                 scheme_type = None, 
                 scheme_param = 1, 
                 sigma = 0.1, 
                 gamma=0.5,
                 bar_h=None, 
                 degree_correction=1, 
                 alpha_c=1, 
                 covariates_1=None, 
                 covariates_2=None,
                 epsilon = 1e-6,
                 rng = None,  
                 verbose=False):

        # a lot of type and value checking
        args = {k: v for k, v in locals().items() if k != "self"}
        
        self._type_check(**args)
        
        self.scheme_dict = {'DM':1, 'DP':2, 'PY':3, 'GN':4}
        
        self._process_args(**args)

    def _type_check(self, **kwargs):
        prior_a = kwargs.get('prior_a')
        prior_b = kwargs.get('prior_b')
        scheme_type = kwargs.get('scheme_type')
        bar_h = kwargs.get('bar_h')
        sigma = kwargs.get('sigma')
        scheme_param = kwargs.get('scheme_param')
        gamma = kwargs.get('gamma')
        
        covariates_1 = kwargs.get('covariates_1')
        covariates_2 = kwargs.get('covariates_2')
        clustering = kwargs.get('clustering')
        bipartite = kwargs.get('bipartite')
        likelihood = kwargs.get('likelihood')
        rng = kwargs.get('rng')
        Y = kwargs.get('Y')
        num_nodes_1, num_nodes_2 = Y.shape 
        
        if not isinstance(prior_a, (int, float)):
            raise TypeError(f'prior a should be a float or int. You provided {type(prior_a)}')
        if prior_a <= 0:
            raise ValueError(f'please provide valid prior a parameter (>0). You provided {prior_a}')
        if not isinstance(prior_b, (int, float)):
            raise TypeError(f'prior b should be a float or int. You provided {type(prior_b)}')
        if prior_b <= 0:
            raise ValueError(f'please provide valid prior b parameter (>0). You provided {prior_b}')

        if scheme_type is None or scheme_type not in ['DP', 'PY', 'GN', 'DM']:
            raise ValueError(f'scheme type must be one of DP, PY, GN, DM. You provided {scheme_type}')

        if bar_h is None:
            bar_h = min(num_nodes_1, num_nodes_2)    

        if scheme_type == 'DM':
            if not isinstance(bar_h, int):
                raise TypeError('maximum number of clusters users must be integer for DM')
            if (bar_h <= 0) or (bar_h > min(num_nodes_1, num_nodes_2)):
                raise ValueError(f"maximum number of clusters for DM for users should be in (0, {num_users}]."
                                 f". You provided {bar_h}")
            if not isinstance(sigma, (int, float)):
                raise TypeError('sigma must be a float or int for DM')
            if sigma >= 0:
                raise ValueError(f'sigma for DM should be negative. You provided {sigma}')
        
        if scheme_type == 'DP':
            if not isinstance(scheme_param, (int, float)):
                raise TypeError('concentration parameter for DP must be float or int')
            if scheme_param <= 0:
                raise ValueError(f'concentration parameter for DP should be positive. You provided {scheme_param}')

        if scheme_type == 'PY':
            if not isinstance(sigma, (int, float)):
                raise TypeError('sigma must be a float or int for PY')
            if (sigma < 0 or sigma >= 1):
                raise ValueError(f'provide sigma in [0, 1) for PY. You provided {sigma}')
            if not isinstance(scheme_param, (int, float)):
                raise TypeError('scheme param must be a float or int for PY')
            if not isinstance(scheme_param, (int, float)):
                raise TypeError('scheme param must be a float or int for PY')
            if scheme_param <= -sigma:
                raise ValueError(f'scheme param should be < -sigma for PY. You provided {scheme_param}')
            if sigma == 0:
                warnings.warn('note: for sigma=0 the PY reduces to DP, use scheme_type=DP for greater efficiency')
        
        if scheme_type == 'GN':
            if not isinstance(gamma, float):
                raise TypeError(f'gamma should be a float. You provided {type(gamma)}')
            if (gamma<=0 or gamma>= 1):
                raise ValueError(f'gamma for GN should be in (0, 1). You provided {gamma}')
        
        if not isinstance(bipartite, bool):
            raise TypeError(f'bipartite must be boolean. You provided {type(bipartite)}')

        if isinstance(likelihood, str):
            likelihood = likelihood.lower()
            if likelihood not in ['bernoulli']:
                raise NotImplementedError(f'likelihood string must be "bernoulli". You provided {likelihood}')
            else:
                likelihood = likelihood_dict[likelihood]()
        elif not isinstance(likelihood, BaseLikelihood):
            raise TypeError(f'likelihood must be a string or BaseLikelihood instance. You provided {type(likelihood)}')
            
        
        if clustering is not None:
            if isinstance(clustering, str):
                clustering = clustering.lower()
                if clustering != 'random':
                    raise ValueError(f'clustering string value must be "random". You provided {clustering}')
            
            elif isinstance(clustering, (np.ndarray, list)):
                clustering = np.array(clustering)
                if clustering.ndim == 1:
                    if bipartite is True:
                        raise ValueError('for bipartite networks clustering must be a tuple of two lists/arrays')
                    else:
                        if len(clustering) != num_nodes_1:
                            raise ValueError(f'clustering length must be equal to number of nodes. You provided {len(clustering)} but should be {num_nodes_1}')
                elif clustering.ndim == 2:
                    if bipartite is False:
                        raise ValueError('for unipartite networks clustering must be a single list/array')
                    else:
                        if clustering.shape[0] != num_nodes_1 and clustering.shape[0] != num_nodes_2:
                            raise ValueError(f'clustering shape must be equal to number of nodes. You provided {clustering.shape} but should be ({num_nodes_1}, ) and ({num_nodes_2}, )')
                else:
                    raise ValueError(f'clustering array must be 1D or 2D. You provided {clustering.ndim}D array')
            else:
                raise TypeError(f'clustering must be a string, list or array. You provided {type(clustering)}')

        if covariates_1 is not None:
            if not isinstance(covariates_1, (list)):
                raise TypeError('covariates for users must be provided as a list of tuples')
            for cov in covariates_1:
                if not isinstance(cov, tuple):
                    raise TypeError('each covariate for users must be provided as a tuple')
                if not isinstance(cov[0], str):
                    raise TypeError('covariate name and type for users must be provided as a string')
                if not isinstance(cov[1], (list, np.ndarray)):
                    raise TypeError('covariate values for users must be provided as a list or array')
                if len(cov[1]) != num_nodes_1:
                    raise ValueError(f'covariate length is {len(cov[1])} but should be {num_nodes_1}')
        
        if covariates_2 is not None:
            if not isinstance(covariates_2, (list)):
                raise TypeError('covariates for items must be provided as a list of tuples')
            for cov in covariates_2:
                if not isinstance(cov, tuple):
                    raise TypeError('each covariate for items must be provided as a tuple')
                if not isinstance(cov[0], str):
                    raise TypeError('covariate name and type for items must be provided as a string')
                if not isinstance(cov[1], (list, np.ndarray)):
                    raise TypeError('covariate values for items must be provided as a list or array')
                if len(cov[1]) != num_nodes_2:
                    raise ValueError(f'covariate length is {len(cov[1])} but should be {num_nodes_2}')
        
        if rng is not None:
            if not isinstance(rng, (np.random.Generator, int)):
                raise TypeError('rng must be a numpy random Generator or an integer seed')
    
    def _process_args(self, **kwargs):
        self.Y = kwargs.get('Y')
        self.bipartite = kwargs.get('bipartite')

        self.num_nodes_1, self.num_nodes_2 = self.Y.shape
        self.prior_a = kwargs.get('prior_a')
        self.prior_b = kwargs.get('prior_b')
        self.verbose = kwargs.get('verbose')
        self.epsilon = kwargs.get('epsilon')

        self.scheme_type = kwargs.get('scheme_type')
        self.scheme_param = kwargs.get('scheme_param')
        self.bar_h = kwargs.get('bar_h')
        self.gamma = kwargs.get('gamma')
        self.sigma = kwargs.get('sigma')
        self.degree_correction = kwargs.get('degree_correction')

        self.alpha_c = kwargs.get('alpha_c')
        self.alpha_0 = np.sum(np.array(self.alpha_c))
        self.covariates_1 = kwargs.get('covariates_1')
        self.covariates_2 = kwargs.get('covariates_2')

        self.train_llk = None
        self.mcmc_draws_users = None
        self.mcmc_draws_items = None
        
        self.estimated_items = None
        self.estimated_users = None
        
        self.estimated_theta = None
        
        self.cov_nch_1 = None
        self.cov_nch_2 = None
        
        
        rng = kwargs.get('rng')
        if rng is None:
            self.rng = np.random.default_rng()
        elif isinstance(rng, int):
            self.rng = np.random.default_rng(rng)
        else:
            self.rng = rng
        
        likelihood = kwargs.get('likelihood')
        if isinstance(likelihood, str):
            likelihood = likelihood.lower()
            likelihood = likelihood_dict[likelihood]
            likelihood = likelihood()
            
        # process covariates
        self.cov_names_1, self.cov_types_1, self.cov_values_1 = None, None, None
        if self.covariates_1 is not None:
            self.cov_names_1, self.cov_types_1, self.cov_values_1 = self._process_cov(self.covariates_1)

        self.cov_names_2, self.cov_types_2, self.cov_values_2 = None, None, None
        if self.covariates_2 is not None:
            self.cov_names_2, self.cov_types_2, self.cov_values_2 = self._process_cov(self.covariates_2)
        
        clustering = kwargs.get('clustering')
        # process clustering
        if clustering is not None:
            if isinstance(clustering, str):
                clustering = clustering.lower()
            if clustering == "random":
                if self.bipartite is True:
                    clustering_1 = self._init_cluster_random(num_nodes=self.num_nodes_1, 
                                                             cov_values=self.cov_values_1, 
                                                             cov_types=self.cov_types_1)
                    clustering_2 = self._init_cluster_random(num_nodes=self.num_nodes_2, 
                                                             cov_values=self.cov_values_2, 
                                                             cov_types=self.cov_types_2)
                else:
                    clustering_1 = self._init_cluster_random(num_nodes=self.num_nodes_1,
                                                            cov_values=self.cov_values_1, 
                                                            cov_types=self.cov_types_1) 

                    clustering_2 = None
            else:
                if self.bipartite is True:
                    clustering_1 = clustering[0]
                    clustering_2 = clustering[1]
                else:
                    clustering_1 = clustering
                    clustering_2 = None
        else:
            if self.bipartite is True:
                clustering_1 = np.arange(self.num_nodes_1)
                clustering_2 = np.arange(self.num_nodes_2)
            else:
                clustering_1 = np.arange(self.num_nodes_1)
                clustering_2 = None

        self._process_clusters(clustering=clustering_1, side=1)
        if self.bipartite is True:
            self._process_clusters(clustering=clustering_2, side=2)

        # if there are covs compute nch
        if self.covariates_1 is not None:
            self.cov_nch_users = self._compute_nch(self.cov_values_1, 
                                                   self.clustering_1, 
                                                   self.num_clusters_1)
            
        if self.covariates_2 is not None:
            self.cov_nch_items = self._compute_nch(self.cov_values_2, 
                                                   self.clustering_2, 
                                                   self.num_clusters_2)


    def _process_clusters(self, clustering, side=1):
        """Computes cluster metrics.

        Parameters
        ----------
        clustering : array-like
            Cluster assignments for users.

        """
        occupied_clusters, out_frequencies = np.unique(clustering, return_counts=True)
        out_num_clusters = len(occupied_clusters)
        out_clustering = np.array(clustering)
        
        setattr(self, f"clustering_{side}", out_clustering)
        setattr(self, f"num_clusters_{side}", out_num_clusters)
        setattr(self, f"frequencies_{side}", out_frequencies)

        return

    def _init_cluster_random(self, num_nodes, cov_values=None, cov_types=None):
        """Initialises random clustering structure according to the prior.

        Parameters
        ----------
        clustering : array-like, optional
            Initial clustering structure. If 'random' random initialisation is performed, 
            by default None
        cov_values : list, optional
            list of covariate values, by default None
        cov_types : list, optional
            list of covariate types, by default None

        Returns
        -------
        clustering : array-like
            Final clustering.
        """
        
        clustering = [0]
        num_clusters = 1
        current_num_nodes = 1
        frequencies = [1]

        if self.verbose is True:
            print('initialsing user clusters random')

        nch = None
        if cov_values is not None:
            nch = self._compute_nch(cov_values, clustering, num_clusters)

        # sequential assignment of clusters
        for i in range(1, num_nodes):
            # prior contribution
            prior_probs = sampling_scheme(V=current_num_nodes,
                                    H=num_clusters,
                                    frequencies=np.array(frequencies),
                                    bar_h=self.bar_h,
                                    scheme_type=self.scheme_dict[self.scheme_type],
                                    scheme_param=self.scheme_param, 
                                    sigma=self.sigma, 
                                    gamma=self.gamma)
            
            logits_cov = 0
            if nch is not None:
                logits_cov = compute_log_probs_cov(probs=probs, 
                                                      idx=i, 
                                                      cov_types=cov_types, 
                                                      cov_nch=nch, 
                                                      cov_values=cov_values, 
                                                      nh=np.array(frequencies), 
                                                      alpha_c=self.alpha_c, 
                                                      alpha_0=self.alpha_0)

            # convert back using exp and normalise
            logits = np.log(prior_probs+self.epsilon)+logits_cov
            logits = logits - max(logits)
            probs = np.exp(logits)
            probs = probs/probs.sum()

            assignment = self.rng.choice(len(probs), p=probs)
            if assignment >= num_clusters:
                # make new cluster
                num_clusters += 1
                frequencies.append(1)
                
                if nch is not None:
                    for cov in range(len(cov_values)):
                        n_unique = len(np.unique(cov_values[cov]))
                        temp = np.zeros(n_unique)
                        c = cov_values[cov][i]
                        temp[c] += 1
                        nch[cov] = np.column_stack([nch[cov], temp.reshape(-1, 1)])
            else:
                frequencies[assignment] += 1
                if nch is not None:
                    for cov in range(len(cov_values)):
                        c = cov_values[cov][i]
                        nch[cov][c, assignment] += 1

            clustering.append(assignment)
            current_num_nodes += 1

        clustering = np.array(clustering)

        # safety check
        assert current_num_nodes == num_nodes
        assert len(clustering) == num_nodes
        assert len(np.unique(clustering)) == num_clusters
        
        return clustering, frequencies, num_clusters

    def _compute_mhk(self, clustering_1=None, clustering_2=None):
        """Computes the MHK matrix using (fast) sparse matrix multiplication.

        Parameters
        ----------
        clustering_1 : list, optional
            First dimension clustering, if None uses self.clustering_1. By default None
        clustering_2 : list, optional
            Second dimension clustering, if None uses self.clustering_2. By default None

        Returns
        -------
        mhk : np.array
            MHK matrix
        """
        
        if clustering_1 is None:
            clustering_1 = self.clustering_1
            num_nodes_1 = self.num_nodes_1
            num_clusters_1 = self.num_clusters_1
        else:
            num_nodes_1 = len(clustering_1)
            num_clusters_1 = len(np.unique(clustering_1))

        if clustering_2 is None:
            clustering_2 = self.clustering_2
            num_nodes_2 = self.num_nodes_2
            num_clusters_2 = self.num_clusters_2
        else:
            num_nodes_2 = len(clustering_2)
            num_clusters_2 = len(np.unique(clustering_2))

        # using sparse matrices for speed, this sums up entries of
        # Y in depending on their block assignment.
        # m[h,k] is the sum of entries for blocks (h, k)
        clusters_1 = sparse.csr_matrix(
            (np.ones(num_nodes_1),
            (range(num_nodes_1),
            clustering_1)),
            shape=(num_nodes_1, num_clusters_1))

        clusters_2 = sparse.csr_matrix(
            (np.ones(num_nodes_2),
            (range(num_nodes_2),
            clustering_2)),
            shape=(num_nodes_2, num_clusters_2))

        mhk = clusters_1.T @ self.Y @ clusters_2
        return mhk
     
    def _compute_yuk(self):
        """Computes the YUK matrix.

        Returns
        -------
        yuk : np.array
            YUK matrix
        """
        # using sparse matrices for speed, this sums up entries of
        # Y in depending on their block assignment.
        # y[u,k] is the sum of entries for user u and cluster k (in items)
        item_clusters = sparse.csr_matrix(
            (np.ones(self.num_items),
            (range(self.num_items),
            self.item_clustering)),
            shape=(self.num_items, self.num_clusters_items))
        
        yuk = self.Y @ item_clusters
        return yuk
    
    def _compute_yih(self):
        """Computes the YIH matrix.

        Returns
        -------
        yih : np.array
            YIH matrix
        """
        # using sparse matrices for speed, this sums up entries of
        # Y in depending on their block assignment.
        # y[i,h] is the sum of entries for item i and cluster h (in users)
        user_clusters = sparse.csr_matrix(
            (np.ones(self.num_users),
            (range(self.num_users),
            self.user_clustering)),
            shape=(self.num_users, self.num_clusters_users))

        yih = self.Y.T @ user_clusters
        return yih
    
    
    def _process_cov(self, cov_list):
        """Processes a list of covariates.

        Parameters
        ----------
        cov_list : list of tuples
            list of tuples (covname_covtype, covvalues)

        Returns
        -------
        tuple: (cov_names, cov_types, cov_values)
            cov_names: list of covariate names
            cov_types: list of covariate types
            cov_values: list of covariate values
        """
        cov_names = []
        cov_types = []
        cov_values = []
        for cov in cov_list:
            cov_name, cov_type = cov[0].split('_')
            cov_names.append(cov_name)
            cov_types.append(cov_type)
            cov_values.append(np.array(cov[1]))  

        if isinstance(self.alpha_c, (int, float)):
            temp = []
            for i in range(len(cov_names)):
                unique_cov_values = len(np.unique(cov_values[i]))
                temp.extend([self.alpha_c for _ in range(unique_cov_values)])
                
            self.alpha_c = np.array(temp)
            self.alpha_0 = np.sum(self.alpha_c)
            
        return np.array(cov_names), np.array(cov_types), np.array(cov_values)
     
    def _compute_nch(self, cov_values, clustering, n_clusters):
        """Computes the NCH matrix.

        Parameters
        ----------
        cov_values : list
            list of covariate values
        clustering : list
            list of cluster assignments
        n_clusters : int
            number of clusters

        Returns
        -------
        list
            list of NCH matrices
        """
        cov_nch = []
        for cov in range(len(cov_values)):
            uniques = np.unique(cov_values[cov])
            nch = np.zeros((len(uniques), n_clusters))
            for h in range(n_clusters):
                mask = (clustering==h)
                for c in uniques:
                    nch[c, h] = (cov_values[cov][mask]==c).sum()    
            cov_nch.append(nch)
        return np.array(cov_nch)
    
    def gibbs_step(self):
        """Performs a Gibbs sampling step."""
        # do nothing for baseline
        return
    
    def compute_log_likelihood(self):
        """Computes the log-likelihood of the model.

        Returns
        -------
        float
            Log-likelihood value
        """
        ll = self.likelihood.compute_log_likelihood(
            frequencies_1=self.frequencies_users,
            frequencies_2=self.frequencies_items,
            mhk=self._compute_mhk(),
            clustering_1=self.clustering_1,
            clustering_2=self.clustering_2,
            degree_correction=self.degree_correction
        )
        
        return ll
    
    def fit(self, n_iters, verbose=0):
        """Trains the model using Gibbs sampling.

        Parameters
        ----------
        n_iters : int
            Number of iterations for Gibbs sampling.
        verbose : int, optional
            Verbosity level, by default 0. 0: no output, 1: every 10% of iterations,
            2: also print frequencies, 3: also print cluster assignments

        Returns
        -------
        tuple: (llks, mcmc_draws_1, mcmc_draws_2)
            llks : np.ndarray
                Log-likelihood values at each iteration
            mcmc_draws_1 : np.ndarray
                MCMC draws for cluster assignments
            mcmc_draws_2 : np.ndarray
                MCMC draws for second dimension (if bipartite) cluster assignments
        """
        if not isinstance(n_iters, int) or n_iters <= 0:
            raise ValueError('n_iters must be a positive integer')
        
        self.n_iters = n_iters
        
        ll = self.compute_log_likelihood()
        
        if verbose>0:
            print('starting log likelihood', ll)
        llks = np.zeros(n_iters+1)
        
        mcmc_draws_1 = np.zeros((n_iters+1, self.num_users), dtype=np.int32)
        mcmc_draws_2 = np.zeros((n_iters+1, self.num_items), dtype=np.int32)
        
        mcmc_frequencies_1 = []
        mcmc_frequencies_2 = []
        
        llks[0] = ll
        mcmc_draws_1[0] = self.user_clustering.copy()
        mcmc_draws_2[0] = self.item_clustering.copy()
        mcmc_frequencies_1.append(self.frequencies_1.copy())
        mcmc_frequencies_2.append(self.frequencies_2.copy())

        check = time.perf_counter()
        for it in range(n_iters):
    
            self.gibbs_step()
            ll = self.compute_log_likelihood()
            
            llks[it+1] += ll
            mcmc_draws_1[it+1] += self.clustering_1
            mcmc_draws_2[it+1] += self.clustering_2
            mcmc_frequencies_1.append(self.frequencies_1.copy())
            mcmc_frequencies_2.append(self.frequencies_2.copy())

            if verbose >= 1:
                if it % (n_iters // 10) == 0:
                    print(it, llks[it+1])
                    print('time', time.perf_counter()-check)
                    check = time.perf_counter()
                if verbose >= 2:
                    print('user freq ', self.frequencies_1)
                    if self.bipartite is True:
                        print('item freq ', self.frequencies_2)
                    if verbose >= 3:
                        print('user cluster ', self.clustering_1)
                        if self.bipartite is True:
                            print('item cluster ', self.clustering_2)

        if verbose>0:
            print('end llk: ', llks[-1])
        
        self.train_llk = llks
        self.mcmc_draws_1 = mcmc_draws_1
        self.mcmc_draws_2 = mcmc_draws_2
        self.mcmc_draws_1_frequencies = mcmc_frequencies_1
        self.mcmc_draws_2_frequencies = mcmc_frequencies_2

        return llks, mcmc_draws_1, mcmc_draws_2
    
    def pred_cluster():
        pass
    
    def estimate_edge_probabilities(self, burn_in=0, thinning=1):
        pass
    
    def edge_llk(self):
        pass


    def estimate_cluster_assignment_mode(self, burn_in = 0, thinning = 1):
        """Estimate cluster assignments using the mode.

        Parameters
        ----------
        burn_in : int, optional
            Number of initial samples to discard, by default 0
        thinning : int, optional
            Thinning factor for MCMC samples, by default 1

        Returns
        -------
        tuple (user_cluster_assignments, item_cluster_assignments)
            user_cluster_assignments : np.ndarray
                Estimated cluster assignments for users
            item_cluster_assignments : np.ndarray
                Estimated cluster assignments for items

        Raises
        ------
        Exception
            If the model has not been trained.
        """
        
        if self.mcmc_draws_1 is None:
            raise Exception('model must be trained first')
        
        assignment_1 = -np.ones(self.num_nodes_1, dtype=np.int64)
        for u in range(self.num_nodes_1):
            assignment_1[u] = int(mode(self.mcmc_draws_1[burn_in::thinning, u])[0])
        
        self.clustering_1[:] = assignment_1
        _, frequencies_1 = np.unique(assignment_1, return_counts=True)
        self.frequencies_1 = frequencies_1
        
        # store estimation method
        self.estimation_method = 'mode'
        
        if self.bipartite is False:
            return assignment_1

        assignment_2 = -np.ones(self.num_nodes_2, dtype=np.int64)
        for i in range(self.num_nodes_2):
            assignment_2[i] = int(mode(self.mcmc_draws_2[burn_in::thinning, i])[0])
                
        self.clustering_2[:] = assignment_2
        _, frequencies_2 = np.unique(assignment_2, return_counts=True)
        self.frequencies_2 = frequencies_2

        return assignment_1, assignment_2
    
    def estimate_cluster_assignment_vi(self, method='avg', max_k=None, burn_in=0, thinning=1):
        """Estimate cluster assignments minimizing the variation of information.
        
        Uses a greedy algorithm as described in Wade and Ghahramani (2018))

        Parameters
        ----------
        method : str, optional
            Estimation method to use, by default 'avg'
        max_k : int, optional
            Maximum number of clusters to consider for greedy optimization, by default int(np.ceil(psm.shape[0] / 8))
        burn_in : int, optional
            Number of initial samples to discard, by default 0
        thinning : int, optional
            Thinning factor for MCMC samples, by default 1

        Returns
        -------
        tuple (user_cluster_assignments, item_cluster_assignments, vi_value_users, vi_value_items)
            user_cluster_assignments : np.ndarray
                Estimated cluster assignments for users
            item_cluster_assignments : np.ndarray
                Estimated cluster assignments for items
            vi_value_users : float
                Variation of information value for user clustering
            vi_value_items : float
                Variation of information value for item clustering

        Raises
        ------
        Exception
            If the model has not been trained.
        """
        
        if method not in ['avg', 'comp', 'all']:
            raise Exception('invalid method')
        
        cc_matrix_1, cc_matrix_2 = None, None
        est_cluster_1, est_cluster_2 = None, None
        vi_value_1, vi_value_2 = None, None
        
        cc_matrix_1 = self._compute_co_clustering_matrix(mcmc_draws=self.mcmc_draws_1,
                                                         burn_in=burn_in, 
                                                         thinning=thinning)
            
        psm_1 = cc_matrix_1/np.max(cc_matrix_1)

        res_side_1 = minVI(psm_1, 
                           cls_draw = self.mcmc_draws_1[burn_in::thinning], 
                           method=method, 
                           max_k=max_k)
        
        est_cluster_1 = res_side_1['cl']
        vi_value_1 = res_side_1['value']
        
        self.clustering_1[:] = est_cluster_1
        unique_users, frequencies_users = np.unique(est_cluster_1, return_counts=True)
        self.frequencies_1 = frequencies_users
        self.num_clusters_1 = len(unique_users)
        
        self.estimation_method = 'vi'
        
        if self.bipartite is False:
            return est_cluster_1, vi_value_1            

        # repeat for other side if bipartite
        cc_matrix_2 = self._compute_co_clustering_matrix(mcmc_draws=self.mcmc_draws_2,
                                                        burn_in=burn_in, 
                                                             thinning=thinning)
        psm_items = cc_matrix_2/np.max(cc_matrix_2)

        res_side_2 = minVI(psm_items, 
                           cls_draw = self.mcmc_draws_2[burn_in::thinning],
                           method=method, max_k=max_k)
        est_cluster_2 = res_side_2['cl']
        vi_value_2 = res_side_2['value']

        self.clustering_2[:] = est_cluster_2
        unique_items, frequencies_items = np.unique(est_cluster_2, return_counts=True)
        self.frequencies_2 = frequencies_items
        self.num_clusters_2 = len(unique_items)
                
        return est_cluster_1, vi_value_1, est_cluster_2, vi_value_2
        
    def _compute_co_clustering_matrix(self, mcmc_draws, burn_in=0, thinning=1):
        """Aux function to call the optimised function on relevant sample"""
        if self.mcmc_draws_users is None:
            raise Exception('model must be trained first')

        cc_matrix = compute_co_clustering_matrix(mcmc_draws[burn_in::thinning])
        
        return cc_matrix
    
    def point_predict(self, pairs, seed=None):
        """Predict ratings for user-item pairs.

        Parameters
        ----------
        pairs : list of tuples
            List of (user, item) pairs for which to predict ratings.
        seed : int, optional
            Random seed for reproducibility, by default None

        Returns
        -------
        preds : list
            List of predicted ratings corresponding to the input pairs.
        """
        
        if not isinstance(pairs, list):
            raise TypeError('pairs must be a list of tuples')
        for p in pairs:
            if not isinstance(p, tuple) or len(p) != 2:
                raise ValueError('each pair must be a tuple of (user, item)')
            u, i = p
            if not (0 <= u < self.num_users):
                raise ValueError(f'user index {u} out of bounds')
            if not (0 <= i < self.num_items):
                raise ValueError(f'item index {i} out of bounds')
        
        if seed is None:
            np.random.seed(self.seed)
        elif seed == -1:
            pass
        else:
            np.random.seed(seed)
        
        uniques_ratings, frequencies_ratings = np.unique(self.Y, return_counts=True)
        
        preds = []
        for u, i in pairs:
            # baseline: predict with predictive posterior mean
            preds.append(self.rng.choice(uniques_ratings, 
                                          p=frequencies_ratings/np.sum(frequencies_ratings)))

        return preds
    
    
    def predict_with_ranking(self, users):
        """Predict items for users based on cluster with highest score.

        Parameters
        ----------
        users : list
            List of users for whom to predict items.

        Returns
        -------
        list
            List of predicted item indices for each user.
        """
        top_cluster =[]
        for u in users:
            # baseline: completely random
            num = np.random.randint(1, self.num_items)
            choice = np.random.choice(self.num_items, num, replace=False)
            top_cluster.append(choice)

        return top_cluster
    
    

    def predict_k(self, users, k):
        """Predict k items for each user based on cluster with highest score.

        Parameters
        ----------
        users : list
            List of users for whom to predict items.
        k : int
            Number of items to predict for each user.

        Returns
        -------
        list
            List of predicted item indices for each user.
        """
        if not isinstance(users, (list, np.ndarray, int)):
            raise TypeError('users must be a a user index or a list/array of user indices')
        if not isinstance(users, (list, np.ndarray)):
            users = [users]
        for u in users:
            if not (0 <= u < self.num_users):
                raise ValueError(f'user index {u} out of bounds')
        
        if not isinstance(k, int) or k <= 0:
            raise ValueError('k must be a positive integer')
        
        out = []
        for u in users:
            # baseline: completely random
            choice = np.random.choice(self.num_items, k, replace=False)
            out.append(choice)

        return out
    