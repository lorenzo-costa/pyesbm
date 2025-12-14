import time
import numpy as np
from scipy.stats import mode

from pyesbm.utilities import (
    compute_co_clustering_matrix,
    minVI,
    compute_mhk,
    compute_y_values,
    ClusterProcessor,
    waic_calculation,
)

from pyesbm.esbm_config import ESBMconfig

from pyesbm.utilities.plotting_functions import plot_trace

from pyesbm.utilities.vi_functs import credibleball


#########################################
# baseline class
########################################
class BaseESBM(ESBMconfig):
    """Baseline ESBM model

    Parameters
    ----------
    Y : 2D array
        adjacency matrix (if None automatically generated), by default None
    likelihood : BaseLikelihood object
        likelihood object
    prior : BasePrior object
        prior object
    bipartite : bool
        whether the network is bipartite, by default True
    clustering : list or array-like
        initial clustering assignments, by default None
    covariates_1 : CovariateClass
        covariate object for first dimension, by default None
    covariates_2 : CovariateClass
        covariate object for second dimension, by default None
    verbose : bool
        whether to print verbose output for user-related computations, by default False
    rng : np.random.Generator or int, optional
        random number generator or seed, by default None

    Attributes
    ----------
    num_nodes_1 : int
        number of nodes in first dimension
    num_nodes_2 : int
        number of nodes in second dimension
    num_clusters_1 : int
        number of clusters in first dimension
    num_clusters_2 : int
        number of clusters in second dimension
    clustering_1 : np.ndarray
        cluster assignments for first dimension
    clustering_2 : np.ndarray
        cluster assignments for second dimension
    frequencies_1 : np.ndarray
        frequencies for first dimension
    frequencies_2 : np.ndarray
        frequencies for second dimension
    train_llk : np.ndarray
        log-likelihood values during training
    estimation_method : str
        method used for cluster assignment estimation
    """

    def __init__(
        self,
        Y,
        likelihood,
        prior,
        *,
        bipartite=True,
        clustering=None,
        covariates_1=None,
        covariates_2=None,
        epsilon=1e-6,
        rng=None,
        verbose=False,
    ):
        super().__init__(
            Y=Y,
            likelihood=likelihood,
            prior=prior,
            bipartite=bipartite,
            clustering=clustering,
            covariates_1=covariates_1,
            covariates_2=covariates_2,
            rng=rng,
            epsilon=epsilon,
            verbose=verbose,
        )

        if rng is None:
            self.rng = np.random.default_rng()
        elif isinstance(rng, int):
            self.rng = np.random.default_rng(rng)
        else:
            self.rng = rng

        if isinstance(clustering, (list, np.ndarray)):
            if self.bipartite is True:
                clustering_1 = np.array(clustering[0])
                clustering_2 = np.array(clustering[1])
            else:
                clustering_1 = np.array(clustering)
                clustering_2 = clustering_1.copy()
        else:
            clustering_1 = clustering
            clustering_2 = clustering

        cluster_init_1 = ClusterProcessor(
            self.num_nodes_1,
            clustering_1,
            self.prior,
            covariates=self.covariates_1,
            verbose=self.verbose,
            epsilon=self.epsilon,
            rng=self.rng,
        )

        cluster_init_1 = cluster_init_1.clustering
        self._process_clusters(clustering=cluster_init_1, side=1)

        if self.bipartite is True:
            cluster_init_2 = ClusterProcessor(
                self.num_nodes_2,
                clustering_2,
                self.prior,
                covariates=self.covariates_2,
                verbose=self.verbose,
                epsilon=self.epsilon,
                rng=self.rng,
            )

            cluster_init_2 = cluster_init_2.clustering
            self._process_clusters(clustering=cluster_init_2, side=2)
        else:
            self._process_clusters(clustering=cluster_init_1, side=2)

        # if there are covs compute nch
        if self.covariates_1 is not None:
            self.covariates_1.get_nch(
                clustering=self.clustering_1, num_clusters=self.num_clusters_1
            )

        if self.covariates_2 is not None:
            self.covariates_2.get_nch(
                clustering=self.clustering_2, num_clusters=self.num_clusters_2
            )

        self.estimation_method = None

    def gibbs_step(self, side=1):
        """Performs a single Gibbs sampling step for the specified side."""

        if side not in [1, 2]:
            raise ValueError("side must be 1 or 2")

        frequencies = self.frequencies_1 if side == 1 else self.frequencies_2
        frequencies_other_side = self.frequencies_2 if side == 1 else self.frequencies_1

        num_clusters = self.num_clusters_1 if side == 1 else self.num_clusters_2
        num_clusters_other_side = (
            self.num_clusters_2 if side == 1 else self.num_clusters_1
        )

        num_nodes = self.num_nodes_1 if side == 1 else self.num_nodes_2
        num_nodes_other_side = self.num_nodes_2 if side == 1 else self.num_nodes_1

        clustering = self.clustering_1 if side == 1 else self.clustering_2
        clustering_other_side = self.clustering_2 if side == 1 else self.clustering_1

        covariates = self.covariates_1 if side == 1 else self.covariates_2

        # use a dict to keep track of which quantities to pass
        computed_quantities = {
            "frequencies": frequencies,
            "frequencies_other_side": frequencies_other_side,
            "num_clusters": num_clusters,
            "num_clusters_other_side": num_clusters_other_side,
            "num_nodes": num_nodes,
            "num_nodes_other_side": num_nodes_other_side,
            "clustering": clustering,
            "clustering_other_side": clustering_other_side,
            "covariates": covariates,
            "nch": covariates.get_nch(clustering=clustering, num_clusters=num_clusters)
            if covariates is not None
            else None,
            "mhk": None,
            "side": side,
            "bipartite": self.bipartite,
        }

        for i in range(num_nodes):
            if self.verbose is True:
                print(f"\nProcessing node {i} on side {side}")
                print(
                    f"frequencies before removal: {computed_quantities['frequencies']}"
                )
                print(
                    f"num_clusters before removal: {computed_quantities['num_clusters']}"
                )
                print(f"clustering before removal: {computed_quantities['clustering']}")
                print(f"nch before removal: {computed_quantities['nch']}")
                print(f"mhk shape before removal: {computed_quantities['mhk'].shape}")

            computed_quantities["node_idx"] = i
            computed_quantities["current_cluster"] = clustering[i]

            self._remove_node_from_cluster(computed_quantities)

            # prior contribution

            prior_probs = self.prior.compute_probs(
                model=self,  # pass whole model
                # pass all quantities (prior will pick only those needed)
                **computed_quantities,
            )

            # likelihood contribution
            llk_logits = self.likelihood.update_logits(
                num_components=len(prior_probs),  # probs may have added one cluster
                model=self,  # likelihoood has access to entire model info
                **computed_quantities,  # likelihood has access to quantities computed in remove_node
            )

            # covariate contribution
            cov_logits = 0
            if covariates is not None:
                cov_logits = covariates.compute_logits(
                    model=self, num_components=len(prior_probs), **computed_quantities
                )

            logits = np.log(prior_probs + self.epsilon) + llk_logits + cov_logits
            probs = np.exp(logits - np.max(logits))
            probs = probs / np.sum(probs)

            assignment = self.rng.choice(len(probs), p=probs)

            if self.verbose is True:
                print(
                    f"Node {i} on side {side} assigned to cluster {assignment}, {computed_quantities['num_clusters']} clusters now"
                )

            self._add_to_cluster(assignment, computed_quantities)

        clustering = computed_quantities["clustering"]
        num_clusters = computed_quantities["num_clusters"]
        frequencies = computed_quantities["frequencies"]
        nch = computed_quantities["nch"]

        if nch is not None:
            covariates.nch = nch

        setattr(self, f"clustering_{side}", clustering)
        setattr(self, f"frequencies_{side}", frequencies)
        setattr(self, f"num_clusters_{side}", num_clusters)

        return computed_quantities

    def compute_log_likelihood(self):
        """Computes the log-likelihood of the model.

        Returns
        -------
        float
            Log-likelihood value
        """

        mhk = compute_mhk(self.Y, self.clustering_1, self.clustering_2)

        ll = self.likelihood.compute_llk(
            frequencies=self.frequencies_1,
            frequencies_other_side=self.frequencies_2,
            mhk=mhk,
            clustering=self.clustering_1,
            clustering_other_side=self.clustering_2,
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
            raise ValueError("n_iters must be a positive integer")

        self.n_iters = n_iters

        ll = self.compute_log_likelihood()

        if verbose > 0:
            print("starting log likelihood", ll)

        llks = np.zeros(n_iters + 1)
        mcmc_draws_1 = np.zeros((n_iters + 1, self.num_nodes_1), dtype=np.int32)
        mcmc_draws_2 = np.zeros((n_iters + 1, self.num_nodes_2), dtype=np.int32)

        mcmc_frequencies_list_1 = []
        mcmc_frequencies_list_2 = []

        llks[0] = ll
        mcmc_draws_1[0] = self.clustering_1.copy()
        mcmc_draws_2[0] = self.clustering_2.copy()

        mcmc_frequencies_list_1.append(self.frequencies_1.copy())
        mcmc_frequencies_list_2.append(self.frequencies_2.copy())

        check = time.perf_counter()
        for it in range(n_iters):
            out = self.gibbs_step(side=1)
            if self.bipartite is True:
                self.gibbs_step(side=2)
            else:
                self.clustering_2 = self.clustering_1.copy()
                self.frequencies_2 = self.frequencies_1.copy()
                self.num_clusters_2 = self.num_clusters_1
            ll = self.compute_log_likelihood()

            llks[it + 1] += ll
            mcmc_draws_1[it + 1] += self.clustering_1
            mcmc_draws_2[it + 1] += self.clustering_2
            mcmc_frequencies_list_1.append(self.frequencies_1.copy())
            mcmc_frequencies_list_2.append(self.frequencies_2.copy())

            if verbose >= 1:
                if it % (n_iters // 10) == 0:
                    print(it, llks[it + 1])
                    print("time", time.perf_counter() - check)
                    check = time.perf_counter()
                if verbose >= 2:
                    print("user freq ", self.frequencies_1)
                    if self.bipartite is True:
                        print("item freq ", self.frequencies_2)
                    if verbose >= 3:
                        print("user cluster ", self.clustering_1)
                        if self.bipartite is True:
                            print("item cluster ", self.clustering_2)

        if verbose > 0:
            print("end llk: ", llks[-1])

        self.train_llk = llks
        self.mcmc_draws_1 = mcmc_draws_1
        self.mcmc_draws_2 = mcmc_draws_2
        self.mcmc_draws_1_frequencies = mcmc_frequencies_list_1
        self.mcmc_draws_2_frequencies = mcmc_frequencies_list_2

        return llks, mcmc_draws_1, mcmc_draws_2

    def compute_waic(self, burn_in=0, thinning=1):
        """Computes the WAIC for the model.

        Parameters
        ----------
        burn_in : int, optional
            Number of initial samples to discard, by default 0
        thinning : int, optional
            Thinning factor for MCMC samples, by default 1

        Returns
        -------
        float
            WAIC value
        """
        if self.mcmc_draws_1 is None:
            raise Exception("model must be trained first")

        num_iters = (self.n_iters - burn_in) // thinning + 1
        llk_array = np.zeros((self.Y.shape[0] * self.Y.shape[1], num_iters))
        for it in range(burn_in, self.n_iters, thinning):
            cl1 = self.mcmc_draws_1[it]
            cl2 = self.mcmc_draws_2[it]
            fr1 = self.mcmc_draws_1_frequencies[it]
            fr2 = self.mcmc_draws_2_frequencies[it]
            mhk = compute_mhk(self.Y, cl1, cl2)
            llk_array[:, it - 1] = self.likelihood.sample_llk_edges(
                Y=self.Y,
                mhk=mhk,
                frequencies_1=fr1,
                frequencies_2=fr2,
                clustering_1=cl1,
                clustering_2=cl2,
                bipartite=self.bipartite,
                rng=self.rng,
            )

        waic_value = waic_calculation(llk_array)

        return waic_value

    def pred_cluster(self):
        pass

    def estimate_edge_probabilities(self, burn_in=0, thinning=1):
        pass

    def edge_llk(self):
        pass

    def estimate_cluster_assignment_mode(self, burn_in=0, thinning=1):
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
            raise Exception("model must be trained first")

        assignment_1 = -np.ones(self.num_nodes_1, dtype=np.int64)
        for u in range(self.num_nodes_1):
            assignment_1[u] = int(mode(self.mcmc_draws_1[burn_in::thinning, u])[0])

        self.clustering_1[:] = assignment_1
        _, frequencies_1 = np.unique(assignment_1, return_counts=True)
        self.frequencies_1 = frequencies_1

        # store estimation method
        self.estimation_method = "mode"

        if self.bipartite is False:
            return assignment_1

        assignment_2 = -np.ones(self.num_nodes_2, dtype=np.int64)
        for i in range(self.num_nodes_2):
            assignment_2[i] = int(mode(self.mcmc_draws_2[burn_in::thinning, i])[0])

        self.clustering_2[:] = assignment_2
        _, frequencies_2 = np.unique(assignment_2, return_counts=True)
        self.frequencies_2 = frequencies_2

        return assignment_1, assignment_2

    def estimate_cluster_assignment_vi(
        self, method="avg", max_k=None, burn_in=0, thinning=1
    ):
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

        if method not in ["avg", "comp", "all"]:
            raise Exception("invalid method")
        if self.mcmc_draws_1 is None:
            raise Exception("model must be trained first")

        cc_matrix_1, cc_matrix_2 = None, None
        est_cluster_1, est_cluster_2 = None, None
        vi_value_1, vi_value_2 = None, None

        cc_matrix_1 = compute_co_clustering_matrix(self.mcmc_draws_1[burn_in::thinning])
        psm_1 = cc_matrix_1 / np.max(cc_matrix_1)

        res_side_1 = minVI(
            psm_1,
            cls_draw=self.mcmc_draws_1[burn_in::thinning],
            method=method,
            max_k=max_k,
        )

        est_cluster_1 = res_side_1["cl"]
        vi_value_1 = res_side_1["value"]

        self.clustering_1[:] = est_cluster_1
        unique_users, frequencies_users = np.unique(est_cluster_1, return_counts=True)
        self.frequencies_1 = frequencies_users
        self.num_clusters_1 = len(unique_users)

        self.estimation_method = "vi"

        if self.bipartite is True:
            # repeat for other side if bipartite
            cc_matrix_2 = compute_co_clustering_matrix(
                self.mcmc_draws_2[burn_in::thinning]
            )
            psm_items = cc_matrix_2 / np.max(cc_matrix_2)

            res_side_2 = minVI(
                psm_items,
                cls_draw=self.mcmc_draws_2[burn_in::thinning],
                method=method,
                max_k=max_k,
            )
            est_cluster_2 = res_side_2["cl"]
            vi_value_2 = res_side_2["value"]

            self.clustering_2[:] = est_cluster_2
            unique_items, frequencies_items = np.unique(
                est_cluster_2, return_counts=True
            )
            self.frequencies_2 = frequencies_items
            self.num_clusters_2 = len(unique_items)
        else:
            est_cluster_2 = est_cluster_1.copy()
            vi_value_2 = vi_value_1

            self.clustering_2[:] = est_cluster_2
            unique_items, frequencies_items = np.unique(
                est_cluster_2, return_counts=True
            )
            self.frequencies_2 = frequencies_items
            self.num_clusters_2 = len(unique_items)

        return est_cluster_1, vi_value_1, est_cluster_2, vi_value_2

    def plot_trace(
        self, title="Log-likelihood Trace", start=0, save_path=None, figsize=(6, 4)
    ):
        """Plot trace of log-likelihood during training.

        Parameters
        ----------
        start : int, optional
            Starting iteration for plotting, by default 0
        save_path : str, optional
            Path to save the plot, by default None
        figsize : tuple, optional
            Figure size for the plot, by default (6, 4)
        """

        if self.train_llk is None:
            raise Exception("model must be trained first")
        plot_trace(
            self.train_llk[start:],
            save_path=save_path,
            figsize=figsize,
            title=title,
            xlabel="Iteration",
            ylabel="Log-likelihood",
        )

    def point_predict(self, pairs, rng=None):
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
        if self.estimation_method is None:
            raise Exception("cluster assignments must be estimated before prediction")

        mhk = compute_mhk(self.Y, self.clustering_1, self.clustering_2)
        preds = self.likelihood.point_predict(
            mhk=mhk,
            frequencies_1=self.frequencies_1,
            frequencies_2=self.frequencies_2,
            clustering_1=self.clustering_1,
            clustering_2=self.clustering_2,
            pairs=pairs,
        )
        return preds

    def compute_llk_edges(self, iter=None):
        """Compute log-likelihood for edges in the graph.

        Parameters
        ----------
        iter : int, optional
            MCMC iteration to use for computation, by default None

        Returns
        -------
        llk_edges : np.ndarray
            Log-likelihoods for the edges in the graph.
        """
        if iter is not None:
            if self.mcmc_draws_1 is None:
                raise Exception("model must be trained first")
            clustering_1 = self.mcmc_draws_1[iter]
            clustering_2 = self.mcmc_draws_2[iter]
            frequencies_1 = self.mcmc_draws_1_frequencies[iter]
            frequencies_2 = self.mcmc_draws_2_frequencies[iter]
        else:
            clustering_1 = self.clustering_1
            clustering_2 = self.clustering_2
            frequencies_1 = self.frequencies_1
            frequencies_2 = self.frequencies_2

        mhk = compute_mhk(self.Y, clustering_1, clustering_2)
        llk_edges = self.likelihood.sample_llk_edges(
            Y=self.Y,
            mhk=mhk,
            frequencies_1=frequencies_1,
            frequencies_2=frequencies_2,
            clustering_1=clustering_1,
            clustering_2=clustering_2,
            bipartite=self.bipartite,
            rng=self.rng,
        )
        return llk_edges

    def credible_ball(self, burn_in=0, thinning=1, alpha=0.05):
        """Compute credible ball for cluster assignments.

        Parameters
        ----------
        burn_in : int, optional
            Number of initial samples to discard, by default 0
        thinning : int, optional
            Thinning factor for MCMC samples, by default 1
        alpha : float, optional
            Significance level for credible ball, by default 0.05

        Returns
        -------
        dict
            Dictionary containing credible ball information
        """
        if self.mcmc_draws_1 is None:
            raise Exception("model must be trained first")

        if self.estimation_method != "vi":
            raise Exception(
                "cluster assignments must be estimated using VI before computing credible ball"
            )

        estimated_clusters_1 = self.clustering_1
        estimated_clusters_2 = self.clustering_2

        cc_matrix_1 = compute_co_clustering_matrix(self.mcmc_draws_1[burn_in::thinning])
        cc_matrix_2 = compute_co_clustering_matrix(self.mcmc_draws_2[burn_in::thinning])

        cb1 = credibleball(
            c_star=estimated_clusters_1, cls_draw=cc_matrix_1, c_dist="VI", alpha=alpha
        )

        cb2 = credibleball(
            c_star=estimated_clusters_2, cls_draw=cc_matrix_2, c_dist="VI", alpha=alpha
        )

        return cb1, cb2

    def _process_clusters(self, clustering, side=1):
        """Computes cluster metrics.

        Parameters
        ----------
        clustering : array-like
            Cluster assignments for users.

        """
        out_clustering = None
        out_num_clusters = None
        out_frequencies = None

        if clustering is not None:
            occupied_clusters, out_frequencies = np.unique(
                clustering, return_counts=True
            )
            out_num_clusters = len(occupied_clusters)
            out_clustering = np.array(clustering)

        setattr(self, f"clustering_{side}", out_clustering)
        setattr(self, f"num_clusters_{side}", out_num_clusters)
        setattr(self, f"frequencies_{side}", out_frequencies)

        return

    def _remove_node_from_cluster(self, computed_quantities):
        """
        Removes a node from its current cluster and updates relevant quantities.
        Optionally returns mhk and y_values if needed.
        """

        side = computed_quantities["side"]

        node_idx = computed_quantities["node_idx"]
        current_cluster = computed_quantities["current_cluster"]

        mhk = computed_quantities["mhk"]
        nch = computed_quantities["nch"]

        frequencies = computed_quantities["frequencies"]
        frequencies_other_side = computed_quantities["frequencies_other_side"]

        num_clusters = computed_quantities["num_clusters"]
        num_clusters_other_side = computed_quantities["num_clusters_other_side"]
        num_nodes = computed_quantities["num_nodes"]
        num_nodes_other_side = computed_quantities["num_nodes_other_side"]

        clustering = computed_quantities["clustering"]
        clustering_other_side = computed_quantities["clustering_other_side"]

        covariates = computed_quantities["covariates"]

        mhk_minus = None
        y_values = None
        nch_minus = None

        frequencies_minus = frequencies.copy()
        frequencies_minus[current_cluster] -= 1
        frequencies[current_cluster] -= 1

        frequencies_other_side_minus = frequencies_other_side

        if self.likelihood.needs_mhk is True:
            # in principle compute_mhk() should be called using clustering and other_clustering
            # since the matrix mhk should be symmetric.
            # This however works only if we pass the full adjacency matrix.
            # However, for bipartite graph, it is more efficient to pass only one block
            # such that we can operate on a smaller matrix.
            # this requires some care in the updates below.
            if mhk is None:
                mhk = compute_mhk(self.Y, self.clustering_1, self.clustering_2)

        if nch is not None:
            nch_minus = []
            for cov in range(len(nch)):
                c = np.where(covariates.covariates[cov].cov_values[node_idx] == 1)[0][0]
                nch_minus.append(nch[cov].copy())
                nch_minus[-1][c, current_cluster] -= 1

        if self.likelihood.needs_yvalues is True:
            y_values = compute_y_values(
                Y=self.Y.T if side == 1 else self.Y,
                clustering=clustering_other_side,
                num_nodes=num_nodes_other_side,
                num_clusters=num_clusters_other_side,
            )

        if frequencies_minus[current_cluster] == 0:
            frequencies_minus = np.concatenate(
                [
                    frequencies_minus[:current_cluster],
                    frequencies_minus[current_cluster + 1 :],
                ]
            )
            num_clusters -= 1

            if self.likelihood.needs_mhk is True:
                if side == 1:
                    mhk_minus = np.vstack(
                        [mhk[:current_cluster], mhk[current_cluster + 1 :]]
                    )
                else:
                    mhk_minus = np.hstack(
                        [mhk[:, :current_cluster], mhk[:, current_cluster + 1 :]]
                    )

            if nch is not None:
                for cov in range(len(nch)):
                    nch_minus[cov] = np.hstack(
                        [
                            nch[cov][:, :current_cluster],
                            nch[cov][:, current_cluster + 1 :],
                        ]
                    )

        else:
            if self.likelihood.needs_mhk is True:
                mhk_minus = mhk.copy()
                if side == 1:
                    mhk_minus[current_cluster] -= y_values[node_idx]
                else:
                    mhk_minus[:, current_cluster] -= y_values[node_idx]

        computed_quantities["frequencies"] = frequencies
        computed_quantities["frequencies_minus"] = frequencies_minus
        computed_quantities["frequencies_other_side"] = frequencies_other_side
        computed_quantities["frequencies_other_side_minus"] = (
            frequencies_other_side_minus
        )
        computed_quantities["num_clusters"] = num_clusters
        computed_quantities["mhk"] = mhk
        computed_quantities["mhk_minus"] = mhk_minus
        computed_quantities["nch"] = nch
        computed_quantities["nch_minus"] = nch_minus
        computed_quantities["y_values"] = y_values
        computed_quantities["node_idx"] = node_idx

        return

    def _add_to_cluster(self, assignment, computed_quantities):
        current_cluster = computed_quantities["current_cluster"]
        frequencies = computed_quantities["frequencies"]
        frequencies_other_side = computed_quantities["frequencies_other_side"]
        frequencies_minus = computed_quantities["frequencies_minus"]
        num_clusters = computed_quantities["num_clusters"]
        clustering = computed_quantities["clustering"]
        node_idx = computed_quantities["node_idx"]
        side = computed_quantities["side"]
        mhk = computed_quantities["mhk"]
        mhk_minus = computed_quantities["mhk_minus"]
        y_values = computed_quantities["y_values"]
        nch = computed_quantities["nch"]
        covariates = computed_quantities["covariates"]
        nch_minus = computed_quantities["nch_minus"]

        if assignment == current_cluster:
            if frequencies[current_cluster] == 0:
                num_clusters += 1
            frequencies[current_cluster] += 1
        # change cluster
        else:
            if frequencies[current_cluster] == 0:
                # maintains cluster indices contiguous
                mask = np.where(clustering >= current_cluster)
                clustering[mask] -= 1

            if assignment >= num_clusters:
                clustering[node_idx] = assignment

                num_clusters += 1
                frequencies_minus = np.append(frequencies_minus, 1)

                if side == 1:
                    mhk = np.vstack([mhk_minus, y_values[node_idx]])
                else:
                    mhk = np.column_stack([mhk_minus, y_values[node_idx]])

                if nch is not None:
                    for cov in range(len(nch)):
                        c = np.where(
                            covariates.covariates[cov].cov_values[node_idx] == 1
                        )[0][0]
                        padding = np.zeros((nch[cov].shape[0], 1))
                        nch_minus[cov] = np.column_stack([nch_minus[cov], padding])
                        nch_minus[cov][c, assignment] += 1
                    nch = nch_minus
            else:
                frequencies_minus[assignment] += 1
                clustering[node_idx] = assignment
                if side == 1:
                    mhk_minus[assignment] += y_values[node_idx]
                else:
                    mhk_minus[:, assignment] += y_values[node_idx]
                mhk = mhk_minus

                if nch is not None:
                    for cov in range(len(nch)):
                        c = np.where(
                            covariates.covariates[cov].cov_values[node_idx] == 1
                        )[0][0]
                        nch_minus[cov][c, assignment] += 1
                    nch = nch_minus
            frequencies = frequencies_minus

        computed_quantities["frequencies"] = frequencies
        computed_quantities["num_clusters"] = num_clusters
        computed_quantities["clustering"] = clustering
        computed_quantities["mhk"] = mhk
        computed_quantities["nch"] = nch

        return computed_quantities
