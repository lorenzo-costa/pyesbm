import numpy as np


class ClusterProcessor:
    def __init__(self, 
                 num_nodes, 
                 clustering, 
                 prior,
                 covariates=None,
                 verbose=False,
                 epsilon=1e-6,
                 rng=None):

        self.num_nodes = num_nodes
        self.prior = prior
        self.verbose = verbose
        self.epsilon = epsilon
        self.rng = rng if rng is not None else np.random.default_rng()

        # process clustering
        if clustering is not None:
            if isinstance(clustering, str):
                clustering = clustering.lower()
                if clustering == "random":
                    self.clustering, _, _ = self._init_cluster_random(num_nodes, covariates)
            else:
                self.clustering = clustering
        else:
            self.clustering = np.arange(self.num_nodes)

    def _init_cluster_random(self, num_nodes, covariates=None):
        """Initialises random clustering structure according to the prior.

        Parameters
        ----------
        clustering : array-like, optional
            Initial clustering structure. If 'random' random initialisation is performed,
            by default None
        covariates : object, optional
            covariates object

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
            print("initialsing user clusters random")

        nch = covariates.get_nch() if covariates is not None else None
    
        # sequential assignment of clusters
        for i in range(1, num_nodes):
            # prior contribution
            prior_probs = self.prior.sample(
                num_nodes=current_num_nodes,
                num_clusters=num_clusters,
                frequencies=np.array(frequencies),
            )
            
            # covariate contribution
            logits_cov = 0
            if nch is not None:
                logits_cov = covariates.compute_logits(
                    num_nodes=current_num_nodes,
                    idx=i,
                    frequencies=np.array(frequencies))

            # convert back using exp and normalise
            logits = np.log(prior_probs + self.epsilon) + logits_cov
            logits = logits - max(logits)
            probs = np.exp(logits)
            probs = probs / probs.sum()

            assignment = self.rng.choice(len(probs), p=probs)
            if assignment >= num_clusters:
                # make new cluster
                num_clusters += 1
                frequencies.append(1)

                if covariates is not None:
                    covariates.add_cluster(i)
            else:
                frequencies[assignment] += 1
                if covariates is not None:
                    covariates.update_nch(i, assignment)

            clustering.append(assignment)
            current_num_nodes += 1

        clustering = np.array(clustering)

        # safety check
        assert current_num_nodes == num_nodes
        assert len(clustering) == num_nodes
        assert len(np.unique(clustering)) == num_clusters

        return clustering, frequencies, num_clusters