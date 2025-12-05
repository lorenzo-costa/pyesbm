"""
Implement different covariate models
"""

# covariates.py
import numpy as np
from pyesbm.utilities import compute_log_probs_categorical


class CovariateClass:
    def __init__(self, alpha_c, covariates, num_nodes, cov_names=None):
        self._arg_validation(alpha_c, covariates, num_nodes)

        self.covariates = covariates
        self._process_cov(covariates)

        if isinstance(alpha_c, (int, float)):
            temp = []
            for vals in self.cov_values:
                n_unique = len(np.unique(vals))
                temp.extend([alpha_c] * n_unique)
            self.alpha_c = np.array(temp)
        else:
            self.alpha_c = alpha_c

        self.alpha_0 = np.sum(self.alpha_c)

        self.num_nodes = num_nodes

        self.nch = None

    def get_nch(self, clustering=None, num_clusters=None):
        if self.nch is None:
            if clustering is None or num_clusters is None:
                raise ValueError(
                    "nch not initialised, clustering and num_clusters must be provided to compute nch."
                )
            self.nch = self._compute_nch(clustering, num_clusters)

        if clustering is not None and num_clusters is not None:
            self.nch = self._compute_nch(clustering, num_clusters)

        return self.nch

    def add_cluster(self, idx):
        nch = list(self.get_nch())
        for cov in range(len(self.cov_values)):
            n_unique = len(np.unique(self.cov_values[cov]))
            temp = np.zeros(n_unique)
            c = self.cov_values[cov][idx]
            temp[int(c)] += 1
            nch[cov] = np.column_stack((nch[cov], temp.reshape(-1, 1)))

        self.nch = np.array(nch)

        return nch

    def update_nch(self, idx, new_cluster):
        nch = self.get_nch()
        for cov in range(len(self.cov_values)):
            c = self.cov_values[cov][idx]
            nch[cov][c, new_cluster] += 1

        self.nch = nch
        return nch

    def compute_logits(
        self, num_components, node_idx, frequencies_minus, nch_minus=None, **kwargs
    ):
        """Compute log probabilities for the covariate part of the likelihood.

        Parameters
        ----------
        probs : array-like
            Array of probabilities for each cluster.
        idx : int
            Index of the current data point.
        frequencies : array-like
            Frequencies of each covariate level.

        Returns
        -------
        log_probs : array-like
            Log probabilities for each cluster.
        """

        if nch_minus is None:
            nch_minus = self.get_nch()

        logits = compute_log_probs_categorical(
            num_components=num_components,
            idx=node_idx,
            cov_types=self.cov_types,
            cov_nch=nch_minus,
            cov_values=self.cov_values,
            nh=frequencies_minus,
            alpha_c=self.alpha_c,
            alpha_0=self.alpha_0,
        )

        return logits

    def _arg_validation(self, alpha_c, covariates, num_nodes):
        if not isinstance(alpha_c, (int, float, np.ndarray)):
            raise TypeError(
                f"alpha_c must be int, float or np.ndarray. You provided {type(alpha_c)}"
            )

        if not isinstance(num_nodes, int):
            raise TypeError(f"num_nodes must be int. You provided {type(num_nodes)}")

        if not isinstance(covariates, list):
            raise TypeError(
                f"covariates must be a list. You provided {type(covariates)}"
            )

        for cov in covariates:
            if not isinstance(cov, tuple) or len(cov) != 2:
                raise ValueError(
                    f"Each covariate must be a tuple of (name, values). You provided {cov}"
                )
            name, values = cov
            if not isinstance(name, str):
                raise TypeError(
                    f"Covariate name must be a string. You provided {type(name)}"
                )
            if not isinstance(values, (list, np.ndarray)):
                raise TypeError(
                    f"Covariate values must be a list or np.ndarray. You provided {type(values)}"
                )
            if len(values) == 0:
                raise ValueError("Covariate values cannot be empty.")

    def _process_cov(self, cov_list):
        cov_names, cov_types, cov_values = [], [], []
        for cov in cov_list:
            name_type = cov[0].split("_", 1)
            cov_name, cov_type = name_type[0], name_type[1]
            cov_names.append(cov_name)
            cov_types.append(cov_type)
            cov_values.append(np.array(cov[1]))

        self.cov_names = np.array(cov_names)
        self.cov_types = np.array(cov_types)
        self.cov_values = np.array(cov_values)

    def _compute_nch(self, clustering, n_clusters):
        cov_nch = []
        for vals in self.cov_values:
            uniques = np.unique(vals)
            nch = np.zeros((len(uniques), n_clusters), dtype=int)

            for h in range(n_clusters):
                mask = clustering == h
                for c in uniques:
                    nch[int(c), h] = int((vals[mask] == c).sum())
            cov_nch.append(nch)
        return np.array(cov_nch)
