"""
Implement different covariate models
"""

# covariates.py
import numpy as np
from pyesbm.utilities import compute_logits_categorical, compute_logits_count
from scipy.sparse import csr_matrix


class BaseCovariate:
    """Base class for covariate models"""
    def __init__(self, name=None, type=None, importance=1, **kwargs):
        if not isinstance(name, str):
            raise TypeError(f"name must be a string. You provided {type(name)}")
        if not isinstance(type, str):
            raise TypeError(f"type must be a string. You provided {type(type)}")
        if not isinstance(importance, (int, float)):
            raise TypeError(
                f"importance must be int or float. You provided {type(importance)}"
            )
        if importance <= 0:
            raise ValueError("importance must be positive.")
        
        self.name = name
        self.type = type
        self.importance = importance

    def compute_logits(self):
        raise NotImplementedError(
            "This is an abstract method. Please implement in subclass."
        )
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name={self.name}, type={self.type})"


class CategoricalCovariate(BaseCovariate):
    """Categorical covariate model.

    Arguments
    ---------
    cov_array : array-like
        Array of categorical covariate values for each node.
    alpha_c : int, float, np.ndarray, list, optional
        Concentration parameters for each category. If int or float, the same value is used for
        all categories. Default is 1.
    name : str, optional
        Name of the covariate. Default is "categorical_covariate".
    importance : int, float, optional
        Importance weight for the covariate. Default is 1.

    Methods
    -------
    compute_logits(num_components, node_idx, frequencies_minus, nch_minus, **kwargs)
        Compute logits for gibbs update.
    """
    
    def __init__(self, cov_array, alpha_c=1, name=None, **kwargs):
        
        name = name if name is not None else "categorical_covariate"
        
        super().__init__(name=name, type="categorical")
        
        if not isinstance(alpha_c, (int, float, np.ndarray, list)):
            raise TypeError(
                f"alpha_c must be int, float, np.ndarray or list. You provided {type(alpha_c)}"
            )

        if not isinstance(cov_array, (list, np.ndarray)):
            raise TypeError(
                f"cov_array must be a list or np.ndarray. You provided {type(cov_array)}"
            )

        if isinstance(cov_array, list):
            cov_array = np.array(cov_array)

        self.name = name
        
        if isinstance(alpha_c, (int, float)):
            n_unique = len(np.unique(cov_array))
            self.alpha_c = np.array([alpha_c] * n_unique)
        else:
            self.alpha_c = np.array(alpha_c)
        self.alpha_0 = np.sum(self.alpha_c)

        num_classes = np.max(cov_array) + 1
        self.cov_values = np.eye(num_classes)[cov_array]

        self.cov_type = "categorical"

    def compute_logits(self, 
                       num_components, 
                       node_idx, 
                       frequencies_minus, 
                       nch_minus, 
                       **kwargs):
        """Compute logits for categorical covariate.

        Parameters
        ----------
        num_components : int
            Number of components.
        node_idx : int
            Index of the node.
        frequencies_minus : np.ndarray
            Frequencies for the nodes.
        nch_minus : np.ndarray
            Count of neighbors for the nodes.

        Returns
        -------
        np.ndarray
            Logits for the categorical covariate.
        """
        logits = compute_logits_categorical(
            num_components=num_components,
            idx=node_idx,
            nch_minus=nch_minus,
            cov_values=self.cov_values,
            nh_minus=frequencies_minus,
            alpha_c=self.alpha_c,
            alpha_0=self.alpha_0,
        )
        
        logits = logits * self.importance

        return logits


class CountCovariate(BaseCovariate):
    """Count covariate model.

    Arguments
    ---------
    cov_array : array-like
        Array of count covariate values for each node.
    a : int, float, optional
        Hyperparameter a for the Beta prior. Default is 1.0.
    b : int, float, optional
        Hyperparameter b for the Beta prior. Default is 1.0.
    name : str, optional
        Name of the covariate. Default is "count_covariate".
    importance : int, float, optional
        Importance weight for the covariate. Default is 1.
    
    Methods
    -------
    compute_logits(num_components, node_idx, frequencies, frequencies_minus, nch, nch_minus, **kwargs)
        Compute logits for gibbs update.
    """
    def __init__(self, 
                 cov_array, 
                 name=None, 
                 a=1.0, 
                 b=1.0,
                 importance=1,
                 **kwargs):
        
        name = name if name is not None else "count_covariate"
        
        super().__init__(name=name, importance=importance, type="count")
        
        if not isinstance(a, (int, float)) or not isinstance(b, (int, float)):
            raise TypeError(
                f"a and b must be int or float. You provided {type(a)} and {type(b)}"
            )

        if name is None:
            name = "count_covariate"

        if not isinstance(name, str):
            raise TypeError(f"name must be a string. You provided {type(name)}")

        if not isinstance(cov_array, (list, np.ndarray)):
            raise TypeError(
                f"cov_array must be a list or np.ndarray. You provided {type(cov_array)}"
            )
        if isinstance(cov_array, list):
            cov_array = np.array(cov_array)

        self.name = name
        num_classes = np.max(cov_array) + 1
        self.cov_values = (np.arange(num_classes) <= cov_array[:, None]).astype(int)

        self.a = a
        self.b = b

        self.cov_type = "count"

    def compute_logits(
        self,
        num_components,
        node_idx,
        frequencies,
        frequencies_minus,
        nch,
        nch_minus,
        **kwargs,
    ):
        """Compute logits for count covariate.

        Parameters
        ----------
        num_components : int
            The number of components (clusters).
        node_idx : int
            The index of the current node.
        frequencies : np.ndarray
            The frequencies of each covariate level.
        frequencies_minus : np.ndarray
            The frequencies of each covariate level minus the current node.
        nch : np.ndarray
            The count of observations in each cluster for each covariate.
        nch_minus : np.ndarray
            The count of observations in each cluster for each covariate minus the current node.

        Returns
        -------
        np.ndarray
            The computed logits for the count covariate.
        """

        logits = compute_logits_count(
            num_components=num_components,
            idx=node_idx,
            nch=nch,
            nch_minus=nch_minus,
            cov_values=self.cov_values,
            nh=frequencies,
            nh_minus=frequencies_minus,
            a=self.a,
            b=self.b,
        )

        logits = logits * self.importance

        return logits

class ContinuousCovariate(BaseCovariate):
    def __init__(self, 
                 cov_array, 
                 name=None,
                 importance=1,
                 **kwargs):

        name = name if name is not None else "continuous_covariate"

        super().__init__(name=name, importance=importance, type="continuous")

        if not isinstance(cov_array, (list, np.ndarray)):
            raise TypeError(
                f"cov_array must be a list or np.ndarray. You provided {type(cov_array)}"
            )
        if isinstance(cov_array, list):
            cov_array = np.array(cov_array)

        self.name = name
        self.cov_values = cov_array.reshape(-1, 1)

        self.cov_type = "continuous"

    def compute_logits(self, **kwargs):
        """Compute logits for continuous covariate."""
        raise NotImplementedError(
            "Continuous covariate likelihood not implemented yet."
        )

class CovariateModel:
    def __init__(self, covariates):
        self.covariates = covariates
        self.nch = None

    def get_nch(self, clustering=None, num_clusters=None):
        """Get or compute the nch matrix."""
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
        """Add a new cluster and update nch matrix."""
        nch = self.get_nch()
        for cov in range(len(self.covariates)):
            cov_values = self.covariates[cov].cov_values
            n_unique = cov_values.shape[1]
            temp = np.zeros(n_unique)
            c = np.where(cov_values[idx] == 1)[0][0]
            temp[int(c)] += 1
            nch[cov] = np.column_stack((nch[cov], temp.reshape(-1, 1)))

        self.nch = nch

        return nch

    def update_nch(self, idx, new_cluster):
        """Update nch matrix when moving a node to a new cluster."""
        nch = self.get_nch()
        for cov in range(len(self.covariates)):
            cov_values = self.covariates[cov].cov_values
            c = np.where(cov_values[idx] == 1)[0][0]
            nch[cov][c, new_cluster] += 1

        self.nch = nch
        return nch

    def compute_logits(
        self,
        num_components,
        node_idx,
        frequencies,
        frequencies_minus,
        nch,
        nch_minus,
        **kwargs,
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

        logits = np.zeros(num_components)

        for cov in range(len(self.covariates)):
            out = self.covariates[cov].compute_logits(
                num_components=num_components,
                node_idx=node_idx,
                frequencies=frequencies,
                frequencies_minus=frequencies_minus,
                nch=nch[cov],
                nch_minus=nch_minus[cov],
            )
            logits += out

        return logits

    def _compute_nch(self, clustering, n_clusters):
        cov_nch = []
        for i in range(len(self.covariates)):
            vals = self.covariates[i].cov_values
            n_samples = len(clustering)
            vals = vals[:n_samples]  # when building clustering use only some vals

            n_clusters = np.max(clustering) + 1

            # row indices = sample indices
            row = np.arange(n_samples)
            col = clustering
            data = np.ones(n_samples)

            cluster_indicator = csr_matrix(
                (data, (row, col)), shape=(n_samples, n_clusters)
            )

            cov_nch.append(vals.T @ cluster_indicator)

        return cov_nch
