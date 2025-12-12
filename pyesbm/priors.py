import warnings
import numpy as np
from scipy.special import digamma, gammaln
from pyesbm.utilities import sampling_scheme
from pyesbm.utilities import probs_gnedin


class BasePrior:
    def __init__(self):
        pass

    def sample(self):
        raise NotImplementedError("This method should be implemented by subclasses.")


class GibbsTypePrior(BasePrior):
    """Gibbs Type Prior for Bayesian Nonparametric Models

    Parameters
    ----------
    scheme_type : str
        Type of the Gibbs-type prior. Options are 'DM' (Dirichlet-Multinomial), 'DP' (Dirichlet Process), 'PY' (Pitman-Yor), 'GN' (Gnedin).
    scheme_param : float
        Additional parameter for the prior distribution.
    sigma : float
        Sigma parameter for Gibbs-type priors.
    gamma : float
        Additional parameter for the GN model.
    bar_h : int
        Maximum number of clusters (for DM prior).
    num_nodes_1 : int
        Number of nodes in the first set.
    num_nodes_2 : int
        Number of nodes in the second set (for bipartite graphs).
    """

    def __init__(
        self,
        scheme_type=None,
        scheme_param=1,
        sigma=0.1,
        gamma=0.5,
        bar_h=None,
        num_nodes_1=None,
        num_nodes_2=None,
    ):
        super().__init__()
        args = {k: v for k, v in locals().items() if k != "self"}

        self._type_check(**args)

        self.scheme_type = scheme_type
        self.scheme_param = scheme_param
        self.sigma = sigma
        self.gamma = gamma
        self.bar_h = bar_h if bar_h is not None else -1
        self.num_nodes_1 = num_nodes_1
        self.num_nodes_2 = num_nodes_2
        self.scheme_dict = {"DM": 1, "DP": 2, "PY": 3, "GN": 4}
    
    def expected_num_clusters(self, n):
        """Compute prior expected number of clusters.
        
        Parameters
        ----------
        n : int
            Number of nodes.
        
        Returns
        -------
        float
            Expected number of clusters.
        """
        
        if self.scheme_type == "DP":
            alpha = self.scheme_param
            # digamma more efficient than sum
            return alpha * (digamma(alpha + n) - digamma(alpha))
        
        if self.scheme_type == "PY":
            alpha = self.scheme_param
            sigma = self.sigma
            log_term = (gammaln(alpha + sigma + n) - gammaln(alpha + sigma) - 
                        gammaln(alpha + n) + gammaln(alpha + 1))
            return (1.0 / sigma) * np.exp(log_term) - (alpha / sigma)

        if self.scheme_type == "DM":
            theta = self.scheme_param
            H = self.bar_h
            A = theta * (1 - 1.0/H)
            B = theta
            
            log_prod = (gammaln(A + n) - gammaln(A)) - (gammaln(B + n) - gammaln(B))
            return H - H * np.exp(log_prod)
        
        if self.scheme_type == "GN":
            gamma = self.gamma
            h_vals = np.arange(1, n + 1) 
            p = probs_gnedin(n, h_vals, gamma=gamma)

            return  np.sum(h_vals * p)

    def _type_check(self, **kwargs):
        scheme_type = kwargs.get("scheme_type")
        scheme_param = kwargs.get("scheme_param")
        sigma = kwargs.get("sigma")
        gamma = kwargs.get("gamma")
        bar_h = kwargs.get("bar_h")
        num_nodes_1 = kwargs.get("num_nodes_1")
        num_nodes_2 = kwargs.get("num_nodes_2")

        if scheme_type is None or scheme_type not in ["DP", "PY", "GN", "DM"]:
            raise ValueError(
                f"scheme type must be one of DP, PY, GN, DM. You provided {scheme_type}"
            )

        if scheme_type == "DM":
            if num_nodes_1 is None:
                raise ValueError("num_nodes_1 must be provided for DM prior")
            if num_nodes_2 is None:
                num_nodes_2 = num_nodes_1

            if not isinstance(bar_h, int):
                raise TypeError(
                    "maximum number of clusters users must be integer for DM"
                )
            if (bar_h <= 0) or (bar_h > min(num_nodes_1, num_nodes_2)):
                raise ValueError(
                    f"maximum number of clusters for DM for users should be in"
                    f"(0, {min(num_nodes_1, num_nodes_2)}]."
                    f". You provided {bar_h}"
                )
            if not isinstance(sigma, (int, float)):
                raise TypeError("sigma must be a float or int for DM")
            if sigma >= 0:
                raise ValueError(
                    f"sigma for DM should be negative. You provided {sigma}"
                )

        if scheme_type == "DP":
            if not isinstance(scheme_param, (int, float)):
                raise TypeError("concentration parameter for DP must be float or int")
            if scheme_param <= 0:
                raise ValueError(
                    f"concentration parameter for DP should be positive. You provided {scheme_param}"
                )

        if scheme_type == "PY":
            if not isinstance(sigma, (int, float)):
                raise TypeError("sigma must be a float or int for PY")
            if sigma < 0 or sigma >= 1:
                raise ValueError(
                    f"provide sigma in [0, 1) for PY. You provided {sigma}"
                )
            if not isinstance(scheme_param, (int, float)):
                raise TypeError("scheme param must be a float or int for PY")
            if not isinstance(scheme_param, (int, float)):
                raise TypeError("scheme param must be a float or int for PY")
            if scheme_param <= -sigma:
                raise ValueError(
                    f"scheme param should be < -sigma for PY. You provided {scheme_param}"
                )
            if sigma == 0:
                warnings.warn(
                    "note: for sigma=0 the PY reduces to DP, use scheme_type=DP for greater efficiency"
                )

        if scheme_type == "GN":
            if not isinstance(gamma, (int, float)):
                raise TypeError(f"gamma should be a float. You provided {type(gamma)}")
            if gamma <= 0 or gamma >= 1:
                raise ValueError(
                    f"gamma for GN should be in (0, 1). You provided {gamma}"
                )

    def compute_probs(self, num_nodes, num_clusters, frequencies_minus, **kwargs):
        if not isinstance(num_nodes, int):
            raise TypeError(f"num_nodes must be int. You provided {type(num_nodes)}")
        if not isinstance(num_clusters, int):
            raise TypeError(
                f"num_clusters must be int. You provided {type(num_clusters)}"
            )
        if not isinstance(frequencies_minus, (list, np.ndarray)):
            raise TypeError(
                f"frequencies must be list or np.ndarray. You provided {type(frequencies_minus)}"
            )

        if not isinstance(frequencies_minus, np.ndarray):
            frequencies_minus = np.array(frequencies_minus)

        out = sampling_scheme(
            V=num_nodes,
            H=num_clusters,
            frequencies=frequencies_minus,
            bar_h=self.bar_h,
            scheme_type=self.scheme_dict[self.scheme_type],
            scheme_param=self.scheme_param,
            sigma=self.sigma,
            gamma=self.gamma,
        )

        return out


class DirichletMultinomialPrior(GibbsTypePrior):
    """Dirichlet-Multinomial prior for clustering.

    Parameters
    ----------
    bar_h : int
        Maximum number of clusters.
    num_nodes_1 : int
        Number of nodes in the first set.
    num_nodes_2 : int
        Number of nodes in the second set (for bipartite graphs).
    """

    def __init__(self, bar_h, num_nodes_1, num_nodes_2=None):
        super().__init__(
            scheme_type="DM",
            scheme_param=1.0,
            sigma=-1.0,
            gamma=0.5,
            bar_h=bar_h,
            num_nodes_1=num_nodes_1,
            num_nodes_2=num_nodes_2,
        )


class DirichletProcess(GibbsTypePrior):
    """Dirichlet Process prior for clustering.

    Parameters
    ----------
    concentration : float
        Concentration parameter for the Dirichlet Process.
    """

    def __init__(self, concentration=1.0):
        super().__init__(
            scheme_type="DP",
            scheme_param=concentration,
            sigma=0.0,
            gamma=0.5,
            bar_h=None,
            num_nodes_1=None,
            num_nodes_2=None,
        )


class PitmanYorProcess(GibbsTypePrior):
    """Pitman-Yor Process prior for clustering.

    Parameters
    ----------
    sigma : float
        Stick-breaking parameter for the Pitman-Yor Process.
    scheme_param : float
        Second scheme parameter for the Pitman-Yor Process.
    """

    def __init__(self, sigma=0.5, scheme_param=1.0):
        super().__init__(
            scheme_type="PY",
            scheme_param=scheme_param,
            sigma=sigma,
            gamma=0.5,
            bar_h=None,
            num_nodes_1=None,
            num_nodes_2=None,
        )


class GnedinProcess(GibbsTypePrior):
    """Gnedin Process prior for clustering.

    Parameters
    ----------
    gamma : float
        Parameter for the Gnedin Process.
    """

    def __init__(self, gamma=0.5):
        super().__init__(
            scheme_type="GN",
            scheme_param=0.5,
            sigma=-1.0,
            gamma=gamma,
            bar_h=None,
            num_nodes_1=None,
            num_nodes_2=None,
        )