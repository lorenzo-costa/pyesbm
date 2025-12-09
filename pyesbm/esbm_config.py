from pyesbm.likelihoods import BaseLikelihood
import numpy as np
from pyesbm.priors import BasePrior
from pyesbm.covariates import CovariateClass


class ESBMconfig:
    def __init__(
        self,
        Y,
        likelihood,
        prior,
        *,
        bipartite=True,
        clustering=None,
        degree_correction=0,
        alpha_c=1,
        cov_a=1,
        cov_b=1,
        covariates_1=None,
        covariates_2=None,
        epsilon=1e-6,
        rng=None,
        verbose=False,
    ):
        self._type_check(
            Y=Y,
            likelihood=likelihood,
            prior=prior,
            bipartite=bipartite,
            clustering=clustering,
            degree_correction=degree_correction,
            alpha_c=alpha_c,
            covariates_1=covariates_1,
            covariates_2=covariates_2,
            epsilon=epsilon,
            rng=rng,
            verbose=verbose,
            cov_a=cov_a,
            cov_b=cov_b,
        )

        self._process_args(
            Y=Y,
            likelihood=likelihood,
            prior=prior,
            bipartite=bipartite,
            clustering=clustering,
            degree_correction=degree_correction,
            alpha_c=alpha_c,
            covariates_1=covariates_1,
            covariates_2=covariates_2,
            epsilon=epsilon,
            rng=rng,
            verbose=verbose,
            cov_a=cov_a,
            cov_b=cov_b,
        )

    def _type_check(
        self,
        Y,
        likelihood,
        prior,
        bipartite,
        clustering,
        degree_correction,
        alpha_c,
        covariates_1,
        covariates_2,
        epsilon,
        rng,
        verbose,
        cov_a,
        cov_b,
    ):
        if not isinstance(Y, (np.ndarray, list)):
            raise TypeError(f"Y must be a numpy array or list. You provided {type(Y)}")
        if not isinstance(bipartite, bool):
            raise TypeError(
                f"bipartite must be a boolean. You provided {type(bipartite)}"
            )
        if not isinstance(likelihood, BaseLikelihood):
            raise TypeError(
                f"likelihood must be a BaseLikelihood instance. You provided {type(likelihood)}"
            )
        if not isinstance(degree_correction, (int, float)):
            raise TypeError(
                f"degree_correction must be int or float. You provided {type(degree_correction)}"
            )
        if not isinstance(prior, BasePrior):
            raise TypeError(
                f"prior must be a BasePrior instance. You provided {type(prior)}"
            )
        if rng is not None:
            if not isinstance(rng, (np.random.Generator, int)):
                raise TypeError(
                    f"rng must be a numpy random Generator or int. You provided {type(rng)}"
                )
        if covariates_1 is not None:
            if not isinstance(covariates_1, CovariateClass):
                raise TypeError(
                    f"covariates_1 must be a CovariateClass instance. You provided {type(covariates_1)}"
                )
        if covariates_2 is not None:
            if not isinstance(covariates_2, CovariateClass):
                raise TypeError(
                    f"covariates_2 must be a CovariateClass instance. You provided {type(covariates_2)}"
                )
        
        if not isinstance(epsilon, (int, float)):
            raise TypeError(
                f"epsilon must be int or float. You provided {type(epsilon)}"
            )
        num_nodes_1, num_nodes_2 = Y.shape

        if not isinstance(degree_correction, (int, float)):
            raise TypeError(
                f"degree_correction must be int or float. You provided {type(degree_correction)}"
            )

        if not isinstance(bipartite, bool):
            raise TypeError(
                f"bipartite must be boolean. You provided {type(bipartite)}"
            )

        if not isinstance(likelihood, BaseLikelihood):
            raise TypeError(
                f"likelihood must be a BaseLikelihood instance. You provided {type(likelihood)}"
            )

        if not isinstance(prior, BasePrior):
            raise TypeError(
                f"prior must be a BasePrior instance. You provided {type(prior)}"
            )

        if clustering is not None:
            if isinstance(clustering, str):
                clustering = clustering.lower()
                if clustering != "random":
                    raise ValueError(
                        f'clustering string value must be "random". You provided {clustering}'
                    )

            elif isinstance(clustering, np.ndarray):
                if clustering.ndim == 1:
                    if bipartite is True:
                        raise ValueError(
                            f"for bipartite networks clustering must be an array of dim (2, {num_nodes_1})"
                        )
                    else:
                        if clustering.shape[0] != num_nodes_1:
                            raise ValueError(
                                f"clustering length must be equal to number of nodes. You provided {clustering.shape[0]} but should be {num_nodes_1}"
                            )
                elif clustering.ndim == 2:
                    if bipartite is False:
                        raise ValueError(
                            "for unipartite networks clustering must be a single list/array"
                        )
                    else:
                        if (
                            clustering[0].shape[0] != num_nodes_1
                            and clustering[1].shape[0] != num_nodes_2
                        ):
                            raise ValueError(
                                f"clustering shape must be equal to number of nodes. You provided {clustering.shape} but should be ({num_nodes_1}, ) and ({num_nodes_2}, )"
                            )
                else:
                    raise ValueError(
                        f"clustering array must be 1D or 2D. You provided {clustering.ndim}D array"
                    )

            elif isinstance(clustering, list):
                if bipartite is True:
                    if len(clustering) != 2:
                        raise ValueError(
                            "for bipartite networks clustering must be a list of two lists/arrays"
                        )
                    else:
                        clustering[0] = np.array(clustering[0])
                        clustering[1] = np.array(clustering[1])

                        if (
                            clustering[0].shape[0] != num_nodes_1
                            and clustering[1].shape[0] != num_nodes_2
                        ):
                            raise ValueError(
                                f"clustering length must be equal to number of nodes. You provided {clustering[0].shape[0]} and {clustering[1].shape[0]} but should be {num_nodes_1} and {num_nodes_2}"
                            )
                else:
                    if len(clustering) != num_nodes_1:
                        raise ValueError(
                            f"clustering length must be equal to number of nodes. You provided {len(clustering)} but should be {num_nodes_1}"
                        )
            else:
                raise TypeError(
                    f"clustering must be a string, list or array. You provided {type(clustering)}"
                )

        if rng is not None:
            if not isinstance(rng, (np.random.Generator, int)):
                raise TypeError(
                    "rng must be a numpy random Generator or an integer seed"
                )

    def _process_args(
        self,
        Y,
        likelihood,
        prior,
        bipartite,
        clustering,
        degree_correction,
        alpha_c,
        covariates_1,
        covariates_2,
        epsilon,
        rng,
        verbose,
        cov_a,
        cov_b,
    ):
        self.Y = Y
        self.prior = prior
        self.likelihood = likelihood
        self.bipartite = bipartite
        self.degree_correction = degree_correction

        self.num_nodes_1, self.num_nodes_2 = self.Y.shape
        self.verbose = verbose
        self.epsilon = epsilon

        self.covariates_1 = covariates_1
        self.covariates_2 = covariates_2

        self.train_llk = None
        self.mcmc_draws_users = None
        self.mcmc_draws_items = None

        self.estimated_items = None
        self.estimated_users = None

        self.estimated_theta = None
