################################################################################
# functions to sample data
################################################################################

import numpy as np


def generate_poisson_data(
    prior_shape, prior_rate, clustering_1, clustering_2=None, bipartite=False, rng=None
):
    """Generate count-valued network from a Poisson-Gamma model.

    Parameters
    ----------
    prior_shape : float
        The shape parameter for the gamma prior.
    prior_rate : float
        The rate parameter for the gamma prior.
    clustering_1 : np.ndarray
        The cluster assignments for the first set of nodes.
    clustering_2 : np.ndarray, optional
        The cluster assignments for the second set of nodes, by default None
    bipartite : bool, optional
        Whether the graph is bipartite, by default False. If True and clustering_2 is None,
        a ValueError is raised.
    rng : np.random.Generator, optional
        The random number generator to use, by default None

    Returns
    -------
    np.ndarray
        The generated Poisson SBM data. For bipartite return only one non-zero block
        of the adjacency matrix, else the full symmetric adjacency matrix.
    """
    if bipartite is True and clustering_2 is None:
        raise ValueError("clustering_2 must be provided for bipartite graphs.")

    if rng is None:
        rng = np.random.default_rng()

    if bipartite is True:
        d1 = np.unique(clustering_1).size
        d2 = np.unique(clustering_2).size

        theta = rng.gamma(prior_shape, prior_rate, size=(d1, d2))
        Y_params = theta[clustering_1][:, clustering_2]  # shape (n1, n2)
        Y = rng.poisson(Y_params)

    else:
        clustering_2 = clustering_1
        d = np.unique(clustering_1).size
        n = clustering_1.size

        theta = rng.gamma(prior_shape, prior_rate, size=(d, d))  # shape (d,d)

        Y_params = theta[clustering_1][:, clustering_2]  # shape (n, n)

        # sample only the lower triangle (no self-loops)
        L = np.tril(np.ones((n, n), dtype=bool), k=-1)
        Y_lower = rng.poisson(Y_params[L])

        Y = np.zeros((n, n), dtype=int)
        Y[L] = Y_lower
        Y = Y + Y.T

    return Y


def generate_bernoulli_data(
    prior_alpha, prior_beta, clustering_1, clustering_2=None, bipartite=False, rng=None
):
    """Generate binary network from a Bernoulli model.

    Parameters
    ----------
    prior_alpha : float
        Alpha parameter for the Beta prior.
    prior_beta : float
        Beta parameter for the Beta prior.
    clustering_1 : np.ndarray
        Cluster assignments for the first set of nodes.
    clustering_2 : np.ndarray, optional
        Cluster assignments for the second set of nodes, by default None
    bipartite : bool, optional
        Whether the graph is bipartite, by default False
    rng : np.random.Generator, optional
        The random number generator to use, by default None

    Returns
    -------
    np.ndarray
        The generated binary network. For bipartite return only one non-zero block
        of the adjacency matrix, else the full symmetric adjacency matrix.
    """
    if rng is None:
        rng = np.random.default_rng()

    if bipartite is True:
        d1 = np.unique(clustering_1).size
        d2 = np.unique(clustering_2).size

        theta = rng.beta(prior_alpha, prior_beta, size=(d1, d2))
        Y_params = theta[clustering_1][:, clustering_2]  # shape (n1, n2)
        Y = rng.binomial(1, Y_params)

    else:
        clustering_2 = clustering_1
        d = np.unique(clustering_1).size
        n = clustering_1.size

        theta = rng.beta(prior_alpha, prior_beta, size=(d, d))

        Y_params = theta[clustering_1][:, clustering_2]  # shape (n, n)

        # sample only the lower triangle (no self-loops)
        L = np.tril(np.ones((n, n), dtype=bool), k=-1)
        Y_lower = rng.binomial(1, Y_params[L])

        Y = np.zeros((n, n), dtype=int)
        Y[L] = Y_lower
        Y = Y + Y.T

    return Y
