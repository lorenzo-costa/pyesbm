#####################
# misc functions
#################
import numba as nb
import numpy as np


def compute_precision(true, preds):
    try:
        return len(set(true).intersection(set(preds))) / len(preds)
    except ZeroDivisionError:
        return 0.0


def compute_recall(true, preds):
    try:
        return len(set(true).intersection(set(preds))) / len(true)
    except ZeroDivisionError:
        return 0.0


@nb.jit(nopython=True, parallel=False)
def compute_co_clustering_matrix(mcmc_draws_users):
    """
    Compute co-clustering matrix from MCMC draws.
    """
    n_iters, num_users = mcmc_draws_users.shape

    co_clustering_matrix_users = np.zeros((num_users, num_users))
    for it in nb.prange(n_iters):
        for user_one in range(num_users):
            for user_two in range(num_users):
                if mcmc_draws_users[it, user_one] == mcmc_draws_users[it, user_two]:
                    co_clustering_matrix_users[user_one, user_two] += 1

    return co_clustering_matrix_users

def generate_poisson_data(prior_shape, 
                          prior_rate,
                          clustering_1, 
                          clustering_2=None,
                          bipartite=False, 
                          rng=None):
    
    if rng is None:
        rng = np.random.default_rng()
        
    if bipartite is True:
        d1 = np.unique(clustering_1).size
        d2 = np.unique(clustering_2).size
        
        theta = rng.gamma(prior_shape, prior_rate, size=(d1, d2))
        Y_params = theta[clustering_1][:, clustering_2]     # shape (n1, n2)
        Y = rng.poisson(Y_params)
        
    else:
        clustering_2 = clustering_1
        d1 = np.unique(clustering_1).size
        d2 = np.unique(clustering_2).size
        n = clustering_1.size

        # block intensity matrix
        theta = rng.gamma(prior_shape, prior_rate, size=(d1, d2))

        # map per-node cluster assignments → Poisson rate matrix
        Y_params = theta[clustering_1][:, clustering_2]     # shape (n, n)

        # sample only the lower triangle (no self-loops)
        L = np.tril(np.ones((n, n), dtype=bool), k=-1)
        Y_lower = rng.poisson(Y_params[L])

        # build full symmetric adjacency matrix
        Y = np.zeros((n, n), dtype=int)
        Y[L] = Y_lower
        Y = Y + Y.T
        
    return Y 

def generate_bernoulli_data(prior_alpha, 
                          prior_beta,
                          clustering_1, 
                          clustering_2=None,
                          bipartite=False, 
                          rng=None):
    
    if rng is None:
        rng = np.random.default_rng()
        
    if bipartite is True:
        d1 = np.unique(clustering_1).size
        d2 = np.unique(clustering_2).size
        
        theta = rng.beta(prior_alpha, prior_beta, size=(d1, d2))
        Y_params = theta[clustering_1][:, clustering_2]     # shape (n1, n2)
        Y = rng.binomial(1, Y_params)
        
    else:
        clustering_2 = clustering_1
        d1 = np.unique(clustering_1).size
        d2 = np.unique(clustering_2).size
        n = clustering_1.size

        # block probability matrix
        theta = rng.beta(prior_alpha, prior_beta, size=(d1, d2))

        # map per-node cluster assignments → Bernoulli probability matrix
        Y_params = theta[clustering_1][:, clustering_2]     # shape (n, n)

        # sample only the lower triangle (no self-loops)
        L = np.tril(np.ones((n, n), dtype=bool), k=-1)
        Y_lower = rng.binomial(1, Y_params[L])

        # build full symmetric adjacency matrix
        Y = np.zeros((n, n), dtype=int)
        Y[L] = Y_lower
        Y = Y + Y.T
        
    return Y
