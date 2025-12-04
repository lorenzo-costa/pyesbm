###########################################
# this file contains a bunch of numba optimised functions
# to compute stuff
##############################################

import numba as nb
from math import lgamma
import numpy as np


###########################################
@nb.jit(nopython=True, fastmath=True)
def compute_llk_poisson(
    a,
    b,
    nh,
    nk,
    eps,
    mhk,
    clustering_1,
    clustering_2,
    dg_u,
    dg_i,
    dg_cl_u,
    dg_cl_i,
    degree_param_users=0.5,
    degree_param_items=0.5,
    degree_corrected=False,
):
    """Function to compute log-likelihood of current clustering

    To understand the parameters, see the detailed descriptions in
    results/text/lorenzocosta_thesis.pdf.

    Parameters
    ----------
    a : float
        Shape parameter for gamma prior
    b : float
        Rate parameter for gamma prior
    nh : array-like
        User cluster sizes
    nk : array-like
        Item cluster sizes
    eps : float
        Small value to avoid division by zero
    mhk : array-like
        mhk matrix
    user_clustering : array-like
        User clustering assignments
    item_clustering : array-like
        Item clustering assignments
    dg_u : array-like
        User degree sequence
    dg_i : array-like
        Item degree sequence
    dg_cl_u : array-like
        Sum of user degrees for users within each cluster
    dg_cl_i : array-like
        Sum of item degrees for items within each cluster
    degree_param_users : float, optional
        Degree-correction parameter for users, by default 0.5
    degree_param_items : float, optional
        Degree-correction parameter for items, by default 0.5
    degree_corrected : bool, optional
        Whether to apply degree correction, by default False

    Returns
    -------
    out : float
        log-likelihood value
    """

    out = 0.0
    if degree_corrected is True:
        for h in range(len(nh)):
            idx = np.where(clustering_1 == h)[0]
            for i in idx:
                out += lgamma(degree_param_users + dg_u[i] + eps)
            out -= lgamma(nh[h] * degree_param_users + dg_cl_u[h] + eps)
            out += dg_cl_u[h] * np.log(nh[h])
            out += lgamma(nh[h] * degree_param_users)
            out -= nh[h] * lgamma(degree_param_users)

        for k in range(len(nk)):
            idx = np.where(clustering_2 == k)[0]
            for i in idx:
                out += lgamma(degree_param_items + dg_i[i] + eps)
            out -= lgamma(nk[k] * degree_param_items + dg_cl_i[k] + eps)
            out += dg_cl_i[k] * np.log(nk[k])
            out += lgamma(nk[k] * degree_param_items)
            out -= nk[k] * lgamma(degree_param_items)

    for h in range(len(nh)):
        for k in range(len(nk)):
            out += (
                lgamma(mhk[h, k] + a + eps) - 
                (mhk[h, k] + a) * np.log((nh[h] * nk[k] + b))
                )

    return out

def compute_llk_bernoulli(a,
    b,
    nh,
    nk,
    eps,
    mhk,
    clustering_1,
    clustering_2,
    dg_u,
    dg_i,
    dg_cl_u,
    dg_cl_i,
    degree_param_users=0.5,
    degree_param_items=0.5,
    degree_corrected=False,
    ):
    
    """Function to compute log-likelihood of current clustering for Bernoulli likelihood"""
    out = 0.0
    if degree_corrected is True:
        for h in range(len(nh)):
            idx = np.where(clustering_1 == h)[0]
            for i in idx:
                out += lgamma(degree_param_users + dg_u[i] + eps)
            out -= lgamma(nh[h] * degree_param_users + dg_cl_u[h] + eps)
            out += dg_cl_u[h] * np.log(nh[h])
            out += lgamma(nh[h] * degree_param_users)
            out -= nh[h] * lgamma(degree_param_users)

        for k in range(len(nk)):
            idx = np.where(clustering_2 == k)[0]
            for i in idx:
                out += lgamma(degree_param_items + dg_i[i] + eps)
            out -= lgamma(nk[k] * degree_param_items + dg_cl_i[k] + eps)
            out += dg_cl_i[k] * np.log(nk[k])
            out += lgamma(nk[k] * degree_param_items)
            out -= nk[k] * lgamma(degree_param_items)
    
    for h in range(len(nh)):
        for k in range(len(nk)):
            out += (
                lgamma(mhk[h, k] + a + eps) + 
                lgamma(nh[h]*nk[k] - mhk[h,k] + b + eps) - 
                lgamma(nh[h]*nk[k] + a + b)
                )

    return out


###################################
# gibbs-type prior sampling scheme
####################################


@nb.jit(nopython=True)
def sampling_scheme(V, H, frequencies, bar_h, scheme_type, scheme_param, sigma, gamma):
    """Probability of sampling each cluster (and a new one) under Gibbs-type priors.

    Parameters
    ----------
    V : int
        number of data points (nodes)
    H : int
        number of clusters
    frequencies : array-like
        cluster frequencies
    bar_h : int
        maximum number of clusters (for DM prior)
    scheme_type : str
        type of the prior distribution
    scheme_param : float
        additional parameter for Gibbs-type priors
    sigma : float
        sigma parameter for Gibbs-type priors
    gamma : float
        additional parameter for the GN model

    Returns
    -------
    probs : array-like
        probabilities of sampling each cluster and a new cluster

    """

    if scheme_type == 1:
        if H < bar_h:
            probs = np.zeros(len(frequencies) + 1)
            for i in range(len(frequencies)):
                probs[i] = frequencies[i] - sigma
            probs[-1] = -sigma * (bar_h - H)
        else:
            probs = np.zeros(len(frequencies))
            for i in range(len(frequencies)):
                probs[i] = frequencies[i] - sigma

    if scheme_type == 2:
        probs = np.zeros(len(frequencies) + 1)
        for i in range(len(frequencies)):
            probs[i] = frequencies[i]
        probs[-1] = scheme_param

    if scheme_type == 3:
        probs = np.zeros(len(frequencies) + 1)
        for i in range(len(frequencies)):
            probs[i] = frequencies[i] - sigma
        probs[-1] = scheme_param + H * sigma

    if scheme_type == 4:
        probs = np.zeros(len(frequencies) + 1)
        for i in range(len(frequencies)):
            probs[i] = (frequencies[i] + 1) * (V - H + gamma)
        probs[-1] = H * (H - gamma)

    return probs


#################################
# log probability computation for gibbs sampling steps
#################################

@nb.njit(fastmath=True, parallel=False)
def update_prob_poissongamma(
    num_components,
    mhk,
    frequencies_primary,
    frequencies_secondary,
    y_values,
    epsilon,
    a,
    b,
    max_clusters,
    side,
    bipartite,
    degree_corrected,
    degree_cluster_minus,
    degree_node,
    degree_param,
):
    log_probs = np.zeros(num_components)
    a_plus_epsilon = a + epsilon
    lgamma_a = lgamma(a)
    log_b = np.log(b)
    lgamma_a_log_b = -lgamma_a + a * log_b

    for i in range(max_clusters):
        p_i = 0.0
        freq_i = frequencies_primary[i]
        
        for j in range(len(frequencies_secondary)):
            # swap indices based on side (issue with mhk, see compute_mhk for details)
            if side==1:
                h, k = i, j
            else:
                k, h = i, j
                
            mhk_val = mhk[h, k]
            y_val = y_values[j]

            mhk_plus_a = mhk_val + a_plus_epsilon
            mhk_plus_y_plus_a = mhk_val + y_val + a_plus_epsilon

            if bipartite is True:
                log_freq_prod1 = np.log(b + freq_i * frequencies_secondary[j])
                log_freq_prod2 = np.log(b + (freq_i + 1) * frequencies_secondary[j])
            else:
                log_freq_prod1 = np.log(b + freq_i * frequencies_secondary[j])
                log_freq_prod2 = np.log(b + (freq_i + 1) * (frequencies_secondary[j]+1))
                
            p_i += (
                lgamma(mhk_plus_y_plus_a)
                - lgamma(mhk_plus_a)
                + (mhk_plus_a - epsilon) * log_freq_prod1
                - (mhk_plus_y_plus_a - epsilon) * log_freq_prod2
            )

        log_probs[i] += p_i

        if degree_corrected is True:
            first = lgamma(
                frequencies_primary[i] * degree_param + degree_cluster_minus[i]
            )
            second = lgamma(
                (frequencies_primary[i] + 1) * degree_param
                + degree_cluster_minus[i]
                + degree_node
            )

            third = lgamma((frequencies_primary[i] + 1) * degree_param)
            fourth = lgamma(frequencies_primary[i] * degree_param)

            fifth = (degree_cluster_minus[i] + degree_node) * np.log(
                frequencies_primary[i] + 1
            )
            sixth = degree_cluster_minus[i] * np.log(frequencies_primary[i])

            log_probs[i] += first - second + third - fourth + fifth - sixth

    # Handle new cluster case
    if len(log_probs) > max_clusters:
        p_new = 0.0
        for j in range(len(frequencies_secondary)):
            y_val = y_values[j]
            p_new += (
                lgamma(y_val + a_plus_epsilon)
                + lgamma_a_log_b
                - (y_val + a) * np.log(b + frequencies_secondary[j])
            )

        log_probs[max_clusters] += p_new
        if degree_corrected is True:
            log_probs[max_clusters] += lgamma(degree_param) - lgamma(
                degree_param + degree_node
            )

    return log_probs

@nb.njit(fastmath=True, parallel=False)
def update_prob_betabernoulli(
    num_components,
    mhk,
    frequencies_primary,
    frequencies_secondary,
    y_values,
    epsilon,
    a,
    b,
    max_clusters,
    side,
    degree_corrected,
    degree_cluster_minus,
    degree_node,
    degree_param,
):
    
    log_probs = np.zeros(num_components)
    
    for i in range(max_clusters):
        p_i = 0.0
        freq_i = frequencies_primary[i]
        
        for j in range(len(frequencies_secondary)):
            # swap indices based on side (issue with mhk, see compute_mhk for details)
            if side==1:
                h, k = i, j
            else:
                k, h = i, j
            
            freq_j = frequencies_secondary[j]

            mhk_val = mhk[h, k]
            mhk_bar = freq_i * freq_j - mhk_val
            y_val = y_values[j]
            y_val_bar = freq_j - y_val
            
            mhk_a = mhk_val + a + epsilon
            mhk_bar_b = mhk_bar + b + epsilon

            mhk_a_y = mhk_a + y_val
            mhk_bar_b_y = mhk_bar_b + y_val_bar

            p_i += (
                lgamma(mhk_a_y)
                + lgamma(mhk_bar_b_y)
                - lgamma(mhk_a_y + mhk_bar_b_y)
                + lgamma(mhk_a + mhk_bar_b)
                - lgamma(mhk_a)
                - lgamma(mhk_bar_b)
            )

        log_probs[i] += p_i

    if len(log_probs) > max_clusters:
        p_new = 0.0
        for j in range(len(frequencies_secondary)):
            y_val = y_values[j]
            freq_j = frequencies_secondary[j]

            mhk_bar = freq_j
            y_val = y_values[j]
            y_val_bar = freq_j - y_val
            
            mhk_a = a + epsilon
            mhk_bar_b = mhk_bar + b + epsilon

            mhk_a_y = mhk_a + y_val
            mhk_bar_b_y = mhk_bar_b + y_val_bar
            
            p_new += (
                lgamma(mhk_a_y)
                + lgamma(mhk_bar_b_y)
                - lgamma(mhk_a_y + mhk_bar_b_y)
                + lgamma(mhk_a + mhk_bar_b)
                - lgamma(mhk_a)
                - lgamma(mhk_bar_b)
            )

        log_probs[max_clusters] += p_new
    
    return log_probs


@nb.jit(nopython=True)
def compute_log_probs_cov(
    probs, idx, cov_types, cov_nch, cov_values, nh, alpha_c, alpha_0
):
    """Numba-optimized function to compute contribution of covariates to log probabilities.

    Parameters
    ----------
    probs : array-like
        array of probabilities
    idx : int
        index of the user/item being considered
    cov_types : array-like
        types of covariates ('cat' for categorical)
    cov_nch : array-like
        nch matrices for categorical covariates
    cov_values : array-like
        covariate values for each user/item
    nh : array-like
        cluster sizes
    alpha_c : array-like
        alpha_c parameters for categorical covariates
    alpha_0 : float
        sum of alpha_c parameters

    Returns
    -------
    log_probs : array-like
        log probabilities contribution from covariates
    """

    log_probs = np.zeros_like(probs)
    for i in nb.prange(len(cov_types)):
        if cov_types[i] == "cat":
            c = cov_values[i][idx]
            nch = cov_nch[i]
            for h in range(len(nh)):
                log_probs[h] += np.log(nch[c, h] + alpha_c[c]) - np.log(nh[h] + alpha_0)
            log_probs[-1] += np.log(alpha_c[c]) - np.log(alpha_0)

    return log_probs
