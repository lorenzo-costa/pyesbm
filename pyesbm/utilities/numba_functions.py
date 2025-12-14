###########################################
# this file contains a bunch of numba optimised functions
# to compute stuff
##############################################

import numba as nb
from math import lgamma
import numpy as np

import os

# warning raised within numba, silence it globally
os.environ["KMP_WARNINGS"] = "0"


###########################################
# log-likelihood computation functions
##############################################
@nb.jit(nopython=True, fastmath=True)
def compute_llk_poisson(
    a,
    b,
    nh,
    nk,
    eps,
    mhk,
):
    """Function to compute log-likelihood of current clustering

    To understand the parameters, see the detailed descriptions in
    examples/lorenzocosta_thesis.pdf.

    Parameters
    ----------
    a : float
        Shape parameter for gamma prior
    b : float
        Rate parameter for gamma prior
    nh : array-like
        cluster sizes side 1
    nk : array-like
        cluster sizes side 2
    eps : float
        Small value to avoid division by zero
    mhk : array-like
        mhk matrix

    Returns
    -------
    out : float
        log-likelihood value
    """

    out = 0.0

    for h in range(len(nh)):
        for k in range(len(nk)):
            out += lgamma(mhk[h, k] + a + eps) - (mhk[h, k] + a) * np.log(
                (nh[h] * nk[k] + b)
            )

    return out


@nb.jit(nopython=True, fastmath=True)
def compute_llk_bernoulli(
    a,
    b,
    nh,
    nk,
    eps,
    mhk,
):
    """Compute log-likelihood of current clustering for Bernoulli likelihood.

    Parameters
    ----------
    a : float
        Shape parameter for beta prior
    b : float
        Rate parameter for beta prior
    nh : array-like
        cluster sizes side 1
    nk : array-like
        cluster sizes side 2
    eps : float
        Small value to avoid division by zero
    mhk : array-like
        MHK matrix

    Returns
    -------
    out : float
        log-likelihood value
    """
    out = 0.0

    for h in range(len(nh)):
        for k in range(len(nk)):
            out += (
                lgamma(mhk[h, k] + a + eps)
                + lgamma(nh[h] * nk[k] - mhk[h, k] + b + eps)
                - lgamma(nh[h] * nk[k] + a + b)
            )

    return out


###################################
# gibbs-type prior functions
####################################
@nb.jit(nopython=True, fastmath=True)
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


@nb.jit(nopython=True, fastmath=True)
def probs_gnedin(V, h_vals, gamma=0.5):
    """Probability of having h clusters under Gnedin Process prior."""

    n = len(h_vals)
    result = np.empty(n, dtype=np.float64)

    # pre-compute constants invariant to the loop
    lgamma_V_plus_1 = lgamma(V + 1)
    lgamma_1_minus_gamma = lgamma(1.0 - gamma)
    lgamma_V_plus_gamma = lgamma(V + gamma)
    log_gamma = np.log(gamma)

    for i in range(n):
        val_h = h_vals[i]

        l_choose = lgamma_V_plus_1 - lgamma(val_h + 1) - lgamma(V - val_h + 1)

        log_res = (
            l_choose
            + lgamma(val_h - gamma)
            - lgamma_1_minus_gamma
            + log_gamma
            + lgamma(V + gamma - val_h)
            - lgamma_V_plus_gamma
        )

        result[i] = np.exp(log_res)

    return result


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
):
    """Update log probabilities for Poisson-Gamma model.

    Parameters
    ----------
    num_components : int
        Number of components (clusters)
    mhk : array-like
        Co-occurrence matrix
    frequencies_primary : array-like
        Frequencies of primary items
    frequencies_secondary : array-like
        Frequencies of secondary items
    y_values : array-like
        Y values
    epsilon : float
        Epsilon value to avoid numerical issues
    a : float
        Hyperparameter for the Poisson-Gamma model
    b : float
        Hyperparameter for the Poisson-Gamma model
    max_clusters : int
        Maximum number of clusters
    side : int
        Side indicator (1 or 2)

    Returns
    -------
    log_probs : array-like
        Updated log probabilities
    """
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
            if side == 1:
                h, k = i, j
            else:
                k, h = i, j

            mhk_val = mhk[h, k]
            y_val = y_values[j]

            mhk_plus_a = mhk_val + a_plus_epsilon
            mhk_plus_y_plus_a = mhk_val + y_val + a_plus_epsilon

            log_freq_prod1 = np.log(b + freq_i * frequencies_secondary[j])
            log_freq_prod2 = np.log(b + (freq_i + 1) * frequencies_secondary[j])

            p_i += (
                lgamma(mhk_plus_y_plus_a)
                - lgamma(mhk_plus_a)
                + (mhk_plus_a - epsilon) * log_freq_prod1
                - (mhk_plus_y_plus_a - epsilon) * log_freq_prod2
            )

        log_probs[i] += p_i

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
):
    """Update the log probabilities for the Beta-Bernoulli distribution.

    Parameters
    ----------
    num_components : int
        Number of components in the mixture
    mhk : array-like
        Co-occurrence matrix
    frequencies_primary : array-like
        Frequencies of primary observations
    frequencies_secondary : array-like
        Frequencies of secondary observations
    y_values : array-like
        Observed values
    epsilon : float
        Small value to avoid numerical issues
    a : float
        Hyperparameter for the Poisson-Gamma model
    mhk : array-like
        Co-occurrence matrix
    frequencies_primary : array-like
        Frequencies of primary observations
    frequencies_secondary : array-like
        Frequencies of secondary observations
    y_values : array-like
        Observed values
    epsilon : float
        Small value to avoid numerical issues
    a : float
        Hyperparameter for the Poisson-Gamma model
    b : float
        Hyperparameter for the Beta-Bernoulli model
    max_clusters : int
        Maximum number of clusters
    side : int
        Side indicator (1 or 2)

    Returns
    -------
    array-like
        Log probabilities for each component
    """
    log_probs = np.zeros(num_components)

    for i in range(max_clusters):
        p_i = 0.0
        freq_i = frequencies_primary[i]

        for j in range(len(frequencies_secondary)):
            # swap indices based on side (issue with mhk, see compute_mhk for details)
            if side == 1:
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


##########################################
# covariate log-probability computation functions
##########################################


@nb.jit(nopython=True)
def compute_logits_count(
    num_components, idx, nch, nch_minus, cov_values, nh, nh_minus, a, b
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

    log_probs = np.zeros(num_components)

    num_cov_values = nch.shape[0]

    my_val = nch * np.arange(num_cov_values).reshape(-1, 1)
    my_val = my_val.sum(axis=0)

    my_val_minus = nch_minus * np.arange(num_cov_values).reshape(-1, 1)
    my_val_minus = my_val_minus.sum(axis=0)

    for h in range(len(nh_minus)):
        first = lgamma(a + my_val[h])
        second = lgamma(a + my_val_minus[h])
        third = (a + my_val_minus[h]) * np.log(nh_minus[h] + b)
        fourth = (a + my_val[h]) * np.log(nh[h] + b)

        log_probs[h] += first - second + third - fourth

    xi = cov_values[idx].sum()
    first = lgamma(a + xi)
    second = lgamma(a)
    third = a * np.log(b)
    fourth = (a + xi) * np.log(b + 1)
    log_probs[-1] += first - second + third - fourth

    return log_probs


@nb.jit(nopython=True)
def compute_logits_categorical(
    num_components,
    idx,
    nch_minus,
    cov_values,
    nh_minus,
    alpha_c,
    alpha_0,
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

    log_probs = np.zeros(num_components)
    c = np.where(cov_values[idx] == 1)[0][0]
    for h in range(len(nh_minus)):
        c = int(c)
        log_probs[h] += np.log(nch_minus[c, h] + alpha_c[c]) - np.log(
            nh_minus[h] + alpha_0
        )
    log_probs[-1] += np.log(alpha_c[c]) - np.log(alpha_0)

    return log_probs


##########################################
# waic
##########################################


@nb.jit(nopython=True)
def waic_calculation(x):
    """
    Numba-optimized function for WAIC comp
    """
    n_samples, n_iterations = x.shape

    mean_exp_x = np.zeros(n_samples)
    log_mean_exp_x = np.zeros(n_samples)
    mean_x = np.zeros(n_samples)
    var_x = np.zeros(n_samples)

    for i in range(n_samples):
        sum_exp = 0.0
        a = np.max(x[i, :])
        for j in range(n_iterations):
            sum_exp += np.exp(x[i, j] - a)
        mean_exp_x[i] = sum_exp / n_iterations
        log_mean_exp_x[i] = np.log(mean_exp_x[i])

    for i in range(n_samples):
        sum_x = 0.0
        for j in range(n_iterations):
            sum_x += x[i, j]
        mean_x[i] = sum_x / n_iterations

    for i in range(n_samples):
        sum_squared_diff = 0.0
        for j in range(n_iterations):
            diff = x[i, j] - mean_x[i]
            sum_squared_diff += diff * diff
        var_x[i] = sum_squared_diff / (n_iterations - 1)

    lppd = 0.0
    for i in range(n_samples):
        lppd += log_mean_exp_x[i]

    pWAIC2 = 0.0
    for i in range(n_samples):
        pWAIC2 += var_x[i]

    WAIC = -2 * lppd + 2 * pWAIC2

    return WAIC
