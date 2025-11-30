"""
Auxiliary matrix operations for pyesbm.
"""
import numpy as np
from scipy import sparse

def compute_mhk(Y, clustering_1, clustering_2):
    """Computes the MHK matrix using (fast) sparse matrix multiplication.

    Parameters
    ----------
    clustering_1 : list, optional
        First dimension clustering, if None uses self.clustering_1. By default None
    clustering_2 : list, optional
        Second dimension clustering, if None uses self.clustering_2. By default None

    Returns
    -------
    mhk : np.array
        MHK matrix
    """

    num_nodes_1 = len(clustering_1)
    num_clusters_1 = len(np.unique(clustering_1))

    num_nodes_2 = len(clustering_2)
    num_clusters_2 = len(np.unique(clustering_2))

    # using sparse matrices for speed, this sums up entries of
    # Y in depending on their block assignment.
    # m[h,k] is the sum of entries for blocks (h, k)
    clusters_1 = sparse.csr_matrix(
        (np.ones(num_nodes_1), (range(num_nodes_1), clustering_1)),
        shape=(num_nodes_1, num_clusters_1),
    )

    clusters_2 = sparse.csr_matrix(
        (np.ones(num_nodes_2), (range(num_nodes_2), clustering_2)),
        shape=(num_nodes_2, num_clusters_2),
    )

    mhk = clusters_1.T @ Y @ clusters_2
    return mhk

def compute_yuk(Y, item_clustering, num_clusters_items):
    """Computes the YUK matrix.

    Returns
    -------
    yuk : np.array
        YUK matrix
    """
    # using sparse matrices for speed, this sums up entries of
    # Y in depending on their block assignment.
    # y[u,k] is the sum of entries for user u and cluster k (in items)
    item_clusters = sparse.csr_matrix(
        (np.ones(item_clustering.shape[0]), (range(item_clustering.shape[0]), item_clustering)),
        shape=(item_clustering.shape[0], num_clusters_items),
    )

    yuk = Y @ item_clusters
    return yuk

def compute_yih(Y, num_users, user_clustering, num_clusters_users):
    """Computes the YIH matrix.

    Returns
    -------
    yih : np.array
        YIH matrix
    """
    # using sparse matrices for speed, this sums up entries of
    # Y in depending on their block assignment.
    # y[i,h] is the sum of entries for item i and cluster h (in users)
    user_clusters = sparse.csr_matrix(
        (np.ones(num_users), (range(num_users), user_clustering)),
        shape=(num_users, num_clusters_users),
    )

    yih = Y.T @ user_clusters
    return yih