"""
Auxiliary matrix operations for pyesbm.
"""

import numpy as np
from scipy import sparse


def compute_mhk(Y, clustering_1, clustering_2):
    """Computes the MHK matrix using (fast) sparse matrix multiplication.

    Mhk matrix stores the sum of edges between cluster h and cluster k.
    For unipartite graphs, clustering_1 and clustering_2 should be the same.
    For bipartite graphs, clustering_1 and clustering_2 should be for different dimensions

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


def compute_y_values(Y, clustering, num_nodes, num_clusters):
    """Computes the YUK matrix.

    Yuk matrix stores the sum of edges between node u and cluster k.
    For unipartite graphs, k representes clusters in the same dimension as u so pass
    clustering, num_nodes and num_clusters for the same dimension.
    For bipartite graphs, k represents clusters in the opposite dimension as u so pass
    clustering, num_nodes and num_clusters for the opposite dimension.

    Returns
    -------
    yuk : np.array
        YUK matrix
    """
    clusters = sparse.csr_matrix(
        (np.ones(num_nodes), (range(num_nodes), clustering)),
        shape=(num_nodes, num_clusters),
    )
    y_values = Y.T @ clusters
    return y_values
