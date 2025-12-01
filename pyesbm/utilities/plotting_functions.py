################################################
# auxialiaries for plotting

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle


# function to plot the heatmap representation
def plot_heatmap(
    model,
    covariates_1=None,
    covariates_2=None,
    add_labels=True,
    size=(12, 10),
    capped=None,
    save_path=None,
    bipartite=True,
    triangular_mask=False,
):
    if covariates_1 is not None:
        covariates_1 = covariates_1.copy()
    if covariates_2 is not None:
        covariates_2 = covariates_2.copy()

    clustering_1 = model.clustering_1.copy()
    clustering_2 = model.clustering_2.copy() if bipartite else model.clustering_1.copy()

    Y = model.Y
    if capped is not None:
        Y = np.clip(Y, 0, capped)

    # Sort clusters by size
    cluster_sizes_1 = {}
    for cluster in clustering_1:
        cluster_sizes_1[cluster] = cluster_sizes_1.get(cluster, 0) + 1

    cluster_sizes_2 = {}
    for cluster in clustering_2:
        cluster_sizes_2[cluster] = cluster_sizes_2.get(cluster, 0) + 1

    sorted_clusters_1 = sorted(
        cluster_sizes_1.keys(), key=lambda x: cluster_sizes_1[x], reverse=True
    )
    sorted_clusters_2 = sorted(
        cluster_sizes_2.keys(), key=lambda x: cluster_sizes_2[x], reverse=True
    )

    cluster_rank_1 = {cluster: i for i, cluster in enumerate(sorted_clusters_1)}
    cluster_rank_2 = {cluster: i for i, cluster in enumerate(sorted_clusters_2)}

    idx_sort_1 = sorted(
        np.arange(Y.shape[0]), key=lambda i: cluster_rank_1[clustering_1[i]]
    )
    idx_sort_2 = sorted(
        np.arange(Y.shape[1]), key=lambda i: cluster_rank_2[clustering_2[i]]
    )

    sorted_cluster_list_1 = [clustering_1[i] for i in idx_sort_1]
    sorted_cluster_list_2 = [clustering_2[i] for i in idx_sort_2]

    cluster_boundaries_1 = [0]
    prev_cluster = sorted_cluster_list_1[0]
    for i, cluster in enumerate(sorted_cluster_list_1[1:], 1):
        if cluster != prev_cluster:
            cluster_boundaries_1.append(i)
            prev_cluster = cluster
    cluster_boundaries_1.append(len(sorted_cluster_list_1))

    cluster_boundaries_2 = [0]
    prev_cluster = sorted_cluster_list_2[0]
    for i, cluster in enumerate(sorted_cluster_list_2[1:], 1):
        if cluster != prev_cluster:
            cluster_boundaries_2.append(i)
            prev_cluster = cluster
    cluster_boundaries_2.append(len(sorted_cluster_list_2))

    # Create figure with appropriate size and layout for covariates
    fig = plt.figure(figsize=size)

    # Set the layout based on whether we have covariates
    if covariates_1 is not None or covariates_2 is not None:
        width_ratios = [0.02, 1] if covariates_1 is not None else [0, 1]
        height_ratios = [1, 0.02] if covariates_2 is not None else [1, 0]
        gs = fig.add_gridspec(
            2,
            2,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
            wspace=0.01,
            hspace=0.01,
        )

        # Main heatmap
        ax_heatmap = fig.add_subplot(gs[0, 1])

        # User covariates on the left
        if covariates_1 is not None:
            ax_cov_1 = fig.add_subplot(gs[0, 0], sharey=ax_heatmap)

        # Item covariates on the bottom
        if covariates_2 is not None:
            ax_cov_2 = fig.add_subplot(gs[1, 1], sharex=ax_heatmap)
    else:
        ax_heatmap = fig.add_subplot(111)

    # Plot main heatmap
    Y_sorted = Y[idx_sort_1, :][:, idx_sort_2]

    if not bipartite:
        mask = np.triu(np.ones_like(Y_sorted, dtype=bool))
    else:
        mask = None

    heatmap = sns.heatmap(
        Y_sorted,
        mask=mask if triangular_mask else None,              
        ax=ax_heatmap,
        cbar_kws={"shrink": 0.8}
    )

    # Add cluster boundaries
    for boundary in cluster_boundaries_1:
        ax_heatmap.axhline(y=boundary, color="white", linewidth=2)
    if bipartite is True:
        for boundary in cluster_boundaries_2:
            ax_heatmap.axvline(x=boundary, color="white", linewidth=2)

    # Add cluster labels if requested
    if add_labels:
        for i in range(len(cluster_boundaries_1) - 1):
            cluster_label = sorted_clusters_1[i]
            mid_point = (
                cluster_boundaries_1[i] + cluster_boundaries_1[i + 1]
            ) / 2
            ax_heatmap.text(
                -0.5,
                mid_point,
                f"C{cluster_label}",
                verticalalignment="center",
                horizontalalignment="right",
                fontsize=12,
            )
        if bipartite is True:
            for i in range(len(cluster_boundaries_2) - 1):
                cluster_label = sorted_clusters_2[i]
                mid_point = (
                    cluster_boundaries_2[i] + cluster_boundaries_2[i + 1]
                ) / 2
                ax_heatmap.text(
                    mid_point,
                    Y.shape[0] + 0.5,
                    f"C{cluster_label}",
                    verticalalignment="top",
                    horizontalalignment="center",
                    fontsize=12,
                )

    ax_heatmap.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        labelbottom=False,
        left=False,
        right=False,
        labelleft=False,
    )

    # Process and plot user covariates if provided
    if covariates_1 is not None:
        sorted_user_covs = [covariates_1[i] for i in idx_sort_1]

        # Get unique categories and assign colors
        unique_user_cats = sorted(set(sorted_user_covs))
        user_cmap = plt.get_cmap("tab10", len(unique_user_cats))
        user_color_dict = {cat: user_cmap(i) for i, cat in enumerate(unique_user_cats)}

        # Create color matrix for user covariates
        n_users = len(sorted_user_covs)
        user_cov_matrix = np.zeros((n_users, 1))
        for i, cov in enumerate(sorted_user_covs):
            user_cov_matrix[i, 0] = unique_user_cats.index(cov)

        # Plot user covariates with proper alignment
        # The extent parameter is set to match the axis limits of the heatmap
        user_img = ax_cov_1.imshow(
            user_cov_matrix,
            aspect="auto",
            origin="upper",
            cmap=user_cmap,
            extent=[0, 1, n_users, 0],
        )

        ax_cov_1.set_xticks([])
        ax_cov_1.set_yticks([])

        # Add a legend for user covariates
        user_legend_elements = [
            Rectangle((0, 0), 1, 1, color=user_color_dict[cat], label=str(cat))
            for cat in unique_user_cats
        ]
        ax_cov_1.legend(
            handles=user_legend_elements,
            title="User Covariates",
            bbox_to_anchor=(0.1, 0.1),
            loc="center right",
        )

    if covariates_2 is not None:
        # Extract covariates for sorted items
        if isinstance(covariates_2, dict):
            sorted_item_covs = [covariates_2[i] for i in idx_sort_2]
        else:
            sorted_item_covs = [covariates_2[i] for i in idx_sort_2]

        # Get unique categories and assign colors
        unique_item_cats = sorted(set(sorted_item_covs))
        item_cmap = plt.get_cmap("tab10", len(unique_item_cats))
        item_color_dict = {cat: item_cmap(i) for i, cat in enumerate(unique_item_cats)}

        # Create color matrix for item covariates
        n_items = len(sorted_item_covs)
        item_cov_matrix = np.zeros((1, n_items))
        for i, cov in enumerate(sorted_item_covs):
            item_cov_matrix[0, i] = unique_item_cats.index(cov)

        # Plot item covariates with proper alignment
        # The extent parameter is set to match the axis limits of the heatmap
        heatmap_pos = ax_heatmap.get_position()
        ax_cov_2.set_position(
            [
                heatmap_pos.x0,
                ax_cov_2.get_position().y0,
                heatmap_pos.width,
                ax_cov_2.get_position().height,
            ]
        )

        # Plot item covariates ensuring correct width
        item_img = ax_cov_2.imshow(
            item_cov_matrix,
            aspect="auto",
            origin="upper",
            cmap=item_cmap,
            extent=[0, n_items, 1, 0],
        )

        ax_cov_2.set_xticks([])
        ax_cov_2.set_yticks([])

        item_legend_elements = [
            Rectangle((0, 0), 1, 1, color=item_color_dict[cat], label=str(cat))
            for cat in unique_item_cats
        ]
        ax_cov_2.legend(
            handles=item_legend_elements,
            title="Item Covariates",
            bbox_to_anchor=(0.25, -5),
            loc="center right",
        )

    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()
