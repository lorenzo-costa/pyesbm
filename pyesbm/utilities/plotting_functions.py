################################################
# auxialiaries for plotting

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import matplotlib.patheffects as pe

# --- Plotting configuration ---
rcparams = {
    "font.family": "serif",
    "text.usetex": True,
    "font.serif": ["Computer Modern Roman"],
    "font.size": 9,
    "figure.titlesize": 11,
    "legend.fontsize": 10,
    "legend.title_fontsize": 10.5,
    "lines.linewidth": 1,
    "axes.linewidth": 0.5,
    "axes.facecolor": "white",
    "axes.grid": False,
    "lines.markersize": 3,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
}

plt.rcParams.update(rcparams)


# function to plot the heatmap representation
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

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
    sort_clusters_by_size=None,
):
    if model.covariates_1 is not None:
        covariates_1_name = model.covariates_1.covariates[0].name
        covariates_1_type = model.covariates_1.covariates[0].cov_type
        if covariates_1_type == 'categorical':
            covariates_1 = model.covariates_1.covariates[0].cov_values
            covariates_1 = np.where(covariates_1>0)[1]
        elif covariates_1_type == 'count':
            covariates_1 = model.covariates_1.covariates[0].cov_values.sum(axis=1)

    if model.covariates_2 is not None:
        covariates_2_name = model.covariates_2.covariates[0].name
        covariates_2_type = model.covariates_2.covariates[0].cov_type
        if covariates_2_type == 'categorical':
            covariates_2 = model.covariates_2.covariates[0].cov_values
            covariates_2 = np.where(covariates_2>0)[1]
        elif covariates_2_type == 'count':
            covariates_2 = model.covariates_2.covariates[0].cov_values.sum(axis=1)

    if sort_clusters_by_size is None:
        if bipartite is True:
            sort_clusters_by_size = True
        else:
            sort_clusters_by_size = False

    clustering_1 = model.clustering_1.copy()
    clustering_2 = model.clustering_2.copy() if bipartite else model.clustering_1.copy()

    Y = model.Y
    if capped is not None:
        Y = np.clip(Y, 0, capped)

    # Sort clusters by size
    if sort_clusters_by_size is True:
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
    else:
        cluster_sizes_1 = {}
        for cluster in clustering_1:
            cluster_sizes_1[cluster] = cluster_sizes_1.get(cluster, 0) + 1

        if bipartite:
            cluster_sizes_2 = {}
            for cluster in clustering_2:
                cluster_sizes_2[cluster] = cluster_sizes_2.get(cluster, 0) + 1
        else:
            cluster_sizes_2 = cluster_sizes_1

        sorted_clusters_1 = sorted(
            cluster_sizes_1.keys(), key=lambda x: cluster_sizes_1[x], reverse=True
        )
        sorted_clusters_2 = sorted(
            cluster_sizes_2.keys(), key=lambda x: cluster_sizes_2[x], reverse=True
        )

        idx_sort_1 = np.argsort(clustering_1)
        idx_sort_2 = np.argsort(clustering_2)

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

    # Set the layout
    if covariates_1 is not None or covariates_2 is not None:
        width_ratios = [0.02, 1] if covariates_1 is not None else [0, 1]
        
        # CHANGED: If Cov 2 exists, Row 0 is thin (Top Bar), Row 1 is fat (Heatmap)
        height_ratios = [0.02, 1] if covariates_2 is not None else [0, 1]
        
        gs = fig.add_gridspec(
            2,
            2,
            width_ratios=width_ratios,
            height_ratios=height_ratios,
            wspace=0.01,
            hspace=0.01,
        )

        # CHANGED: Heatmap moves to Bottom Right [1, 1]
        ax_heatmap = fig.add_subplot(gs[1, 1])

        # CHANGED: User covariates (Left) move to Bottom Left [1, 0]
        if covariates_1 is not None:
            ax_cov_1 = fig.add_subplot(gs[1, 0], sharey=ax_heatmap)

        # CHANGED: Item covariates (Top) move to Top Right [0, 1]
        if covariates_2 is not None:
            ax_cov_2 = fig.add_subplot(gs[0, 1], sharex=ax_heatmap)
    else:
        ax_heatmap = fig.add_subplot(111)

    # Plot main heatmap
    Y_sorted = Y[idx_sort_1, :][:, idx_sort_2]

    if not bipartite:
        mask = np.triu(np.ones_like(Y_sorted, dtype=bool))
    else:
        mask = None

    # brute force seaborn not to use white
    if np.max(Y_sorted) == 1:
        Y_sorted[0, 0] = 2

    heatmap = sns.heatmap(
        Y_sorted,
        mask=mask if triangular_mask else None,
        ax=ax_heatmap,
        cbar_kws={"shrink": 0.8},
        cmap="rocket",
    )

    # Add cluster boundaries
    for boundary in cluster_boundaries_1:
        ax_heatmap.axhline(y=boundary, color="white", linewidth=2)
    for boundary in cluster_boundaries_2:
        ax_heatmap.axvline(x=boundary, color="white", linewidth=2)

    # Add cluster labels if requested
    if add_labels:
        for i in range(len(cluster_boundaries_1) - 1):
            cluster_label = sorted_clusters_1[i]
            mid_point = Y.shape[0] - (cluster_boundaries_1[i] + cluster_boundaries_1[i + 1]) / 2
            ax_heatmap.text(
                -0.05,  # slightly above top edge
                mid_point / Y.shape[0],  # convert mid_point into [0,1]
                f"C{cluster_label}",
                transform=ax_heatmap.transAxes,
                verticalalignment="center",
                horizontalalignment="right",
                fontsize=12,
                zorder=10,
            )
        for i in range(len(cluster_boundaries_2) - 1):
            cluster_label = sorted_clusters_2[i]
            mid_point = (cluster_boundaries_2[i] + cluster_boundaries_2[i + 1]) / 2
            ax_heatmap.text(
                mid_point / Y.shape[1],  # convert mid_point into [0,1]
                1.12,  # slightly above top edge
                f"C{cluster_label}",
                transform=ax_heatmap.transAxes,
                verticalalignment="top",
                horizontalalignment="center",
                fontsize=12,
                zorder=10,
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
        if covariates_1_type == 'categorical':
            # Get unique categories and assign colors
            unique_user_cats = sorted(set(sorted_user_covs))
            user_cmap = plt.get_cmap("inferno", len(unique_user_cats))
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
                Rectangle((0, 0), 0.5, 0.5, color=user_color_dict[cat], label=str(cat))
                for cat in unique_user_cats
            ]
            ax_cov_1.legend(
                handles=user_legend_elements,
                title=covariates_1_name,
                bbox_to_anchor=(-3, 0.15),
                loc="center right",
            )
            
        elif covariates_1_type == "count":
            cov_arr = np.array(sorted_user_covs).astype(float)

            min_val = np.min(cov_arr)
            max_val = np.max(cov_arr)
            n_levels = int(max_val - min_val + 1)  # discrete bins by integer value

            # Discrete colormap (viridis or any you prefer)
            count_cmap = plt.get_cmap("inferno", n_levels)

            # 1-column image
            user_cov_matrix = cov_arr.reshape(-1, 1)

            # Plot discrete gradient
            user_img = ax_cov_1.imshow(
                user_cov_matrix,
                aspect="auto",
                origin="upper",
                cmap=count_cmap,
                extent=[0, 1, len(cov_arr), 0],
                vmin=min_val - 0.5,
                vmax=max_val + 0.5,
            )

            ax_cov_1.set_xticks([])
            ax_cov_1.set_yticks([])
            
            # Add a discrete colorbar below the covariate bar
            box = ax_cov_1.get_position()

            legend_height = 0.025 

            ax_grad = fig.add_axes([
                box.x0 - 0.1,               
                box.y0 - 0.05,
                box.width+0.05,            
                legend_height
            ])

            gradient = np.linspace(min_val, max_val, 256).reshape(1, -1)

            ax_grad.imshow(
                gradient,
                aspect="auto",
                cmap=count_cmap,
                extent=[min_val, max_val, 0, 1],
            )

            ax_grad.set_yticks([])

            ax_grad.set_xticks([min_val, max_val])
            ax_grad.set_xticklabels(
                [f"{min_val}", f"{max_val}"],
                fontsize=11
            )

            ax_grad.set_title(
                f"{covariates_1_name}",
                fontsize=12,
                pad=4                
            )

            # # Use colorbar instead of legend (numeric variable)
            # cbar = plt.colorbar(
            #     user_img,
            #     ax=ax_cov_1,
            #     fraction=0.15,
            #     pad=0.2,
            # )
            
            # cbar.set_label(covariates_1_name)
            # cbar.set_ticks(np.arange(min_val, max_val + 1))

    if covariates_2 is not None:
        # Extract covariates for sorted items
        if isinstance(covariates_2, dict):
            sorted_item_covs = [covariates_2[i] for i in idx_sort_2]
        else:
            sorted_item_covs = [covariates_2[i] for i in idx_sort_2]
            
        if covariates_2_type == 'categorical':
            # Get unique categories and assign colors
            unique_item_cats = sorted(set(sorted_item_covs))
            item_cmap = plt.get_cmap("inferno", len(unique_item_cats))
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
                title=covariates_2_name,
                bbox_to_anchor=(1.2, 5),
                loc="center right",
            )
        elif covariates_2_type == "count":
            cov_arr = np.array(sorted_item_covs).astype(float)

            min_val = np.min(cov_arr)
            max_val = np.max(cov_arr)
            n_levels = int(max_val - min_val + 1)

            count_cmap = plt.get_cmap("inferno", n_levels)

            # Reshape for horizontal bar (1 row, N columns)
            item_cov_matrix = cov_arr.reshape(1, -1)

            # Align position exactly with heatmap width
            heatmap_pos = ax_heatmap.get_position()
            ax_cov_2.set_position(
                [
                    heatmap_pos.x0,
                    ax_cov_2.get_position().y0,
                    heatmap_pos.width,
                    ax_cov_2.get_position().height,
                ]
            )

            # Plot discrete gradient (horizontal extent)
            item_img = ax_cov_2.imshow(
                item_cov_matrix,
                aspect="auto",
                origin="upper",
                cmap=count_cmap,
                extent=[0, len(cov_arr), 1, 0],
                vmin=min_val - 0.5,
                vmax=max_val + 0.5,
            )

            ax_cov_2.set_xticks([])
            ax_cov_2.set_yticks([])

            # Add discrete colorbar/legend below
            box = ax_cov_2.get_position()
            
            legend_height = 0.025

            ax_grad = fig.add_axes([
                box.x0 + 0.625,
                box.y0+0.05, 
                box.width-0.54,
                legend_height
            ])

            gradient = np.linspace(min_val, max_val, 256).reshape(1, -1)

            ax_grad.imshow(
                gradient,
                aspect="auto",
                cmap=count_cmap,
                extent=[min_val, max_val, 0, 1],
            )

            ax_grad.set_yticks([])
            ax_grad.set_xticks([min_val, max_val])
            ax_grad.set_xticklabels(
                [f"{min_val:.0f}", f"{max_val:.0f}"],
                fontsize=11
            )
            
            ax_grad.set_title(
                f"{covariates_2_name}",
                fontsize=12,
                pad=4 # Adjust padding as needed
            )
            
            # Label below the gradient bar
            #ax_grad.set_xlabel(covariates_2_name, fontsize=12)


    if save_path is not None:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()