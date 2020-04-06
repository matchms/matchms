#
# Spec2Vec
#
# Copyright 2019 Netherlands eScience Center
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

# Import libraries
import numpy as np
import networkx as nx
import community
from networkx.algorithms.connectivity import minimum_st_edge_cut  # , minimum_st_node_cut
from networkx.algorithms.flow import shortest_augmenting_path
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib

# ----------------------------------------------------------------------------
# ---------------- Graph / networking related functions ----------------------
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# ---------------------- Functions to refine network -------------------------
# ----------------------------------------------------------------------------

# ----------------------------------------------------------------------------
# --------------------- Graph related plotting functions ---------------------
# ----------------------------------------------------------------------------

def plot_clustering_performance(data_collection,
                                labels,
                                total_num_nodes,
                                thres_well=0.6,
                                thres_poor=0.4,
                                title=None,
                                filename=None,
                                size_xy=(8, 5)):
    """ Plot cluster evaluations for all conditions found in data_collection.
    Cluster will be classified as "well clustered" if the mean(similarity) across
    all nodes is > thres_well. Or as "poorly clustered" if < thres_poor.
    Clusters with only one node (singletons) will be counted as "non-clustered".

    Args:
    --------
    data_collection: list of pandas.DataFrame()
        List of DataFrames as created by evaluate_clusters().
    labels: list
        List of labels for the different conditions found in data_collection.
    total_num_nodes: int
        Give the total number of nodes present in the network.
    thres_well: float
        Threshold above which clusters will be classified as "well clustered". Default = 0.6.
    thres_poor: float
        Threshold below which clusters will be classified as "poorly clustered". Default = 0.4.
    title: str
        Title for plot. Default = None
    filename: str
        If not none: save figure to file with given name.
    size_xy: tuple
        Figure size. Default is (8,5).
    """

    performance_data = []
    ymax = total_num_nodes
    legend_labels = [
        'well clustered nodes', 'poorly clustered nodes', 'non-clustered nodes'
    ]

    for cluster_data in data_collection:
        nodes_clustered_well = np.sum(
            cluster_data[(cluster_data['num_nodes'] > 1)
                         & (cluster_data['ref_sim_mean_nodes'] > thres_well)]
            ['num_nodes'].values)
        nodes_clustered_poor = np.sum(
            cluster_data[(cluster_data['num_nodes'] > 1)
                         & (cluster_data['ref_sim_mean_nodes'] < thres_poor)]
            ['num_nodes'].values)
        nodes_not_clustered = np.sum(
            cluster_data[(cluster_data['num_nodes'] < 2)]['num_nodes'].values)

        performance_data.append(
            [nodes_clustered_well, nodes_clustered_poor, nodes_not_clustered])

    fig = plt.figure(figsize=size_xy)
    ax = plt.subplot(111)
    plt.plot(labels, [x[0] / ymax for x in performance_data],
             'o-',
             color='crimson',
             label=legend_labels[0])
    plt.plot(labels, [x[1] / ymax for x in performance_data],
             'o-',
             color='teal',
             label=legend_labels[1])
    plt.plot(labels, [x[2] / ymax for x in performance_data],
             'o-',
             color='darkblue',
             alpha=0.6,
             label=legend_labels[2])
    plt.title(title)
    plt.ylabel("Fraction of total nodes")
    plt.xlabel("networking conditions")
    plt.legend()

    # Place legend
    # chartbox = ax.get_position()
    # ax.set_position([chartbox.x0, chartbox.y0, chartbox.width*0.8, chartbox.height])
    # ax.legend(loc='upper center', bbox_to_anchor=(1.25, 1))

    # Save figure to file
    if filename is not None:
        plt.savefig(filename, dpi=600)


def plot_cluster(g, filename=None):
    """ Very basic plotting function to inspect small to medium sized clusters (or networks).

    Args:
    --------
    g: networkx.Graph
        Networkx generated graph containing nodes and edges.
    filename: str
        If not none: save figure to file with given name.
    """
    if len(g.nodes) > 1:
        edges = [(u, v) for (u, v, d) in g.edges(data=True)]
        weights = [d['weight'] for (u, v, d) in g.edges(data=True)]
        weights = weights - 0.95 * np.min(weights)
        weights = weights / np.max(weights)

        # Positions for all nodes
        pos = nx.spring_layout(g)

        plt.figure(figsize=(12, 12))

        # Nodes
        nx.draw_networkx_nodes(g, pos, node_size=100)

        # Edges
        nx.draw_networkx_edges(g,
                               pos,
                               edgelist=edges,
                               width=4 * weights,
                               alpha=0.5)

        # Labels
        nx.draw_networkx_labels(g, pos, font_size=5, font_family='sans-serif')

        plt.axis('off')
        plt.show()

        if filename is not None:
            plt.savefig(filename, dpi=600)
    else:
        print("Given graph has not enough nodes to plot network.")


# ----------------------------------------------------------------------------
# -------------------------- Small helper functions --------------------------
# ----------------------------------------------------------------------------


def row_counts(array):
    """
    Function to find unique rows and count their occurences.
    """
    dt = np.dtype((np.void, array.dtype.itemsize * array.shape[1]))
    b = np.ascontiguousarray(array).view(dt)
    unq, cnt = np.unique(b, return_counts=True)
    unq = unq.view(array.dtype).reshape(-1, array.shape[1])

    return unq, cnt
