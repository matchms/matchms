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


def refine_network(graph_main,
                   similars_idx,
                   similars,
                   weigh_bounds=(0.6, 1),
                   filename=None,
                   max_cluster_size=100,
                   min_cluster_size=10,
                   max_search_steps=1000,
                   max_cuts=2,
                   max_split_iterations=10,
                   basic_splitting=True,
                   dilation=False):
    """
    Args:
    -------
    """
    # Split graph into separate clusters
    graphs = list(nx.connected_component_subgraphs(graph_main))

    links_removed = []
    links_added = []

    # n_cluster = len(graphs)
    cluster_max = np.max([len(x.nodes) for x in graphs])
    counter = 0

    print(20 * '---')
    while cluster_max > max_cluster_size and counter < max_split_iterations:
        print("Splitting iteration:", counter + 1, "Max cluster size =",
              cluster_max, '\n')
        graph_main, links = split_cluster(graph_main.copy(),
                                          max_cluster_size=max_cluster_size,
                                          min_cluster_size=min_cluster_size,
                                          max_search_steps=max_search_steps,
                                          max_cuts=max_cuts,
                                          multiple_cuts_per_level=True)
        links_removed.extend(links)

        # Split updated graph into separate clusters
        graphs = list(nx.connected_component_subgraphs(graph_main))
        cluster_max = np.max([len(x.nodes) for x in graphs])
        counter += 1

    if basic_splitting:
        print(20 * '---')
        print("Extra splitting step to sanitize clusters.")
        graph_main, links = split_cluster(
            graph_main,
            max_cluster_size=2 *
            min_cluster_size,  # ! here we try to 'sanitize most clusters'
            min_cluster_size=min_cluster_size,
            max_search_steps=max_search_steps,
            max_cuts=1,
            multiple_cuts_per_level=False)
        links_removed.extend(links)

    if dilation:
        print(20 * '---')
        print("Runing dilation function for smaller clusters <",
              min_cluster_size)
        graph_main, links = dilate_cluster(graph_main,
                                           similars_idx,
                                           similars,
                                           max_cluster_size=max_cluster_size,
                                           min_cluster_size=min_cluster_size,
                                           min_weight=weigh_bounds[0])
        links_added.extend(links)

    if filename is not None:
        # Export graph for drawing (e.g. using Cytoscape)
        nx.write_graphml(graph_main, filename)
        print("Network stored as graphml file under: ", filename)

    return graph_main, links_added, links_removed


# ----------------------------------------------------------------------------
# -------------------- Functions to evaluate networks ------------------------
# ----------------------------------------------------------------------------


def evaluate_clusters(graph_main, m_sim_ref):
    """ Evaluate separate clusters of network based on given reference matrix.

    Args:
    -------
    graph_main: networkx graph
        Graph, e.g. made using create_network() function. Based on networkx.
    m_sim_ref: numpy array
        2D array with all reference similarity values between all-vs-all nodes.
    """

    # Split graph into separate clusters
    graphs = list(nx.connected_component_subgraphs(graph_main))

    num_nodes = []
    num_edges = []
    ref_sim_mean_edges = []
    ref_sim_var_edges = []
    ref_sim_mean_nodes = []
    ref_sim_var_nodes = []

    # Loop through clusters
    for graph in graphs:
        num_nodes.append(len(graph.nodes))
        if len(graph.edges) > 0:  # no edges for singletons
            num_edges.append(len(graph.edges))

            edges = list(graph.edges)
            mol_sim_edges = np.array([m_sim_ref[x] for x in edges])
            mol_sim_edges = np.nan_to_num(mol_sim_edges)
            ref_sim_mean_edges.append(np.mean(mol_sim_edges))
            ref_sim_var_edges.append(np.var(mol_sim_edges))
        else:
            num_edges.append(0)
            ref_sim_mean_edges.append(0)
            ref_sim_var_edges.append(0)

        nodes = list(graph.nodes)
        mean_mol_sims = []
        for node in nodes:
            mean_mol_sims.append(m_sim_ref[node, nodes])

        ref_sim_mean_nodes.append(np.mean(mean_mol_sims))
        ref_sim_var_nodes.append(np.var(mean_mol_sims))

    zipped = zip(num_nodes, num_edges, ref_sim_mean_edges, ref_sim_var_edges, ref_sim_mean_nodes, ref_sim_var_nodes)
    cluster_data = pd.DataFrame(list(zipped), columns=['num_nodes', 'num_edges', 'ref_sim_mean_edges',
                                                       'ref_sim_var_edges', 'ref_sim_mean_nodes', 'ref_sim_var_nodes'])
    return cluster_data


def evaluate_clusters_louvain(graph_main, m_sim_ref, resolution=1.0):
    """ Cluster given network using Louvain algorithm.
    Then evaluate clusters of network based on given reference matrix.

    Args:
    -------
    graph_main: networkx.Graph
        Graph, e.g. made using create_network() function. Based on networkx.
    m_sim_ref: numpy array
        2D array with all reference similarity values between all-vs-all nodes.
    resolution: float
        Louvain algorithm resolution parameter. Will change size of communities.
        See also: https://python-louvain.readthedocs.io/en/latest/api.html Default=1.0
    """
    plt.style.use('ggplot')
    # Find clusters using Louvain algorithm (and python-louvain library)
    communities = community.best_partition(graph_main,
                                           weight='weight',
                                           resolution=resolution)
    nx.set_node_attributes(graph_main, communities, 'modularity')

    clusters = []
    for cluster_id in set(communities.values()):
        cluster = [
            nodes for nodes in communities.keys()
            if communities[nodes] == cluster_id
        ]
        clusters.append(cluster)

    num_nodes = []
    ref_sim_mean_nodes = []
    ref_sim_var_nodes = []

    for cluster in clusters:
        num_nodes.append(len(cluster))
        mean_mol_sims = []
        for node in cluster:
            mean_mol_sims.append(m_sim_ref[node, cluster])

        ref_sim_mean_nodes.append(np.mean(mean_mol_sims))
        ref_sim_var_nodes.append(np.var(mean_mol_sims))

    cluster_data = pd.DataFrame(
        list(zip(num_nodes, ref_sim_mean_nodes, ref_sim_var_nodes)),
        columns=['num_nodes', 'ref_sim_mean_nodes', 'ref_sim_var_nodes'])

    return graph_main, cluster_data


# ----------------------------------------------------------------------------
# --------------------- Graph related plotting functions ---------------------
# ----------------------------------------------------------------------------
def plots_cluster_evaluations(cluster_data_collection,
                              m_sim_ref,
                              total_num_nodes,
                              size_bins,
                              labels,
                              title,
                              filename=None):
    """ Plot cluster sizes and mean node similarity.

    Args:
    --------
    cluster_data_collection:  list
        List of cluster data for all scenarios to be plotted.
    m_sim_ref: numpy array
        2D array with all reference similarity values between all-vs-all nodes.
    total_num_nodes: int
        Total number of nodes of graph.
    size_bins: list of int
        List of bins for cluster sizes.
    labels: list of str
        List of labels for all scenarios in list of cluster_data_collection.
    title: str
        Title for plot. Default = None
    filename: str
        If not None: save figure to file with given name.
    """

    plt.style.use('ggplot')
    fig = plt.figure(figsize=(12, 5))
    ax = plt.subplot(111)

    num_plots = len(cluster_data_collection)
    cmap = matplotlib.cm.get_cmap('inferno')  # 'Spectral')
    bins = [0] + [x + 1 for x in size_bins]
    x_labels = ['<' + str(bins[1])]
    x_labels += [
        str(bins[i]) + '-' + str(bins[i + 1] - 1)
        for i in range(1,
                       len(bins) - 2)
    ]
    x_labels += ['>' + str(bins[-2] - 1)]

    for count, cluster_data in enumerate(cluster_data_collection):

        num_elements = []
        mean_edge_sim = []
        mean_node_sim = []
        for i in range(len(bins) - 1):
            num_elements.append(
                np.sum(cluster_data[(cluster_data['num_nodes'] < bins[i + 1])
                                    & (cluster_data['num_nodes'] > bins[i])]
                       ['num_nodes'].values))

            if 'ref_sim_mean_edges' in cluster_data.columns:
                mean_edge_sim.append(
                    np.mean(
                        cluster_data[(cluster_data['num_nodes'] < bins[i + 1])
                                     & (cluster_data['num_nodes'] > bins[i])]
                        ['ref_sim_mean_edges'].values))

            mean_node_sim.append(
                np.mean(cluster_data[(cluster_data['num_nodes'] < bins[i + 1])
                                     & (cluster_data['num_nodes'] > bins[i])]
                        ['ref_sim_mean_nodes'].values))

        num_elements[0] = total_num_nodes - np.sum(num_elements[1:])
        if 'ref_sim_mean_edges' in cluster_data.columns:
            if np.isnan(mean_edge_sim[0]):
                mean_edge_sim[0] = 0

        plt.scatter(x_labels,
                    mean_node_sim,
                    s=num_elements,
                    facecolor="None",
                    edgecolors=[cmap(count / num_plots)],
                    lw=3,
                    alpha=0.7,
                    label=labels[count])

    plt.xlabel('cluster size')
    plt.ylabel('mean molecular similarity of nodes in cluster')
    chartbox = ax.get_position()
    ax.set_position(
        [chartbox.x0, chartbox.y0, chartbox.width * 0.8, chartbox.height])
    lgnd = ax.legend(loc='upper center', bbox_to_anchor=(1.12, 1))
    for i in range(num_plots):
        lgnd.legendHandles[i]._sizes = [30]

    plt.title(title)

    # Save figure to file
    if filename is not None:
        plt.savefig(filename, dpi=600)


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
