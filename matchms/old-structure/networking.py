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


def create_network(similars_idx,
                   similars,
                   max_links=10,
                   cutoff=0.7,
                   link_method='single'):
    """
    Function to create network from given top-n similarity values.

    Args:
    --------
    similars_idx: numpy array
        Array with indices of top-n most similar nodes.
    similars: numpy array
        Array with similarity values of top-n most similar nodes.
    max_links: int
        Maximum number of links to add per node. Default = 10.
        Due to incoming links, total number of links per node can be higher.
    cutoff: float
        Threshold for given similarities. Edges/Links will only be made for
        similarities > cutoff. Default = 0.7.
    link_method: str
        Chose between 'single' and 'mutual'. 'single will add all links based
        on individual nodes. 'mutual' will only add links if that link appears
        in the given top-n list for both nodes.
    """

    dimension = similars_idx.shape[0]

    # Initialize network graph, add nodes
    msnet = nx.Graph()
    msnet.add_nodes_from(np.arange(0, dimension))

    # Add edges based on global threshold for weights
    for i in range(0, dimension):
        idx = np.where(similars[i, :] > cutoff)[0][:max_links]
        if link_method == "single":
            new_edges = [(i, int(similars_idx[i, x]), float(similars[i, x]))
                         for x in idx if similars_idx[i, x] != i]
        elif link_method == "mutual":
            new_edges = [(i, int(similars_idx[i, x]), float(similars[i, x]))
                         for x in idx
                         if similars_idx[i, x] != i and i in similars_idx[x, :]
                         ]
        else:
            print("Link method not kown")
        msnet.add_weighted_edges_from(new_edges)

    return msnet


def sample_cuts(graph, max_steps=1000, max_cuts=1):
    """ Function to help find critical links in the given graph.
    Critical links here are links which -once removed- would disconnect considerable
    parts of the network. Those links are searched for by counting minimum cuts between
    a large number of node pairs (up to max_steps pairs will be explored).
    If more pairs exist than max_steps allows to explore, pick max_steps random pairs.

    Args:
    -------
    graph: networkx graph
        Graph of individual cluster (created using networkx).
    max_steps
        Up to max_steps pairs will be explored to search for cuts. Default = 1000.
    max_cuts
        Maximum numbers of links allowed to be cut. Default = 1.
    """

    num_nodes = graph.number_of_nodes()
    # num_edges = graph.number_of_edges()

    # Make list of all pairs within graph
    nodes = np.array(graph.nodes)
    pairs = np.array(np.meshgrid(nodes, nodes)).T
    remove_diagonal = np.array([(i * num_nodes + i) for i in range(num_nodes)])
    pairs = np.delete(pairs.reshape(-1, 2), remove_diagonal, axis=0)

    sampled_cuts = []
    if pairs.shape[0] <= max_steps:
        max_steps = pairs.shape[0]
    else:
        # If more pairs exist than max_steps allows to explore, pick max_steps random pairs.
        choices = np.random.choice(np.arange(pairs.shape[0]),
                                   max_steps,
                                   replace=False)
        pairs = pairs[choices, :]

    for pair in pairs:
        cuts = minimum_st_edge_cut(graph,
                                   pair[0],
                                   pair[1],
                                   flow_func=shortest_augmenting_path)
        # nx.node_connectivity(graphs[4], 592, 376)
        # cuts = nx.minimum_st_edge_cut(graph, pair[0], pair[1])
        # cuts = nx.minimum_edge_cut(graph, pair[0], pair[1])#, flow_func=shortest_augmenting_path)
        if len(cuts) <= max_cuts:
            sampled_cuts.append(cuts)

    return sampled_cuts


def weak_link_finder(graph, max_steps=1000, max_cuts=1):
    """ Function to detect critical links in the given graph.
    Critical links here are links which -once removed- would disconnect considerable
    parts of the network. Those links are searched for by counting minimum cuts between
    a large number of node pairs (up to max_steps pairs will be explored).
    If more pairs exist than max_steps allows to explore, pick max_steps random pairs.

    Args:
    -------
    graph: networkx graph
        Graph of individual cluster (created using networkx).
    max_steps
        Up to max_steps pairs will be explored to search for cuts. Default = 1000.
    max_cuts
        Maximum numbers of links allowed to be cut. Default = 1.
    """

    sampled_cuts = sample_cuts(graph, max_steps=max_steps, max_cuts=max_cuts)

    sampled_cuts_len = [len(x) for x in sampled_cuts]
    proposed_cuts = []
    for min_cuts in list(set(sampled_cuts_len)):
        sampled_cuts_select = [
            list(x)[:min_cuts] for x in sampled_cuts if len(x) == min_cuts
        ]

        sampled_cuts_select = np.array(sampled_cuts_select)
        # Sort array
        if min_cuts > 1:
            sampled_cuts_select = np.sort(np.sort(sampled_cuts_select, axis=2),
                                          axis=1)
        else:
            sampled_cuts_select = np.sort(sampled_cuts_select, axis=2)

        # Find unique cuts and count occurences
        cuts_unique, cuts_count = row_counts(
            sampled_cuts_select.reshape(-1, min_cuts * 2))

        # Return most promising cuts
        proposed_cuts.append((min_cuts, cuts_unique, cuts_count))

    return proposed_cuts


def dilate_cluster(graph_main,
                   similars_idx,
                   similars,
                   max_cluster_size=100,
                   min_cluster_size=10,
                   max_per_node=1,
                   max_per_cluster=None,
                   min_weight=0.5):
    """ Add more links to clusters that are < min_cluster_size.
    This function is in particular made to avoid small remaining clusters or singletons.

    Will only add links if they won't lead to clusters > max_cluster_size,
    and if the links have weights > min_weight.
    Starts iteratively from highest weight links that are not yet part of the network
    (out of given top-n links).

    Args:
    --------
    graph_main: networkx graph
        Graph, e.g. made using create_network() function. Based on networkx.
    similars_idx: numpy array
        Array with indices of top-n most similar nodes.
    similars: numpy array
        Array with similarity values of top-n most similar nodes.
    max_cluster_size: int
        Maximum desired size of clusters. Default = 100.
    min_cluster_size: int
        Minimum desired size of clusters. Default = 10.
    max_per_node: int
        Only add the top max_addition ones per cluster. Default = 1.
    max_per_cluster: int, None
        Only add the top max_addition ones per cluster. Ignore if set to None. Default = None.
    min_weight: float
        Set minimum weight to be considered for making link. Default = 0.5.
    """

    links_added = []

    # Split graph into separate clusters
    graphs = list(nx.connected_component_subgraphs(graph_main))

    for graph in graphs:
        cluster_size = len(graph.nodes)
        if cluster_size < min_cluster_size:
            best_scores = []
            potential_links = []

            for ID in graph.nodes:
                nodes_connected = []
                for key in graph[ID].keys():
                    nodes_connected.append(key)

                potential_new_links = [(i, x)
                                       for i, x in enumerate(similars_idx[ID])
                                       if x not in nodes_connected and x != ID]
                best_score_arr = similars[ID][[
                    x[0] for x in potential_new_links
                ]]
                select = np.where(
                    best_score_arr >= min_weight)[0][:max_per_node]
                # if best_score >= min_weight:
                if select.shape[0] > 0:
                    for s in select:
                        best_scores.append(best_score_arr[s])
                        potential_link = (ID,
                                          [x[1]
                                           for x in potential_new_links][s])
                        potential_links.append(potential_link)

            if max_per_cluster is None:
                selected_candidates = np.argsort(best_scores)[::-1]
            else:
                # Only add the top max_addition ones
                selected_candidates = np.argsort(
                    best_scores)[::-1][:max_per_cluster]

            for ID in selected_candidates:
                # node_id = list(graph.nodes)[ID]
                node_id = potential_links[ID][0]

                # Only add link if no cluster > max_cluster_size is formed by it
                if (len(
                        nx.node_connected_component(graph_main,
                                                    potential_links[ID][1])) +
                        cluster_size) <= max_cluster_size:
                    # Actual adding of new links
                    graph_main.add_edge(node_id,
                                        potential_links[ID][1],
                                        weight=best_scores[ID])
                    links_added.append((node_id, potential_links[ID][1]))
                    # Update cluster_size to keep track of growing clusters
                    cluster_size = len(
                        nx.node_connected_component(graph_main,
                                                    potential_links[ID][1]))

    return graph_main, links_added


def erode_clusters(graph_main, max_cluster_size=100, keep_weights_above=0.8):
    """ Remove links from clusters that are > max_cluster_size.
    This function is in particular made to avoid small remaining clusters or singletons.

    Will only add links if they won't lead to clusters > max_cluster_size,
    and if the links have weights > min_weight.
    Starts iteratively from highest weight links that are not yet part of the network.

    Args:
    --------
    graph_main: networkx graph
        Graph, e.g. made using create_network() function. Based on networkx.
    max_cluster_size: int
        Maximum desired size of clusters. Default = 100.
    keep_weights_above: float
        Set threshold above which weights will not be removed. Default = 0.8.
    """

    links_removed = []

    # Split graph into separate clusters
    graphs = list(nx.connected_component_subgraphs(graph_main))

    for graph in graphs:
        cluster_size = len(graph.nodes)
        while cluster_size > max_cluster_size:

            edges = list(graph.edges)
            edges_weights = np.array(
                [graph[x[0]][x[1]]['weight'] for x in edges])

            weakest_edge = edges_weights.argsort()[0]
            if edges_weights[weakest_edge] < keep_weights_above:
                print("Remove edge:", edges[weakest_edge][0],
                      edges[weakest_edge][1])
                graph.remove_edge(edges[weakest_edge][0],
                                  edges[weakest_edge][1])
                graph_main.remove_edge(edges[weakest_edge][0],
                                       edges[weakest_edge][1])
                links_removed.append(edges[weakest_edge])

            # If link removal caused split of cluster:
            if not nx.is_connected(graph):
                subgraphs = list(nx.connected_component_subgraphs(graph))
                print("Getting from cluster with", len(graph.nodes),
                      "nodes, to clusters with",
                      [len(x.nodes) for x in subgraphs], "nodes.")
                idx1 = np.argmax([len(x.nodes) for x in subgraphs])
                graph = subgraphs[idx1]  # keep largest subcluster here

            cluster_size = len(graph.nodes)

    return graph_main, links_removed


def add_intra_cluster_links(graph_main, m_sim, min_weight=0.5, max_links=20):
    """ Add links within each separate cluster if weights above min_weight.

    Args:
    -------
    graph_main: networkx graph
        Graph, e.g. made using create_network() function. Based on networkx.
    m_sim: numpy array
        2D array with all reference similarity values between all-vs-all nodes.
    min_weight: float
        Set minimum weight to be considered for making link. Default = 0.5.
    """
    # Split graph into separate clusters
    graphs = list(nx.connected_component_subgraphs(graph_main))

    for graph in graphs:
        nodes = list(graph.nodes)
        nodes0 = nodes.copy()
        for node in nodes:
            del nodes0[0]
            weights = m_sim[node, nodes0]
            weights_select = weights.argsort()[::-1][:max_links]
            weights_select = np.where(weights[weights_select] >= min_weight)[0]
            new_edges = [(node, nodes0[x], weights[x]) for x in weights_select]

            graph_main.add_weighted_edges_from(new_edges)

    return graph_main


def split_cluster(graph_main,
                  max_cluster_size=100,
                  min_cluster_size=10,
                  max_search_steps=1000,
                  max_cuts=1,
                  multiple_cuts_per_level=True):
    """
    Function to split clusters at weak links.

    Args:
    ---------
    graph_main: networkx graph
        Graph, e.g. made using create_network() function. Based on networkx.
    max_cluster_size: int
        Maximum desired size of clusters. Default = 100.
    min_cluster_size: int
        Minimum desired size of clusters. Default = 10.
    max_steps
        Up to max_steps pairs will be explored to search for cuts. Default = 1000.
    max_cuts
        Maximum numbers of links allowed to be cut. Default = 1.
    multiple_cuts_per_level
        If true allow multiple cuts to be done per level and run. Default = True.
    """

    # Split graph into separate clusters
    graphs = list(nx.connected_component_subgraphs(graph_main))

    links_removed = []
    for i, graph in enumerate(graphs):
        if len(graph.nodes) > max_cluster_size:
            # Detect potential weak links
            weak_links = weak_link_finder(graph,
                                          max_steps=max_search_steps,
                                          max_cuts=max_cuts)

            split_done = False
            j = 0
            new_graph = graph.copy()
            while not split_done and j < len(weak_links):

                # Test best candidates

                new_graph_testing = new_graph.copy()
                pairs = weak_links[j][1]
                pair_counts = weak_links[j][2]
                pairs = pairs[pair_counts.argsort()[::-1]]
                # print(i,j, pairs)

                # ----------------------------------------------
                # Check if pairs have already been removed in former iteration
                # ----------------------------------------------
                pairs_still_present = []
                for i, pair in enumerate(pairs):
                    all_edges_present = True
                    for m in range(int(pairs.shape[1] / 2)):
                        edge = (pair[m * 2], pair[m * 2 + 1])
                        if edge not in new_graph_testing.edges:
                            all_edges_present = False
                    if all_edges_present:
                        pairs_still_present.append(i)
                    pairs_still_present = list(set(pairs_still_present))
                pairs = pairs[
                    pairs_still_present]  # Remove pairs which have been cut out already

                # ----------------------------------------------
                # Test removing proposed links for all pairs
                # ----------------------------------------------
                if len(pairs) > 0:
                    min_size_after_cutting = []
                    for pair in pairs:
                        new_graph_testing = new_graph.copy()

                        # Remove edges in pair
                        for m in range(int(pairs.shape[1] / 2)):
                            new_graph_testing.remove_edge(
                                pair[m * 2], pair[m * 2 + 1])

                        # Check if created subclustes are big enough:
                        subgraphs = list(
                            nx.connected_component_subgraphs(
                                new_graph_testing))
                        min_size_after_cutting.append(
                            min([len(x.nodes) for x in subgraphs]))

                    # Select best partition of graph (creating most similar sized subclusters)
                    min_size_after_cutting = np.array(min_size_after_cutting)
                    best_partition = np.argmax(min_size_after_cutting)
                else:
                    min_size_after_cutting = [0]
                    best_partition = 0

                # ----------------------------------------------
                # Actual removal of links
                # ----------------------------------------------
                if min_size_after_cutting[best_partition] >= min_cluster_size:
                    new_graph_testing = new_graph.copy()
                    pair = pairs[best_partition]

                    # Remove edges in selected pair
                    for m in range(int(pairs.shape[1] / 2)):
                        # Remove edge from current cluster:
                        new_graph_testing.remove_edge(pair[m * 2],
                                                      pair[m * 2 + 1])
                        # Remove edge from main graph:
                        graph_main.remove_edge(pair[m * 2], pair[m * 2 + 1])
                        links_removed.append((pair[m * 2], pair[m * 2 + 1]))
                    subgraphs = list(
                        nx.connected_component_subgraphs(new_graph_testing))

                    if int(pairs.shape[1] / 2) > 1:
                        print("Removed", int(pairs.shape[1] / 2), "edges:",
                              pair)
                    else:
                        print("Removed", int(pairs.shape[1] / 2), "edge:",
                              pair)

                    print("Getting from cluster with", len(new_graph.nodes),
                          "nodes, to clusters with",
                          [len(x.nodes) for x in subgraphs], "nodes.")
                    idx1 = np.argmax([len(x.nodes) for x in subgraphs])
                    new_graph = subgraphs[idx1]  # keep largest subcluster here

                    if len(new_graph.nodes) <= max_cluster_size:
                        split_done = True
                    else:
                        pass

                # Check if more suited cuts are expected for the same number of cuts
                if len(min_size_after_cutting) > 1:
                    idx = np.argsort(min_size_after_cutting)[::-1][1]
                    if min_size_after_cutting[
                            idx] >= min_cluster_size and multiple_cuts_per_level:
                        pass
                    else:
                        j += 1
                else:
                    j += 1

    return graph_main, links_removed


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
