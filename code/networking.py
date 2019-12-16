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
from networkx.algorithms.connectivity import minimum_st_node_cut, minimum_st_edge_cut
from networkx.algorithms.flow import shortest_augmenting_path
import pandas as pd

## ----------------------------------------------------------------------------
## ---------------- Graph / networking related functions ----------------------
## ----------------------------------------------------------------------------


def create_network(similars_idx,
                   similars,
                   max_links = 10,
                   cutoff = 0.7,
                   link_method = 'single'):
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
    MSnet = nx.Graph()               
    MSnet.add_nodes_from(np.arange(0, dimension))   

    # Add edges based on global threshold for weights
    for i in range(0, dimension):      
        idx = np.where(similars[i,:] > cutoff)[0][:max_links]
        if link_method == "single":
            new_edges = [(i, int(similars_idx[i,x]), float(similars[i,x])) for x in idx if similars_idx[i,x] != i]
        elif link_method == "mutual":
            new_edges = [(i, int(similars_idx[i,x]), float(similars[i,x])) for x in idx if similars_idx[i,x] != i and i in similars_idx[x,:]]
        else:
            print("Link method not kown")
        MSnet.add_weighted_edges_from(new_edges)
        
    return MSnet


def sample_cuts(graph,
                max_steps = 1000,
                max_cuts = 1):
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
    #num_edges = graph.number_of_edges()
    
    # Make list of all pairs within graph
    nodes = np.array(graph.nodes)
    pairs = np.array(np.meshgrid(nodes, nodes)).T
    remove_diagonal = np.array([(i*num_nodes + i) for i in range(num_nodes)])
    pairs = np.delete(pairs.reshape(-1,2), remove_diagonal, axis=0)
    
    sampled_cuts = []
    if pairs.shape[0] <= max_steps:
        max_steps = pairs.shape[0]
    else:
        # If more pairs exist than max_steps allows to explore, pick max_steps random pairs.
        choices = np.random.choice(np.arange(pairs.shape[0]), max_steps, replace=False)
        pairs = pairs[choices,:]
        
    for pair in pairs:
        cuts = minimum_st_edge_cut(graph, pair[0], pair[1], flow_func=shortest_augmenting_path)
        #nx.node_connectivity(graphs[4], 592, 376)
        #cuts = nx.minimum_st_edge_cut(graph, pair[0], pair[1])
        #cuts = nx.minimum_edge_cut(graph, pair[0], pair[1])#, flow_func=shortest_augmenting_path)
        if len(cuts) <= max_cuts:
            sampled_cuts.append(cuts)
            
    return sampled_cuts



def weak_link_finder(graph,
                     max_steps = 1000,
                     max_cuts = 1):
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
    
    sampled_cuts = sample_cuts(graph,
                                max_steps = max_steps,
                                max_cuts = max_cuts)
    
    sampled_cuts_len = [len(x) for x in sampled_cuts]
    proposed_cuts = []
    for min_cuts in list(set(sampled_cuts_len)):
        sampled_cuts_select = [list(x)[:min_cuts] for x in sampled_cuts if len(x) == min_cuts]
        
        sampled_cuts_select = np.array(sampled_cuts_select)
        # Sort array
        if min_cuts > 1:
            sampled_cuts_select = np.sort(np.sort(sampled_cuts_select, axis=2), axis=1)
        else:
            sampled_cuts_select = np.sort(sampled_cuts_select, axis=2)    
        
        # Find unique cuts and count occurences
        cuts_unique, cuts_count = row_counts(sampled_cuts_select.reshape(-1,min_cuts*2))
        
        # Return most promising cuts 
        proposed_cuts.append((min_cuts, cuts_unique, cuts_count))

    return proposed_cuts


def dilate_cluster(graph_main,
                   similars_idx,
                   similars,
                   max_cluster_size = 100,   
                   min_cluster_size = 10,
                   max_addition = None,
                   min_weight = 0.5):
    """ Add more links to clusters that are < min_cluster_size.
    This function is in particular made to avoid small remaining clusters or singletons. 
    
    Will only add links if they won't lead to clusters > max_cluster_size,
    and if the links have weights > min_weight.
    Starts iteratively from highest weight links that are not yet part of the network.
    
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
    max_addition: int, None 
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

                potential_new_links = [(i, x) for i, x in enumerate(similars_idx[ID]) if x not in nodes_connected and x != ID]
                best_score = similars[ID][list(zip(*potential_new_links))[0][0]]
                if best_score >= min_weight:
                    best_scores.append(best_score)
                    potential_link = list(zip(*potential_new_links))[1][0]
                    potential_links.append(potential_link)
                       
            if max_addition is None:
                selected_candidates = np.argsort(best_scores)[::-1]
            else:
                # Only add the top max_addition ones
                selected_candidates = np.argsort(best_scores)[::-1][:max_addition]
            
            for ID in selected_candidates:
                node_ID = list(graph.nodes)[ID]
                
                # Only add link if no cluster > max_cluster_size is formed by it
                if (len(nx.node_connected_component(graph_main, potential_links[ID])) + cluster_size) <= max_cluster_size:
                    # Actual adding of new links
                    graph_main.add_edge(node_ID, potential_links[ID], weight=best_scores[ID])
                    links_added.append((node_ID, potential_links[ID]))
                    # Update cluster_size to keep track of growing clusters
                    cluster_size = len(nx.node_connected_component(graph_main, potential_links[ID]))
    
    return graph_main, links_added


def erode_clusters(graph_main,
                   max_cluster_size = 100,   
                   keep_weights_above = 0.8):
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
            edges_weights = np.array([graph[x[0]][x[1]]['weight'] for x in edges])

            weakest_edge = edges_weights.argsort()[0]
            if edges_weights[weakest_edge] < keep_weights_above:
                print("Remove edge:", edges[weakest_edge][0], edges[weakest_edge][1])
                graph.remove_edge(edges[weakest_edge][0], edges[weakest_edge][1])
                graph_main.remove_edge(edges[weakest_edge][0], edges[weakest_edge][1])
                links_removed.append(edges[weakest_edge])

            # If link removal caused split of cluster:
            if not nx.is_connected(graph):
                subgraphs = list(nx.connected_component_subgraphs(graph))
                print("Getting from cluster with", len(graph.nodes), "nodes, to clusters with",
                     [len(x.nodes) for x in subgraphs], "nodes.")
                idx1 = np.argmax([len(x.nodes) for x in subgraphs])
                graph = subgraphs[idx1] # keep largest subcluster here
            
            cluster_size = len(graph.nodes)
    
    return graph_main, links_removed



def add_intra_cluster_links(graph_main,
                           M_sim, 
                           min_weight = 0.5):
    """ Add links within each separate cluster if weights above min_weight.
    
    Args:
    -------
    graph_main: networkx graph
        Graph, e.g. made using create_network() function. Based on networkx.
    M_sim: numpy array
        2D array with all reference similarity values between all-vs-all nodes.
    min_weight: float
        Set minimum weight to be considered for making link. Default = 0.5.
    """
    
    # Split graph into separate clusters
    graphs = list(nx.connected_component_subgraphs(graph_main))    
    
    for graph in graphs:
        nodes = graph.nodes
        for node in nodes:
            M_sim[node, nodes]    


def split_cluster(graph_main,
                 max_cluster_size = 100,
                 min_cluster_size = 10,
                 max_search_steps = 1000,
                 max_cuts = 1,
                 multiple_cuts_per_level = True):
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
                                         max_steps = max_search_steps,
                                         max_cuts = max_cuts)
            
            split_done = False
            j = 0
            new_graph = graph.copy()
            while not split_done and j < len(weak_links):

                # Test best candidates

                new_graph_testing = new_graph.copy()
                pairs = weak_links[j][1]
                pair_counts = weak_links[j][2] 
                pairs = pairs[pair_counts.argsort()[::-1]]
                #print(i,j, pairs)

                # ----------------------------------------------
                # Check if pairs have already been removed in former iteration
                # ----------------------------------------------
                pairs_still_present = []
                for i, pair in enumerate(pairs):
                    all_edges_present = True
                    for m in range(int(pairs.shape[1]/2)):
                        edge = (pair[m*2], pair[m*2+1])
                        if edge not in new_graph_testing.edges:
                            all_edges_present = False             
                    if all_edges_present:
                        pairs_still_present.append(i)          
                    pairs_still_present = list(set(pairs_still_present))     
                pairs = pairs[pairs_still_present] # Remove pairs which have been cut out already

                # ----------------------------------------------
                # Test removing proposed links for all pairs
                # ----------------------------------------------
                if len(pairs) > 0:
                    min_size_after_cutting = []
                    for pair in pairs:
                        new_graph_testing = new_graph.copy()

                        # Remove edges in pair
                        for m in range(int(pairs.shape[1]/2)):
                            new_graph_testing.remove_edge(pair[m*2], pair[m*2+1])

                        # Check if created subclustes are big enough:
                        subgraphs = list(nx.connected_component_subgraphs(new_graph_testing))
                        min_size_after_cutting.append(min([len(x.nodes) for x in subgraphs]))

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
                    for m in range(int(pairs.shape[1]/2)):
                        # Remove edge from current cluster:
                        new_graph_testing.remove_edge(pair[m*2], pair[m*2+1])
                        # Remove edge from main graph:
                        graph_main.remove_edge(pair[m*2], pair[m*2+1])
                        links_removed.append((pair[m*2], pair[m*2+1]))
                    subgraphs = list(nx.connected_component_subgraphs(new_graph_testing))

                    if int(pairs.shape[1]/2) > 1:
                        print("Removed", int(pairs.shape[1]/2), "edges:", pair)
                    else:
                        print("Removed", int(pairs.shape[1]/2), "edge:", pair)
                    
                    print("Getting from cluster with", len(new_graph.nodes), "nodes, to clusters with",
                         [len(x.nodes) for x in subgraphs], "nodes.")
                    idx1 = np.argmax([len(x.nodes) for x in subgraphs])
                    new_graph = subgraphs[idx1] # keep largest subcluster here

                    if len(new_graph.nodes) <= max_cluster_size:
                        split_done = True
                    else:
                        pass

                # Check if more suited cuts are expected for the same number of cuts
                if len(min_size_after_cutting) > 1:
                    idx = np.argsort(min_size_after_cutting)[::-1][1]
                    if min_size_after_cutting[idx] >= min_cluster_size and multiple_cuts_per_level:
                        pass
                    else:
                        j += 1
                else:
                    j += 1
            
    return graph_main, links_removed


## ----------------------------------------------------------------------------
## ---------------------- Functions to refine network -------------------------
## ----------------------------------------------------------------------------

def refine_network(graph_main,
                   similars_idx,
                   similars,
                   weigh_bounds = (0.6, 1),
                   filename = None,
                   max_cluster_size = 100,
                   min_cluster_size = 10,
                   max_search_steps = 1000,
                   max_cuts = 2,
                   max_split_iterations = 10,
                   basic_splitting = True,
                   dilation = False):
    """
    Args:
    -------
    """
    # Split graph into separate clusters
    graphs = list(nx.connected_component_subgraphs(graph_main))
    
    links_removed = []
    links_added = []
    
    #n_cluster = len(graphs)
    cluster_max = np.max([len(x.nodes) for x in graphs])
    counter = 0
    
    print(20 * '---')
    while cluster_max > max_cluster_size and counter < max_split_iterations:
        print("Splitting iteration:", counter+1, "Max cluster size =", cluster_max, '\n')
        graph_main, links = split_cluster(graph_main.copy(),
                                 max_cluster_size = max_cluster_size,
                                 min_cluster_size = min_cluster_size,
                                 max_search_steps = max_search_steps,
                                 max_cuts = max_cuts,
                                 multiple_cuts_per_level = True)
        links_removed.extend(links)
        
        # Split updated graph into separate clusters
        graphs = list(nx.connected_component_subgraphs(graph_main))
        cluster_max = np.max([len(x.nodes) for x in graphs])
        counter += 1
        
    if basic_splitting:
        print(20 * '---')
        print("Extra splitting step to sanitize clusters.")
        graph_main, links = split_cluster(graph_main,
                                 max_cluster_size = 2*min_cluster_size, #! here we try to 'sanitize most clusters'
                                 min_cluster_size = min_cluster_size,
                                 max_search_steps = max_search_steps,
                                 max_cuts = 1,
                                 multiple_cuts_per_level = False)
        links_removed.extend(links)
        
    if dilation:
        print(20 * '---')
        print("Runing dilation function for smaller clusters <", min_cluster_size)
        graph_main, links = dilate_cluster(graph_main,
                               similars_idx,
                               similars,
                               max_cluster_size = max_cluster_size,   
                               min_cluster_size = min_cluster_size,
                               max_addition = None,
                               min_weight = weigh_bounds[0])
        links_added.extend(links)
    
    if filename is not None:
        # Export graph for drawing (e.g. using Cytoscape)
        nx.write_graphml(graph_main, filename)
        print("Network stored as graphml file under: ", filename)

    return graph_main, links_added, links_removed


## ----------------------------------------------------------------------------
## -------------------- Functions to evaluate networks ------------------------
## ----------------------------------------------------------------------------
    

def evaluate_clusters(graph_main,
                     M_sim_ref):
    """ Evaluate separate clusters of network based on given reference matrix.
    
    Args:
    -------
    graph_main: networkx graph
        Graph, e.g. made using create_network() function. Based on networkx.
    M_sim_ref: numpy array
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
        if len(graph.edges) > 0: # no edges for singletons
            num_edges.append(len(graph.edges)) 
            
            edges = list(graph.edges)
            mol_sim_edges = np.array([M_sim_ref[x] for x in edges])
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
            mean_mol_sims.append(M_sim_ref[node, nodes])

        ref_sim_mean_nodes.append(np.mean(mean_mol_sims))
        ref_sim_var_nodes.append(np.var(mean_mol_sims))
    
    cluster_data = pd.DataFrame(list(zip(num_nodes,
                                 num_edges,
                                 ref_sim_mean_edges,
                                 ref_sim_var_edges,
                                 ref_sim_mean_nodes,
                                 ref_sim_var_nodes)), columns=['num_nodes',
                                                             'num_edges',
                                                             'ref_sim_mean_edges',
                                                             'ref_sim_var_edges',
                                                             'ref_sim_mean_nodes',
                                                             'ref_sim_var_nodes'])
    return cluster_data



## ----------------------------------------------------------------------------
## -------------------------- Small helper functions --------------------------
## ----------------------------------------------------------------------------
    
def row_counts(array):
    """
    Function to find unique rows and count their occurences.
    """
    dt = np.dtype((np.void, array.dtype.itemsize * array.shape[1]))
    b = np.ascontiguousarray(array).view(dt)
    unq, cnt = np.unique(b, return_counts=True)
    unq = unq.view(array.dtype).reshape(-1, array.shape[1])
    
    return unq, cnt