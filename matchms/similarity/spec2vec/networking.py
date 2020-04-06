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
