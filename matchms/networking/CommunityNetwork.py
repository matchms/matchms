import json
from typing import Optional
import networkx as nx
import numpy as np
import igraph as ig
import leidenalg
from matchms import Scores
from .networking_functions import get_top_hits


class CommunityNetwork:
    """
    Community-based Similarity Network for mass spectral data.

    This class creates a k-nearest neighbors graph from spectral similarity scores,
    computes communities using the Leiden algorithm, and optionally removes inter-community edges.

    Parameters
    ----------
    identifier_key : str, default "spectrum_id"
        Metadata key used as a unique identifier for each spectrum.
    k : int, default 20
        Number of nearest neighbors to consider for each spectrum.
    score_cutoff : Optional[float], default None
        Threshold for similarity scores. Only edges with a similarity score greater or equal to this
        threshold are included. If None, no score-based filtering is applied.
    remove_inter_community_links : bool, default True
        If True, all edges connecting spectra within the same community are removed.
    """
    def __init__(self,
                 identifier_key: str = "spectrum_id",
                 k: int = 20,
                 score_cutoff: Optional[float] = None,
                 remove_inter_community_links: bool = True):
        self.identifier_key = identifier_key
        self.k = k
        self.score_cutoff = score_cutoff
        self.remove_inter_community_links = remove_inter_community_links
        self.graph: Optional[nx.Graph] = None

    def create_network(self, scores: Scores, score_name: Optional[str] = None) -> None:
        """
        Create a k-nearest neighbors graph from spectral similarities, compute communities using the Leiden algorithm,
        and optionally remove inter-community edges.

        Parameters
        ----------
        scores : Scores
            Matchms Scores object containing all pairwise spectral similarities.
        score_name : Optional[str], default None
            Name of the score attribute to be used. If None, it is automatically determined.
        
        Raises
        ------
        TypeError
            If the provided scores are not symmetric.
        ValueError
            If queries and references in scores do not match.
        """
        if score_name is None:
            score_name = scores.scores.guess_score_name()

        # Validate that scores are symmetric.
        if scores.queries.shape != scores.references.shape:
            raise TypeError("Expected symmetric scores: queries and references shapes differ.")
        if not np.all(scores.queries == scores.references):
            raise ValueError("Queries and references in scores do not match.")

        # Build the kNN graph from the similarities.
        self.graph = self._create_knn_graph(scores, score_name)

        # Compute communities using the Leiden algorithm.
        communities = self._compute_communities(self.graph)
        # Store the community assignment as node attributes.
        nx.set_node_attributes(self.graph, communities, "community")

        # Optionally remove all inter-community edges.
        if self.remove_inter_community_links:
            self.graph = self._remove_inter_community_links(communities)

    def _create_knn_graph(self, scores: Scores, score_name: str) -> nx.Graph:
        """
        Create a k-nearest neighbors graph from spectral similarities.

        Parameters
        ----------
        scores : Scores
            Matchms Scores object containing all pairwise spectral similarities.
        score_name : str
            Name of the score attribute to be used.

        Returns
        -------
        nx.Graph
            A NetworkX graph with spectra as nodes and edges representing similarity relationships.
        """
        # Extract unique spectrum identifiers.
        unique_ids = list({spec.get(self.identifier_key) for spec in scores.queries})
        graph = nx.Graph()
        graph.add_nodes_from(unique_ids)

        # Get k nearest hits for each spectrum using matchms's get_top_hits utility.
        similars_idx, similars_scores = get_top_hits(scores,
                                                     identifier_key=self.identifier_key,
                                                     top_n=self.k,
                                                     search_by="queries",
                                                     score_name=score_name,
                                                     ignore_diagonal=True)

        for i, spec in enumerate(scores.queries):
            query_id = spec.get(self.identifier_key)
            # Retrieve candidate identifiers and corresponding similarity scores.
            candidate_ids = np.array([scores.references[idx].get(self.identifier_key)
                                      for idx in similars_idx[query_id]])
            candidate_scores = similars_scores[query_id]
            # Apply the score cutoff if specified.
            if self.score_cutoff is not None:
                valid_idx = np.where(candidate_scores >= self.score_cutoff)[0]
            else:
                valid_idx = np.arange(len(candidate_scores))
            # Add edges for valid candidate neighbors (avoiding self-loops).
            for idx in valid_idx:
                neighbor_id = candidate_ids[idx]
                if neighbor_id != query_id:
                    graph.add_edge(query_id, neighbor_id, weight=float(candidate_scores[idx]))
        return graph

    def _compute_communities(self, graph: nx.Graph) -> dict:
        """
        Compute communities in the graph using the Leiden algorithm.

        Parameters
        ----------
        graph : nx.Graph
            The NetworkX graph for which communities will be computed.

        Returns
        -------
        dict
            A dictionary mapping each node to its community identifier.
        """
        # Map NetworkX nodes to indices for igraph.
        mapping = {node: idx for idx, node in enumerate(graph.nodes())}
        reverse_mapping = {idx: node for node, idx in mapping.items()}
        # Prepare edges and weights for the igraph graph.
        edges = [(mapping[u], mapping[v]) for u, v in graph.edges()]
        weights = [data.get("weight", 1.0) for _, _, data in graph.edges(data=True)]

        # Create an undirected igraph graph.
        ig_graph = ig.Graph(edges=edges, directed=False)
        ig_graph.vs["name"] = list(mapping.keys())
        if weights:
            ig_graph.es["weight"] = weights

        # Compute communities using the Leiden algorithm with modularity optimization.
        partition = leidenalg.find_partition(ig_graph,
                                             leidenalg.ModularityVertexPartition,
                                             weights="weight")
        membership = partition.membership
        # Map community assignments back to original node names.
        communities = {reverse_mapping[idx]: community for idx, community in enumerate(membership)}
        return communities

    def _remove_inter_community_links(self, communities: dict) -> None:
        """
        Remove edges connecting nodes in different communities from the graph.

        Parameters
        ----------
        communities : dict
            A dictionary mapping each node to its community identifier.

        Returns
        -------
        nx.Graph
            The modified graph with inter-community edges removed.
        """
        inter_edges = [(u, v) for u, v in list(self.graph.edges())
                       if communities.get(u) != communities.get(v)]
        self.graph.remove_edges_from(inter_edges)
        return self.graph

    def export_to_file(self, filename: str, graph_format: str = "graphml") -> None:
        """
        Save the network to a file in the specified format.

        Parameters
        ----------
        filename : str
            Path to the file where the graph will be saved.
        graph_format : str, default "graphml"
            Format to export the graph. Supported formats: 'cyjs', 'gexf', 'gml', 'graphml', 'json'.
        """
        if not self.graph:
            raise ValueError("No network graph available. Run create_network() first.")

        writer = self._generate_writer(graph_format)
        writer(filename)

    def _generate_writer(self, graph_format: str):
        writers = {
            "cyjs": self._export_to_cyjs,
            "gexf": self._export_to_gexf,
            "gml": self._export_to_gml,
            "graphml": self.export_to_graphml,
            "json": self._export_to_node_link_json
        }
        if graph_format not in writers:
            raise ValueError("Unsupported format. Use one of: 'cyjs', 'gexf', 'gml', 'graphml', 'json'.")
        return writers[graph_format]

    def export_to_graphml(self, filename: str) -> None:
        """
        Export the network as a GraphML file.

        Parameters
        ----------
        filename : str
            The filename where the graph will be saved.
        """
        nx.write_graphml_lxml(self.graph, filename)

    def _export_to_cyjs(self, filename: str) -> None:
        """
        Export the network in Cytoscape.js format.
        
        Parameters
        ----------
        filename : str
            The filename where the graph will be saved.
        """
        cyjs_data = nx.cytoscape_data(self.graph)
        self._write_to_json(cyjs_data, filename)

    def _export_to_node_link_json(self, filename: str) -> None:
        """
        Export the network in node-link JSON format.
        
        Parameters
        ----------
        filename : str
            The filename where the graph will be saved.
        """
        node_link_data = nx.node_link_data(self.graph, edges="links")
        self._write_to_json(node_link_data, filename)

    @staticmethod
    def _write_to_json(graph_data: dict, filename: str) -> None:
        """
        Write a graph (as a dictionary) to a JSON file.

        Parameters
        ----------
        graph_data : dict
            The graph data to write.
        filename : str
            The filename where the graph will be saved.
        """
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(graph_data, file)

    def _export_to_gexf(self, filename: str) -> None:
        """
        Export the network as a GEXF file.
        
        Parameters
        ----------
        filename : str
            The filename where the graph will be saved.
        """
        nx.write_gexf(self.graph, filename)

    def _export_to_gml(self, filename: str) -> None:
        """
        Export the network as a GML file.
        
        Parameters
        ----------
        filename : str
            The filename where the graph will be saved.
        """
        nx.write_gml(self.graph, filename)
