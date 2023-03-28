import json
from typing import Optional
import networkx as nx
import numpy as np
from matchms import Scores
from .networking_functions import get_top_hits


class SimilarityNetwork:
    """Create a spectral network from spectrum similarities.

    For example

    .. testcode::

        import numpy as np
        from matchms import Spectrum, calculate_scores
        from matchms.similarity import ModifiedCosine
        from matchms.networking import SimilarityNetwork

        spectrum_1 = Spectrum(mz=np.array([100, 150, 200.]),
                              intensities=np.array([0.7, 0.2, 0.1]),
                              metadata={"precursor_mz": 100.0,
                                        "test_id": "one"})
        spectrum_2 = Spectrum(mz=np.array([104.9, 140, 190.]),
                              intensities=np.array([0.4, 0.2, 0.1]),
                              metadata={"precursor_mz": 105.0,
                                        "test_id": "two"})

        # Use factory to construct a similarity function
        modified_cosine = ModifiedCosine(tolerance=0.2)
        spectrums = [spectrum_1, spectrum_2]
        scores = calculate_scores(spectrums, spectrums, modified_cosine)
        ms_network = SimilarityNetwork(identifier_key="test_id")
        ms_network.create_network(scores, score_name="ModifiedCosine_score")

        nodes = list(ms_network.graph.nodes())
        nodes.sort()
        print(nodes)

    Should output

    .. testoutput::

        ['one', 'two']

    """
    def __init__(self, identifier_key: str = "spectrum_id",
                 top_n: int = 20,
                 max_links: int = 10,
                 score_cutoff: float = 0.7,
                 link_method: str = 'single',
                 keep_unconnected_nodes: bool = True):
        """
        Parameters
        ----------
        identifier_key
            Metadata key for unique identifier for each spectrum in scores.
            Will also be used for the naming the network nodes. Default is 'spectrum_id'.
        top_n
            Consider edge between spectrumA and spectrumB if score falls into
            top_n for spectrumA or spectrumB (link_method="single"), or into
            top_n for spectrumA and spectrumB (link_method="mutual"). From those
            potential links, only max_links will be kept, so top_n must be >= max_links.
        max_links
            Maximum number of links to add per node. Default = 10.
            Due to incoming links, total number of links per node can be higher.
            The links are populated by looping over the query spectrums.
            Important side note: The max_links restriction is strict which means that
            if scores around max_links are equal still only max_links will be added
            which can results in some random variations (sorting spectra with equal
            scores results in a random order of such elements).
        score_cutoff
            Threshold for given similarities. Edges/Links will only be made for
            similarities > score_cutoff. Default = 0.7.
        link_method
            Chose between 'single' and 'mutual'. 'single will add all links based
            on individual nodes. 'mutual' will only add links if that link appears
            in the given top-n list for both nodes.
        keep_unconnected_nodes
            If set to True (default) all spectra will be included as nodes even
            if they have no connections/edges of other spectra. If set to False
            all nodes without connections will be removed.
        """
        # pylint: disable=too-many-arguments
        self.identifier_key = identifier_key
        self.top_n = top_n
        self.max_links = max_links
        self.score_cutoff = score_cutoff
        self.link_method = link_method
        self.keep_unconnected_nodes = keep_unconnected_nodes
        self.graph: Optional[nx.Graph] = None
        """NetworkX graph. Set after calling create_network()"""

    def create_network(self, scores: Scores, score_name: str = None):
        """
        Function to create network from given top-n similarity values. Expects that
        similarities given in scores are from an all-vs-all comparison including all
        possible pairs.

        Parameters
        ----------
        scores
            Matchms Scores object containing all spectrums and pair similarities for
            generating a network.
        """
        if score_name is None:
            score_name = scores.scores.guess_score_name()
        assert self.top_n >= self.max_links, "top_n must be >= max_links"
        assert np.all(scores.queries == scores.references), \
            "Expected symmetric scores object with queries==references"
        unique_ids = list({s.get(self.identifier_key) for s in scores.queries})

        # Initialize network graph, add nodes
        msnet = nx.Graph()
        msnet.add_nodes_from(unique_ids)

        # Collect location and score of highest scoring candidates for queries and references
        similars_idx, similars_scores = get_top_hits(scores, identifier_key=self.identifier_key,
                                                     top_n=self.top_n,
                                                     search_by="queries",
                                                     score_name=score_name,
                                                     ignore_diagonal=True)

        # Add edges based on global threshold (cutoff) for weights
        for i, spec in enumerate(scores.queries):
            query_id = spec.get(self.identifier_key)
            ref_candidates = np.array([scores.references[x].get(self.identifier_key)
                                          for x in similars_idx[query_id]])
            idx = np.where((similars_scores[query_id] >= self.score_cutoff) &
                              (ref_candidates != query_id))[0][:self.max_links]
            if self.link_method == "single":
                new_edges = [(query_id, str(ref_candidates[x]),
                              float(similars_scores[query_id][x])) for x in idx]
            elif self.link_method == "mutual":
                new_edges = [(query_id, str(ref_candidates[x]),
                              float(similars_scores[query_id][x]))
                             for x in idx if i in similars_idx[ref_candidates[x]][:]]
            else:
                raise ValueError("Link method not kown")

            msnet.add_weighted_edges_from(new_edges)

        if not self.keep_unconnected_nodes:
            msnet.remove_nodes_from(list(nx.isolates(msnet)))
        self.graph = msnet

    def export_to_file(self, filename: str, graph_format: str = "graphml"):
        """
        Save the network to a file with chosen format.

        Parameters
        ----------
        filename
            Path to file to write to.
        graph_format
            Format, in which to store the network graph. Supported formats are: "cyjs", "gexf", "gml", "graphml", "json".
            Default is "graphml".
        """
        if not self.graph:
            raise ValueError("No network found. Make sure to first run .create_network() step")

        writer = self._generate_writer(graph_format)
        writer(filename)

    def _generate_writer(self, graph_format: str):
        writer = {"cyjs": self._export_to_cyjs,
                  "gexf": self._export_to_gexf,
                  "gml": self._export_to_gml,
                  "graphml": self.export_to_graphml,
                  "json": self._export_to_node_link_json}

        assert graph_format in writer, "Format not supported.\n" \
                                       "Please use one of supported formats: 'cyjs', 'gexf', 'gml', 'graphml', 'json'"
        return writer[graph_format]

    def export_to_graphml(self, filename: str):
        """Save the network as .graphml file.

        Parameters
        ----------
        filename
            Specify filename for exporting the graph.

        """
        nx.write_graphml_lxml(self.graph, filename)

    def _export_to_cyjs(self, filename: str):
        """Save the network in cyjs format."""
        graph = nx.cytoscape_data(self.graph)
        return self._write_to_json(graph, filename)

    def _export_to_node_link_json(self, filename: str):
        """Save the network in node-link format."""
        graph = nx.node_link_data(self.graph)
        return self._write_to_json(graph, filename)

    @staticmethod
    def _write_to_json(graph: dict, filename: str):
        """Save the network as JSON file.

        Parameters
        ----------
        graph
            JSON-dictionary type graph to save.
        filename
            Specify filename for exporting the graph.

        """
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(graph, file)

    def _export_to_gexf(self, filename: str):
        """Save the network as .gexf file."""
        nx.write_gexf(self.graph, filename)

    def _export_to_gml(self, filename: str):
        """Save the network as .gml file."""
        nx.write_gml(self.graph, filename)
