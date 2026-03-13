import json
from typing import Optional, Sequence
import networkx as nx
import numpy as np
from matchms import Scores
from .networking_functions import get_top_hits


class SimilarityNetwork:
    """Create a similarity network from all-vs-all spectrum similarities.

    For example

    .. testcode::

        import numpy as np
        from matchms import Spectrum, calculate_scores
        from matchms.similarity import ModifiedCosineGreedy
        from matchms.networking import SimilarityNetwork

        spectrum_1 = Spectrum(
            mz=np.array([100, 150, 200.0]),
            intensities=np.array([0.7, 0.2, 0.1]),
            metadata={"precursor_mz": 100.0, "test_id": "one"},
        )
        spectrum_2 = Spectrum(
            mz=np.array([104.9, 140, 190.0]),
            intensities=np.array([0.4, 0.2, 0.1]),
            metadata={"precursor_mz": 105.0, "test_id": "two"},
        )

        modified_cosine = ModifiedCosineGreedy(tolerance=0.2)
        spectra = [spectrum_1, spectrum_2]
        scores = calculate_scores(spectra, spectra, modified_cosine)

        identifiers = [s.get("test_id") for s in spectra]

        ms_network = SimilarityNetwork()
        ms_network.create_network(scores, identifiers=identifiers, score_name="score")

        nodes = list(ms_network.graph.nodes())
        nodes.sort()
        print(nodes)

    Should output

    .. testoutput::

        ['one', 'two']
    """

    def __init__(
        self,
        top_n: int = 20,
        max_links: int = 10,
        score_cutoff: float = 0.7,
        link_method: str = "single",
        keep_unconnected_nodes: bool = True,
    ):
        """
        Parameters
        ----------
        top_n
            Consider an edge between node A and node B if the score falls into
            the top_n hits of A or B (``link_method="single"``), or into the
            top_n hits of both A and B (``link_method="mutual"``).
            From those potential links, only ``max_links`` are kept per node,
            so ``top_n`` must be >= ``max_links``.
        max_links
            Maximum number of outgoing links to add per node. Default is 10.
            Due to incoming links, total degree can be higher.
        score_cutoff
            Threshold for similarities. Edges are only created for
            similarities >= ``score_cutoff``.
        link_method
            Choose between ``"single"`` and ``"mutual"``.
            - ``"single"`` adds all eligible top-k links.
            - ``"mutual"`` only adds a link if both nodes rank each other
              within their respective top-k lists.
        keep_unconnected_nodes
            If True (default), all identifiers are included as nodes even if
            they have no edges. If False, isolated nodes are removed.
        """
        self.top_n = top_n
        self.max_links = max_links
        self.score_cutoff = score_cutoff
        self.link_method = link_method
        self.keep_unconnected_nodes = keep_unconnected_nodes
        self.graph: Optional[nx.Graph] = None

    def create_network(
        self,
        scores: Scores,
        identifiers: Sequence[str],
        score_name: Optional[str] = None,
    ) -> None:
        """Create a similarity network from a square all-vs-all Scores object.

        Parameters
        ----------
        scores
            Matchms Scores object containing all-vs-all similarities.
            The score matrix must be square.
        identifiers
            Node identifiers corresponding to the rows/columns of the score matrix.
            Must have length equal to ``scores.shape[0]``.
        score_name
            Name of the score field to use. If None:
            - scalar Scores: the only field is used
            - multi-field Scores: ``"score"`` is used if present
        """
        if self.top_n < self.max_links:
            raise ValueError("top_n must be >= max_links.")
        if self.link_method not in {"single", "mutual"}:
            raise ValueError("link_method must be either 'single' or 'mutual'.")

        n_rows, n_cols = scores.shape
        if n_rows != n_cols:
            raise TypeError("Expected square all-vs-all scores for network creation.")
        if len(identifiers) != n_rows:
            raise ValueError(
                f"identifiers must have length {n_rows}, but got {len(identifiers)}."
            )

        if len(set(identifiers)) != len(identifiers):
            raise ValueError("identifiers must be unique.")

        msnet = nx.Graph()
        msnet.add_nodes_from(identifiers)

        similars_idx, similars_scores = get_top_hits(
            scores=scores,
            top_n=self.top_n,
            axis=1,
            score_name=score_name,
            identifiers=identifiers,
            ignore_diagonal=True,
        )

        for i, source_id in enumerate(identifiers):
            candidate_indices = similars_idx[source_id]
            candidate_scores = similars_scores[source_id]

            if len(candidate_indices) == 0:
                continue

            target_ids = np.array([identifiers[j] for j in candidate_indices], dtype=object)

            keep = np.where(candidate_scores >= self.score_cutoff)[0][: self.max_links]

            if self.link_method == "single":
                new_edges = [
                    (source_id, str(target_ids[k]), float(candidate_scores[k]))
                    for k in keep
                ]
            else:  # mutual
                new_edges = []
                for k in keep:
                    target_idx = candidate_indices[k]
                    target_id = identifiers[target_idx]
                    if i in similars_idx[target_id][: self.top_n]:
                        new_edges.append(
                            (source_id, str(target_id), float(candidate_scores[k]))
                        )

            msnet.add_weighted_edges_from(new_edges)

        if not self.keep_unconnected_nodes:
            msnet.remove_nodes_from(list(nx.isolates(msnet)))

        self.graph = msnet

    def export_to_file(self, filename: str, graph_format: str = "graphml"):
        """Save the network to a file.

        Parameters
        ----------
        filename
            Path to output file.
        graph_format
            Output format. Supported formats are:
            ``"cyjs"``, ``"gexf"``, ``"gml"``, ``"graphml"``, ``"json"``.
        """
        if self.graph is None:
            raise ValueError("No network found. Make sure to first run create_network().")

        writer = self._generate_writer(graph_format)
        writer(filename)

    def _generate_writer(self, graph_format: str):
        writer = {
            "cyjs": self._export_to_cyjs,
            "gexf": self._export_to_gexf,
            "gml": self._export_to_gml,
            "graphml": self.export_to_graphml,
            "json": self._export_to_node_link_json,
        }

        if graph_format not in writer:
            raise ValueError(
                "Format not supported. Please use one of: "
                "'cyjs', 'gexf', 'gml', 'graphml', 'json'."
            )
        return writer[graph_format]

    def export_to_graphml(self, filename: str):
        """Save the network as GraphML."""
        nx.write_graphml_lxml(self.graph, filename)

    def _export_to_cyjs(self, filename: str):
        """Save the network in Cytoscape JSON format."""
        graph = nx.cytoscape_data(self.graph)
        self._write_to_json(graph, filename)

    def _export_to_node_link_json(self, filename: str):
        """Save the network in node-link JSON format."""
        graph = nx.node_link_data(self.graph, edges="links")
        self._write_to_json(graph, filename)

    @staticmethod
    def _write_to_json(graph: dict, filename: str):
        """Save the network as JSON file."""
        with open(filename, "w", encoding="utf-8") as file:
            json.dump(graph, file)

    def _export_to_gexf(self, filename: str):
        """Save the network as GEXF."""
        nx.write_gexf(self.graph, filename)

    def _export_to_gml(self, filename: str):
        """Save the network as GML."""
        nx.write_gml(self.graph, filename)
