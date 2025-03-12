import os
import tempfile
import numpy as np
import pytest
from matchms import Spectrum, calculate_scores
from matchms.similarity import FingerprintSimilarity, ModifiedCosine
from matchms.networking import CommunityNetwork

@pytest.fixture(params=["cyjs", "gexf", "gml", "graphml", "json"])
def graph_format(request):
    return request.param

@pytest.fixture()
def filename(graph_format):
    filename = f"test.{graph_format}"
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, filename)
        yield filepath

def create_dummy_spectra():
    """Create dummy spectra with two groups."""
    fingerprints1 = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]]
    fingerprints2 = [[1, 0, 1], [0, 1, 1], [1, 1, 1]]
    spectra = []
    # First group of spectra (references)
    for i, fp in enumerate(fingerprints1):
        spectra.append(Spectrum(mz=np.array([100, 200.]),
                                intensities=np.array([0.7, 0.1 * i]),
                                metadata={"spectrum_id": f"ref_spec_{i}",
                                          "fingerprint": np.array(fp),
                                          "smiles": "C1=CC=C2C(=C1)NC(=N2)C3=CC=CO3",
                                          "precursor_mz": 100 + 50 * i}))
    # Second group of spectra (queries)
    for i, fp in enumerate(fingerprints2):
        spectra.append(Spectrum(mz=np.array([100, 200.]),
                                intensities=np.array([0.5, 0.1 * i]),
                                metadata={"spectrum_id": f"query_spec_{i}",
                                          "fingerprint": np.array(fp),
                                          "smiles": "CC1=C(C=C(C=C1)NC(=O)N(C)C)Cl",
                                          "precursor_mz": 110 + 50 * i}))
    return spectra

def create_dummy_scores():
    """
    Create asymmetric Scores object (references and queries differ).
    Used for testing that non-symmetric inputs trigger errors.
    """
    spectra = create_dummy_spectra()
    references = spectra[:5]
    queries = spectra[5:]
    similarity_measure = FingerprintSimilarity("dice")
    scores = calculate_scores(references, queries, similarity_measure)
    return scores

def create_dummy_scores_symmetric():
    """
    Create a symmetric Scores object using all spectra.
    """
    spectra = create_dummy_spectra()
    similarity_measure = FingerprintSimilarity("dice")
    scores = calculate_scores(spectra, spectra, similarity_measure)
    return scores

def create_dummy_scores_symmetric_faulty():
    """
    Create a faulty symmetric Scores object (by modifying one query)
    to test if mismatches trigger an error.
    """
    scores = create_dummy_scores_symmetric()
    faulty_spec = Spectrum(mz=np.array([100, 400.]),
                           intensities=np.array([0.5, 0.1 * 4]),
                           metadata={"spectrum_id": "faulty_spec",
                                     "fingerprint": np.array([[1, 0, 0],
                                                              [0, 1, 0],
                                                              [0, 0, 1],
                                                              [1, 1, 0],
                                                              [1, 0, 1]]),
                                     "smiles": "CC1=C(C=C(C=C1)NC(=O)N(C)C)Cl",
                                     "precursor_mz": 110 + 50 * 4})
    scores.queries[0] = faulty_spec
    return scores

def create_dummy_scores_symmetric_modified_cosine():
    """
    Create a symmetric Scores object using the ModifiedCosine similarity.
    """
    spectra = create_dummy_spectra()
    similarity_measure = ModifiedCosine()
    scores = calculate_scores(spectra, spectra, similarity_measure)
    return scores

def test_create_network_symmetric_wrong_input():
    """Test that non-symmetric Scores objects raise errors."""
    scores = create_dummy_scores()
    ensnet = CommunityNetwork()
    with pytest.raises(TypeError, match="Expected symmetric scores"):
        ensnet.create_network(scores)

    scores_faulty = create_dummy_scores_symmetric_faulty()
    with pytest.raises(ValueError, match="Queries and references in scores do not match"):
        ensnet.create_network(scores_faulty)

def test_create_network_symmetric():
    """Test building a network from a symmetric Scores object."""
    cutoff = 0.7
    # Using k=3 to keep the test graph small and deterministic
    scores = create_dummy_scores_symmetric()
    ensnet = CommunityNetwork(score_cutoff=cutoff, k=3)
    ensnet.create_network(scores)
    graph = ensnet.graph
    assert graph is not None, "Graph should not be None after creation."
    nodes_list = list(graph.nodes())
    assert len(nodes_list) == 8, "Expected 8 nodes in the graph."
    # Check that each node has an assigned community attribute.
    for node in nodes_list:
        assert "community" in graph.nodes[node], f"Node {node} missing 'community' attribute."
    # Verify that all edges have a weight >= cutoff.
    for u, v, data in graph.edges(data=True):
        assert data.get("weight", 0) >= cutoff, "Edge weight below cutoff found."

def test_remove_inter_community_links():
    """Test that inter-community edges are removed when remove_inter_community_links=True."""
    cutoff = 0.7
    scores = create_dummy_scores_symmetric()
    ensnet = CommunityNetwork(score_cutoff=cutoff, k=3, remove_inter_community_links=True)
    ensnet.create_network(scores)
    graph = ensnet.graph
    for u, v in graph.edges():
        community_u = graph.nodes[u].get("community")
        community_v = graph.nodes[v].get("community")
        assert community_u == community_v, f"Inter-community edge found between {u} and {v}."

def test_keep_inter_community_links():
    """
    Test that when remove_inter_community_links is False the graph may contain edges connecting nodes
    of the same community.
    """
    cutoff = 0.7
    scores = create_dummy_scores_symmetric()
    ensnet = CommunityNetwork(score_cutoff=cutoff, k=3, remove_inter_community_links=False)
    ensnet.create_network(scores)
    graph = ensnet.graph
    # Depending on the community detection, it is possible that no inter-community edge is present.
    # Therefore, we only check that the graph structure is intact.
    assert graph.number_of_edges() > 0, "Expected some edges in the network."

def test_create_network_higher_cutoff():
    """Test that a higher cutoff results in fewer (or equal) edges."""
    cutoff_low = 0.7
    cutoff_high = 0.9
    scores = create_dummy_scores_symmetric()
    ensnet_low = CommunityNetwork(score_cutoff=cutoff_low, k=3, remove_inter_community_links=False)
    ensnet_low.create_network(scores)
    ensnet_high = CommunityNetwork(score_cutoff=cutoff_high, k=3, remove_inter_community_links=False)
    ensnet_high.create_network(scores)
    edges_low = list(ensnet_low.graph.edges())
    edges_high = list(ensnet_high.graph.edges())
    assert len(edges_high) <= len(edges_low), "Expected fewer or equal edges with higher cutoff."

def test_create_network_different_k():
    """Test that using a larger k value produces a graph with at least as many edges as with a smaller k."""
    scores = create_dummy_scores_symmetric()
    ensnet_k_small = CommunityNetwork(score_cutoff=0.7, k=1, remove_inter_community_links=False)
    ensnet_k_small.create_network(scores)
    ensnet_k_large = CommunityNetwork(score_cutoff=0.7, k=5, remove_inter_community_links=False)
    ensnet_k_large.create_network(scores)
    assert ensnet_k_small.graph.number_of_edges() <= ensnet_k_large.graph.number_of_edges(), (
        "Graph with larger k should have at least as many edges."
    )

def test_export_to_file(filename, graph_format):
    """Test exporting the graph to a file."""
    cutoff = 0.7
    scores = create_dummy_scores_symmetric_modified_cosine()
    ensnet = CommunityNetwork(score_cutoff=cutoff, k=3, remove_inter_community_links=False)
    ensnet.create_network(scores, score_name="ModifiedCosine_score")
    ensnet.export_to_file(filename, graph_format)
    assert os.path.isfile(filename), "Exported network file not found."
