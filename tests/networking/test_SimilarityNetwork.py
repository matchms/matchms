import os
import tempfile
import numpy as np
import pytest
from matchms import Spectrum, create_scores_object_and_calculate_scores
from matchms.networking import SimilarityNetwork
from matchms.similarity import FingerprintSimilarity, ModifiedCosine


@pytest.fixture(params=["cyjs", "gexf", "gml", "graphml", "json"])
def graph_format(request):
    yield request.param


@pytest.fixture()
def filename(graph_format):
    filename = f"test.{graph_format}"
    with tempfile.TemporaryDirectory() as temp_dir:
        filepath = os.path.join(temp_dir, filename)
        yield filepath


def create_dummy_spectra():
    """Create dummy spectra"""
    fingerprints1 = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]]
    fingerprints2 = [[1, 0, 1], [0, 1, 1], [1, 1, 1]]
    spectra = []
    for i, fp in enumerate(fingerprints1):
        spectra.append(Spectrum(mz=np.array([100, 200.]),
                                  intensities=np.array([0.7, 0.1 * i]),
                                  metadata={"spectrum_id": 'ref_spec_'+str(i),
                                            "fingerprint": np.array(fp),
                                            "smiles": 'C1=CC=C2C(=C1)NC(=N2)C3=CC=CO3',
                                            "precursor_mz": 100+50*i}))
    for i, fp in enumerate(fingerprints2):
        spectra.append(Spectrum(mz=np.array([100, 200.]),
                                  intensities=np.array([0.5, 0.1 * i]),
                                  metadata={"spectrum_id": 'query_spec_'+str(i),
                                            "fingerprint": np.array(fp),
                                            "smiles": 'CC1=C(C=C(C=C1)NC(=O)N(C)C)Cl',
                                            "precursor_mz": 110+50*i}))
    return spectra


def create_dummy_scores():
    """Creat asymmetric scores object (references != queries)"""
    spectra = create_dummy_spectra()
    references = spectra[:5]
    queries = spectra[5:]

    # Create Scores object by calculating dice scores
    similarity_measure = FingerprintSimilarity("dice")
    scores = create_scores_object_and_calculate_scores(references, queries, similarity_measure)
    return scores


def create_dummy_scores_symmetric():
    spectra = create_dummy_spectra()

    # Create Scores object by calculating dice scores
    similarity_measure = FingerprintSimilarity("dice")
    scores = create_scores_object_and_calculate_scores(spectra, spectra, similarity_measure)
    return scores


def create_dummy_scores_symmetric_faulty():
    scores = create_dummy_scores_symmetric()
    faulty_spec = Spectrum(mz=np.array([100, 400.]),
                           intensities=np.array([0.5, 0.1 * 4]),
                           metadata={"spectrum_id": 'query_spec_400',
                                     "fingerprint": np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]]),
                                     "smiles": 'CC1=C(C=C(C=C1)NC(=O)N(C)C)Cl',
                                     "precursor_mz": 110 + 50 * 4})
    scores.queries[0] = faulty_spec

    return scores

def create_dummy_scores_symmetric_modified_cosine():
    spectra = create_dummy_spectra()

    # Create Scores object by calculating dice scores
    similarity_measure = ModifiedCosine()
    scores = create_scores_object_and_calculate_scores(spectra, spectra, similarity_measure)
    return scores


def test_create_network_symmetric_wrong_input():
    """Test if function is used with non-symmetric scores object"""
    scores = create_dummy_scores()
    msnet = SimilarityNetwork()
    with pytest.raises(TypeError) as msg:
        msnet.create_network(scores)

    expected_msg = "Expected symmetric scores"
    assert expected_msg in str(msg), "Expected different exception"

    scores = create_dummy_scores_symmetric_faulty()
    with pytest.raises(ValueError) as msg:
        msnet.create_network(scores)

    expected_msg = "Queries and references do not match"
    assert expected_msg in str(msg), "Expected different exception"


def test_create_network_symmetric():
    """Test creating a graph from a symmetric Scores object"""
    cutoff = 0.7
    scores = create_dummy_scores_symmetric()
    msnet = SimilarityNetwork(score_cutoff=cutoff)
    msnet.create_network(scores)

    nodes_list = list(msnet.graph.nodes())
    edges_list = list(msnet.graph.edges())
    edges_list.sort()
    nodes_without_edges = ['ref_spec_0',
                           'ref_spec_1',
                           'ref_spec_2']
    assert len(nodes_list) == 8, "Expected different number of nodes"
    assert len(edges_list) == 5, "Expected different number of edges"
    assert np.all([(x[0] not in nodes_without_edges) for x in edges_list]), \
        "Expected this node to have no edges"
    assert np.all([(x[1] not in nodes_without_edges) for x in edges_list]), \
        "Expected this node to have no edges"


def test_create_network_symmetric_remove_unconnected_nodes():
    """Test if unconnected nodes are removed"""
    cutoff = 0.7
    scores = create_dummy_scores_symmetric()
    msnet = SimilarityNetwork(score_cutoff=cutoff, keep_unconnected_nodes=False)
    msnet.create_network(scores)

    nodes_list = list(msnet.graph.nodes())
    edges_list = list(msnet.graph.edges())
    edges_list.sort()
    nodes_with_edges = ['query_spec_0', 'ref_spec_4',
                        'query_spec_2', 'ref_spec_3', 'query_spec_1']
    assert len(nodes_list) == 5, "Expected different number of nodes"
    assert np.all([(x in nodes_with_edges) for x in nodes_list]), \
        "Expected this node to have edges"


def test_create_network_symmetric_modified_cosine():
    """Test creating a graph from a symmetric Scores object using ModifiedCosine"""
    cutoff = 0.7
    scores = create_dummy_scores_symmetric_modified_cosine()
    msnet = SimilarityNetwork(score_cutoff=cutoff)
    msnet.create_network(scores, score_name="ModifiedCosine_score")

    edges_list = list(msnet.graph.edges())
    edges_list.sort()
    assert len(edges_list) == 28, "Expected different number of edges"


def test_create_network_export_to_file(filename, graph_format):
    """Test creating a graph file from a symmetric Scores object using ModifiedCosine"""
    cutoff = 0.7
    scores = create_dummy_scores_symmetric_modified_cosine()
    msnet = SimilarityNetwork(score_cutoff=cutoff)
    msnet.create_network(scores, score_name="ModifiedCosine_score")
    msnet.export_to_file(filename, graph_format)

    assert os.path.isfile(filename), "network file not found"


def test_create_network_symmetric_higher_cutoff():
    cutoff = 0.9
    scores = create_dummy_scores_symmetric()
    msnet = SimilarityNetwork(score_cutoff=cutoff)
    msnet.create_network(scores)

    edges_list = list(msnet.graph.edges())
    edges_list.sort()
    assert len(edges_list) == 1, "Expected only one link"
    assert edges_list[0][0] in ['query_spec_0', 'ref_spec_4'], \
        "Expected different node to have a link"
    assert edges_list[0][1] in ['query_spec_0', 'ref_spec_4'], \
        "Expected different node to have a link"


def test_create_network_symmetric_mutual_method():
    """Test creating a graph from a Scores object"""
    # pylint: disable=protected-access
    cutoff = 0.7
    scores = create_dummy_scores_symmetric()
    scores_arr = scores.to_array()
    # change some scores
    scores_arr[7, 6] = scores_arr[6, 7] = 0.85
    scores_arr[7, 5] = scores_arr[5, 7] = 0.75
    scores_arr[7, 3] = scores_arr[3, 7] = 0.7
    scores._scores.add_dense_matrix(scores_arr, "modified_score")

    msnet = SimilarityNetwork(score_cutoff=cutoff, top_n=3,
                              max_links=3, link_method="mutual")
    msnet.create_network(scores, score_name="modified_score")
    nodes_with_edges = ['query_spec_0', 'query_spec_1', 'query_spec_2', 'ref_spec_4']
    edges_list = list(msnet.graph.edges())
    edges_list.sort()
    assert len(edges_list) == 4, "Expected four links"
    assert np.all([(x[0] in nodes_with_edges) for x in edges_list]), "Expected different edges in graph"
    assert np.all([(x[1] in nodes_with_edges) for x in edges_list]), "Expected different edges in graph"


def test_create_network_symmetric_max_links_1():
    """Test creating a graph from a Scores object using max_links=1"""
    cutoff = 0.7
    scores = create_dummy_scores_symmetric()
    msnet = SimilarityNetwork(score_cutoff=cutoff, max_links=1, link_method="single")
    msnet.create_network(scores)

    edges_list = list(msnet.graph.edges())
    edges_list.sort()
    nodes_without_edges = ['ref_spec_0',
                           'ref_spec_1',
                           'ref_spec_2']
    assert len(edges_list) == 3, "Expected different number of edges"
    assert np.all([(x[0] not in nodes_without_edges) for x in edges_list]), \
        "Expected this node to have no edges"
    assert np.all([(x[1] not in nodes_without_edges) for x in edges_list]), \
        "Expected this node to have no edges"
