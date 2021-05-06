import os
import tempfile
import numpy as np
import pytest
from matchms import Spectrum
from matchms import calculate_scores
from matchms.networking import SimilarityNetwork
from matchms.similarity import FingerprintSimilarity
from matchms.similarity import ModifiedCosine


@pytest.yield_fixture
def filename():
    with tempfile.TemporaryDirectory() as temp_dir:
        filename = os.path.join(temp_dir, "test.graphml")
        yield filename


def create_dummy_spectrums():
    """Create dummy spectrums"""
    fingerprints1 = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1]]
    fingerprints2 = [[1, 0, 1], [0, 1, 1], [1, 1, 1]]
    spectrums = []
    for i, fp in enumerate(fingerprints1):
        spectrums.append(Spectrum(mz=np.array([100, 200.]),
                                  intensities=np.array([0.7, 0.1 * i]),
                                  metadata={"spectrumid": 'ref_spec_'+str(i),
                                            "fingerprint": np.array(fp),
                                            "smiles": 'C1=CC=C2C(=C1)NC(=N2)C3=CC=CO3',
                                            "precursor_mz": 100+50*i}))
    for i, fp in enumerate(fingerprints2):
        spectrums.append(Spectrum(mz=np.array([100, 200.]),
                                  intensities=np.array([0.5, 0.1 * i]),
                                  metadata={"spectrumid": 'query_spec_'+str(i),
                                            "fingerprint": np.array(fp),
                                            "smiles": 'CC1=C(C=C(C=C1)NC(=O)N(C)C)Cl',
                                            "precursor_mz": 110+50*i}))
    return spectrums


def create_dummy_scores():
    """Creat asymmetric scores object (references != queries)"""
    spectrums = create_dummy_spectrums()
    references = spectrums[:5]
    queries = spectrums[5:]

    # Create Scores object by calculating dice scores
    similarity_measure = FingerprintSimilarity("dice")
    scores = calculate_scores(references, queries, similarity_measure)
    return scores


def create_dummy_scores_symmetric():
    spectrums = create_dummy_spectrums()

    # Create Scores object by calculating dice scores
    similarity_measure = FingerprintSimilarity("dice")
    scores = calculate_scores(spectrums, spectrums, similarity_measure)
    return scores


def create_dummy_scores_symmetric_modified_cosine():
    spectrums = create_dummy_spectrums()

    # Create Scores object by calculating dice scores
    similarity_measure = ModifiedCosine()
    scores = calculate_scores(spectrums, spectrums, similarity_measure)
    return scores


def test_create_network_symmetric_wrong_input():
    """Test if function is used with non-symmetric scores object"""
    scores = create_dummy_scores()
    msnet = SimilarityNetwork()
    with pytest.raises(AssertionError) as msg:
        msnet.create_network(scores)

    expected_msg = "Expected symmetric scores object with queries==references"
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
    msnet.create_network(scores)

    edges_list = list(msnet.graph.edges())
    edges_list.sort()
    assert len(edges_list) == 28, "Expected different number of edges"


def test_create_network_export_to_graphml(filename):
    """Test creating a graph from a symmetric Scores object using ModifiedCosine"""
    cutoff = 0.7
    scores = create_dummy_scores_symmetric_modified_cosine()
    msnet = SimilarityNetwork(score_cutoff=cutoff)
    msnet.create_network(scores)
    msnet.export_to_graphml(filename)

    assert os.path.isfile(filename), "graphml file not found"


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
    cutoff = 0.7
    scores = create_dummy_scores_symmetric()
    # change some scores
    scores.scores[7, 6] = scores.scores[6, 7] = 0.85
    scores.scores[7, 5] = scores.scores[5, 7] = 0.75
    scores.scores[7, 3] = scores.scores[3, 7] = 0.7

    msnet = SimilarityNetwork(score_cutoff=cutoff, top_n=3,
                              max_links=3, link_method="mutual")
    msnet.create_network(scores)
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
