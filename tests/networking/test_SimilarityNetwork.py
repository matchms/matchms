import os
import tempfile
import numpy as np
import pytest
from matchms import Scores, Spectrum, calculate_scores
from matchms.networking import SimilarityNetwork
from matchms.similarity import FlashCosine


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
    spectra = []
    for i in range(5):
        spectra.append(Spectrum(mz=np.array([100, 200., 210. + 10 * i]),
                                  intensities=np.array([0.2 * i, 0.1 * i, 1.0]),
                                  metadata={"spectrum_id": 'ref_spec_'+str(i),
                                            "smiles": 'C1=CC=C2C(=C1)NC(=N2)C3=CC=CO3',
                                            "precursor_mz": 100+20*i}))
    for i in range(3):
        spectra.append(Spectrum(mz=np.array([100., 200.01]),
                                  intensities=np.array([0.5, 0.1 * i]),
                                  metadata={"spectrum_id": 'query_spec_'+str(i),
                                            "smiles": 'CC1=C(C=C(C=C1)NC(=O)N(C)C)Cl',
                                            "precursor_mz": 120+20*i}))
    return spectra


def create_dummy_scores():
    """Create asymmetric scores object."""
    spectra = create_dummy_spectra()
    spectra_1 = spectra[:5]
    spectra_2 = spectra[5:]

    similarity_measure = FlashCosine(matching_mode="fragment")
    scores = calculate_scores(spectra_1, spectra_2, similarity_measure)
    return scores, spectra_1, spectra_2


def create_dummy_scores_symmetric():
    spectra = create_dummy_spectra()
    similarity_measure = FlashCosine(matching_mode="fragment")
    scores = calculate_scores(spectra, spectra, similarity_measure)
    return scores, spectra


def _identifiers(spectra):
    return [s.get("spectrum_id") for s in spectra]


def test_create_network_requires_square_scores():
    scores, spectra_1, _ = create_dummy_scores()
    msnet = SimilarityNetwork()

    with pytest.raises(TypeError) as msg:
        msnet.create_network(scores, identifiers=_identifiers(spectra_1))

    assert "Expected square all-vs-all scores" in str(msg.value)


def test_create_network_identifier_length_mismatch():
    scores, spectra = create_dummy_scores_symmetric()
    msnet = SimilarityNetwork()

    with pytest.raises(ValueError) as msg:
        msnet.create_network(scores, identifiers=_identifiers(spectra[:-1]))

    assert "identifiers must have length" in str(msg.value)


def test_create_network_duplicate_identifiers():
    scores, spectra = create_dummy_scores_symmetric()
    identifiers = _identifiers(spectra)
    identifiers[0] = identifiers[1]

    msnet = SimilarityNetwork()
    with pytest.raises(ValueError) as msg:
        msnet.create_network(scores, identifiers=identifiers)

    assert "identifiers must be unique" in str(msg.value)


def test_create_network_symmetric():
    cutoff = 0.6
    scores, spectra = create_dummy_scores_symmetric()
    msnet = SimilarityNetwork(score_cutoff=cutoff)
    msnet.create_network(scores, identifiers=_identifiers(spectra), score_name="score")

    nodes_list = list(msnet.graph.nodes())
    edges_list = list(msnet.graph.edges())
    edges_list.sort()

    nodes_without_edges = ["ref_spec_0", "ref_spec_1", "ref_spec_2", "ref_spec_3"]

    assert len(nodes_list) == 8, "Expected different number of nodes"
    assert len(edges_list) == 5, "Expected different number of edges"
    assert np.all([(x[0] not in nodes_without_edges) for x in edges_list]), (
        "Expected this node to have no edges"
    )
    assert np.all([(x[1] not in nodes_without_edges) for x in edges_list]), (
        "Expected this node to have no edges"
    )


def test_create_network_symmetric_remove_unconnected_nodes():
    cutoff = 0.6
    scores, spectra = create_dummy_scores_symmetric()
    msnet = SimilarityNetwork(score_cutoff=cutoff, keep_unconnected_nodes=False)
    msnet.create_network(scores, identifiers=_identifiers(spectra), score_name="score")

    nodes_list = list(msnet.graph.nodes())
    edges_list = list(msnet.graph.edges())
    edges_list.sort()

    nodes_with_edges = ['ref_spec_4', 'query_spec_0', 'query_spec_1', 'query_spec_2']

    assert len(nodes_list) == 4, "Expected different number of nodes"
    assert np.all([(x in nodes_list) for x in nodes_with_edges]), (
        "Expected this node to have edges"
    )


def test_create_network_symmetric_higher_cutoff():
    cutoff = 0.983
    scores, spectra = create_dummy_scores_symmetric()
    msnet = SimilarityNetwork(score_cutoff=cutoff)
    msnet.create_network(scores, identifiers=_identifiers(spectra), score_name="score")

    edges_list = list(msnet.graph.edges())
    edges_list.sort()

    assert len(edges_list) == 1, "Expected only one link"
    assert edges_list[0][0] in ['query_spec_1', 'query_spec_2'], "Expected different node to have a link"
    assert edges_list[0][1] in ['query_spec_1', 'query_spec_2'], "Expected different node to have a link"


def test_create_network_symmetric_mutual_method():
    cutoff = 0.7
    scores, spectra = create_dummy_scores_symmetric()

    scores_arr = scores["score"].to_array().copy()
    scores_arr[7, 6] = scores_arr[6, 7] = 0.85
    scores_arr[7, 5] = scores_arr[5, 7] = 0.75
    scores_arr[7, 3] = scores_arr[3, 7] = 0.7

    modified_scores = Scores({"modified_score": scores_arr})

    msnet = SimilarityNetwork(score_cutoff=cutoff, top_n=3, max_links=3, link_method="mutual")
    msnet.create_network(modified_scores, identifiers=_identifiers(spectra), score_name="modified_score")

    nodes_with_edges = ["query_spec_0", "query_spec_1", "query_spec_2", "ref_spec_3"]
    edges_list = list(msnet.graph.edges())
    edges_list.sort()

    assert len(edges_list) == 4, "Expected four links"
    assert np.all([(x[0] in nodes_with_edges) for x in edges_list]), "Expected different edges in graph"
    assert np.all([(x[1] in nodes_with_edges) for x in edges_list]), "Expected different edges in graph"


def test_create_network_symmetric_max_links_1():
    cutoff = 0.6
    scores, spectra = create_dummy_scores_symmetric()
    msnet = SimilarityNetwork(score_cutoff=cutoff, max_links=1, link_method="single")
    msnet.create_network(scores, identifiers=_identifiers(spectra), score_name="score")

    edges_list = list(msnet.graph.edges())
    edges_list.sort()

    nodes_without_edges = ["ref_spec_0", "ref_spec_1", "ref_spec_2", "ref_spec_3"]

    assert len(edges_list) == 3, "Expected different number of edges"
    assert np.all([(x[0] not in nodes_without_edges) for x in edges_list]), (
        "Expected this node to have no edges"
    )
    assert np.all([(x[1] not in nodes_without_edges) for x in edges_list]), (
        "Expected this node to have no edges"
    )


def test_create_network_invalid_link_method():
    scores, spectra = create_dummy_scores_symmetric()
    msnet = SimilarityNetwork(link_method="unknown")

    with pytest.raises(ValueError) as msg:
        msnet.create_network(scores, identifiers=_identifiers(spectra), score_name="score")

    assert "link_method must be either" in str(msg.value)


def test_create_network_top_n_smaller_than_max_links():
    scores, spectra = create_dummy_scores_symmetric()
    msnet = SimilarityNetwork(top_n=2, max_links=3)

    with pytest.raises(ValueError) as msg:
        msnet.create_network(scores, identifiers=_identifiers(spectra), score_name="score")

    assert "top_n must be >= max_links" in str(msg.value)


def test_export_without_graph_raises():
    msnet = SimilarityNetwork()
    with pytest.raises(ValueError) as msg:
        msnet.export_to_file("dummy.graphml")

    assert "No network found" in str(msg.value)


def test_export_invalid_format_raises():
    scores, spectra = create_dummy_scores_symmetric()
    msnet = SimilarityNetwork()
    msnet.create_network(scores, identifiers=_identifiers(spectra), score_name="score")

    with pytest.raises(ValueError) as msg:
        msnet.export_to_file("dummy.out", graph_format="unsupported")

    assert "Format not supported" in str(msg.value)
