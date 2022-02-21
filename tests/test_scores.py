import numpy as np
import pytest
from matchms import Scores, calculate_scores
from matchms.similarity import CosineGreedy, IntersectMz
from matchms.similarity.BaseSimilarity import BaseSimilarity
from .builder_Spectrum import SpectrumBuilder


class DummySimilarityFunction(BaseSimilarity):
    """Simple dummy score, only contain pair-wise implementation."""
    score_datatype = [("score", np.unicode_, 16), ("len", np.int32)]

    def __init__(self):
        """constructor"""

    def pair(self, reference, query):
        """necessary pair computation method"""
        s = reference + query
        return np.array([(s, len(s))], dtype=self.score_datatype)


class DummySimilarityFunctionParallel(BaseSimilarity):
    """Simple dummy score, contains pair-wise and matrix implementation."""
    score_datatype = [("score", np.unicode_, 16), ("len", "int")]

    def __init__(self):
        """constructor"""

    def pair(self, reference, query):
        """necessary pair computation method"""
        s = reference + query
        return np.array([(s, len(s))], dtype=self.score_datatype)

    def matrix(self, references, queries, is_symmetric: bool = False):
        """additional matrix computation method"""
        shape = len(references), len(queries)
        s = np.empty(shape, dtype=self.score_datatype)
        for index_reference, reference in enumerate(references):
            for index_query, query in enumerate(queries):
                rq = reference + query
                s[index_reference, index_query] = rq, len(rq)
        return s


def spectra():
    builder = SpectrumBuilder()
    spectrum_1 = builder.with_mz(np.array([100, 150, 200.])).with_intensities(
        np.array([0.7, 0.2, 0.1])).with_metadata({'id': 'spectrum1'}).build()
    spectrum_2 = builder.with_mz(np.array([100, 140, 190.])).with_intensities(
        np.array([0.4, 0.2, 0.1])).with_metadata({'id': 'spectrum2'}).build()
    spectrum_3 = builder.with_mz(np.array([110, 140, 195.])).with_intensities(
        np.array([0.6, 0.2, 0.1])).with_metadata({'id': 'spectrum3'}).build()
    spectrum_4 = builder.with_mz(np.array([100, 150, 200.])).with_intensities(
        np.array([0.6, 0.1, 0.6])).with_metadata({'id': 'spectrum4'}).build()

    return spectrum_1, spectrum_2, spectrum_3, spectrum_4


def test_scores_single_pair():
    """Test single pair input."""
    dummy_similarity_function = DummySimilarityFunction()
    scores = Scores(references=["A"],
                    queries=["B"])
    scores.calculate(dummy_similarity_function)
    actual_1 = scores.scores[0, 0, 'DummySimilarityFunction_score']
    actual_2 = scores.scores[0, 0, 'DummySimilarityFunction_len']
    assert actual_1 == "AB", "Expected different scores."
    assert actual_2 == 2, "Expected different scores."


def test_scores_calculate():
    dummy_similarity_function = DummySimilarityFunction()
    scores = Scores(references=["r0", "r1", "r2"],
                    queries=["q0", "q1"])
    scores.calculate(dummy_similarity_function)
    actual = list(scores)
    expected = [
        ("r0", "q0", ["r0q0", 4]),
        ("r0", "q1", ["r0q1", 4]),
        ("r1", "q0", ["r1q0", 4]),
        ("r1", "q1", ["r1q1", 4]),
        ("r2", "q0", ["r2q0", 4]),
        ("r2", "q1", ["r2q1", 4])
    ]
    assert actual == expected, "Expected different scores."


def test_scores_calculate_parallel():
    dummy_similarity_function = DummySimilarityFunctionParallel()
    scores = Scores(references=["r0", "r1", "r2"],
                    queries=["q0", "q1"])
    scores.calculate(dummy_similarity_function)
    actual = list(scores)
    expected = [
        ("r0", "q0", ["r0q0", 4]),
        ("r0", "q1", ["r0q1", 4]),
        ("r1", "q0", ["r1q0", 4]),
        ("r1", "q1", ["r1q1", 4]),
        ("r2", "q0", ["r2q0", 4]),
        ("r2", "q1", ["r2q1", 4])
    ]
    assert actual == expected, "Expected different scores."


def test_scores_init_with_list():
    scores = Scores(references=["r0", "r1", "r2"],
                    queries=["q0", "q1"])
    assert scores.shape == (3, 2, 0), "Expected different scores shape."


def test_scores_init_with_numpy_array():
    scores = Scores(references=np.asarray(["r0", "r1", "r2"]),
                    queries=np.asarray(["q0", "q1"]))
    assert scores.shape == (3, 2, 0), "Expected different scores shape."


def test_scores_init_with_queries_dict():
    with pytest.raises(AssertionError) as msg:
        _ = Scores(references=["r0", "r1", "r2"],
                   queries=dict(k0="q0", k1="q1"))

    assert str(msg.value) == "Expected input argument 'queries' to be list or tuple or numpy.ndarray."


def test_scores_init_with_references_dict():
    with pytest.raises(AssertionError) as msg:
        _ = Scores(references=dict(k0="r0", k1="r1", k2="r2"),
                   queries=["q0", "q1"])

    assert str(msg.value) == "Expected input argument 'references' to be list or tuple or numpy.ndarray."


def test_scores_init_with_tuple():
    scores = Scores(references=("r0", "r1", "r2"),
                    queries=("q0", "q1"))
    assert scores.shape == (3, 2, 0), "Expected different scores shape."


def test_scores_next():

    dummy_similarity_function = DummySimilarityFunction()
    scores = Scores(references=["r", "rr", "rrr"],
                    queries=["q", "qq"]).calculate(dummy_similarity_function)

    actual = list(scores)
    expected = [
        ("r", "q", ["rq", 2]),
        ("r", "qq", ["rqq", 3]),
        ("rr", "q", ["rrq", 3]),
        ("rr", "qq", ["rrqq", 4]),
        ("rrr", "q", ["rrrq", 4]),
        ("rrr", "qq", ["rrrqq", 5])
    ]
    assert actual == expected, "Expected different scores."


def test_scores_by_referencey():
    "Test scores_by_reference method."
    spectrum_1, spectrum_2, spectrum_3, spectrum_4 = spectra()
    references = [spectrum_1, spectrum_2, spectrum_3]
    queries = [spectrum_3, spectrum_4]

    scores = calculate_scores(references, queries, CosineGreedy())
    assert scores.shape == (3, 2, 2)
    name_score = scores.score_names[0]
    selected_scores = scores.scores_by_reference(spectrum_2, name_score)

    expected_result = [(scores.queries[i], scores.scores[1, i]) for i in range(2)]
    assert selected_scores == expected_result, "Expected different scores."


def test_scores_by_reference_sorted():
    "Test scores_by_reference method with sort=True."
    spectrum_1, spectrum_2, spectrum_3, spectrum_4 = spectra()
    references = [spectrum_1, spectrum_2, spectrum_3]
    queries = [spectrum_3, spectrum_4, spectrum_2]

    scores = calculate_scores(references, queries, CosineGreedy())
    name_score = scores.score_names[0]
    selected_scores = scores.scores_by_reference(spectrum_2, name_score, sort=True)

    expected_result = [(scores.queries[i], scores.scores[1, i]) for i in [2, 1, 0]]
    assert selected_scores == expected_result, "Expected different scores."
    scores_only = np.array([x[1] for x in selected_scores]).tolist()
    scores_expected = [(1.0, 3), (0.61297133, 1), (0.13631964, 1)]
    assert np.allclose(scores_only, scores_expected, atol=1e-8), \
        "Expected different sorted scores."


def test_scores_by_referencey_non_tuple_score():
    "Test scores_by_reference method."
    spectrum_1, spectrum_2, spectrum_3, spectrum_4 = spectra()
    references = [spectrum_1, spectrum_2, spectrum_3]
    queries = [spectrum_3, spectrum_4]

    scores = calculate_scores(references, queries, IntersectMz())
    selected_scores = scores.scores_by_reference(spectrum_2)

    expected_result = [(scores.queries[i], scores.scores[1, i]) for i in range(2)]
    assert selected_scores == expected_result, "Expected different scores."


def test_scores_by_query():
    "Test scores_by_query method."
    spectrum_1, spectrum_2, spectrum_3, spectrum_4 = spectra()
    references = [spectrum_1, spectrum_2, spectrum_3]
    queries = [spectrum_2, spectrum_3, spectrum_4]

    scores = calculate_scores(references, queries, CosineGreedy())
    name_score = scores.score_names[0]
    selected_scores = scores.scores_by_query(spectrum_4, name_score)

    expected_result = [(scores.references[i], scores.scores[i, 2]) for i in range(2)]
    assert selected_scores == expected_result, "Expected different scores."


def test_scores_by_query_sorted():
    "Test scores_by_query method with sort=True."
    builder = SpectrumBuilder()
    spectrum_1 = builder.with_mz(np.array([100, 150, 200.])).with_intensities(
        np.array([0.7, 0.2, 0.1])).with_metadata({'id': 'spectrum1'}).build()
    spectrum_2 = builder.with_mz(np.array([100, 140, 190.])).with_intensities(
        np.array([0.4, 0.2, 0.1])).with_metadata({'id': 'spectrum2'}).build()
    spectrum_3 = builder.with_mz(np.array([100, 140, 195.])).with_intensities(
        np.array([0.6, 0.2, 0.1])).with_metadata({'id': 'spectrum3'}).build()
    spectrum_4 = builder.with_mz(np.array([100, 150, 200.])).with_intensities(
        np.array([0.6, 0.1, 0.6])).with_metadata({'id': 'spectrum4'}).build()

    references = [spectrum_1, spectrum_2, spectrum_3]
    queries = [spectrum_2, spectrum_3, spectrum_4]

    scores = calculate_scores(references, queries, CosineGreedy())
    name_score = scores.score_names[0]
    selected_scores = scores.scores_by_query(spectrum_4, name_score, sort=True)

    expected_result = [(scores.references[i], scores.scores[i, 2]) for i in [0, 2, 1]]
    assert selected_scores == expected_result, "Expected different scores."
    assert np.allclose(np.array([x[1] for x in selected_scores]).tolist(),
                       [(0.79636414, 3), (0.65803523, 1), (0.61297133, 1)])


def test_scores_by_query_non_tuple_score():
    "Test scores_by_query method."
    spectrum_1, spectrum_2, spectrum_3, spectrum_4 = spectra()
    references = [spectrum_1, spectrum_2, spectrum_3]
    queries = [spectrum_2, spectrum_3, spectrum_4]

    scores = calculate_scores(references, queries, IntersectMz())
    name_score = scores.score_names[0]
    selected_scores = scores.scores_by_query(spectrum_4, name_score)

    expected_result = [(scores.references[i], scores.scores[i, 2]) for i in range(2)]
    assert selected_scores == expected_result, "Expected different scores."


def test_sort_without_name_exception():
    spectrum_1, spectrum_2, spectrum_3, spectrum_4 = spectra()
    references = [spectrum_1, spectrum_2, spectrum_3]
    queries = [spectrum_2, spectrum_3, spectrum_4]

    scores = calculate_scores(references, queries, CosineGreedy())
    with pytest.raises(IndexError) as exception:
        _ = scores.scores_by_query(spectrum_4, sort=True)
    assert "For sorting, score must be specified" in exception.value.args[0]

    with pytest.raises(IndexError) as exception:
        _ = scores.scores_by_reference(spectrum_3, sort=True)
    assert "For sorting, score must be specified" in exception.value.args[0]
