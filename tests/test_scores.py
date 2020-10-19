import numpy
import pytest
from matchms import Scores
from matchms import Spectrum
from matchms.similarity import CosineGreedy
from matchms.similarity import IntersectMz
from matchms.similarity.BaseSimilarity import BaseSimilarity


class DummySimilarityFunction(BaseSimilarity):
    """Simple dummy score, only contain pair-wise implementation."""
    score_datatype = [("score", "float"), ("len", "int")]

    def __init__(self):
        """constructor"""

    def pair(self, reference, query):
        """necessary pair computation method"""
        s = reference + query
        return s, len(s)


class DummySimilarityFunctionParallel(BaseSimilarity):
    """Simple dummy score, contains pair-wise and matrix implementation."""
    def __init__(self):
        """constructor"""

    def pair(self, reference, query):
        """necessary pair computation method"""
        s = reference + query
        return s, len(s)

    def matrix(self, references, queries, is_symmetric: bool = False):
        """additional matrix computation method"""
        shape = len(references), len(queries)
        s = numpy.empty(shape, dtype="object")
        for index_reference, reference in enumerate(references):
            for index_query, query in enumerate(queries):
                rq = reference + query
                s[index_reference, index_query] = rq, len(rq)
        return s


def test_scores_single_pair():
    """Test single pair input."""
    dummy_similarity_function = DummySimilarityFunction()
    scores = Scores(references=["A"],
                    queries=["B"],
                    similarity_function=dummy_similarity_function)
    scores.calculate()
    actual = scores.scores[0][0]
    expected = ('AB', 2)
    assert actual == expected, "Expected different scores."


def test_scores_calculate():
    dummy_similarity_function = DummySimilarityFunction()
    scores = Scores(references=["r0", "r1", "r2"],
                    queries=["q0", "q1"],
                    similarity_function=dummy_similarity_function)
    scores.calculate()
    actual = list(scores)
    expected = [
        ("r0", "q0", "r0q0", 4),
        ("r0", "q1", "r0q1", 4),
        ("r1", "q0", "r1q0", 4),
        ("r1", "q1", "r1q1", 4),
        ("r2", "q0", "r2q0", 4),
        ("r2", "q1", "r2q1", 4)
    ]
    assert actual == expected, "Expected different scores."


def test_scores_calculate_parallel():
    dummy_similarity_function = DummySimilarityFunctionParallel()
    scores = Scores(references=["r0", "r1", "r2"],
                    queries=["q0", "q1"],
                    similarity_function=dummy_similarity_function)
    scores.calculate()
    actual = list(scores)
    expected = [
        ("r0", "q0", "r0q0", 4),
        ("r0", "q1", "r0q1", 4),
        ("r1", "q0", "r1q0", 4),
        ("r1", "q1", "r1q1", 4),
        ("r2", "q0", "r2q0", 4),
        ("r2", "q1", "r2q1", 4)
    ]
    assert actual == expected, "Expected different scores."


def test_scores_init_with_list():

    dummy_similarity_function = DummySimilarityFunction()
    scores = Scores(references=["r0", "r1", "r2"],
                    queries=["q0", "q1"],
                    similarity_function=dummy_similarity_function)
    assert scores.scores.shape == (3, 2), "Expected different scores shape."


def test_scores_init_with_numpy_array():

    dummy_similarity_function = DummySimilarityFunction()
    scores = Scores(references=numpy.asarray(["r0", "r1", "r2"]),
                    queries=numpy.asarray(["q0", "q1"]),
                    similarity_function=dummy_similarity_function)
    assert scores.scores.shape == (3, 2), "Expected different scores shape."


def test_scores_init_with_queries_dict():

    dummy_similarity_function = DummySimilarityFunction()
    with pytest.raises(AssertionError) as msg:
        _ = Scores(references=["r0", "r1", "r2"],
                   queries=dict(k0="q0", k1="q1"),
                   similarity_function=dummy_similarity_function)

    assert str(msg.value) == "Expected input argument 'queries' to be list or tuple or numpy.ndarray."


def test_scores_init_with_references_dict():

    dummy_similarity_function = DummySimilarityFunction()
    with pytest.raises(AssertionError) as msg:
        _ = Scores(references=dict(k0="r0", k1="r1", k2="r2"),
                   queries=["q0", "q1"],
                   similarity_function=dummy_similarity_function)

    assert str(msg.value) == "Expected input argument 'references' to be list or tuple or numpy.ndarray."


def test_scores_init_with_tuple():

    dummy_similarity_function = DummySimilarityFunction()
    scores = Scores(references=("r0", "r1", "r2"),
                    queries=("q0", "q1"),
                    similarity_function=dummy_similarity_function)
    assert scores.scores.shape == (3, 2), "Expected different scores shape."


def test_scores_next():

    dummy_similarity_function = DummySimilarityFunction()
    scores = Scores(references=["r", "rr", "rrr"],
                    queries=["q", "qq"],
                    similarity_function=dummy_similarity_function).calculate()

    actual = list(scores)
    expected = [
        ("r", "q", "rq", 2),
        ("r", "qq", "rqq", 3),
        ("rr", "q", "rrq", 3),
        ("rr", "qq", "rrqq", 4),
        ("rrr", "q", "rrrq", 4),
        ("rrr", "qq", "rrrqq", 5)
    ]
    assert actual == expected, "Expected different scores."


def test_scores_by_referencey():
    "Test scores_by_reference method."
    spectrum_1 = Spectrum(mz=numpy.array([100, 150, 200.]),
                          intensities=numpy.array([0.7, 0.2, 0.1]),
                          metadata={'id': 'spectrum1'})
    spectrum_2 = Spectrum(mz=numpy.array([100, 140, 190.]),
                          intensities=numpy.array([0.4, 0.2, 0.1]),
                          metadata={'id': 'spectrum2'})
    spectrum_3 = Spectrum(mz=numpy.array([110, 140, 195.]),
                          intensities=numpy.array([0.6, 0.2, 0.1]),
                          metadata={'id': 'spectrum3'})
    spectrum_4 = Spectrum(mz=numpy.array([100, 150, 200.]),
                          intensities=numpy.array([0.6, 0.1, 0.6]),
                          metadata={'id': 'spectrum4'})
    references = [spectrum_1, spectrum_2, spectrum_3]
    queries = [spectrum_3, spectrum_4]

    scores = Scores(references, queries, CosineGreedy()).calculate()
    selected_scores = scores.scores_by_reference(spectrum_2)

    expected_result = [(scores.queries[i], *scores.scores[1, i]) for i in range(2)]
    assert selected_scores == expected_result, "Expected different scores."


def test_scores_by_referencey_non_tuple_score():
    "Test scores_by_reference method."
    spectrum_1 = Spectrum(mz=numpy.array([100, 150, 200.]),
                          intensities=numpy.array([0.7, 0.2, 0.1]),
                          metadata={'id': 'spectrum1'})
    spectrum_2 = Spectrum(mz=numpy.array([100, 140, 190.]),
                          intensities=numpy.array([0.4, 0.2, 0.1]),
                          metadata={'id': 'spectrum2'})
    spectrum_3 = Spectrum(mz=numpy.array([110, 140, 195.]),
                          intensities=numpy.array([0.6, 0.2, 0.1]),
                          metadata={'id': 'spectrum3'})
    spectrum_4 = Spectrum(mz=numpy.array([100, 150, 200.]),
                          intensities=numpy.array([0.6, 0.1, 0.6]),
                          metadata={'id': 'spectrum4'})
    references = [spectrum_1, spectrum_2, spectrum_3]
    queries = [spectrum_3, spectrum_4]

    scores = Scores(references, queries, IntersectMz()).calculate()
    selected_scores = scores.scores_by_reference(spectrum_2)

    expected_result = [(scores.queries[i], scores.scores[1, i]) for i in range(2)]
    assert selected_scores == expected_result, "Expected different scores."


def test_scores_by_query():
    "Test scores_by_query method."
    spectrum_1 = Spectrum(mz=numpy.array([100, 150, 200.]),
                          intensities=numpy.array([0.7, 0.2, 0.1]),
                          metadata={'id': 'spectrum1'})
    spectrum_2 = Spectrum(mz=numpy.array([100, 140, 190.]),
                          intensities=numpy.array([0.4, 0.2, 0.1]),
                          metadata={'id': 'spectrum2'})
    spectrum_3 = Spectrum(mz=numpy.array([110, 140, 195.]),
                          intensities=numpy.array([0.6, 0.2, 0.1]),
                          metadata={'id': 'spectrum3'})
    spectrum_4 = Spectrum(mz=numpy.array([100, 150, 200.]),
                          intensities=numpy.array([0.6, 0.1, 0.6]),
                          metadata={'id': 'spectrum4'})
    references = [spectrum_1, spectrum_2, spectrum_3]
    queries = [spectrum_2, spectrum_3, spectrum_4]

    scores = Scores(references, queries, CosineGreedy()).calculate()
    selected_scores = scores.scores_by_query(spectrum_4)

    expected_result = [(scores.references[i], *scores.scores[i, 2]) for i in range(3)]
    assert selected_scores == expected_result, "Expected different scores."


def test_scores_by_query_non_tuple_score():
    "Test scores_by_query method."
    spectrum_1 = Spectrum(mz=numpy.array([100, 150, 200.]),
                          intensities=numpy.array([0.7, 0.2, 0.1]),
                          metadata={'id': 'spectrum1'})
    spectrum_2 = Spectrum(mz=numpy.array([100, 140, 190.]),
                          intensities=numpy.array([0.4, 0.2, 0.1]),
                          metadata={'id': 'spectrum2'})
    spectrum_3 = Spectrum(mz=numpy.array([110, 140, 195.]),
                          intensities=numpy.array([0.6, 0.2, 0.1]),
                          metadata={'id': 'spectrum3'})
    spectrum_4 = Spectrum(mz=numpy.array([100, 150, 200.]),
                          intensities=numpy.array([0.6, 0.1, 0.6]),
                          metadata={'id': 'spectrum4'})
    references = [spectrum_1, spectrum_2, spectrum_3]
    queries = [spectrum_2, spectrum_3, spectrum_4]

    scores = Scores(references, queries, IntersectMz()).calculate()
    selected_scores = scores.scores_by_query(spectrum_4)

    expected_result = [(scores.references[i], scores.scores[i, 2]) for i in range(3)]
    assert selected_scores == expected_result, "Expected different scores."
