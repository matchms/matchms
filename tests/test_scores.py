import os
import tempfile
import numpy
import pytest
from matchms import Scores, calculate_scores
from matchms.similarity import CosineGreedy, IntersectMz
from matchms.similarity.BaseSimilarity import BaseSimilarity
from .builder_Spectrum import SpectrumBuilder


class DummySimilarityFunction(BaseSimilarity):
    """Simple dummy score, only contain pair-wise implementation."""
    score_datatype = [("score", numpy.unicode_, 16), ("len", numpy.int32)]

    def __init__(self):
        """constructor"""

    def pair(self, reference, query):
        """necessary pair computation method"""
        s = reference + query
        return numpy.array([(s, len(s))], dtype=self.score_datatype)


class DummySimilarityFunctionParallel(BaseSimilarity):
    """Simple dummy score, contains pair-wise and matrix implementation."""
    score_datatype = [("score", numpy.unicode_, 16), ("len", "int")]

    def __init__(self):
        """constructor"""

    def pair(self, reference, query):
        """necessary pair computation method"""
        s = reference + query
        return numpy.array([(s, len(s))], dtype=self.score_datatype)

    def matrix(self, references, queries, is_symmetric: bool = False):
        """additional matrix computation method"""
        shape = len(references), len(queries)
        s = numpy.empty(shape, dtype=self.score_datatype)
        for index_reference, reference in enumerate(references):
            for index_query, query in enumerate(queries):
                rq = reference + query
                s[index_reference, index_query] = rq, len(rq)
        return s


@pytest.fixture(params=["json", "pkl"])
def file_format(request):
    yield request.param


@pytest.fixture()
def filename(file_format):
    with tempfile.TemporaryDirectory() as tmpdir:
        yield os.path.join(tmpdir, f"test_scores.{file_format}")


def spectra():
    builder = SpectrumBuilder()
    spectrum_1 = builder.with_mz(numpy.array([100, 150, 200.])).with_intensities(
        numpy.array([0.7, 0.2, 0.1])).with_metadata({'id': 'spectrum1'}).build()
    spectrum_2 = builder.with_mz(numpy.array([100, 140, 190.])).with_intensities(
        numpy.array([0.4, 0.2, 0.1])).with_metadata({'id': 'spectrum2'}).build()
    spectrum_3 = builder.with_mz(numpy.array([110, 140, 195.])).with_intensities(
        numpy.array([0.6, 0.2, 0.1])).with_metadata({'id': 'spectrum3'}).build()
    spectrum_4 = builder.with_mz(numpy.array([100, 150, 200.])).with_intensities(
        numpy.array([0.6, 0.1, 0.6])).with_metadata({'id': 'spectrum4'}).build()

    return spectrum_1, spectrum_2, spectrum_3, spectrum_4


def test_scores_single_pair():
    """Test single pair input."""
    dummy_similarity_function = DummySimilarityFunction()
    scores = Scores(references=["A"],
                    queries=["B"],
                    similarity_function=dummy_similarity_function)
    scores.calculate()
    actual = scores.scores[0][0]
    expected = numpy.array([('AB', 2)], dtype=dummy_similarity_function.score_datatype)
    assert actual == expected, "Expected different scores."


def test_scores_calculate():
    dummy_similarity_function = DummySimilarityFunction()
    scores = Scores(references=["r0", "r1", "r2"],
                    queries=["q0", "q1"],
                    similarity_function=dummy_similarity_function)
    scores.calculate()
    actual = list(scores)
    expected = [
        ("r0", "q0", numpy.array([("r0q0", 4)], dtype=dummy_similarity_function.score_datatype)),
        ("r0", "q1", numpy.array([("r0q1", 4)], dtype=dummy_similarity_function.score_datatype)),
        ("r1", "q0", numpy.array([("r1q0", 4)], dtype=dummy_similarity_function.score_datatype)),
        ("r1", "q1", numpy.array([("r1q1", 4)], dtype=dummy_similarity_function.score_datatype)),
        ("r2", "q0", numpy.array([("r2q0", 4)], dtype=dummy_similarity_function.score_datatype)),
        ("r2", "q1", numpy.array([("r2q1", 4)], dtype=dummy_similarity_function.score_datatype))
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
        ("r0", "q0", numpy.array([("r0q0", 4)], dtype=dummy_similarity_function.score_datatype)),
        ("r0", "q1", numpy.array([("r0q1", 4)], dtype=dummy_similarity_function.score_datatype)),
        ("r1", "q0", numpy.array([("r1q0", 4)], dtype=dummy_similarity_function.score_datatype)),
        ("r1", "q1", numpy.array([("r1q1", 4)], dtype=dummy_similarity_function.score_datatype)),
        ("r2", "q0", numpy.array([("r2q0", 4)], dtype=dummy_similarity_function.score_datatype)),
        ("r2", "q1", numpy.array([("r2q1", 4)], dtype=dummy_similarity_function.score_datatype))
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
        ("r", "q", numpy.array([("rq", 2)], dtype=dummy_similarity_function.score_datatype)),
        ("r", "qq", numpy.array([("rqq", 3)], dtype=dummy_similarity_function.score_datatype)),
        ("rr", "q", numpy.array([("rrq", 3)], dtype=dummy_similarity_function.score_datatype)),
        ("rr", "qq", numpy.array([("rrqq", 4)], dtype=dummy_similarity_function.score_datatype)),
        ("rrr", "q", numpy.array([("rrrq", 4)], dtype=dummy_similarity_function.score_datatype)),
        ("rrr", "qq", numpy.array([("rrrqq", 5)], dtype=dummy_similarity_function.score_datatype))
    ]
    assert actual == expected, "Expected different scores."


def test_scores_by_referencey():
    "Test scores_by_reference method."
    spectrum_1, spectrum_2, spectrum_3, spectrum_4 = spectra()
    references = [spectrum_1, spectrum_2, spectrum_3]
    queries = [spectrum_3, spectrum_4]

    scores = calculate_scores(references, queries, CosineGreedy())
    selected_scores = scores.scores_by_reference(spectrum_2)

    expected_result = [(scores.queries[i], scores.scores[1, i]) for i in range(2)]
    assert selected_scores == expected_result, "Expected different scores."


def test_scores_by_reference_sorted():
    "Test scores_by_reference method with sort=True."
    spectrum_1, spectrum_2, spectrum_3, spectrum_4 = spectra()
    references = [spectrum_1, spectrum_2, spectrum_3]
    queries = [spectrum_3, spectrum_4, spectrum_2]

    scores = calculate_scores(references, queries, CosineGreedy())
    selected_scores = scores.scores_by_reference(spectrum_2, sort=True)

    expected_result = [(scores.queries[i], scores.scores[1, i]) for i in [2, 1, 0]]
    assert selected_scores == expected_result, "Expected different scores."
    scores_only = numpy.array([x[1]["score"] for x in selected_scores])
    scores_expected = numpy.array([1.0, 0.6129713330865563, 0.1363196353181994])
    assert numpy.allclose(scores_only, scores_expected, atol=1e-8), \
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
    selected_scores = scores.scores_by_query(spectrum_4)

    expected_result = [(scores.references[i], scores.scores[i, 2]) for i in range(3)]
    assert selected_scores == expected_result, "Expected different scores."


def test_scores_by_query_sorted():
    "Test scores_by_query method with sort=True."
    builder = SpectrumBuilder()
    spectrum_1 = builder.with_mz(numpy.array([100, 150, 200.])).with_intensities(
        numpy.array([0.7, 0.2, 0.1])).with_metadata({'id': 'spectrum1'}).build()
    spectrum_2 = builder.with_mz(numpy.array([100, 140, 190.])).with_intensities(
        numpy.array([0.4, 0.2, 0.1])).with_metadata({'id': 'spectrum2'}).build()
    spectrum_3 = builder.with_mz(numpy.array([100, 140, 195.])).with_intensities(
        numpy.array([0.6, 0.2, 0.1])).with_metadata({'id': 'spectrum3'}).build()
    spectrum_4 = builder.with_mz(numpy.array([100, 150, 200.])).with_intensities(
        numpy.array([0.6, 0.1, 0.6])).with_metadata({'id': 'spectrum4'}).build()

    references = [spectrum_1, spectrum_2, spectrum_3]
    queries = [spectrum_2, spectrum_3, spectrum_4]

    scores = calculate_scores(references, queries, CosineGreedy())
    selected_scores = scores.scores_by_query(spectrum_4, sort=True)

    expected_result = [(scores.references[i], scores.scores[i, 2]) for i in [0, 2, 1]]
    assert selected_scores == expected_result, "Expected different scores."


def test_scores_by_query_non_tuple_score():
    "Test scores_by_query method."
    spectrum_1, spectrum_2, spectrum_3, spectrum_4 = spectra()
    references = [spectrum_1, spectrum_2, spectrum_3]
    queries = [spectrum_2, spectrum_3, spectrum_4]

    scores = calculate_scores(references, queries, IntersectMz())
    selected_scores = scores.scores_by_query(spectrum_4)

    expected_result = [(scores.references[i], scores.scores[i, 2]) for i in range(3)]
    assert selected_scores == expected_result, "Expected different scores."


def test_comparing_scores():
    "Test comparing scores objects."
    spectrum_1, spectrum_2, spectrum_3, spectrum_4 = spectra()
    spectrums = [spectrum_1, spectrum_2, spectrum_3, spectrum_4]

    scores_symmetric_cosine = calculate_scores(spectrums[0:3], spectrums[0:3], CosineGreedy())
    scores_symmetric_cosine_copy = calculate_scores(spectrums[0:3], spectrums[0:3], CosineGreedy())
    scores_symmetric_intersect = calculate_scores(spectrums[0:3], spectrums[0:3], IntersectMz())

    assert scores_symmetric_cosine == scores_symmetric_cosine_copy
    assert scores_symmetric_cosine != scores_symmetric_intersect

    scores_asymmetric_cosine = calculate_scores(spectrums[0:3], spectrums, CosineGreedy())
    scores_asymmetric_cosine_copy = calculate_scores(spectrums[0:3], spectrums, CosineGreedy())
    scores_asymmetric_cosine_mirrored = calculate_scores(spectrums, spectrums[0:3], CosineGreedy())

    assert scores_asymmetric_cosine == scores_asymmetric_cosine_copy
    assert scores_asymmetric_cosine != scores_asymmetric_cosine_mirrored

    scores_parametrized = calculate_scores(spectrums, spectrums, CosineGreedy(tolerance=0.1, mz_power=0.5))
    scores_parametrized_mirrored = calculate_scores(spectrums, spectrums, CosineGreedy(tolerance=0.5, mz_power=0.1))

    assert scores_parametrized != scores_parametrized_mirrored
