import os
import tempfile
import numpy as np
import pytest
from matchms import Scores, calculate_scores
from matchms.similarity import CosineGreedy, IntersectMz
from matchms.similarity.BaseSimilarity import BaseSimilarity
from .builder_Spectrum import SpectrumBuilder


class DummySimilarityFunction(BaseSimilarity):
    """Simple dummy score, only contain pair-wise implementation."""

    score_datatype = [("score", np.str_, 16), ("len", np.int32)]

    def __init__(self):
        """constructor"""

    def pair(self, reference, query):
        """necessary pair computation method"""
        s = reference + query
        return np.array([(s, len(s))], dtype=self.score_datatype)


class DummySimilarityFunctionParallel(BaseSimilarity):
    """Simple dummy score, contains pair-wise and matrix implementation."""

    score_datatype = [("score", np.str_, 16), ("len", "int")]

    def __init__(self):
        """constructor"""

    def pair(self, reference, query):
        """necessary pair computation method"""
        s = reference + query
        return np.array([(s, len(s))], dtype=self.score_datatype)

    def matrix(self, references, queries, array_type: str = "numpy", is_symmetric: bool = False):
        """additional matrix computation method"""
        shape = len(references), len(queries)
        s = np.empty(shape, dtype=self.score_datatype)
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
    spectrum_1 = (
        builder.with_mz(np.array([100, 150, 200.0]))
        .with_intensities(np.array([0.7, 0.2, 0.1]))
        .with_metadata({"id": "spectrum1"})
        .build()
    )
    spectrum_2 = (
        builder.with_mz(np.array([100, 140, 190.0]))
        .with_intensities(np.array([0.4, 0.2, 0.1]))
        .with_metadata({"id": "spectrum2"})
        .build()
    )
    spectrum_3 = (
        builder.with_mz(np.array([110, 140, 195.0]))
        .with_intensities(np.array([0.6, 0.2, 0.1]))
        .with_metadata({"id": "spectrum3"})
        .build()
    )
    spectrum_4 = (
        builder.with_mz(np.array([100, 150, 200.0]))
        .with_intensities(np.array([0.6, 0.1, 0.6]))
        .with_metadata({"id": "spectrum4"})
        .build()
    )

    return spectrum_1, spectrum_2, spectrum_3, spectrum_4


def test_scores_single_pair():
    """Test single pair input."""
    dummy_similarity_function = DummySimilarityFunction()
    scores = Scores(references=["A"], queries=["B"])
    scores.calculate(dummy_similarity_function)
    actual_1 = scores.scores[0, 0, "DummySimilarityFunction_score"]
    actual_2 = scores.scores[0, 0, "DummySimilarityFunction_len"]
    assert actual_1 == "AB", "Expected different scores."
    assert actual_2 == 2, "Expected different scores."


def test_scores_calculate():
    dummy_similarity_function = DummySimilarityFunction()
    scores = Scores(references=["r0", "r1", "r2"], queries=["q0", "q1"])
    scores.calculate(dummy_similarity_function)
    actual_data = list(scores)
    expected = [
        ("r0", "q0", ["r0q0", 4]),
        ("r0", "q1", ["r0q1", 4]),
        ("r1", "q0", ["r1q0", 4]),
        ("r1", "q1", ["r1q1", 4]),
        ("r2", "q0", ["r2q0", 4]),
        ("r2", "q1", ["r2q1", 4]),
    ]
    assert actual_data == expected, "Expected different scores."


def test_scores_calculate_parallel():
    dummy_similarity_function = DummySimilarityFunctionParallel()
    scores = Scores(references=["r0", "r1", "r2"], queries=["q0", "q1"])
    scores.calculate(dummy_similarity_function)
    actual = list(scores)
    expected = [
        ("r0", "q0", ["r0q0", 4]),
        ("r0", "q1", ["r0q1", 4]),
        ("r1", "q0", ["r1q0", 4]),
        ("r1", "q1", ["r1q1", 4]),
        ("r2", "q0", ["r2q0", 4]),
        ("r2", "q1", ["r2q1", 4]),
    ]
    assert actual == expected, "Expected different scores."


def test_scores_init_with_list():
    scores = Scores(references=["r0", "r1", "r2"], queries=["q0", "q1"])
    assert scores.shape == (3, 2, 0), "Expected different scores shape."


def test_scores_init_with_numpy_array():
    scores = Scores(references=np.asarray(["r0", "r1", "r2"]), queries=np.asarray(["q0", "q1"]))
    assert scores.shape == (3, 2, 0), "Expected different scores shape."


def test_scores_init_with_queries_dict():
    with pytest.raises(AssertionError) as msg:
        _ = Scores(references=["r0", "r1", "r2"], queries={"k0": "q0", "k1": "q1"})

    assert str(msg.value) == "Expected input argument 'queries' to be list or tuple or np.ndarray."


def test_scores_init_with_references_dict():
    with pytest.raises(AssertionError) as msg:
        _ = Scores(references={"k0": "r0", "k1": "r1", "k2": "r2"}, queries=["q0", "q1"])

    assert str(msg.value) == "Expected input argument 'references' to be list or tuple or np.ndarray."


def test_scores_init_with_tuple():
    scores = Scores(references=("r0", "r1", "r2"), queries=("q0", "q1"))
    assert scores.shape == (3, 2, 0), "Expected different scores shape."


def test_scores_next():
    dummy_similarity_function = DummySimilarityFunction()
    scores = Scores(references=["r", "rr", "rrr"], queries=["q", "qq"]).calculate(dummy_similarity_function)

    actual = list(scores)
    expected = [
        ("r", "q", ["rq", 2]),
        ("r", "qq", ["rqq", 3]),
        ("rr", "q", ["rrq", 3]),
        ("rr", "qq", ["rrqq", 4]),
        ("rrr", "q", ["rrrq", 4]),
        ("rrr", "qq", ["rrrqq", 5]),
    ]
    assert actual == expected, "Expected different scores."


def test_scores_to_dict():
    """Test if export to Python dictionary works as intended"""
    spectrum_1, spectrum_2, spectrum_3, _ = spectra()
    spectrum_1.set("precursor_mz", 123.4)
    references = [spectrum_1, spectrum_2]
    queries = [spectrum_3]
    scores = calculate_scores(references, queries, CosineGreedy())
    scores_dict = scores.to_dict()
    expected_dict = [
        {"id": "spectrum1", "precursor_mz": 123.4, "peaks_json": [[100.0, 0.7], [150.0, 0.2], [200.0, 0.1]]},
        {"id": "spectrum2", "peaks_json": [[100.0, 0.4], [140.0, 0.2], [190.0, 0.1]]},
    ]
    assert len(scores_dict["references"]) == 2
    assert scores_dict["references"] == expected_dict


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
    assert np.allclose(scores_only, scores_expected, atol=1e-8), "Expected different sorted scores."


def test_scores_by_referencey_non_tuple_score():
    "Test scores_by_reference method."
    spectrum_1, spectrum_2, spectrum_3, spectrum_4 = spectra()
    references = [spectrum_1, spectrum_2, spectrum_3]
    queries = [spectrum_3, spectrum_4]

    scores = calculate_scores(references, queries, IntersectMz())
    name_score = scores.score_names[0]
    selected_scores = scores.scores_by_reference(spectrum_2, name_score)

    expected_result = [(scores.queries[i], scores.scores[1, i]) for i in range(2)]
    assert selected_scores == expected_result, "Expected different scores."


def test_scores_by_references_exception():
    spectrum_1, spectrum_2, spectrum_3, spectrum_4 = spectra()
    references = [spectrum_1, spectrum_2, spectrum_3]
    queries = [spectrum_3, spectrum_4]

    builder = SpectrumBuilder()
    faulty_spectrum = (
        builder.with_mz(np.array([200, 350, 400.0]))
        .with_intensities(np.array([0.7, 0.2, 0.1]))
        .with_metadata({"id": "spectrum5"})
        .build()
    )

    scores = calculate_scores(references, queries, CosineGreedy())
    name_score = scores.score_names[0]

    with pytest.raises(ValueError, match="Given input not found in references."):
        scores.scores_by_reference(faulty_spectrum, name_score)


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
    spectrum_1 = (
        builder.with_mz(np.array([100, 150, 200.0]))
        .with_intensities(np.array([0.7, 0.2, 0.1]))
        .with_metadata({"id": "spectrum1"})
        .build()
    )
    spectrum_2 = (
        builder.with_mz(np.array([100, 140, 190.0]))
        .with_intensities(np.array([0.4, 0.2, 0.1]))
        .with_metadata({"id": "spectrum2"})
        .build()
    )
    spectrum_3 = (
        builder.with_mz(np.array([100, 140, 195.0]))
        .with_intensities(np.array([0.6, 0.2, 0.1]))
        .with_metadata({"id": "spectrum3"})
        .build()
    )
    spectrum_4 = (
        builder.with_mz(np.array([100, 150, 200.0]))
        .with_intensities(np.array([0.6, 0.1, 0.6]))
        .with_metadata({"id": "spectrum4"})
        .build()
    )

    references = [spectrum_1, spectrum_2, spectrum_3]
    queries = [spectrum_2, spectrum_3, spectrum_4]

    scores = calculate_scores(references, queries, CosineGreedy())
    name_score = scores.score_names[0]
    selected_scores = scores.scores_by_query(spectrum_4, name_score, sort=True)

    expected_result = [(scores.references[i], scores.scores[i, 2]) for i in [0, 2, 1]]
    assert selected_scores == expected_result, "Expected different scores."
    assert np.allclose(
        np.array([x[1] for x in selected_scores]).tolist(), [(0.79636414, 3), (0.65803523, 1), (0.61297133, 1)]
    )


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


def test_scores_by_query_exception():
    spectrum_1, spectrum_2, spectrum_3, spectrum_4 = spectra()
    references = [spectrum_1, spectrum_2, spectrum_3]
    queries = [spectrum_2, spectrum_3, spectrum_4]

    builder = SpectrumBuilder()
    faulty_spectrum = (
        builder.with_mz(np.array([200, 350, 400.0]))
        .with_intensities(np.array([0.7, 0.2, 0.1]))
        .with_metadata({"id": "spectrum5"})
        .build()
    )

    scores = calculate_scores(references, queries, CosineGreedy())
    name_score = scores.score_names[0]

    with pytest.raises(ValueError, match="Given input not found in queries."):
        scores.scores_by_query(faulty_spectrum, name_score)


@pytest.mark.parametrize(
    "similarity_function_a,similarity_function_b", [(CosineGreedy(), IntersectMz()), (IntersectMz(), CosineGreedy())]
)
def test_comparing_symmetric_scores(similarity_function_a, similarity_function_b):
    "Test comparing symmetric scores objects."
    spectrum_1, spectrum_2, spectrum_3, spectrum_4 = spectra()
    spectra_list = [spectrum_1, spectrum_2, spectrum_3, spectrum_4]

    scores_similarity_a = calculate_scores(spectra_list, spectra_list, similarity_function_a)
    scores_similarity_a_copy = calculate_scores(spectra_list, spectra_list, similarity_function_a)
    scores_similarity_b = calculate_scores(spectra_list, spectra_list, similarity_function_b)

    assert scores_similarity_a == scores_similarity_a_copy
    assert scores_similarity_a != scores_similarity_b


@pytest.mark.parametrize(
    "similarity_function_a,similarity_function_b", [(CosineGreedy(), IntersectMz()), (IntersectMz(), CosineGreedy())]
)
def test_comparing_asymmetric_scores(similarity_function_a, similarity_function_b):
    "Test comparing asymmetric scores objects."
    spectrum_1, spectrum_2, spectrum_3, spectrum_4 = spectra()
    spectra_list = [spectrum_1, spectrum_2, spectrum_3, spectrum_4]

    scores_similarity_a = calculate_scores(spectra_list[0:3], spectra_list, similarity_function_a)
    scores_similarity_a_copy = calculate_scores(spectra_list[0:3], spectra_list, similarity_function_a)
    scores_similarity_b = calculate_scores(spectra_list, spectra_list[0:3], similarity_function_b)

    assert scores_similarity_a == scores_similarity_a_copy
    assert scores_similarity_a != scores_similarity_b


@pytest.mark.parametrize(
    "similarity_function_a,similarity_function_b",
    [
        (CosineGreedy(tolerance=0.5, mz_power=0.5), CosineGreedy(tolerance=0.1, mz_power=0.1)),
        (IntersectMz(scaling=1.0), IntersectMz(scaling=2.0)),
    ],
)
def test_comparing_scores_with_same_shape_different_scores_values(similarity_function_a, similarity_function_b):
    "Test comparing scores objects with same similarity functions but different values of scores."
    spectrum_1, spectrum_2, spectrum_3, spectrum_4 = spectra()
    spectra_list = [spectrum_1, spectrum_2, spectrum_3, spectrum_4]

    scores_parametrized = calculate_scores(spectra_list, spectra_list, similarity_function_a)
    scores_parametrized_mirrored = calculate_scores(spectra_list, spectra_list, similarity_function_b)

    assert scores_parametrized != scores_parametrized_mirrored
