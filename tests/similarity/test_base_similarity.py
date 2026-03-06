import numpy as np
import pytest
from scipy.sparse import coo_array
from matchms.similarity.BaseSimilarity import BaseSimilarity
from tests.builder_Spectrum import SpectrumBuilder


class MockScalarSimilarity(BaseSimilarity):
    """Simple scalar similarity based on precursor_mz distance."""

    score_datatype = np.float64
    score_fields = ("score",)

    def pair(self, spectrum_1, spectrum_2):
        mz_1 = spectrum_1.get("precursor_mz")
        mz_2 = spectrum_2.get("precursor_mz")
        score = 1.0 if mz_1 == mz_2 else 0.5 if abs(mz_1 - mz_2) <= 5 else 0.0
        return np.asarray(score, dtype=self.score_datatype)


class MockStructuredSimilarity(BaseSimilarity):
    """Structured similarity returning (score, matches)."""

    score_datatype = np.dtype([("score", np.float64), ("matches", np.int64)])
    score_fields = ("score", "matches")

    def pair(self, spectrum_1, spectrum_2):
        mz_1 = spectrum_1.get("precursor_mz")
        mz_2 = spectrum_2.get("precursor_mz")

        if mz_1 == mz_2:
            score = 1.0
            matches = 3
        elif abs(mz_1 - mz_2) <= 5:
            score = 0.5
            matches = 2
        else:
            score = 0.0
            matches = 0

        return np.asarray((score, matches), dtype=self.score_datatype)


class MockNonCommutativeSimilarity(BaseSimilarity):
    """Non-commutative scalar similarity."""

    is_commutative = False
    score_datatype = np.float64
    score_fields = ("score",)

    def pair(self, spectrum_1, spectrum_2):
        mz_1 = spectrum_1.get("precursor_mz")
        mz_2 = spectrum_2.get("precursor_mz")
        return np.asarray(mz_1 - mz_2, dtype=self.score_datatype)


@pytest.fixture
def spectra():
    a = "CCC(C)C(C(=O)O)NC(=O)CCl"
    b = "C(C(=O)O)(NC(=O)O)S"

    builder = SpectrumBuilder()
    spectrum_1 = builder.with_mz(np.array([100, 150, 200.])) \
        .with_intensities(np.array([0.7, 0.2, 0.1])) \
        .with_metadata({'id': 'spectrum1', "precursor_mz": 210, "parent_mass": 210, "smiles": a}) \
        .build()
    spectrum_2 = builder.with_mz(np.array([100, 140, 190.])) \
        .with_intensities(np.array([0.4, 0.2, 0.1])) \
        .with_metadata({'id': 'spectrum2', "precursor_mz": 200, "parent_mass": 200, "smiles": a}) \
        .build()
    spectrum_3 = builder.with_mz(np.array([110, 140, 195.])) \
        .with_intensities(np.array([0.6, 0.2, 0.1])) \
        .with_metadata({'id': 'spectrum3', "precursor_mz": 205, "parent_mass": 205, "smiles": b}) \
        .build()
    spectrum_4 = builder.with_mz(np.array([100, 150, 200.])) \
        .with_intensities(np.array([0.6, 0.1, 0.6])) \
        .with_metadata({'id': 'spectrum4', "precursor_mz": 210, "parent_mass": 210, "smiles": b}) \
        .build()

    yield [spectrum_1, spectrum_2, spectrum_3, spectrum_4]


def test_matrix_scalar_self_comparison_returns_symmetric_dense_matrix(spectra):
    similarity = MockScalarSimilarity()

    scores = similarity.matrix(spectra, progress_bar=False)

    expected = np.array([
        [1.0, 0.0, 0.5, 1.0],
        [0.0, 1.0, 0.5, 0.0],
        [0.5, 0.5, 1.0, 0.5],
        [1.0, 0.0, 0.5, 1.0],
    ], dtype=np.float64)

    assert isinstance(scores, np.ndarray)
    assert scores.dtype == np.float64
    np.testing.assert_array_equal(scores, expected)


def test_matrix_scalar_two_input_sets_is_not_assumed_symmetric(spectra):
    similarity = MockScalarSimilarity()

    scores = similarity.matrix(spectra[:2], spectra[2:], progress_bar=False)

    expected = np.array([
        [0.5, 1.0],
        [0.5, 0.0],
    ], dtype=np.float64)

    np.testing.assert_array_equal(scores, expected)


def test_matrix_non_commutative_self_comparison_computes_full_matrix(spectra):
    similarity = MockNonCommutativeSimilarity()

    scores = similarity.matrix(spectra, progress_bar=False)

    expected = np.array([
        [0.0, 10.0, 5.0, 0.0],
        [-10.0, 0.0, -5.0, -10.0],
        [-5.0, 5.0, 0.0, -5.0],
        [0.0, 10.0, 5.0, 0.0],
    ], dtype=np.float64)

    np.testing.assert_array_equal(scores, expected)


def test_matrix_structured_all_fields_returns_dict_of_dense_arrays(spectra):
    similarity = MockStructuredSimilarity()

    scores = similarity.matrix(spectra, progress_bar=False)

    assert isinstance(scores, dict)
    assert set(scores.keys()) == {"score", "matches"}

    expected_score = np.array([
        [1.0, 0.0, 0.5, 1.0],
        [0.0, 1.0, 0.5, 0.0],
        [0.5, 0.5, 1.0, 0.5],
        [1.0, 0.0, 0.5, 1.0],
    ], dtype=np.float64)

    expected_matches = np.array([
        [3, 0, 2, 3],
        [0, 3, 2, 0],
        [2, 2, 3, 2],
        [3, 0, 2, 3],
    ], dtype=np.int64)

    np.testing.assert_array_equal(scores["score"], expected_score)
    np.testing.assert_array_equal(scores["matches"], expected_matches)


def test_matrix_structured_single_field_returns_dense_array(spectra):
    similarity = MockStructuredSimilarity()

    scores = similarity.matrix(spectra, score_fields=("score",), progress_bar=False)

    expected = np.array([
        [1.0, 0.0, 0.5, 1.0],
        [0.0, 1.0, 0.5, 0.0],
        [0.5, 0.5, 1.0, 0.5],
        [1.0, 0.0, 0.5, 1.0],
    ], dtype=np.float64)

    assert isinstance(scores, np.ndarray)
    np.testing.assert_array_equal(scores, expected)


def test_matrix_unknown_score_field_raises(spectra):
    similarity = MockStructuredSimilarity()

    with pytest.raises(ValueError, match="Unknown score field"):
        similarity.matrix(spectra, score_fields=("missing",), progress_bar=False)


def test_sparse_matrix_scalar_self_comparison_returns_coo_array(spectra):
    similarity = MockScalarSimilarity()

    scores = similarity.sparse_matrix(spectra, progress_bar=False)

    assert isinstance(scores, coo_array)

    dense = scores.toarray()
    expected = np.array([
        [1.0, 0.0, 0.5, 1.0],
        [0.0, 1.0, 0.5, 0.0],
        [0.5, 0.5, 1.0, 0.5],
        [1.0, 0.0, 0.5, 1.0],
    ], dtype=np.float64)

    np.testing.assert_array_equal(dense, expected)


def test_sparse_matrix_scalar_with_score_filter(spectra):
    similarity = MockScalarSimilarity()

    scores = similarity.sparse_matrix(
        spectra,
        score_filter=lambda s: s >= 1.0,
        progress_bar=False,
    )

    expected = np.array([
        [1.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 1.0],
    ], dtype=np.float64)

    np.testing.assert_array_equal(scores.toarray(), expected)


def test_sparse_matrix_structured_all_fields_returns_dict_of_coo_arrays(spectra):
    similarity = MockStructuredSimilarity()

    scores = similarity.sparse_matrix(spectra, progress_bar=False)

    assert isinstance(scores, dict)
    assert set(scores.keys()) == {"score", "matches"}
    assert isinstance(scores["score"], coo_array)
    assert isinstance(scores["matches"], coo_array)

    expected_score = np.array([
        [1.0, 0.0, 0.5, 1.0],
        [0.0, 1.0, 0.5, 0.0],
        [0.5, 0.5, 1.0, 0.5],
        [1.0, 0.0, 0.5, 1.0],
    ], dtype=np.float64)

    expected_matches = np.array([
        [3, 0, 2, 3],
        [0, 3, 2, 0],
        [2, 2, 3, 2],
        [3, 0, 2, 3],
    ], dtype=np.int64)

    np.testing.assert_array_equal(scores["score"].toarray(), expected_score)
    np.testing.assert_array_equal(scores["matches"].toarray(), expected_matches)


def test_sparse_matrix_structured_single_field_returns_single_coo_array(spectra):
    similarity = MockStructuredSimilarity()

    scores = similarity.sparse_matrix(
        spectra,
        score_fields=("matches",),
        progress_bar=False,
    )

    assert isinstance(scores, coo_array)

    expected = np.array([
        [3, 0, 2, 3],
        [0, 3, 2, 0],
        [2, 2, 3, 2],
        [3, 0, 2, 3],
    ], dtype=np.int64)

    np.testing.assert_array_equal(scores.toarray(), expected)


def test_sparse_matrix_structured_filter_on_multiple_fields_return_only_score(spectra):
    similarity = MockStructuredSimilarity()

    scores = similarity.sparse_matrix(
        spectra,
        score_fields=("score",),
        score_filter=lambda s: s["score"] > 0.2 and s["matches"] >= 2,
        progress_bar=False,
    )

    expected = np.array([
        [1.0, 0.0, 0.5, 1.0],
        [0.0, 1.0, 0.5, 0.0],
        [0.5, 0.5, 1.0, 0.5],
        [1.0, 0.0, 0.5, 1.0],
    ], dtype=np.float64)

    np.testing.assert_array_equal(scores.toarray(), expected)


def test_sparse_matrix_structured_filter_only_matches_greater_than_two(spectra):
    similarity = MockStructuredSimilarity()

    scores = similarity.sparse_matrix(
        spectra,
        score_fields=("matches",),
        score_filter=lambda s: s["matches"] > 2,
        progress_bar=False,
    )

    expected = np.array([
        [3, 0, 0, 3],
        [0, 3, 0, 0],
        [0, 0, 3, 0],
        [3, 0, 0, 3],
    ], dtype=np.int64)

    np.testing.assert_array_equal(scores.toarray(), expected)


def test_sparse_matrix_scalar_with_score_filter_range(spectra):
    similarity = MockScalarSimilarity()

    scores = similarity.sparse_matrix(
        spectra,
        score_filter=lambda s: 0.4 <= s <= 0.6,
        progress_bar=False,
    )

    expected = np.array([
        [0.0, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5, 0.0],
        [0.5, 0.5, 0.0, 0.5],
        [0.0, 0.0, 0.5, 0.0],
    ], dtype=np.float64)

    np.testing.assert_array_equal(scores.toarray(), expected)


def test_sparse_matrix_structured_with_score_filter_range_on_score_field(spectra):
    similarity = MockStructuredSimilarity()

    scores = similarity.sparse_matrix(
        spectra,
        score_fields=("score",),
        score_filter=lambda s: 0.4 <= s["score"] <= 0.6,
        progress_bar=False,
    )

    expected = np.array([
        [0.0, 0.0, 0.5, 0.0],
        [0.0, 0.0, 0.5, 0.0],
        [0.5, 0.5, 0.0, 0.5],
        [0.0, 0.0, 0.5, 0.0],
    ], dtype=np.float64)

    np.testing.assert_array_equal(scores.toarray(), expected)


def test_sparse_matrix_with_explicit_indices_scalar(spectra):
    similarity = MockScalarSimilarity()

    idx_row = np.array([0, 1, 2])
    idx_col = np.array([2, 2, 3])

    scores = similarity.sparse_matrix(
        spectra,
        idx_row=idx_row,
        idx_col=idx_col,
        progress_bar=False,
    )

    expected = np.zeros((4, 4), dtype=np.float64)
    expected[0, 2] = 0.5
    expected[2, 0] = 0.5
    expected[1, 2] = 0.5
    expected[2, 1] = 0.5
    expected[2, 3] = 0.5
    expected[3, 2] = 0.5

    np.testing.assert_array_equal(scores.toarray(), expected)


def test_sparse_matrix_with_explicit_indices_two_input_sets(spectra):
    similarity = MockScalarSimilarity()

    idx_row = np.array([0, 1])
    idx_col = np.array([0, 1])

    scores = similarity.sparse_matrix(
        spectra[:2],
        spectra[2:],
        idx_row=idx_row,
        idx_col=idx_col,
        progress_bar=False,
    )

    expected = np.zeros((2, 2), dtype=np.float64)
    expected[0, 0] = 0.5
    expected[1, 1] = 0.0

    np.testing.assert_array_equal(scores.toarray(), expected)


def test_sparse_matrix_requires_both_idx_row_and_idx_col(spectra):
    similarity = MockScalarSimilarity()

    with pytest.raises(ValueError, match="idx_row and idx_col must either both be given or both be None"):
        similarity.sparse_matrix(
            spectra,
            idx_row=np.array([0, 1]),
            idx_col=None,
            progress_bar=False,
        )


def test_sparse_matrix_requires_idx_row_idx_col_same_shape(spectra):
    similarity = MockScalarSimilarity()

    with pytest.raises(ValueError, match="idx_row and idx_col must have the same shape"):
        similarity.sparse_matrix(
            spectra,
            idx_row=np.array([0, 1]),
            idx_col=np.array([0]),
            progress_bar=False,
        )


def test_empty_score_fields_raises(spectra):
    similarity = MockStructuredSimilarity()

    with pytest.raises(ValueError, match="score_fields must contain at least one field"):
        similarity.matrix(spectra, score_fields=(), progress_bar=False)


def test_invalid_scalar_score_fields_definition_raises(spectra):
    class InvalidScalarSimilarity(BaseSimilarity):
        score_datatype = np.float64
        score_fields = ("score", "matches")

        def pair(self, spectrum_1, spectrum_2):
            return np.asarray(1.0, dtype=self.score_datatype)

    similarity = InvalidScalarSimilarity()

    with pytest.raises(ValueError, match="Scalar scores must define score_fields"):
        similarity.matrix(spectra, progress_bar=False)


def test_mismatching_structured_score_fields_definition_raises(spectra):
    class InvalidStructuredSimilarity(BaseSimilarity):
        score_datatype = np.dtype([("score", np.float64), ("matches", np.int64)])
        score_fields = ("score",)

        def pair(self, spectrum_1, spectrum_2):
            return np.asarray((1.0, 1), dtype=self.score_datatype)

    similarity = InvalidStructuredSimilarity()

    with pytest.raises(ValueError, match="score_fields does not match the field names in score_datatype"):
        similarity.matrix(spectra, progress_bar=False)