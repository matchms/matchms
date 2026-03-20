import numpy as np
import pytest
from scipy.sparse import coo_array
from matchms.Scores import Scores
from matchms.similarity.MetadataMatch import MetadataMatch
from tests.builder_Spectrum import SpectrumBuilder


@pytest.fixture
def spectra():
    metadata1 = {
        "instrument_type": "orbitrap",
        "retention_time": 100.0,
    }
    metadata2 = {
        "instrument_type": "qtof",
        "retention_time": 100.5,
    }
    metadata3 = {
        "instrument_type": "orbitrap",
        "retention_time": 105.0,
    }
    metadata4 = {
        "retention_time": 99.1,
    }

    s1 = SpectrumBuilder().with_metadata(metadata1).build()
    s2 = SpectrumBuilder().with_metadata(metadata2).build()
    s3 = SpectrumBuilder().with_metadata(metadata3).build()
    s4 = SpectrumBuilder().with_metadata(metadata4).build()
    return [s1, s2, s3, s4]


def test_metadata_match_strings_pair(spectra):
    """Test pairwise metadata matching for string entries."""
    similarity = MetadataMatch(field="instrument_type")

    assert similarity.pair(spectra[0], spectra[1]) == np.array(False, dtype=bool)
    assert similarity.pair(spectra[0], spectra[3]) == np.array(False, dtype=bool)
    assert similarity.pair(spectra[0], spectra[2]) == np.array(True, dtype=bool)


def test_metadata_match_strings_matrix(spectra):
    """Test dense metadata matching for string entries."""
    spectra_1 = spectra[:2]
    spectra_2 = spectra[2:]

    similarity = MetadataMatch(field="instrument_type")
    scores = similarity.matrix(spectra_1, spectra_2)

    expected = np.array([[True, False], [False, False]], dtype=bool)

    assert isinstance(scores, Scores)
    assert scores.is_sparse is False
    assert scores.is_scalar is True
    assert scores.score_fields == ("score",)
    np.testing.assert_array_equal(scores.to_array(), expected)


def test_metadata_match_strings_sparse_matrix(spectra):
    """Test sparse metadata matching for string entries."""
    spectra_1 = spectra[:2]
    spectra_2 = spectra[2:]

    similarity = MetadataMatch(field="instrument_type")
    scores = similarity.sparse_matrix(spectra_1, spectra_2)

    expected = np.array([[True, False], [False, False]], dtype=bool)

    assert isinstance(scores, Scores)
    assert scores.is_sparse is True
    assert scores.is_scalar is True
    assert isinstance(scores.to_coo(), coo_array)
    np.testing.assert_array_equal(scores.to_array(), expected)


def test_metadata_match_strings_self_comparison_is_symmetric(spectra):
    """Implicit self-comparison should equal explicit self-comparison."""
    similarity = MetadataMatch(field="instrument_type")

    scores_self = similarity.matrix(spectra)
    scores_explicit = similarity.matrix(spectra, spectra)

    np.testing.assert_array_equal(scores_self.to_array(), scores_explicit.to_array())


def test_metadata_match_strings_wrong_method_pair_logs_warning(spectra, caplog):
    """Difference matching on strings should return False and log a warning."""
    similarity = MetadataMatch(field="instrument_type", matching_type="difference")

    score = similarity.pair(spectra[0], spectra[2])

    assert score == np.array(False, dtype=bool)
    assert "not compatible with 'difference' method" in caplog.text


def test_metadata_match_strings_wrong_method_matrix_logs_warning(spectra, caplog):
    """Difference matching on strings should yield all False in matrix output."""
    spectra_1 = spectra[:2]
    spectra_2 = spectra[2:]

    similarity = MetadataMatch(field="instrument_type", matching_type="difference")
    scores = similarity.matrix(spectra_1, spectra_2)

    expected = np.array([[False, False], [False, False]], dtype=bool)

    np.testing.assert_array_equal(scores.to_array(), expected)
    assert "not compatible with 'difference' method" in caplog.text


def test_metadata_match_numerical_pair(spectra):
    """Test pairwise metadata matching for numerical entries."""
    similarity = MetadataMatch(
        field="retention_time",
        matching_type="difference",
        tolerance=0.6,
    )

    score = similarity.pair(spectra[0], spectra[1])
    assert score == np.array(True, dtype=bool)


@pytest.mark.parametrize(
    "tolerance, expected",
    [
        (1.0, [[False, True], [False, False]]),
        (2.0, [[False, True], [False, True]]),
        (10.0, [[True, True], [True, True]]),
        (0.1, [[False, False], [False, False]]),
    ],
)
def test_metadata_match_numerical_matrix(spectra, tolerance, expected):
    """Test dense metadata matching for numerical entries."""
    spectra_1 = spectra[:2]
    spectra_2 = spectra[2:]

    similarity = MetadataMatch(
        field="retention_time",
        matching_type="difference",
        tolerance=tolerance,
    )
    scores = similarity.matrix(spectra_1, spectra_2)

    expected = np.array(expected, dtype=bool)

    assert isinstance(scores, Scores)
    assert scores.is_sparse is False
    np.testing.assert_array_equal(scores.to_array(), expected)


@pytest.mark.parametrize(
    "tolerance, expected",
    [
        (1.0, [[False, True], [False, False]]),
        (2.0, [[False, True], [False, True]]),
        (10.0, [[True, True], [True, True]]),
        (0.1, [[False, False], [False, False]]),
    ],
)
def test_metadata_match_numerical_sparse_matrix(spectra, tolerance, expected):
    """Test sparse metadata matching for numerical entries."""
    spectra_1 = spectra[:2]
    spectra_2 = spectra[2:]

    similarity = MetadataMatch(
        field="retention_time",
        matching_type="difference",
        tolerance=tolerance,
    )
    scores = similarity.sparse_matrix(spectra_1, spectra_2)

    expected = np.array(expected, dtype=bool)

    assert isinstance(scores, Scores)
    assert scores.is_sparse is True
    np.testing.assert_array_equal(scores.to_array(), expected)


def test_metadata_match_equal_match_ignores_tolerance_and_logs_warning(spectra, caplog):
    """equal_match should ignore tolerance and log a warning."""
    spectra_1 = spectra[:2]
    spectra_2 = spectra[2:]

    similarity = MetadataMatch(
        field="instrument_type",
        matching_type="equal_match",
        tolerance=5.0,
    )
    scores = similarity.matrix(spectra_1, spectra_2)

    expected = np.array([[True, False], [False, False]], dtype=bool)

    np.testing.assert_array_equal(scores.to_array(), expected)
    assert "Tolerance is set but will be ignored" in caplog.text


def test_metadata_match_equal_match_ignores_tolerance_type_and_logs_warning(spectra, caplog):
    """equal_match should ignore tolerance_type and log a warning."""
    spectra_1 = spectra[:2]
    spectra_2 = spectra[2:]

    similarity = MetadataMatch(
        field="instrument_type",
        matching_type="equal_match",
        tolerance_type="ppm",
    )
    scores = similarity.matrix(spectra_1, spectra_2)

    expected = np.array([[True, False], [False, False]], dtype=bool)

    np.testing.assert_array_equal(scores.to_array(), expected)
    assert "tolerance_type is set but will be ignored" in caplog.text


def test_metadata_match_missing_entries_return_false_in_pair(spectra):
    """Missing metadata entries should return False in pairwise comparison."""
    similarity = MetadataMatch(field="instrument_type")

    score = similarity.pair(spectra[0], spectra[3])
    assert score == np.array(False, dtype=bool)


def test_metadata_match_missing_entries_log_warning_in_matrix(spectra, caplog):
    """Missing metadata entries should be excluded in optimized matrix matching."""
    spectra_1 = spectra[:2]
    spectra_2 = spectra[2:]

    similarity = MetadataMatch(field="instrument_type")
    scores = similarity.matrix(spectra_1, spectra_2)

    expected = np.array([[True, False], [False, False]], dtype=bool)

    np.testing.assert_array_equal(scores.to_array(), expected)
    assert "No instrument_type entry found for spectrum." in caplog.text


def test_metadata_match_sparse_matrix_with_score_filter(spectra):
    """Sparse matrix should honor score_filter."""
    spectra_1 = spectra[:2]
    spectra_2 = spectra[2:]

    similarity = MetadataMatch(field="instrument_type")
    scores = similarity.sparse_matrix(
        spectra_1,
        spectra_2,
        score_filter=lambda s: bool(s),
    )

    expected = np.array([[True, False], [False, False]], dtype=bool)
    np.testing.assert_array_equal(scores.to_array(), expected)

    scores_filtered_out = similarity.sparse_matrix(
        spectra_1,
        spectra_2,
        score_filter=lambda s: False,
    )

    expected_empty = np.array([[False, False], [False, False]], dtype=bool)
    np.testing.assert_array_equal(scores_filtered_out.to_array(), expected_empty)


def test_metadata_match_sparse_matrix_explicit_indices_fallback(spectra):
    """Explicit idx_row/idx_col should use the generic sparse fallback path."""
    similarity = MetadataMatch(field="instrument_type")

    idx_row = np.array([0, 0, 1])
    idx_col = np.array([0, 1, 1])

    scores = similarity.sparse_matrix(
        spectra[:2],
        spectra[2:],
        idx_row=idx_row,
        idx_col=idx_col,
        progress_bar=False,
    )

    expected = np.array([[True, False], [False, False]], dtype=bool)

    assert isinstance(scores, Scores)
    assert scores.is_sparse is True
    np.testing.assert_array_equal(scores.to_array(), expected)


def test_metadata_match_sparse_matrix_requires_matching_idx_shapes(spectra):
    """Explicit sparse indices must have identical shape."""
    similarity = MetadataMatch(field="instrument_type")

    with pytest.raises(ValueError, match="idx_row and idx_col must have the same shape"):
        similarity.sparse_matrix(
            spectra[:2],
            spectra[2:],
            idx_row=np.array([0, 1]),
            idx_col=np.array([0]),
            progress_bar=False,
        )


def test_metadata_match_unknown_score_field_raises(spectra):
    """Only the score field is supported."""
    similarity = MetadataMatch(field="instrument_type")

    with pytest.raises(ValueError, match="Unknown score field"):
        similarity.matrix(spectra[:2], spectra[2:], score_fields=("missing",))


def test_metadata_match_non_score_field_selection_not_supported(spectra):
    """MetadataMatch only supports the score field."""
    similarity = MetadataMatch(field="instrument_type")

    with pytest.raises(ValueError, match="Unknown score field"):
        similarity.matrix(spectra[:2], spectra[2:], score_fields=("matches",))
