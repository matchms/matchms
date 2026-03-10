import json
from pathlib import Path
import numpy as np
import pytest
from scipy.sparse import coo_array
from matchms.Scores import Scores


@pytest.fixture(params=[True, False])
def compressed(request):
    yield request.param


@pytest.fixture
def dense_scalar_scores():
    return Scores({
        "score": np.array([
            [1.0, 0.0, 0.5],
            [0.0, 0.8, 0.0],
        ], dtype=np.float64)
    })


@pytest.fixture
def dense_multi_scores():
    return Scores({
        "score": np.array([
            [1.0, 0.0, 0.5],
            [0.0, 0.8, 0.0],
        ], dtype=np.float64),
        "matches": np.array([
            [3, 0, 2],
            [0, 4, 0],
        ], dtype=np.int64),
    })


@pytest.fixture
def sparse_scalar_scores():
    dense = np.array([
        [1.0, 0.0, 0.5],
        [0.0, 0.8, 0.0],
    ], dtype=np.float64)
    row, col = np.nonzero(dense)
    return Scores({
        "score": coo_array((dense[row, col], (row, col)), shape=dense.shape)
    })


@pytest.fixture
def sparse_multi_scores():
    score = np.array([
        [1.0, 0.0, 0.5],
        [0.0, 0.8, 0.0],
    ], dtype=np.float64)
    matches = np.array([
        [3, 0, 2],
        [0, 4, 0],
    ], dtype=np.int64)

    row_s, col_s = np.nonzero(score)
    row_m, col_m = np.nonzero(matches)

    return Scores({
        "score": coo_array((score[row_s, col_s], (row_s, col_s)), shape=score.shape),
        "matches": coo_array((matches[row_m, col_m], (row_m, col_m)), shape=matches.shape),
    })


def assert_scores_equal(scores_a: Scores, scores_b: Scores):
    assert isinstance(scores_a, Scores)
    assert isinstance(scores_b, Scores)
    assert scores_a.shape == scores_b.shape
    assert scores_a.score_fields == scores_b.score_fields
    assert scores_a.is_sparse == scores_b.is_sparse
    assert scores_a.is_scalar == scores_b.is_scalar

    for field in scores_a.score_fields:
        np.testing.assert_array_equal(
            scores_a.to_array(field),
            scores_b.to_array(field),
        )


def test_save_creates_file_dense_scalar(tmp_path: Path, dense_scalar_scores: Scores):
    filename = tmp_path / "scores_dense_scalar.npz"
    dense_scalar_scores.save(filename)
    assert filename.is_file()


def test_save_creates_file_sparse_scalar(tmp_path: Path, sparse_scalar_scores: Scores):
    filename = tmp_path / "scores_sparse_scalar.npz"
    sparse_scalar_scores.save(filename)
    assert filename.is_file()


def test_dense_scalar_scores_roundtrip(tmp_path: Path, dense_scalar_scores: Scores, compressed: bool):
    filename = tmp_path / "dense_scalar_scores.npz"

    dense_scalar_scores.save(filename, compressed=compressed)
    loaded = Scores.load(filename)

    assert_scores_equal(dense_scalar_scores, loaded)


def test_dense_multi_scores_roundtrip(tmp_path: Path, dense_multi_scores: Scores, compressed: bool):
    filename = tmp_path / "dense_multi_scores.npz"

    dense_multi_scores.save(filename, compressed=compressed)
    loaded = Scores.load(filename)

    assert_scores_equal(dense_multi_scores, loaded)


def test_sparse_scalar_scores_roundtrip(tmp_path: Path, sparse_scalar_scores: Scores, compressed: bool):
    filename = tmp_path / "sparse_scalar_scores.npz"

    sparse_scalar_scores.save(filename, compressed=compressed)
    loaded = Scores.load(filename)

    assert_scores_equal(sparse_scalar_scores, loaded)


def test_sparse_multi_scores_roundtrip(tmp_path: Path, sparse_multi_scores: Scores, compressed: bool):
    filename = tmp_path / "sparse_multi_scores.npz"

    sparse_multi_scores.save(filename, compressed=compressed)
    loaded = Scores.load(filename)

    assert_scores_equal(sparse_multi_scores, loaded)


def test_load_preserves_dense_kind(tmp_path: Path, dense_multi_scores: Scores):
    filename = tmp_path / "dense_scores.npz"
    dense_multi_scores.save(filename)

    loaded = Scores.load(filename)

    assert loaded.is_sparse is False
    assert set(loaded.score_fields) == {"score", "matches"}


def test_load_preserves_sparse_kind(tmp_path: Path, sparse_multi_scores: Scores):
    filename = tmp_path / "sparse_scores.npz"
    sparse_multi_scores.save(filename)

    loaded = Scores.load(filename)

    assert loaded.is_sparse is True
    assert set(loaded.score_fields) == {"score", "matches"}


def test_load_missing_metadata_raises(tmp_path: Path):
    filename = tmp_path / "missing_metadata.npz"
    np.savez(filename, score=np.array([[1.0, 0.0]]))

    with pytest.raises(ValueError, match="does not contain matchms.Scores metadata"):
        Scores.load(filename)


def test_load_wrong_format_raises(tmp_path: Path):
    filename = tmp_path / "wrong_format.npz"
    metadata = {
        "format": "not_matchms.Scores",
        "version": 1,
        "is_sparse": False,
        "score_fields": ["score"],
        "shape": [1, 2],
    }
    np.savez(filename, __scores_metadata__=np.array(json.dumps(metadata)), score=np.array([[1.0, 0.0]]))

    with pytest.raises(ValueError, match="is not a matchms.Scores file"):
        Scores.load(filename)


def test_load_unsupported_version_raises(tmp_path: Path):
    filename = tmp_path / "wrong_version.npz"
    metadata = {
        "format": "matchms.Scores",
        "version": 999,
        "is_sparse": False,
        "score_fields": ["score"],
        "shape": [1, 2],
    }
    np.savez(filename, __scores_metadata__=np.array(json.dumps(metadata)), score=np.array([[1.0, 0.0]]))

    with pytest.raises(ValueError, match="Unsupported matchms.Scores version"):
        Scores.load(filename)


def test_load_missing_dense_field_raises(tmp_path: Path):
    filename = tmp_path / "missing_dense_field.npz"
    metadata = {
        "format": "matchms.Scores",
        "version": 1,
        "is_sparse": False,
        "score_fields": ["score", "matches"],
        "shape": [2, 2],
    }
    np.savez(
        filename,
        __scores_metadata__=np.array(json.dumps(metadata)),
        score=np.array([[1.0, 0.0], [0.0, 0.8]]),
    )

    with pytest.raises(ValueError, match="missing dense data for field 'matches'"):
        Scores.load(filename)


def test_load_missing_sparse_row_raises(tmp_path: Path):
    filename = tmp_path / "missing_sparse_row.npz"
    metadata = {
        "format": "matchms.Scores",
        "version": 1,
        "is_sparse": True,
        "score_fields": ["score"],
        "shape": [2, 2],
    }
    np.savez(
        filename,
        __scores_metadata__=np.array(json.dumps(metadata)),
        score__col=np.array([0, 1]),
        score__data=np.array([1.0, 0.8]),
    )

    with pytest.raises(ValueError, match="missing sparse data for field 'score'"):
        Scores.load(filename)


def test_load_missing_sparse_col_raises(tmp_path: Path):
    filename = tmp_path / "missing_sparse_col.npz"
    metadata = {
        "format": "matchms.Scores",
        "version": 1,
        "is_sparse": True,
        "score_fields": ["score"],
        "shape": [2, 2],
    }
    np.savez(
        filename,
        __scores_metadata__=np.array(json.dumps(metadata)),
        score__row=np.array([0, 1]),
        score__data=np.array([1.0, 0.8]),
    )

    with pytest.raises(ValueError, match="missing sparse data for field 'score'"):
        Scores.load(filename)


def test_load_missing_sparse_data_raises(tmp_path: Path):
    filename = tmp_path / "missing_sparse_data.npz"
    metadata = {
        "format": "matchms.Scores",
        "version": 1,
        "is_sparse": True,
        "score_fields": ["score"],
        "shape": [2, 2],
    }
    np.savez(
        filename,
        __scores_metadata__=np.array(json.dumps(metadata)),
        score__row=np.array([0, 1]),
        score__col=np.array([0, 1]),
    )

    with pytest.raises(ValueError, match="missing sparse data for field 'score'"):
        Scores.load(filename)


def test_load_missing_required_metadata_keys_raises(tmp_path: Path):
    filename = tmp_path / "missing_metadata_keys.npz"
    metadata = {
        "format": "matchms.Scores",
        "version": 1,
        "is_sparse": False,
        # missing score_fields and shape
    }
    np.savez(filename, __scores_metadata__=np.array(json.dumps(metadata)))

    with pytest.raises(ValueError, match="missing metadata keys"):
        Scores.load(filename)


def test_roundtrip_dense_preserves_values_and_dtype(tmp_path: Path):
    scores = Scores({
        "score": np.array([[1.0, 0.0], [0.25, 0.75]], dtype=np.float32),
        "matches": np.array([[5, 0], [1, 4]], dtype=np.int16),
    })

    filename = tmp_path / "dtype_roundtrip_dense.npz"
    scores.save(filename)
    loaded = Scores.load(filename)

    np.testing.assert_array_equal(loaded.to_array("score"), scores.to_array("score"))
    np.testing.assert_array_equal(loaded.to_array("matches"), scores.to_array("matches"))
    assert loaded.to_array("score").dtype == np.float32
    assert loaded.to_array("matches").dtype == np.int16


def test_roundtrip_sparse_preserves_values_and_dtype(tmp_path: Path):
    row = np.array([0, 1], dtype=np.int32)
    col = np.array([0, 1], dtype=np.int32)
    data = np.array([1.0, 0.75], dtype=np.float32)

    scores = Scores({
        "score": coo_array((data, (row, col)), shape=(2, 2)),
    })

    filename = tmp_path / "dtype_roundtrip_sparse.npz"
    scores.save(filename)
    loaded = Scores.load(filename)

    np.testing.assert_array_equal(loaded.to_array(), scores.to_array())
    assert loaded.to_coo().data.dtype == np.float32