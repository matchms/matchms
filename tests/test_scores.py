import numpy as np
import pytest
from scipy.sparse import coo_array
from matchms.Scores import Scores, ScoresField, ScoresMask


@pytest.fixture
def dense_scalar_scores():
    data = {
        "score": np.array([
            [1.0, 0.0, 0.5],
            [0.0, 0.8, 0.0],
        ], dtype=np.float64)
    }
    return Scores(data)


@pytest.fixture
def sparse_scalar_scores():
    dense = np.array([
        [1.0, 0.0, 0.5],
        [0.0, 0.8, 0.0],
    ], dtype=np.float64)
    row, col = np.nonzero(dense)
    coo = coo_array((dense[row, col], (row, col)), shape=dense.shape)
    return Scores({"score": coo})


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


def test_scores_requires_at_least_one_field():
    with pytest.raises(ValueError, match="at least one score field"):
        Scores({})


def test_scores_requires_same_dense_sparse_kind():
    dense = np.array([[1.0, 0.0]])
    sparse = coo_array(([1.0], ([0], [0])), shape=(1, 2))

    with pytest.raises(ValueError, match="either dense or sparse"):
        Scores({"score": dense, "matches": sparse})


def test_scores_requires_same_shape():
    with pytest.raises(ValueError, match="same shape"):
        Scores({
            "score": np.zeros((2, 3)),
            "matches": np.zeros((3, 2)),
        })


def test_scores_basic_properties_dense_scalar(dense_scalar_scores):
    scores = dense_scalar_scores
    assert scores.shape == (2, 3)
    assert scores.score_fields == ("score",)
    assert scores.is_scalar is True
    assert scores.is_sparse is False


def test_scores_basic_properties_sparse_multi(sparse_multi_scores):
    scores = sparse_multi_scores
    assert scores.shape == (2, 3)
    assert scores.score_fields == ("score", "matches")
    assert scores.is_scalar is False
    assert scores.is_sparse is True


def test_scores_repr_contains_shape_and_fields(dense_multi_scores):
    text = repr(dense_multi_scores)
    assert "Scores" in text
    assert "shape=(2, 3)" in text
    assert "score_fields=('score', 'matches')" in text


def test_scores_get_field_returns_scoresfield(dense_multi_scores):
    field = dense_multi_scores["score"]
    assert isinstance(field, ScoresField)
    assert field.shape == (2, 3)
    assert field.is_sparse is False


def test_scores_unknown_field_raises(dense_multi_scores):
    with pytest.raises(KeyError, match="Unknown field"):
        dense_multi_scores.to_array("missing")


def test_scores_to_array_scalar_dense(dense_scalar_scores):
    arr = dense_scalar_scores.to_array()
    expected = np.array([
        [1.0, 0.0, 0.5],
        [0.0, 0.8, 0.0],
    ])
    np.testing.assert_array_equal(arr, expected)


def test_scores_to_array_scalar_sparse(sparse_scalar_scores):
    arr = sparse_scalar_scores.to_array()
    expected = np.array([
        [1.0, 0.0, 0.5],
        [0.0, 0.8, 0.0],
    ])
    np.testing.assert_array_equal(arr, expected)


def test_scores_to_array_multi_requires_field(dense_multi_scores):
    with pytest.raises(KeyError, match="Field name required"):
        dense_multi_scores.to_array()


def test_scores_to_coo_scalar_dense(dense_scalar_scores):
    coo = dense_scalar_scores.to_coo()
    assert isinstance(coo, coo_array)
    np.testing.assert_array_equal(
        coo.toarray(),
        np.array([
            [1.0, 0.0, 0.5],
            [0.0, 0.8, 0.0],
        ])
    )


def test_scores_to_coo_multi_specific_field(dense_multi_scores):
    coo = dense_multi_scores.to_coo("matches")
    assert isinstance(coo, coo_array)
    np.testing.assert_array_equal(
        coo.toarray(),
        np.array([
            [3, 0, 2],
            [0, 4, 0],
        ])
    )


def test_scoresfield_to_array_dense(dense_multi_scores):
    field = dense_multi_scores["score"]
    np.testing.assert_array_equal(
        field.to_array(),
        np.array([
            [1.0, 0.0, 0.5],
            [0.0, 0.8, 0.0],
        ])
    )


def test_scoresfield_to_coo_sparse(sparse_multi_scores):
    field = sparse_multi_scores["matches"]
    coo = field.to_coo()
    assert isinstance(coo, coo_array)
    np.testing.assert_array_equal(
        coo.toarray(),
        np.array([
            [3, 0, 2],
            [0, 4, 0],
        ])
    )


def test_scoresfield_getitem_dense(dense_scalar_scores):
    field = dense_scalar_scores["score"]
    assert field[0, 0] == 1.0
    np.testing.assert_array_equal(field[0, :], np.array([1.0, 0.0, 0.5]))


def test_scores_scalar_getitem_returns_values(dense_scalar_scores):
    scores = dense_scalar_scores
    assert scores[0, 0] == 1.0
    np.testing.assert_array_equal(scores[0, :], np.array([1.0, 0.0, 0.5]))


def test_scores_multi_getitem_tuple_returns_dict(dense_multi_scores):
    item = dense_multi_scores[0, 2]
    assert item == {"score": 0.5, "matches": 2}


def test_scores_mask_dense_construction():
    mask = ScoresMask(
        shape=(2, 3),
        dense_mask=np.array([
            [True, False, True],
            [False, True, False],
        ])
    )
    assert mask.is_sparse is False
    np.testing.assert_array_equal(
        mask.to_dense(),
        np.array([
            [True, False, True],
            [False, True, False],
        ])
    )


def test_scores_mask_sparse_construction():
    mask = ScoresMask(
        shape=(2, 3),
        row=np.array([0, 0, 1]),
        col=np.array([0, 2, 1]),
    )
    assert mask.is_sparse is True
    np.testing.assert_array_equal(
        mask.to_dense(),
        np.array([
            [True, False, True],
            [False, True, False],
        ])
    )


def test_scores_mask_invalid_dense_and_sparse_together():
    with pytest.raises(ValueError, match="either dense or sparse"):
        ScoresMask(
            shape=(2, 2),
            dense_mask=np.ones((2, 2), dtype=bool),
            row=np.array([0]),
            col=np.array([0]),
        )


def test_scores_mask_invalid_sparse_missing_col():
    with pytest.raises(ValueError, match="both row and col"):
        ScoresMask(shape=(2, 2), row=np.array([0]))


def test_scores_mask_invalid_sparse_shape_mismatch():
    with pytest.raises(ValueError, match="same shape"):
        ScoresMask(shape=(2, 2), row=np.array([0, 1]), col=np.array([0]))


def test_scores_mask_and_sparse():
    mask1 = ScoresMask(shape=(2, 3), row=np.array([0, 0, 1]), col=np.array([0, 2, 1]))
    mask2 = ScoresMask(shape=(2, 3), row=np.array([0, 1]), col=np.array([2, 1]))

    combined = mask1 & mask2
    assert combined.is_sparse is True
    np.testing.assert_array_equal(combined.row, np.array([0, 1]))
    np.testing.assert_array_equal(combined.col, np.array([2, 1]))


def test_scores_mask_or_sparse():
    mask1 = ScoresMask(shape=(2, 3), row=np.array([0]), col=np.array([0]))
    mask2 = ScoresMask(shape=(2, 3), row=np.array([1]), col=np.array([2]))

    combined = mask1 | mask2
    assert combined.is_sparse is True
    np.testing.assert_array_equal(combined.to_dense(), np.array([
        [True, False, False],
        [False, False, True],
    ]))


def test_scores_mask_invert_returns_dense():
    mask = ScoresMask(shape=(2, 2), row=np.array([0]), col=np.array([1]))
    inverted = ~mask
    assert inverted.is_sparse is False
    np.testing.assert_array_equal(
        inverted.to_dense(),
        np.array([
            [True, False],
            [True, True],
        ])
    )


def test_scores_mask_incompatible_shapes_raise():
    mask1 = ScoresMask(shape=(2, 2), dense_mask=np.ones((2, 2), dtype=bool))
    mask2 = ScoresMask(shape=(3, 2), dense_mask=np.ones((3, 2), dtype=bool))

    with pytest.raises(ValueError, match="Incompatible mask shapes"):
        _ = mask1 & mask2


def test_dense_field_comparison_returns_dense_mask(dense_multi_scores):
    mask = dense_multi_scores["score"] > 0.5
    assert isinstance(mask, ScoresMask)
    assert mask.is_sparse is False
    np.testing.assert_array_equal(
        mask.to_dense(),
        np.array([
            [True, False, False],
            [False, True, False],
        ])
    )


def test_sparse_field_comparison_gt_returns_sparse_mask(sparse_multi_scores):
    mask = sparse_multi_scores["score"] > 0.5
    assert isinstance(mask, ScoresMask)
    assert mask.is_sparse is True
    np.testing.assert_array_equal(mask.row, np.array([0, 1]))
    np.testing.assert_array_equal(mask.col, np.array([0, 1]))


def test_sparse_field_comparison_ge_returns_sparse_mask(sparse_multi_scores):
    mask = sparse_multi_scores["score"] >= 0.8
    assert mask.is_sparse is True
    np.testing.assert_array_equal(mask.to_dense(), np.array([
        [True, False, False],
        [False, True, False],
    ]))


def test_sparse_field_comparison_lt_falls_back_to_dense_mask(sparse_multi_scores):
    mask = sparse_multi_scores["score"] < 0.7
    assert mask.is_sparse is False
    np.testing.assert_array_equal(
        mask.to_dense(),
        np.array([
            [False, True, True],
            [True, False, True],
        ])
    )


def test_sparse_field_comparison_eq_falls_back_to_dense_mask(sparse_multi_scores):
    mask = sparse_multi_scores["score"] == 0.0
    assert mask.is_sparse is False
    np.testing.assert_array_equal(
        mask.to_dense(),
        np.array([
            [False, True, False],
            [True, False, True],
        ])
    )


def test_scores_filter_with_dense_mask_on_dense_scores(dense_multi_scores):
    mask = np.array([
        [True, False, True],
        [False, True, False],
    ])

    filtered = dense_multi_scores[mask]

    assert isinstance(filtered, Scores)
    assert filtered.is_sparse is False
    np.testing.assert_array_equal(
        filtered.to_array("score"),
        np.array([
            [1.0, 0.0, 0.5],
            [0.0, 0.8, 0.0],
        ])
    )
    np.testing.assert_array_equal(
        filtered.to_array("matches"),
        np.array([
            [3, 0, 2],
            [0, 4, 0],
        ])
    )


def test_scores_filter_with_dense_mask_zeroes_out_dense_scores():
    scores = Scores({
        "score": np.array([
            [1.0, 0.2, 0.5],
            [0.1, 0.8, 0.0],
        ]),
        "matches": np.array([
            [3, 1, 2],
            [1, 4, 0],
        ]),
    })
    mask = np.array([
        [True, False, True],
        [False, True, False],
    ])

    filtered = scores[mask]

    np.testing.assert_array_equal(
        filtered.to_array("score"),
        np.array([
            [1.0, 0.0, 0.5],
            [0.0, 0.8, 0.0],
        ])
    )
    np.testing.assert_array_equal(
        filtered.to_array("matches"),
        np.array([
            [3, 0, 2],
            [0, 4, 0],
        ])
    )


def test_scores_filter_with_sparse_mask_on_sparse_scores_preserves_sparse(sparse_multi_scores):
    mask = sparse_multi_scores["score"] > 0.5
    filtered = sparse_multi_scores[mask]

    assert isinstance(filtered, Scores)
    assert filtered.is_sparse is True
    np.testing.assert_array_equal(
        filtered.to_array("score"),
        np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.8, 0.0],
        ])
    )
    np.testing.assert_array_equal(
        filtered.to_array("matches"),
        np.array([
            [3, 0, 0],
            [0, 4, 0],
        ])
    )


def test_scores_filter_with_combined_sparse_masks(sparse_multi_scores):
    mask = (sparse_multi_scores["score"] > 0.5) & (sparse_multi_scores["matches"] >= 4)
    filtered = sparse_multi_scores[mask]

    np.testing.assert_array_equal(
        filtered.to_array("score"),
        np.array([
            [0.0, 0.0, 0.0],
            [0.0, 0.8, 0.0],
        ])
    )
    np.testing.assert_array_equal(
        filtered.to_array("matches"),
        np.array([
            [0, 0, 0],
            [0, 4, 0],
        ])
    )


def test_scores_filter_with_sparse_or_mask(sparse_multi_scores):
    mask = (sparse_multi_scores["score"] > 0.9) | (sparse_multi_scores["matches"] >= 4)
    filtered = sparse_multi_scores[mask]

    np.testing.assert_array_equal(
        filtered.to_array("score"),
        np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.8, 0.0],
        ])
    )


def test_scores_filter_with_inverted_mask(sparse_multi_scores):
    mask = ~(sparse_multi_scores["score"] > 0.5)
    filtered = sparse_multi_scores[mask]

    np.testing.assert_array_equal(
        filtered.to_array("score"),
        np.array([
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 0.0],
        ])
    )
    np.testing.assert_array_equal(
        filtered.to_array("matches"),
        np.array([
            [0, 0, 2],
            [0, 0, 0],
        ])
    )


def test_scores_filter_mask_shape_mismatch_raises(dense_scalar_scores):
    mask = np.ones((3, 3), dtype=bool)
    with pytest.raises(ValueError, match="Mask has shape"):
        dense_scalar_scores[mask]


def test_scores_filter_with_scoresmask_shape_mismatch_raises(dense_scalar_scores):
    mask = ScoresMask(shape=(3, 3), dense_mask=np.ones((3, 3), dtype=bool))
    with pytest.raises(ValueError, match="Mask has shape"):
        dense_scalar_scores[mask]


def test_scores_scalar_threshold_masking_syntax_dense(dense_scalar_scores):
    filtered = dense_scalar_scores[dense_scalar_scores["score"] > 0.5]
    np.testing.assert_array_equal(
        filtered.to_array(),
        np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.8, 0.0],
        ])
    )


def test_scores_scalar_threshold_masking_syntax_sparse(sparse_scalar_scores):
    filtered = sparse_scalar_scores[sparse_scalar_scores["score"] > 0.5]
    assert filtered.is_sparse is True
    np.testing.assert_array_equal(
        filtered.to_array(),
        np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.8, 0.0],
        ])
    )

def test_dense_scalar_scores_direct_comparison_returns_mask(dense_scalar_scores):
    mask = dense_scalar_scores > 0.5

    assert isinstance(mask, ScoresMask)
    assert mask.is_sparse is False
    np.testing.assert_array_equal(
        mask.to_dense(),
        np.array([
            [True, False, False],
            [False, True, False],
        ])
    )


def test_dense_scalar_scores_direct_comparison_matches_score_field_comparison(dense_scalar_scores):
    mask_direct = dense_scalar_scores > 0.5
    mask_field = dense_scalar_scores["score"] > 0.5

    np.testing.assert_array_equal(mask_direct.to_dense(), mask_field.to_dense())
