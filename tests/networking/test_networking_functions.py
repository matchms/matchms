import numpy as np
from matchms import Spectrum, calculate_scores
from matchms.networking.networking_functions import (
    get_top_hits,
    get_top_hits_by_column,
    get_top_hits_by_row,
)
from matchms.similarity import FlashEntropy


def create_dummy_spectra():
    """Create dummy spectra"""
    spectra = []
    for i in range(5):
        spectra.append(Spectrum(mz=np.array([100, 200.]),
                                  intensities=np.array([0.7, 0.1 * i]),
                                  metadata={"spectrum_id": 'ref_spec_'+str(i),
                                            "smiles": 'C1=CC=C2C(=C1)NC(=N2)C3=CC=CO3',
                                            "precursor_mz": 100+50*i}))
    for i in range(3):
        spectra.append(Spectrum(mz=np.array([100 + i, 210.]),
                                  intensities=np.array([0.5, 0.1 * i]),
                                  metadata={"spectrum_id": 'query_spec_'+str(i),
                                            "smiles": 'CC1=C(C=C(C=C1)NC(=O)N(C)C)Cl',
                                            "precursor_mz": 110+50*i}))
    return spectra


def create_dummy_scores():
    spectra = create_dummy_spectra()
    spectra_1 = spectra[:5]
    spectra_2 = spectra[5:]

    similarity_measure = FlashEntropy(matching_mode="hybrid")
    scores = calculate_scores(spectra_1, spectra_2, similarity_measure)
    return scores, spectra_1, spectra_2


def test_get_top_hits_by_column():
    scores, spectra_1, spectra_2 = create_dummy_scores()
    identifiers = [s.get("spectrum_id") for s in spectra_2]

    idx_col, scores_col = get_top_hits(
        scores,
        top_n=10,
        axis=0,
        identifiers=identifiers,
    )

    expected_scores_col = {
        'query_spec_0': np.array([1.        , 0.80566937, 0.77583811, 0.75247993, 0.73288331]),
        'query_spec_1': np.array([0.34812354, 0.        , 0.        , 0.        , 0.        ]),
        'query_spec_2': np.array([0.612693  , 0.39564064, 0.        , 0.        , 0.        ])
    }
    expected_idx_col = {
        'query_spec_0': np.array([0, 1, 2, 3, 4]),
        'query_spec_1': np.array([1, 0, 2, 3, 4]),
        'query_spec_2': np.array([0, 2, 1, 3, 4]),
    }

    for key, value in scores_col.items():
        assert np.allclose(value, expected_scores_col[key], atol=1e-5), (
            f"Unexpected selected scores for {key}"
        )
    for key, value in idx_col.items():
        assert np.array_equal(value, expected_idx_col[key]), (
            f"Unexpected selected indices for {key}"
        )

    idx_col, scores_col = get_top_hits(
        scores,
        top_n=2,
        axis=0,
        identifiers=identifiers,
    )

    expected_scores_col_top2 = {
        'query_spec_0': np.array([1.0, 0.80566937]),
        'query_spec_1': np.array([0.34812354, 0.0]),
        'query_spec_2': np.array([0.612693, 0.39564064]),
    }

    for key, value in scores_col.items():
        assert np.allclose(value, expected_scores_col_top2[key], atol=1e-5), (
            f"Unexpected selected scores for {key} with top_n=2"
        )

    # Exact result: no tie at cutoff
    assert np.array_equal(idx_col["query_spec_0"], np.array([0, 1]))

    # Exact result: second-best score is unique
    assert np.array_equal(idx_col["query_spec_2"], np.array([0, 2]))

    # Tied zeros at cutoff: first must be the best hit, second can be any zero-score index
    assert idx_col["query_spec_1"][0] == 1
    assert idx_col["query_spec_1"][1] in {0, 2, 3, 4}
    assert len(set(idx_col["query_spec_1"])) == 2


def test_get_top_hits_by_row():
    scores, spectra_1, spectra_2 = create_dummy_scores()
    identifiers = [s.get("spectrum_id") for s in spectra_1]

    idx_row, scores_row = get_top_hits(
        scores,
        top_n=10,
        axis=1,
        identifiers=identifiers,
    )

    expected_scores_row = {
        'ref_spec_0': np.array([1.      , 0.612693, 0.      ]),
        'ref_spec_1': np.array([0.80566937, 0.34812354, 0.        ]),
        'ref_spec_2': np.array([0.77583811, 0.39564064, 0.        ]),
        'ref_spec_3': np.array([0.75247993, 0.        , 0.        ]),
        'ref_spec_4': np.array([0.73288331, 0.        , 0.        ])
    }
    expected_idx_row = {
        'ref_spec_0': np.array([0, 2, 1]),
        'ref_spec_1': np.array([0, 1, 2]),
        'ref_spec_2': np.array([0, 2, 1]),
        'ref_spec_3': np.array([0, 1, 2]),
        'ref_spec_4': np.array([0, 1, 2]),
    }

    for key, value in scores_row.items():
        assert np.allclose(value, expected_scores_row[key], atol=1e-5), (
            f"Unexpected selected scores for {key}"
        )
    for key, value in idx_row.items():
        assert np.array_equal(value, expected_idx_row[key]), (
            f"Unexpected selected indices for {key}"
        )

    idx_row, scores_row = get_top_hits(
        scores,
        top_n=2,
        axis=1,
        identifiers=identifiers,
    )

    expected_scores_row_top2 = {
        'ref_spec_0': np.array([1.0, 0.612693]),
        'ref_spec_1': np.array([0.80566937, 0.34812354]),
        'ref_spec_2': np.array([0.77583811, 0.39564064]),
        'ref_spec_3': np.array([0.75247993, 0.0]),
        'ref_spec_4': np.array([0.73288331, 0.0]),
    }

    for key, value in scores_row.items():
        assert np.allclose(value, expected_scores_row_top2[key], atol=1e-5), (
            f"Unexpected selected scores for {key} with top_n=2"
        )

    # Exact results: no tie affecting top-2 membership
    assert np.array_equal(idx_row["ref_spec_0"], np.array([0, 2]))
    assert np.array_equal(idx_row["ref_spec_1"], np.array([0, 1]))
    assert np.array_equal(idx_row["ref_spec_2"], np.array([0, 2]))

    # Tied zero at cutoff: second can be either of the zero-score indices
    assert idx_row["ref_spec_3"][0] == 0
    assert idx_row["ref_spec_3"][1] in {1, 2}
    assert len(set(idx_row["ref_spec_3"])) == 2

    assert idx_row["ref_spec_4"][0] == 0
    assert idx_row["ref_spec_4"][1] in {1, 2}
    assert len(set(idx_row["ref_spec_4"])) == 2


def test_get_top_hits_by_row_wrapper():
    scores, spectra_1, _ = create_dummy_scores()
    identifiers = [s.get("spectrum_id") for s in spectra_1]

    idx_a, scores_a = get_top_hits(
        scores,
        top_n=3,
        axis=1,
        identifiers=identifiers,
    )
    idx_b, scores_b = get_top_hits_by_row(
        scores,
        top_n=3,
        identifiers=identifiers,
    )

    assert idx_a.keys() == idx_b.keys()
    assert scores_a.keys() == scores_b.keys()
    for key in idx_a:
        assert np.array_equal(idx_a[key], idx_b[key])
        assert np.allclose(scores_a[key], scores_b[key])


def test_get_top_hits_by_column_wrapper():
    scores, _, spectra_2 = create_dummy_scores()
    identifiers = [s.get("spectrum_id") for s in spectra_2]

    idx_a, scores_a = get_top_hits(
        scores,
        top_n=3,
        axis=0,
        identifiers=identifiers,
    )
    idx_b, scores_b = get_top_hits_by_column(
        scores,
        top_n=3,
        identifiers=identifiers,
    )

    assert idx_a.keys() == idx_b.keys()
    assert scores_a.keys() == scores_b.keys()
    for key in idx_a:
        assert np.array_equal(idx_a[key], idx_b[key])
        assert np.allclose(scores_a[key], scores_b[key])


def test_get_top_hits_default_identifiers():
    scores, spectra_1, _ = create_dummy_scores()

    idx_row, scores_row = get_top_hits(scores, top_n=2, axis=1)

    assert set(idx_row.keys()) == set(range(len(spectra_1)))
    assert set(scores_row.keys()) == set(range(len(spectra_1)))
    for key in idx_row:
        assert len(idx_row[key]) <= 2
        assert len(scores_row[key]) <= 2


def test_get_top_hits_ignore_diagonal():
    spectra = create_dummy_spectra()[:5]
    similarity_measure = FlashEntropy(matching_mode="hybrid")
    scores = calculate_scores(spectra, spectra, similarity_measure)
    identifiers = [s.get("spectrum_id") for s in spectra]

    idx_row, scores_row = get_top_hits(
        scores,
        top_n=5,
        axis=1,
        identifiers=identifiers,
        ignore_diagonal=True,
    )

    dense = scores["score"].to_array() if "score" in scores.score_fields else scores.to_array()

    for i, spec_id in enumerate(identifiers):
        assert i not in idx_row[spec_id]
        if len(scores_row[spec_id]) > 0:
            assert not np.any(np.isclose(scores_row[spec_id], dense[i, i]))\
                or dense[i, i] != np.max(scores_row[spec_id])


def test_get_top_hits_ignore_diagonal_requires_square():
    scores, spectra_1, _ = create_dummy_scores()
    identifiers = [s.get("spectrum_id") for s in spectra_1]

    with np.testing.assert_raises(ValueError):
        get_top_hits(
            scores,
            top_n=3,
            axis=1,
            identifiers=identifiers,
            ignore_diagonal=True,
        )


def test_get_top_hits_identifier_length_check():
    scores, _, _ = create_dummy_scores()

    with np.testing.assert_raises(ValueError):
        get_top_hits(
            scores,
            top_n=3,
            axis=1,
            identifiers=["too", "short"],
        )


def test_get_top_hits_invalid_axis():
    scores, _, _ = create_dummy_scores()

    with np.testing.assert_raises(ValueError):
        get_top_hits(scores, top_n=3, axis=2)
