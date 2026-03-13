import numpy as np
import pytest
from matchms import Spectrum
from matchms.similarity import (
    Cosine,
    CosineGreedy,
    CosineHungarian,
    FlashCosine,
)


def _make_test_spectra():
    spectrum_1 = Spectrum(
        mz=np.array([100.0, 150.0, 200.0], dtype="float"),
        intensities=np.array([0.8, 0.5, 0.2], dtype="float"),
        metadata={"precursor_mz": 500.0},
    )
    spectrum_2 = Spectrum(
        mz=np.array([100.0, 150.05, 205.0], dtype="float"),
        intensities=np.array([0.7, 0.4, 0.2], dtype="float"),
        metadata={"precursor_mz": 505.0},
    )
    spectrum_3 = Spectrum(
        mz=np.array([110.0, 160.0, 210.0], dtype="float"),
        intensities=np.array([0.9, 0.3, 0.1], dtype="float"),
        metadata={"precursor_mz": 510.0},
    )
    return spectrum_1, spectrum_2, spectrum_3


def test_cosine_wrapper_score_schema():
    cosine = Cosine()

    assert cosine.is_commutative is True
    assert tuple(cosine.score_fields) == ("score", "matches")

    dtype_names = np.dtype(cosine.score_datatype).names
    assert dtype_names == ("score", "matches")


def test_cosine_pair_matches_greedy_when_use_hungarian_false():
    spectrum_1, spectrum_2, _ = _make_test_spectra()

    wrapped = Cosine(tolerance=0.1, use_hungarian=False)
    direct = CosineGreedy(tolerance=0.1)

    wrapped_score = wrapped.pair(spectrum_1, spectrum_2)
    direct_score = direct.pair(spectrum_1, spectrum_2)

    assert wrapped_score["score"] == pytest.approx(direct_score["score"], abs=1e-12)
    assert wrapped_score["matches"] == direct_score["matches"]


def test_cosine_pair_matches_hungarian_when_use_hungarian_true():
    spectrum_1, spectrum_2, _ = _make_test_spectra()

    wrapped = Cosine(tolerance=0.1, use_hungarian=True)
    direct = CosineHungarian(tolerance=0.1)

    wrapped_score = wrapped.pair(spectrum_1, spectrum_2)
    direct_score = direct.pair(spectrum_1, spectrum_2)

    assert wrapped_score["score"] == pytest.approx(direct_score["score"], abs=1e-12)
    assert wrapped_score["matches"] == direct_score["matches"]


def test_cosine_matrix_matches_flash_cosine_when_use_hungarian_false():
    spectrum_1, spectrum_2, spectrum_3 = _make_test_spectra()
    spectra = [spectrum_1, spectrum_2, spectrum_3]

    wrapped = Cosine(tolerance=0.1, use_hungarian=False)
    direct = FlashCosine(matching_mode="fragment", tolerance=0.1)

    wrapped_scores = wrapped.matrix(spectra, progress_bar=False, n_jobs=1)
    direct_scores = direct.matrix(spectra, progress_bar=False, n_jobs=1)

    assert wrapped_scores.score_fields == ("score", "matches")
    assert direct_scores.score_fields == ("score", "matches")
    assert wrapped_scores.shape == direct_scores.shape == (3, 3)

    assert np.allclose(
        wrapped_scores["score"].to_array(),
        direct_scores["score"].to_array(),
        atol=1e-12,
    )
    assert np.array_equal(
        wrapped_scores["matches"].to_array(),
        direct_scores["matches"].to_array(),
    )


def test_cosine_matrix_matches_hungarian_when_use_hungarian_true():
    spectrum_1, spectrum_2, spectrum_3 = _make_test_spectra()
    spectra = [spectrum_1, spectrum_2, spectrum_3]

    wrapped = Cosine(tolerance=0.1, use_hungarian=True)
    direct = CosineHungarian(tolerance=0.1)

    wrapped_scores = wrapped.matrix(spectra, progress_bar=False)
    direct_scores = direct.matrix(spectra, progress_bar=False)

    assert wrapped_scores.score_fields == ("score", "matches")
    assert direct_scores.score_fields == ("score", "matches")
    assert wrapped_scores.shape == direct_scores.shape == (3, 3)

    assert np.allclose(
        wrapped_scores["score"].to_array(),
        direct_scores["score"].to_array(),
        atol=1e-12,
    )
    assert np.array_equal(
        wrapped_scores["matches"].to_array(),
        direct_scores["matches"].to_array(),
    )


def test_cosine_matrix_score_field_selection():
    spectrum_1, spectrum_2, spectrum_3 = _make_test_spectra()
    spectra = [spectrum_1, spectrum_2, spectrum_3]

    scores = Cosine(use_hungarian=False).matrix(
        spectra,
        score_fields=("score",),
        progress_bar=False,
        n_jobs=1,
    )

    assert scores.score_fields == ("score",)
    assert scores.shape == (3, 3)
    assert np.allclose(np.diag(scores.to_array()), 1.0, atol=1e-12)


def test_cosine_pair_and_matrix_are_consistent_for_single_pair():
    spectrum_1, spectrum_2, _ = _make_test_spectra()

    similarity = Cosine(tolerance=0.1, use_hungarian=True)

    pair_score = similarity.pair(spectrum_1, spectrum_2)
    matrix_scores = similarity.matrix(
        [spectrum_1],
        [spectrum_2],
        progress_bar=False,
    )

    assert matrix_scores.shape == (1, 1)
    assert matrix_scores["score"].to_array()[0, 0] == pytest.approx(pair_score["score"], abs=1e-12)
    assert matrix_scores["matches"].to_array()[0, 0] == pair_score["matches"]


@pytest.mark.parametrize("use_hungarian", [False, True])
def test_cosine_symmetric_matrix_has_expected_fields_and_shape(use_hungarian):
    spectrum_1, spectrum_2, spectrum_3 = _make_test_spectra()
    spectra = [spectrum_1, spectrum_2, spectrum_3]

    kwargs = {"progress_bar": False}
    if not use_hungarian:
        kwargs["n_jobs"] = 1

    scores = Cosine(use_hungarian=use_hungarian).matrix(spectra, **kwargs)

    assert scores.score_fields == ("score", "matches")
    assert scores.shape == (3, 3)

    score_arr = scores["score"].to_array()
    matches_arr = scores["matches"].to_array()

    assert score_arr.shape == (3, 3)
    assert matches_arr.shape == (3, 3)
    assert np.allclose(score_arr, score_arr.T, atol=1e-12)
    assert np.array_equal(matches_arr, matches_arr.T)
