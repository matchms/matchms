import json
import os
import numpy as np
import pytest
from matchms.similarity import CosineHungarian, LinearCosine, get_similarity_function_by_name
from matchms.similarity.linear_cosine_functions import (
    linear_cosine_score,
    sirius_merge_close_peaks,
)
from ..builder_Spectrum import SpectrumBuilder


REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "..", "linear_cosine_reference.json")

with open(REFERENCE_PATH, encoding="utf-8") as f:
    REFERENCE_DATA = json.load(f)

COMPOUND_NAMES = sorted(REFERENCE_DATA["raw_spectra"].keys())
PARAM_SET_IDS = list(range(len(REFERENCE_DATA["parameter_sets"])))


def _raw_spectrum_array(name):
    """Return raw spectrum as (N, 2) numpy array."""
    raw = REFERENCE_DATA["raw_spectra"][name]
    return np.array([raw["mz"], raw["intensities"]], dtype=np.float64).T


def _param_label(idx):
    ps = REFERENCE_DATA["parameter_sets"][idx]
    return f"tol={ps['mz_tolerance']}_mzp={ps['mz_power']}_ip={ps['intensity_power']}"


@pytest.mark.parametrize("param_idx", PARAM_SET_IDS, ids=[_param_label(i) for i in PARAM_SET_IDS])
@pytest.mark.parametrize("compound", COMPOUND_NAMES)
def test_merge_close_peaks(param_idx, compound):
    """Verify that sirius_merge_close_peaks produces the expected merged spectrum."""
    ps = REFERENCE_DATA["parameter_sets"][param_idx]
    tolerance = ps["mz_tolerance"]

    raw = _raw_spectrum_array(compound)
    merged = sirius_merge_close_peaks(raw, tolerance)

    expected = ps["merged_spectra"][compound]
    expected_mz = np.array(expected["mz"], dtype=np.float64)
    expected_int = np.array(expected["intensities"], dtype=np.float64)

    assert len(merged) == len(expected_mz), f"Expected {len(expected_mz)} merged peaks, got {len(merged)}"
    np.testing.assert_allclose(merged[:, 0], expected_mz, rtol=1e-6, err_msg=f"m/z mismatch for {compound}")
    np.testing.assert_allclose(merged[:, 1], expected_int, rtol=1e-5, err_msg=f"intensity mismatch for {compound}")
    # Verify well-separated invariant: consecutive m/z gaps > 2 * tolerance
    if len(merged) > 1:
        gaps = np.diff(merged[:, 0])
        assert np.all(gaps > 2 * tolerance), (
            f"Well-separated invariant violated: min gap {gaps.min():.6f} <= {2 * tolerance}"
        )


def _build_similarity_pairs(param_idx):
    """Build list of (left, right, expected_score, expected_matches) for a parameter set."""
    ps = REFERENCE_DATA["parameter_sets"][param_idx]
    pairs = []
    for sim in ps["similarities"]:
        pairs.append((sim["left"], sim["right"], sim["score"], sim["matches"]))
    return pairs


def _all_score_params():
    """Generate (param_idx, left, right, expected_score, expected_matches) for parametrize."""
    params = []
    ids = []
    for param_idx in PARAM_SET_IDS:
        for left, right, score, matches in _build_similarity_pairs(param_idx):
            params.append((param_idx, left, right, score, matches))
            ids.append(f"{_param_label(param_idx)}_{left}_vs_{right}")
    return params, ids


_SCORE_PARAMS, _SCORE_IDS = _all_score_params()


@pytest.mark.parametrize("param_idx,left,right,expected_score,expected_matches", _SCORE_PARAMS, ids=_SCORE_IDS)
def test_pairwise_scores(param_idx, left, right, expected_score, expected_matches):
    """Verify pairwise LinearCosine scores match the reference."""
    ps = REFERENCE_DATA["parameter_sets"][param_idx]
    tolerance = ps["mz_tolerance"]
    mz_power = ps["mz_power"]
    intensity_power = ps["intensity_power"]

    builder = SpectrumBuilder()
    raw_left = REFERENCE_DATA["raw_spectra"][left]
    raw_right = REFERENCE_DATA["raw_spectra"][right]

    spec_left = (
        builder.with_mz(np.array(raw_left["mz"], dtype="float"))
        .with_intensities(np.array(raw_left["intensities"], dtype="float"))
        .build()
    )
    spec_right = (
        builder.with_mz(np.array(raw_right["mz"], dtype="float"))
        .with_intensities(np.array(raw_right["intensities"], dtype="float"))
        .build()
    )

    linear_cosine = LinearCosine(tolerance=tolerance, mz_power=mz_power, intensity_power=intensity_power)
    result = linear_cosine.pair(spec_left, spec_right)

    assert result["matches"] == expected_matches, f"Expected {expected_matches} matches, got {result['matches']}"
    assert result["score"] == pytest.approx(expected_score, abs=1e-6), (
        f"Expected score {expected_score}, got {result['score']}"
    )


@pytest.mark.parametrize("param_idx", PARAM_SET_IDS, ids=[_param_label(i) for i in PARAM_SET_IDS])
def test_commutativity(param_idx):
    """Verify score(A, B) == score(B, A) for all compound pairs."""
    ps = REFERENCE_DATA["parameter_sets"][param_idx]
    tolerance = ps["mz_tolerance"]
    mz_power = ps["mz_power"]
    intensity_power = ps["intensity_power"]

    linear_cosine = LinearCosine(tolerance=tolerance, mz_power=mz_power, intensity_power=intensity_power)
    builder = SpectrumBuilder()
    spectra = {}
    for name in COMPOUND_NAMES:
        raw = REFERENCE_DATA["raw_spectra"][name]
        spectra[name] = (
            builder.with_mz(np.array(raw["mz"], dtype="float"))
            .with_intensities(np.array(raw["intensities"], dtype="float"))
            .build()
        )

    for i, name_a in enumerate(COMPOUND_NAMES):
        for name_b in COMPOUND_NAMES[i + 1 :]:
            score_ab = linear_cosine.pair(spectra[name_a], spectra[name_b])
            score_ba = linear_cosine.pair(spectra[name_b], spectra[name_a])
            assert score_ab["score"] == pytest.approx(score_ba["score"], abs=1e-9), (
                f"Commutativity failed for {name_a} vs {name_b}"
            )
            assert score_ab["matches"] == score_ba["matches"], (
                f"Match count commutativity failed for {name_a} vs {name_b}"
            )


def _all_hungarian_params():
    """Generate (param_idx, left, right) for Hungarian parity tests."""
    params = []
    ids = []
    for param_idx in PARAM_SET_IDS:
        for left, right, _, _ in _build_similarity_pairs(param_idx):
            params.append((param_idx, left, right))
            ids.append(f"{_param_label(param_idx)}_{left}_vs_{right}")
    return params, ids


_HUNGARIAN_PARAMS, _HUNGARIAN_IDS = _all_hungarian_params()


@pytest.mark.parametrize("param_idx,left,right", _HUNGARIAN_PARAMS, ids=_HUNGARIAN_IDS)
def test_hungarian_matches_linear_cosine_on_merged_spectra(param_idx, left, right):
    """On well-separated (merged) spectra, CosineHungarian must match LinearCosine exactly.

    After sirius_merge_close_peaks, consecutive m/z gaps exceed 2*tolerance, so each
    peak can match at most one peak in the other spectrum. The optimal Hungarian
    assignment must therefore agree with the O(n+m) two-pointer sweep.
    """
    ps = REFERENCE_DATA["parameter_sets"][param_idx]
    tolerance = ps["mz_tolerance"]
    mz_power = ps["mz_power"]
    intensity_power = ps["intensity_power"]

    # Build spectra from the already-merged peaks
    builder = SpectrumBuilder()
    merged_left = ps["merged_spectra"][left]
    merged_right = ps["merged_spectra"][right]

    spec_left = (
        builder.with_mz(np.array(merged_left["mz"], dtype="float"))
        .with_intensities(np.array(merged_left["intensities"], dtype="float"))
        .build()
    )
    spec_right = (
        builder.with_mz(np.array(merged_right["mz"], dtype="float"))
        .with_intensities(np.array(merged_right["intensities"], dtype="float"))
        .build()
    )

    hungarian = CosineHungarian(tolerance=tolerance, mz_power=mz_power, intensity_power=intensity_power)
    linear = LinearCosine(tolerance=tolerance, mz_power=mz_power, intensity_power=intensity_power)

    result_hungarian = hungarian.pair(spec_left, spec_right)
    result_linear = linear.pair(spec_left, spec_right)

    assert result_hungarian["matches"] == result_linear["matches"], (
        f"Match count differs: Hungarian={result_hungarian['matches']}, LinearCosine={result_linear['matches']}"
    )
    assert result_hungarian["score"] == pytest.approx(result_linear["score"], abs=1e-9), (
        f"Score differs: Hungarian={result_hungarian['score']}, LinearCosine={result_linear['score']}"
    )


# ---------------------------------------------------------------------------
# Standalone edge-case tests (no JSON dependency)
# ---------------------------------------------------------------------------


def _build_spectrum(mz, intensities):
    """Helper to build a Spectrum from lists."""
    return (
        SpectrumBuilder()
        .with_mz(np.array(mz, dtype="float"))
        .with_intensities(np.array(intensities, dtype="float"))
        .build()
    )


def test_empty_spectrum():
    """Pairing an empty spectrum with a non-empty one gives score=0, matches=0."""
    empty = _build_spectrum([], [])
    nonempty = _build_spectrum([100.0, 200.0], [0.5, 0.5])
    lc = LinearCosine(tolerance=0.1)
    result = lc.pair(empty, nonempty)
    assert result["score"] == 0.0
    assert result["matches"] == 0


def test_single_peak_match():
    """Two single-peak spectra within tolerance match perfectly."""
    a = _build_spectrum([100.0], [1.0])
    b = _build_spectrum([100.05], [1.0])
    lc = LinearCosine(tolerance=0.1)
    result = lc.pair(a, b)
    assert result["score"] == pytest.approx(1.0, abs=1e-9)
    assert result["matches"] == 1


def test_single_peak_no_match():
    """Two single-peak spectra outside tolerance don't match."""
    a = _build_spectrum([100.0], [1.0])
    b = _build_spectrum([200.0], [1.0])
    lc = LinearCosine(tolerance=0.1)
    result = lc.pair(a, b)
    assert result["score"] == 0.0
    assert result["matches"] == 0


def test_no_overlap():
    """Multi-peak spectra with no overlapping m/z give score=0."""
    a = _build_spectrum([100.0, 200.0, 300.0], [0.5, 0.3, 0.2])
    b = _build_spectrum([400.0, 500.0, 600.0], [0.5, 0.3, 0.2])
    lc = LinearCosine(tolerance=0.1)
    result = lc.pair(a, b)
    assert result["score"] == 0.0
    assert result["matches"] == 0


def test_self_similarity():
    """A spectrum compared to itself should yield score=1.0."""
    spec = _build_spectrum([100.0, 200.0, 300.0], [0.7, 0.2, 0.1])
    lc = LinearCosine(tolerance=0.1)
    result = lc.pair(spec, spec)
    assert result["score"] == pytest.approx(1.0, abs=1e-9)


def test_all_zero_intensities():
    """All-zero intensities produce score=0.0, matches=0."""
    a = _build_spectrum([100.0, 200.0], [0.0, 0.0])
    b = _build_spectrum([100.0, 200.0], [0.0, 0.0])
    lc = LinearCosine(tolerance=0.1)
    result = lc.pair(a, b)
    assert result["score"] == 0.0
    assert result["matches"] == 0


# ---------------------------------------------------------------------------
# Direct unit test for linear_cosine_score
# ---------------------------------------------------------------------------


def test_linear_cosine_score_direct():
    """Hand-crafted well-separated spectra with analytically derivable score.

    spec1: peaks at m/z 100 (intensity 0.6), 200 (intensity 0.8)
    spec2: peaks at m/z 100 (intensity 0.3), 200 (intensity 0.4)
    With mz_power=0 and intensity_power=1:
      products = intensities themselves
      dot = 0.6*0.3 + 0.8*0.4 = 0.18 + 0.32 = 0.50
      norm1 = sqrt(0.36 + 0.64) = 1.0
      norm2 = sqrt(0.09 + 0.16) = 0.5
      score = 0.50 / (1.0 * 0.5) = 1.0
      matches = 2
    """
    spec1 = np.array([[100.0, 0.6], [200.0, 0.8]])
    spec2 = np.array([[100.0, 0.3], [200.0, 0.4]])
    score, matches = linear_cosine_score(spec1, spec2, 0.1, 0.0, 1.0)
    assert score == pytest.approx(1.0, abs=1e-9)
    assert matches == 2


def test_linear_cosine_score_partial_match():
    """Only one of two peaks matches; verify score analytically.

    spec1: m/z 100 (int 0.6), m/z 200 (int 0.8)
    spec2: m/z 100 (int 0.5), m/z 300 (int 0.5)
    mz_power=0, intensity_power=1:
      dot = 0.6*0.5 = 0.30
      norm1 = sqrt(0.36+0.64) = 1.0
      norm2 = sqrt(0.25+0.25) = sqrt(0.5)
      score = 0.30 / (1.0 * sqrt(0.5)) = 0.30 / 0.7071... ≈ 0.42426
      matches = 1
    """
    spec1 = np.array([[100.0, 0.6], [200.0, 0.8]])
    spec2 = np.array([[100.0, 0.5], [300.0, 0.5]])
    score, matches = linear_cosine_score(spec1, spec2, 0.1, 0.0, 1.0)
    expected = 0.30 / np.sqrt(0.5)
    assert score == pytest.approx(expected, abs=1e-9)
    assert matches == 1


# ---------------------------------------------------------------------------
# Class-level integration tests
# ---------------------------------------------------------------------------


def test_matrix_self_similarity():
    """matrix() diagonal should be score=1.0 for identical spectra."""
    spectra = [
        _build_spectrum([100.0, 200.0], [0.7, 0.3]),
        _build_spectrum([150.0, 250.0], [0.5, 0.5]),
        _build_spectrum([100.0, 300.0], [0.9, 0.1]),
    ]
    lc = LinearCosine(tolerance=0.1)
    result = lc.matrix(spectra, spectra, is_symmetric=True, progress_bar=False)
    for i in range(len(spectra)):
        assert result[i, i]["score"] == pytest.approx(1.0, abs=1e-9)


def test_matrix_matches_pair():
    """matrix() results should match pair() results."""
    spectra = [
        _build_spectrum([100.0, 200.0], [0.7, 0.3]),
        _build_spectrum([100.0, 250.0], [0.5, 0.5]),
    ]
    lc = LinearCosine(tolerance=0.1)
    mat = lc.matrix(spectra, spectra, is_symmetric=False, progress_bar=False)
    for i in range(2):
        for j in range(2):
            pair_result = lc.pair(spectra[i], spectra[j])
            assert mat[i, j]["score"] == pytest.approx(pair_result["score"], abs=1e-9)
            assert mat[i, j]["matches"] == pair_result["matches"]


def test_get_similarity_by_name():
    """get_similarity_function_by_name returns LinearCosine class."""
    cls = get_similarity_function_by_name("LinearCosine")
    assert cls is LinearCosine


def test_to_dict_round_trip():
    """to_dict() keys match constructor params."""
    lc = LinearCosine(tolerance=0.2, mz_power=1.0, intensity_power=0.5)
    d = lc.to_dict()
    assert d["__Similarity__"] == "LinearCosine"
    assert d["tolerance"] == 0.2
    assert d["mz_power"] == 1.0
    assert d["intensity_power"] == 0.5
