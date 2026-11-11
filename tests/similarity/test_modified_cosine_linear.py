import json
import os
import numpy as np
import pytest
from matchms import Spectrum
from matchms.similarity import (
    CosineLinear,
    ModifiedCosineGreedy,
    ModifiedCosineHungarian,
    ModifiedCosineLinear,
    get_similarity_function_by_name,
)
from matchms.similarity.cosine_linear_functions import modified_linear_cosine_score, sirius_merge_close_peaks


REFERENCE_PATH = os.path.join(os.path.dirname(__file__), "..", "modified_cosine_linear_reference.json")

with open(REFERENCE_PATH, encoding="utf-8") as f:
    REFERENCE_DATA = json.load(f)

PARAM_SET_IDS = list(range(len(REFERENCE_DATA["parameter_sets"])))


def _param_label(idx):
    ps = REFERENCE_DATA["parameter_sets"][idx]
    return f"tol={ps['mz_tolerance']}_mzp={ps['mz_power']}_ip={ps['intensity_power']}"


def _similarity(param_idx):
    """Similarity matching the reference data, which was scored on raw peaks."""
    ps = REFERENCE_DATA["parameter_sets"][param_idx]
    return ModifiedCosineLinear(
        tolerance=ps["mz_tolerance"],
        mz_power=ps["mz_power"],
        intensity_power=ps["intensity_power"],
        remove_precursor=False,
        noise_cutoff=None,
    )


def _build_spectrum(mz, intensities, precursor_mz):
    return Spectrum(
        mz=np.array(mz, dtype="float"),
        intensities=np.array(intensities, dtype="float"),
        metadata={"precursor_mz": precursor_mz},
    )


def _raw_spectrum(name):
    raw = REFERENCE_DATA["raw_spectra"][name]
    return _build_spectrum(raw["mz"], raw["intensities"], raw["precursor_mz"])


def _merged_spectrum(name, tolerance):
    raw = REFERENCE_DATA["raw_spectra"][name]
    peaks = np.array([raw["mz"], raw["intensities"]], dtype=np.float64).T
    merged = sirius_merge_close_peaks(peaks, tolerance)
    return _build_spectrum(merged[:, 0], merged[:, 1], raw["precursor_mz"])


def _all_score_params():
    params = []
    ids = []
    for param_idx in PARAM_SET_IDS:
        for sim in REFERENCE_DATA["parameter_sets"][param_idx]["similarities"]:
            params.append((param_idx, sim["left"], sim["right"], sim["score"], sim["matches"]))
            ids.append(f"{_param_label(param_idx)}_{sim['left']}_vs_{sim['right']}")
    return params, ids


_SCORE_PARAMS, _SCORE_IDS = _all_score_params()


@pytest.mark.parametrize("param_idx,left,right,expected_score,expected_matches", _SCORE_PARAMS, ids=_SCORE_IDS)
def test_pairwise_scores_match_reference(param_idx, left, right, expected_score, expected_matches):
    result = _similarity(param_idx).pair(_raw_spectrum(left), _raw_spectrum(right))

    assert result["matches"] == expected_matches
    assert result["score"] == pytest.approx(expected_score, abs=1e-9)


@pytest.mark.parametrize("param_idx", PARAM_SET_IDS, ids=[_param_label(i) for i in PARAM_SET_IDS])
def test_commutativity(param_idx):
    similarity = _similarity(param_idx)

    for sim in REFERENCE_DATA["parameter_sets"][param_idx]["similarities"]:
        left = _raw_spectrum(sim["left"])
        right = _raw_spectrum(sim["right"])
        score_ab = similarity.pair(left, right)
        score_ba = similarity.pair(right, left)
        assert score_ab["score"] == pytest.approx(score_ba["score"], abs=1e-9), f"{sim['left']} vs {sim['right']}"
        assert score_ab["matches"] == score_ba["matches"], f"{sim['left']} vs {sim['right']}"


def _shifted_pairs_with_matches(param_idx):
    ps = REFERENCE_DATA["parameter_sets"][param_idx]
    raw = REFERENCE_DATA["raw_spectra"]
    return [
        (sim["left"], sim["right"])
        for sim in ps["similarities"]
        if sim["matches"] > 0
        and abs(raw[sim["left"]]["precursor_mz"] - raw[sim["right"]]["precursor_mz"]) > ps["mz_tolerance"]
    ]


_HUNGARIAN_PAIRS = _shifted_pairs_with_matches(0)


@pytest.mark.parametrize("left,right", _HUNGARIAN_PAIRS, ids=[f"{left}_vs_{right}" for left, right in _HUNGARIAN_PAIRS])
def test_hungarian_matches_modified_cosine_linear_on_merged_spectra(left, right):
    """On well-separated spectra the optimal assignment is reachable in linear time."""
    ps = REFERENCE_DATA["parameter_sets"][0]
    tolerance = ps["mz_tolerance"]
    spec_left = _merged_spectrum(left, tolerance)
    spec_right = _merged_spectrum(right, tolerance)

    hungarian = ModifiedCosineHungarian(
        tolerance=tolerance,
        mz_power=ps["mz_power"],
        intensity_power=ps["intensity_power"],
        remove_precursor=False,
        noise_cutoff=None,
    )
    result_hungarian = hungarian.pair(spec_left, spec_right)
    result_linear = _similarity(0).pair(spec_left, spec_right)

    assert result_hungarian["matches"] == result_linear["matches"]
    assert result_hungarian["score"] == pytest.approx(result_linear["score"], abs=1e-9)


def test_symmetric_matrix_matches_pair():
    """matrix() prepares each spectrum once, including precursor removal, and must agree with pair()."""
    spectra = [_raw_spectrum(name) for name in ("aspirin", "aspirin_ch2", "ladder_a", "ladder_b", "cocaine")]
    similarity = ModifiedCosineLinear(tolerance=0.1)

    symmetric = similarity.matrix(spectra, progress_bar=False)

    for i, reference in enumerate(spectra):
        for j, query in enumerate(spectra):
            expected = similarity.pair(reference, query)
            assert symmetric[i, j]["score"] == pytest.approx(expected["score"], abs=1e-12)
            assert symmetric[i, j]["matches"] == expected["matches"]


@pytest.mark.parametrize("query_precursor_mz", [500.0, 500.1, 499.9])
def test_reduces_to_cosine_linear_when_precursor_delta_within_tolerance(query_precursor_mz):
    reference = _build_spectrum([100.0, 101.0, 102.0], [1.0, 0.9, 0.8], 500.0)
    query = _build_spectrum([100.0, 101.0, 104.0], [1.0, 0.9, 0.8], query_precursor_mz)

    modified_score = ModifiedCosineLinear(tolerance=0.1).pair(reference, query)
    cosine_score = CosineLinear(tolerance=0.1).pair(reference, query)

    assert modified_score["score"] == pytest.approx(cosine_score["score"], abs=1e-12)
    assert modified_score["matches"] == cosine_score["matches"]


def test_optimal_assignment_on_conflicting_direct_and_shifted_matches():
    """The direct edge (0.64) loses to the two conflicting shifted edges (0.48 + 0.48)."""
    reference = _build_spectrum([100.0, 114.01565], [0.6, 0.8], 300.0)
    query = _build_spectrum([114.01565, 128.0313], [0.8, 0.6], 314.01565)

    exact = ModifiedCosineLinear(tolerance=0.1).pair(reference, query)
    greedy = ModifiedCosineGreedy(tolerance=0.1).pair(reference, query)

    assert exact["score"] == pytest.approx(0.96, abs=1e-12)
    assert exact["matches"] == 2
    assert greedy["score"] == pytest.approx(0.64, abs=1e-12)


def test_zero_intensity_peaks_are_not_matches():
    reference = _build_spectrum([100.0, 200.0], [0.0, 2.0], 200.0)
    query = _build_spectrum([100.0, 200.0], [3.0, 4.0], 200.0)

    result = ModifiedCosineLinear(tolerance=0.1, remove_precursor=False, noise_cutoff=None).pair(reference, query)

    assert result["score"] == pytest.approx(0.8, abs=1e-12)
    assert result["matches"] == 1


def test_empty_spectrum():
    empty = _build_spectrum([], [], 300.0)
    nonempty = _build_spectrum([100.0, 200.0], [0.5, 0.5], 314.0)

    result = ModifiedCosineLinear(tolerance=0.1).pair(empty, nonempty)

    assert result["score"] == 0.0
    assert result["matches"] == 0


def test_score_function_rejects_peaks_closer_than_twice_the_tolerance():
    spec1 = np.array([[100.0, 1.0], [100.15, 1.0]])
    spec2 = np.array([[100.0, 1.0]])

    with pytest.raises(ValueError, match="well-separated"):
        modified_linear_cosine_score(spec1, spec2, 300.0, 300.0, 0.1, 0.0, 1.0)


def test_get_similarity_by_name():
    assert get_similarity_function_by_name("ModifiedCosineLinear") is ModifiedCosineLinear


def test_to_dict_round_trip():
    similarity = ModifiedCosineLinear(tolerance=0.2, mz_power=1.0, intensity_power=0.5)
    assert similarity.to_dict() == {
        "__Similarity__": "ModifiedCosineLinear",
        "tolerance": 0.2,
        "mz_power": 1.0,
        "intensity_power": 0.5,
        "noise_cutoff": 0.01,
        "remove_precursor": True,
        "offset_to_precursor": -1.6,
    }
