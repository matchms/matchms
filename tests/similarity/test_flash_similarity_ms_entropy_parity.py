import numpy as np
import pytest
from ms_entropy import calculate_entropy_similarity
from matchms.reference_spectra import (
    aspirin,
    cocaine,
    glucose,
    hydroxy_cholesterol,
    phenylalanine,
    salicin,
)
from matchms.similarity.FlashSimilarity import FlashSimilarity


def _reference_spectra():
    return [aspirin(), cocaine(), glucose(), hydroxy_cholesterol(), phenylalanine(), salicin()]


def _normalize_peaks_for_ms_entropy(spectrum):
    peaks = np.asarray(spectrum.peaks.to_numpy, dtype=np.float32).reshape(-1, 2)
    if peaks.shape[0] == 0:
        return peaks

    valid = np.bitwise_and(peaks[:, 0] > 0, peaks[:, 1] > 0)
    peaks = peaks[valid]
    if peaks.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)

    peaks = peaks[np.argsort(peaks[:, 0], kind="mergesort")]
    total = np.sum(peaks[:, 1], dtype=np.float64)
    if total <= 0.0:
        return np.zeros((0, 2), dtype=np.float32)
    peaks[:, 1] = (peaks[:, 1] / total).astype(np.float32, copy=False)
    return np.ascontiguousarray(peaks, dtype=np.float32)


def test_flash_entropy_fragment_matches_ms_entropy_on_reference_spectra():
    tolerance_da = 0.02
    refs = _reference_spectra()
    refs_for_ms_entropy = [_normalize_peaks_for_ms_entropy(s) for s in refs]

    flash = FlashSimilarity(
        score_type="spectral_entropy",
        matching_mode="fragment",
        tolerance=tolerance_da,
        use_ppm=False,
        remove_precursor=False,
        noise_cutoff=0.0,
        normalize_to_half=True,
        merge_within=0.0,
        dtype=np.float64,
    )
    matrix_scores = flash.matrix(refs, refs, array_type="numpy", n_jobs=0)

    for i, spec_a in enumerate(refs):
        for j, spec_b in enumerate(refs):
            expected = float(
                calculate_entropy_similarity(
                    refs_for_ms_entropy[i],
                    refs_for_ms_entropy[j],
                    ms2_tolerance_in_da=tolerance_da,
                    ms2_tolerance_in_ppm=-1,
                    clean_spectra=False,
                )
            )
            score_pair = float(flash.pair(spec_a, spec_b))
            score_matrix = float(matrix_scores[i, j])

            assert score_pair == pytest.approx(expected, rel=1e-6, abs=1e-6)
            assert score_matrix == pytest.approx(expected, rel=1e-6, abs=1e-6)


def test_flash_entropy_fragment_matches_ms_entropy_on_reference_spectra_ppm():
    tolerance_ppm = 20.0
    refs = _reference_spectra()
    refs_for_ms_entropy = [_normalize_peaks_for_ms_entropy(s) for s in refs]

    flash = FlashSimilarity(
        score_type="spectral_entropy",
        matching_mode="fragment",
        tolerance=tolerance_ppm,
        use_ppm=True,
        remove_precursor=False,
        noise_cutoff=0.0,
        normalize_to_half=True,
        merge_within=0.0,
        dtype=np.float64,
    )
    matrix_scores = flash.matrix(refs, refs, array_type="numpy", n_jobs=0)

    for i, spec_a in enumerate(refs):
        for j, spec_b in enumerate(refs):
            expected = float(
                calculate_entropy_similarity(
                    refs_for_ms_entropy[i],
                    refs_for_ms_entropy[j],
                    ms2_tolerance_in_da=-1,
                    ms2_tolerance_in_ppm=tolerance_ppm,
                    clean_spectra=False,
                )
            )
            score_pair = float(flash.pair(spec_a, spec_b))
            score_matrix = float(matrix_scores[i, j])

            assert score_pair == pytest.approx(expected, rel=1e-6, abs=1e-6)
            assert score_matrix == pytest.approx(expected, rel=1e-6, abs=1e-6)
