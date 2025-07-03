import numpy as np
import pytest
from matchms.similarity.SpectralEntropy import compute_entropy


def test_basic_known_outcome():
    # Two single peaks within tolerance should match perfectly => similarity = 1
    spec1_mz = np.array([100.0])
    spec1_int = np.array([1.0])
    spec2_mz = np.array([101.0])
    spec2_int = np.array([1.0])
    score = compute_entropy(spec1_mz, spec1_int, spec2_mz, spec2_int,
                             tolerance=2.0, use_ppm=False, total_norm=True)
    assert pytest.approx(1.0, rel=1e-7) == score


def test_no_matching_peaks():
    # Peaks far apart, no matches => similarity = 0
    spec1_mz = np.array([100.0, 110.0])
    spec1_int = np.array([1.0, 1.0])
    spec2_mz = np.array([200.0, 210.0])
    spec2_int = np.array([1.0, 1.0])
    score = compute_entropy(spec1_mz, spec1_int, spec2_mz, spec2_int,
                             tolerance=0.1, use_ppm=False, total_norm=True)
    assert pytest.approx(0.0, abs=1e-7) == score


def test_identical_spectra():
    # Identical spectra => similarity = 1
    spec1_mz = np.array([100.0, 200.0])
    spec1_int = np.array([1.0, 3.0])
    spec2_mz = np.array([100.0, 200.0])
    spec2_int = np.array([1.0, 3.0])
    score = compute_entropy(spec1_mz, spec1_int, spec2_mz, spec2_int,
                             tolerance=0.1, use_ppm=False, total_norm=True)
    assert pytest.approx(1.0, rel=1e-7) == score


def test_different_normalizations():
    # Test total sum = 1 vs max peak = 1 normalization
    spec1_mz = np.array([100.0])
    spec1_int = np.array([0.01])
    spec2_mz = np.array([100.0, 200.0])
    spec2_int = np.array([100.0, 100.0])
    score = compute_entropy(
        spec1_mz, spec1_int, spec2_mz, spec2_int,
        tolerance=0.1, use_ppm=False, total_norm=False)
    assert score == 0.5

    score = compute_entropy(
        spec1_mz, spec1_int, spec2_mz, spec2_int,
        tolerance=0.1, use_ppm=False, total_norm=True)
    assert score > 0.6
