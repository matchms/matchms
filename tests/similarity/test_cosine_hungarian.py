import numpy as np
import pytest
from matchms import Spectrum
from matchms.reference_spectra import cocaine, glucose, hydroxy_cholesterol, phenylalanine
from matchms.similarity import CosineHungarian


def test_cosine_hungarian_cocaine_glucose():
    """Compare output cosine score with own calculation on reference spectra."""
    glucose_spectrum = glucose()
    cocaine_spectrum = cocaine()

    cosine_hungarian = CosineHungarian(tolerance=0.1, mz_power=1.0, intensity_power=1.0)

    (similarity, shared_peaks) = cosine_hungarian.pair(glucose_spectrum, cocaine_spectrum)[()]
    assert similarity == 0.0
    assert shared_peaks == 0

    cosine_hungarian = CosineHungarian(tolerance=5.0, mz_power=0.0, intensity_power=1.0)

    (similarity, shared_peaks) = cosine_hungarian.pair(glucose_spectrum, cocaine_spectrum)[()]
    assert similarity == 0.453757948440651, "Expected different cosine score: {}".format(similarity)
    assert shared_peaks == 5, "Expected different number of matching peaks: {}".format(
        shared_peaks
    )

def test_cosine_hungarian_phenylalanine_hydroxy_cholesterol():
    """Compare output cosine score with own calculation on reference spectra."""
    phenylalanine_spectrum = phenylalanine()
    hydroxy_cholesterol_spectrum = hydroxy_cholesterol()

    cosine_hungarian = CosineHungarian(tolerance=0.1, mz_power=1.0, intensity_power=1.0)

    (similarity, shared_peaks) = cosine_hungarian.pair(phenylalanine_spectrum, hydroxy_cholesterol_spectrum)[()]
    assert similarity < 0.0001
    assert shared_peaks == 3

    cosine_hungarian = CosineHungarian(tolerance=5.0, mz_power=0.0, intensity_power=1.0)

    (similarity, shared_peaks) = cosine_hungarian.pair(phenylalanine_spectrum, hydroxy_cholesterol_spectrum)[()]
    assert similarity < 0.01, "Expected different cosine score: {}".format(similarity)
    assert shared_peaks == 19, "Expected different number of matching peaks: {}".format(
        shared_peaks
    )


def test_cosine_hungarian_without_parameters():
    """Compare output cosine score with own calculation on simple dummy spectra."""
    spectrum_1 = Spectrum(
        mz=np.array([100, 200, 300, 500, 510], dtype="float"),
        intensities=np.array([0.1, 0.2, 1.0, 0.3, 0.4], dtype="float"),
    )

    spectrum_2 = Spectrum(
        mz=np.array([100, 200, 290, 490, 510], dtype="float"),
        intensities=np.array([0.1, 0.2, 1.0, 0.3, 0.4], dtype="float"),
    )
    cosine_hungarian = CosineHungarian()
    score = cosine_hungarian.pair(spectrum_1, spectrum_2)

    # Derive expected cosine score
    expected_matches = [
        0,
        1,
        4,
    ]  # Those peaks have matching mz values (within given tolerance)
    multiply_matching_intensities = (
        spectrum_1.peaks.intensities[expected_matches]
        * spectrum_2.peaks.intensities[expected_matches]
    )
    denominator = np.sqrt((spectrum_1.peaks.intensities**2).sum()) * np.sqrt(
        (spectrum_2.peaks.intensities**2).sum()
    )
    expected_score = multiply_matching_intensities.sum() / denominator

    assert score["score"] == pytest.approx(
        expected_score, 0.0001
    ), "Expected different cosine score."
    assert score["matches"] == len(
        expected_matches
    ), "Expected different number of matching peaks."


def test_cosine_hungarian_matrix_without_parameters():
    """Compare output cosine score with expected scores."""
    spectrum_1 = Spectrum(
        mz=np.array([100, 200, 300, 500, 510], dtype="float"),
        intensities=np.array([0.1, 0.2, 1.0, 0.3, 0.4], dtype="float"),
    )

    spectrum_2 = Spectrum(
        mz=np.array([100, 200, 290, 490, 510], dtype="float"),
        intensities=np.array([0.1, 0.2, 1.0, 0.3, 0.4], dtype="float"),
    )
    cosine_hungarian = CosineHungarian()
    scores = cosine_hungarian.matrix([spectrum_1, spectrum_2], [spectrum_1, spectrum_2])

    assert (
        scores[0, 0]["score"] == scores[1, 1]["score"] == 1.0
    ), "Expected different cosine score."
    assert (
        scores[0, 0]["matches"] == scores[1, 1]["matches"] == 5
    ), "Expected different cosine matches."
    assert (
        scores[0, 1]["score"] == scores[1, 0]["score"] == pytest.approx(0.1615384, 1e-6)
    ), "Expected different cosine score."
    assert (
        scores[0, 1]["matches"] == scores[1, 0]["matches"] == 3
    ), "Expected different cosine matches."


def test_cosine_hungarian_with_tolerance_0_2():
    """Compare output cosine score for tolerance 0.2 with own calculation on simple dummy spectra."""
    spectrum_1 = Spectrum(
        mz=np.array([100, 299, 300, 301, 510], dtype="float"),
        intensities=np.array([0.1, 1.0, 0.2, 0.3, 0.4], dtype="float"),
    )

    spectrum_2 = Spectrum(
        mz=np.array([100, 300, 301, 511], dtype="float"),
        intensities=np.array([0.1, 1.0, 0.3, 0.4], dtype="float"),
    )
    cosine_hungarian = CosineHungarian(tolerance=0.2)
    score = cosine_hungarian.pair(spectrum_1, spectrum_2)

    # Derive expected cosine score
    expected_matches = [
        [0, 2, 3],
        [0, 1, 2],
    ]  # Those peaks have matching mz values (within given tolerance)
    multiply_matching_intensities = (
        spectrum_1.peaks.intensities[expected_matches[0]]
        * spectrum_2.peaks.intensities[expected_matches[1]]
    )
    denominator = np.sqrt((spectrum_1.peaks.intensities**2).sum()) * np.sqrt(
        (spectrum_2.peaks.intensities**2).sum()
    )
    expected_score = multiply_matching_intensities.sum() / denominator

    assert score["score"] == pytest.approx(
        expected_score, 0.0001
    ), "Expected different cosine score."
    assert score["matches"] == len(
        expected_matches[0]
    ), "Expected different number of matching peaks."


def test_cosine_hungarian_with_tolerance_2_0():
    """Compare output cosine score for tolerance 2.0 with own calculation on simple dummy spectra."""
    spectrum_1 = Spectrum(
        mz=np.array([100, 299, 300, 301, 510], dtype="float"),
        intensities=np.array([0.1, 1.0, 0.2, 0.3, 0.4], dtype="float"),
    )

    spectrum_2 = Spectrum(
        mz=np.array([100, 300, 301, 511], dtype="float"),
        intensities=np.array([0.1, 1.0, 0.3, 0.4], dtype="float"),
    )
    cosine_hungarian = CosineHungarian(tolerance=2.0)
    score = cosine_hungarian.pair(spectrum_1, spectrum_2)

    # Derive expected cosine score
    expected_matches = [
        [0, 1, 3, 4],
        [0, 1, 2, 3],
    ]  # Those peaks have matching mz values (within given tolerance)
    multiply_matching_intensities = (
        spectrum_1.peaks.intensities[expected_matches[0]]
        * spectrum_2.peaks.intensities[expected_matches[1]]
    )
    denominator = np.sqrt((spectrum_1.peaks.intensities**2).sum()) * np.sqrt(
        (spectrum_2.peaks.intensities**2).sum()
    )
    expected_score = multiply_matching_intensities.sum() / denominator

    assert score["score"] == pytest.approx(
        expected_score, 0.0001
    ), "Expected different cosine score."
    assert score["matches"] == len(
        expected_matches[0]
    ), "Expected different number of matching peaks."


def test_cosine_hungarian_order_of_arguments():
    """Compare cosine scores for A,B versus B,A, which should give the same score."""
    spectrum_1 = Spectrum(
        mz=np.array([100, 200, 299, 300, 301, 500, 510], dtype="float"),
        intensities=np.array([0.02, 0.02, 1.0, 0.2, 0.4, 0.04, 0.2], dtype="float"),
        metadata={},
    )

    spectrum_2 = Spectrum(
        mz=np.array([100, 200, 300, 301, 500, 512], dtype="float"),
        intensities=np.array([0.02, 0.02, 1.0, 0.2, 0.04, 0.2], dtype="float"),
        metadata={},
    )

    cosine_hungarian = CosineHungarian(tolerance=2.0)
    score_1_2 = cosine_hungarian.pair(spectrum_1, spectrum_2)
    score_2_1 = cosine_hungarian.pair(spectrum_2, spectrum_1)

    assert (
        score_1_2["score"] == score_2_1["score"]
    ), "Expected that the order of the arguments would not matter."
    assert (
        score_1_2 == score_2_1
    ), "Expected that the order of the arguments would not matter."


def test_cosine_hungarian_case_where_greedy_would_fail():
    """Test case that would fail for cosine greedy implementations."""
    spectrum_1 = Spectrum(
        mz=np.array([100.005, 100.016], dtype="float"),
        intensities=np.array([1.0, 0.9], dtype="float"),
        metadata={},
    )

    spectrum_2 = Spectrum(
        mz=np.array([100.005, 100.01], dtype="float"),
        intensities=np.array([0.9, 1.0], dtype="float"),
        metadata={},
    )

    cosine_hungarian = CosineHungarian(tolerance=0.01)
    score = cosine_hungarian.pair(spectrum_1, spectrum_2)
    assert score["score"] == pytest.approx(
        0.994475, 0.0001
    ), "Expected different cosine score."
    assert score["matches"] == 2, "Expected different number of matching peaks."


def test_cosine_hungarian_case_without_matches():
    """Test case for spectra without any matching peaks."""
    spectrum_1 = Spectrum(
        mz=np.array([100, 200], dtype="float"),
        intensities=np.array([1.0, 0.1], dtype="float"),
        metadata={},
    )

    spectrum_2 = Spectrum(
        mz=np.array([110, 210], dtype="float"),
        intensities=np.array([1.0, 0.1], dtype="float"),
        metadata={},
    )

    cosine_hungarian = CosineHungarian()
    score = cosine_hungarian.pair(spectrum_1, spectrum_2)
    assert score["score"] == 0.0, "Expected different cosine score."
    assert score["matches"] == 0, "Expected different number of matching peaks."


def test_cosine_hungarian_with_peak_powers():
    """Compare output cosine score with own calculation on simple dummy spectra.
    Here testing the options to raise peak intensities to given powers.
    """
    mz_power = 0.5
    intensity_power = 2.0
    spectrum_1 = Spectrum(
        mz=np.array([100, 200, 300, 500, 510], dtype="float"),
        intensities=np.array([0.1, 0.2, 1.0, 0.3, 0.4], dtype="float"),
    )

    spectrum_2 = Spectrum(
        mz=np.array([100, 200, 290, 490, 510], dtype="float"),
        intensities=np.array([0.1, 0.2, 1.0, 0.3, 0.4], dtype="float"),
    )
    cosine_hungarian = CosineHungarian(
        tolerance=1.0, mz_power=mz_power, intensity_power=intensity_power
    )
    score = cosine_hungarian.pair(spectrum_1, spectrum_2)

    # Derive expected cosine score
    matches = [0, 1, 4]  # Those peaks have matching mz values (within given tolerance)
    intensity1 = spectrum_1.peaks.intensities
    mz1 = spectrum_1.peaks.mz
    intensity2 = spectrum_2.peaks.intensities
    mz2 = spectrum_2.peaks.mz
    multiply_matching_intensities = (
        (mz1[matches] ** mz_power)
        * (intensity1[matches] ** intensity_power)
        * (mz2[matches] ** mz_power)
        * (intensity2[matches] ** intensity_power)
    )
    denominator = np.sqrt(
        (((mz1**mz_power) * (intensity1**intensity_power)) ** 2).sum()
    ) * np.sqrt((((mz2**mz_power) * (intensity2**intensity_power)) ** 2).sum())
    expected_score = multiply_matching_intensities.sum() / denominator

    assert score["score"] == pytest.approx(
        expected_score, 0.0001
    ), "Expected different cosine score."
    assert score["matches"] == len(
        matches
    ), "Expected different number of matching peaks."


def test_cosine_hungarian_phantom_pair_regression():
    """Test that phantom pairs from the Hungarian assignment are excluded from match count.

    The Hungarian algorithm assigns min(rows, cols) pairs. When the optimal
    assignment prefers a strong match that leaves another peak with no real
    partner, the remaining assignment lands on a sentinel 1.0 cell (a "phantom
    pair"). Phantom pairs contribute 0 to the score but must not inflate the
    match count.

    Cost matrix for this test (tolerance=2.0):

        |  R1(100)  |  R2(103)
    ----|-----------|----------
    L1(100) |  0.9      |  1.0  (no pair: |100-103|=3 > 2)
    L2(102) |  0.0      |  0.5

    Hungarian picks L1->R2 (1.0) + L2->R1 (0.0) = total 1.0, which beats
    L1->R1 (0.9) + L2->R2 (0.5) = total 1.4. The L1->R2 assignment is a
    phantom. Old code reported matches=2; correct answer is matches=1.
    """
    spectrum_1 = Spectrum(
        mz=np.array([100, 102], dtype="float"),
        intensities=np.array([0.1, 1.0], dtype="float"),
    )
    spectrum_2 = Spectrum(
        mz=np.array([100, 103], dtype="float"),
        intensities=np.array([1.0, 0.5], dtype="float"),
    )

    cosine_hungarian = CosineHungarian(tolerance=2.0)
    score = cosine_hungarian.pair(spectrum_1, spectrum_2)

    assert score["matches"] == 1, (
        f"Expected 1 real match but got {score['matches']}. "
        "Phantom pairs from Hungarian assignment should be excluded."
    )
    # Score must reflect the real match (L2->R1, product=1.0) regardless of
    # phantom filtering — phantoms contribute 0 by construction.
    assert score["score"] > 0


def test_cosine_hungarian_single_peak_spectra():
    """Test with single-peak spectra that are within tolerance."""
    spectrum_1 = Spectrum(
        mz=np.array([100.0], dtype="float"),
        intensities=np.array([1.0], dtype="float"),
    )
    spectrum_2 = Spectrum(
        mz=np.array([100.05], dtype="float"),
        intensities=np.array([0.8], dtype="float"),
    )

    cosine_hungarian = CosineHungarian(tolerance=0.1)
    score = cosine_hungarian.pair(spectrum_1, spectrum_2)

    assert score["matches"] == 1, "Expected exactly 1 match for single-peak spectra."
    # Cosine similarity normalises: (1.0*0.8) / (sqrt(1.0^2) * sqrt(0.8^2)) = 1.0
    assert score["score"] == pytest.approx(1.0, 0.0001), "Expected perfect cosine score for single matching pair."


def test_cosine_hungarian_all_peaks_matching():
    """Test where all peaks match (fully connected), ensuring no phantoms appear."""
    spectrum_1 = Spectrum(
        mz=np.array([100, 200, 300], dtype="float"),
        intensities=np.array([0.5, 0.8, 1.0], dtype="float"),
    )
    spectrum_2 = Spectrum(
        mz=np.array([100, 200, 300], dtype="float"),
        intensities=np.array([0.5, 0.8, 1.0], dtype="float"),
    )

    cosine_hungarian = CosineHungarian(tolerance=0.1)
    score = cosine_hungarian.pair(spectrum_1, spectrum_2)

    assert score["matches"] == 3, "Expected all 3 peaks to match."
    assert score["score"] == pytest.approx(1.0, 0.0001), "Expected perfect cosine score."
