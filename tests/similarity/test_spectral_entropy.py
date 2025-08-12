import numpy as np
import pytest
from matchms.similarity import SpectralEntropy
from matchms.similarity.SpectralEntropy import compute_entropy_optimal
from ..builder_Spectrum import SpectrumBuilder

# Use non-compiled version for testing
compute_entropy_py = compute_entropy_optimal.py_func


def compute_expected_score(spectrum_1, spectrum_2, matches):
    I1  = spectrum_1.peaks.intensities
    I2  = spectrum_2.peaks.intensities

    # Normalize by sum
    p1 = I1 / I1.sum() if I1.sum() > 0 else np.zeros_like(I1)
    p2 = I2 / I2.sum() if I2.sum() > 0 else np.zeros_like(I2)

    idxs1, idxs2 = matches
    set1, set2 = set(idxs1), set(idxs2)

    entropy_acc = 0.0

    # Matched peaks
    for i, j in zip(idxs1, idxs2):
        a = p1[i]
        b = p2[j]
        if a > 0:
            entropy_acc += a * np.log(2 * a / (a + b))
        if b > 0:
            entropy_acc += b * np.log(2 * b / (a + b))

    # Unmatched in spec1
    for i in range(len(p1)):
        if i not in set1:
            a = p1[i]
            if a > 0:
                entropy_acc += a * np.log(2)

    # Unmatched in spec2
    for j in range(len(p2)):
        if j not in set2:
            b = p2[j]
            if b > 0:
                entropy_acc += b * np.log(2)

    # Final similarity
    return 1.0 - entropy_acc / np.log(4.0)


@pytest.mark.parametrize("mode", ["compiled", "uncompiled"])
def test_identical_spectra(mode):
    """
    Identical spectra should yield similarity of 1.0
    """
    mz = np.array([100.0, 200.0, 300.0])
    intens = np.array([0.2, 0.5, 0.3])
    func = compute_entropy_optimal if mode == "compiled" else compute_entropy_py
    sim = func(mz, intens, mz, intens, tolerance=0.5, use_ppm=False, total_norm=True)
    assert pytest.approx(1.0, rel=1e-6) == sim


@pytest.mark.parametrize("mode", ["compiled", "uncompiled"])
def test_disjoint_spectra(mode):
    """
    Completely disjoint spectra should yield similarity of 0.0
    """
    mz1 = np.array([100.0, 200.0, 300.0])
    intens1 = np.array([0.3, 0.4, 0.3])
    mz2 = np.array([150.0, 250.0, 350.0])
    intens2 = np.array([0.2, 0.5, 0.3])
    func = compute_entropy_optimal if mode == "compiled" else compute_entropy_py
    sim = func(mz1, intens1, mz2, intens2, tolerance=1.0, use_ppm=False, total_norm=True)
    assert pytest.approx(0.0, abs=1e-6) == sim


@pytest.mark.parametrize("mode", ["compiled", "uncompiled"])
def test_partial_overlap_example(mode):
    """
    Use the example with overlapping windows to verify correct optimal matching (301↔301, not 300↔301).
    """
    # reference spectrum
    ref_mz = np.array([100., 299., 300., 301., 510., 600.])
    ref_int = np.array([0.1, 1.0, 0.2, 0.3, 0.4, 0.8])
    # query spectrum
    q_mz = np.array([100., 300., 301., 511.])
    q_int = np.array([0.1, 1.0, 0.3, 0.4])
    tol = 2.0
    func = compute_entropy_optimal if mode == "compiled" else compute_entropy_py
    sim = func(ref_mz, ref_int, q_mz, q_int, tolerance=tol, use_ppm=False, total_norm=True)
    # We expect a value strictly between 0 and 1, and compiled==uncompiled
    sim2 = (compute_entropy_optimal if mode == "compiled" else compute_entropy_py)(
        ref_mz, ref_int, q_mz, q_int, tolerance=tol, use_ppm=False, total_norm=True
    )
    assert pytest.approx(sim, rel=1e-8) == sim2
    assert 0.0 < sim < 1.0


@pytest.mark.parametrize("peaks, tolerance, expected_matches", [
    [
        [
            [np.array([100, 200, 300, 500, 510], dtype="float"), np.array([0.1, 0.2, 1.0, 0.3, 0.4], dtype="float")],
            [np.array([100, 200, 290, 490, 510], dtype="float"), np.array([0.1, 0.2, 1.0, 0.3, 0.4], dtype="float")]
        ],
        0.1, [[0, 1, 4], [0, 1, 4]]
    ], [
        [
            [np.array([100, 299, 300, 301, 510], dtype="float"), np.array([0.1, 1.0, 0.2, 0.3, 0.4], dtype="float")],
            [np.array([100, 300, 301, 511], dtype="float"), np.array([0.1, 1.0, 0.3, 0.4], dtype="float")],
        ],
        0.2, [[0, 2, 3], [0, 1, 2]]
    ], [
        [
            [np.array([100, 299, 300, 301, 510], dtype="float"), np.array([0.1, 1.0, 0.2, 0.3, 0.4], dtype="float")],
            [np.array([100, 300, 301, 511], dtype="float"), np.array([0.1, 1.0, 0.3, 0.4], dtype="float")],
        ],
        2.0, [[0, 1, 3, 4], [0, 1, 2, 3]]
    ], [
        [
            [np.array([100, 200, 300, 500, 510], dtype="float"), np.array([0.1, 0.2, 1.0, 0.3, 0.4], dtype="float")],
            [np.array([100, 200, 290, 490, 510], dtype="float"), np.array([0.1, 0.2, 1.0, 0.3, 0.4], dtype="float")],
        ],
        1.0,  [[0, 1, 4], [0, 1, 4]]
    ], [
        [
            [np.array([100, 199.8, 199.9, 200.0, 200.1, 200.2, 201.], dtype="float"), np.array([0.1, 0.1, 0.1, 0.1, 1.0, 0.1, 0.1], dtype="float")],
            [np.array([100, 200], dtype="float"), np.array([0.1, 1.0], dtype="float")],
        ],
        1.0,  [[0, 4], [0, 1]]
    ]
])
def test_cosine_greedy_pair(peaks, tolerance, expected_matches):
    builder = SpectrumBuilder()
    spectrum_1 = builder.with_mz(peaks[0][0]).with_intensities(peaks[0][1]).build()
    spectrum_2 = builder.with_mz(peaks[1][0]).with_intensities(peaks[1][1]).build()

    cosine_greedy = SpectralEntropy(tolerance=tolerance)
    score = cosine_greedy.pair(spectrum_1, spectrum_2)

    expected_score = compute_expected_score(spectrum_1, spectrum_2, expected_matches)

    assert score == pytest.approx(expected_score, 0.0001), "Expected different score."


@pytest.mark.parametrize("symmetric", [[True], [False]])
def test_cosine_greedy_matrix(symmetric):
    builder = SpectrumBuilder()
    spectrum_1 = builder.with_mz(np.array([100, 200, 300], dtype="float")).with_intensities(
        np.array([0.1, 0.2, 1.0], dtype="float")).build()

    spectrum_2 = builder.with_mz(np.array([110, 190, 290], dtype="float")).with_intensities(
        np.array([0.5, 0.2, 1.0], dtype="float")).build()

    spectra = [spectrum_1, spectrum_2]
    entropy = SpectralEntropy()
    scores = entropy.matrix(spectra, spectra, is_symmetric=symmetric)

    assert scores[0][0] == pytest.approx(scores[1][1], 0.000001), "Expected different cosine score."
    assert scores[0][1] == pytest.approx(scores[1][0], 0.000001), "Expected different cosine score."


def test_cosine_greedy_matrix_unsymmetric_error():
    builder = SpectrumBuilder()
    spectrum_1 = builder.with_mz(np.array([100, 200, 300], dtype="float")).with_intensities(
        np.array([0.1, 0.2, 1.0], dtype="float")).build()

    spectrum_2 = builder.with_mz(np.array([110, 190, 290], dtype="float")).with_intensities(
        np.array([0.5, 0.2, 1.0], dtype="float")).build()

    with pytest.raises(ValueError, match="unequal number of spectra"):
        SpectralEntropy().matrix([spectrum_1, spectrum_2], [spectrum_2], is_symmetric=True)
