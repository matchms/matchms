import numpy as np
import pytest
from matchms.similarity import CosineGreedy
from .builder_Spectrum import SpectrumBuilder


def compute_expected_score(mz_power, intensity_power, spectrum_1, spectrum_2, matches):
    intensity1 = spectrum_1.peaks.intensities
    mz1 = spectrum_1.peaks.mz
    intensity2 = spectrum_2.peaks.intensities
    mz2 = spectrum_2.peaks.mz
    multiply_matching_intensities = (mz1[matches[0]] ** mz_power) * (intensity1[matches[0]] ** intensity_power) \
        * (mz2[matches[1]] ** mz_power) * (intensity2[matches[1]] ** intensity_power)
    denominator = numpy.sqrt((((mz1 ** mz_power) * (intensity1 ** intensity_power)) ** 2).sum()) \
        * numpy.sqrt((((mz2 ** mz_power) * (intensity2 ** intensity_power)) ** 2).sum())
    expected_score = multiply_matching_intensities.sum() / denominator
    return expected_score


@pytest.mark.parametrize("peaks, tolerance, mz_power, intensity_power, expected_matches", [
    [
        [
            [numpy.array([100, 200, 300, 500, 510], dtype="float"), numpy.array([0.1, 0.2, 1.0, 0.3, 0.4], dtype="float")],
            [numpy.array([100, 200, 290, 490, 510], dtype="float"), numpy.array([0.1, 0.2, 1.0, 0.3, 0.4], dtype="float")]
        ],
        0.1, 0.0, 1.0, [[0, 1, 4], [0, 1, 4]]
    ], [
        [
            [numpy.array([100, 299, 300, 301, 510], dtype="float"), numpy.array([0.1, 1.0, 0.2, 0.3, 0.4], dtype="float")],
            [numpy.array([100, 300, 301, 511], dtype="float"), numpy.array([0.1, 1.0, 0.3, 0.4], dtype="float")],
        ],
        0.2, 0.0, 1.0, [[0, 2, 3], [0, 1, 2]]
    ], [
        [
            [numpy.array([100, 299, 300, 301, 510], dtype="float"), numpy.array([0.1, 1.0, 0.2, 0.3, 0.4], dtype="float")],
            [numpy.array([100, 300, 301, 511], dtype="float"), numpy.array([0.1, 1.0, 0.3, 0.4], dtype="float")],
        ],
        2.0, 0.0, 1.0, [[0, 1, 3, 4], [0, 1, 2, 3]]
    ], [
        [
            [numpy.array([100, 200, 300, 500, 510], dtype="float"), numpy.array([0.1, 0.2, 1.0, 0.3, 0.4], dtype="float")],
            [numpy.array([100, 200, 290, 490, 510], dtype="float"), numpy.array([0.1, 0.2, 1.0, 0.3, 0.4], dtype="float")],
        ],
        1.0, 0.5, 2.0, [[0, 1, 4], [0, 1, 4]]
    ]
])
def test_cosine_greedy_pair(peaks, tolerance, mz_power, intensity_power, expected_matches):
    builder = SpectrumBuilder()
    spectrum_1 = builder.with_mz(peaks[0][0]).with_intensities(peaks[0][1]).build()
    spectrum_2 = builder.with_mz(peaks[1][0]).with_intensities(peaks[1][1]).build()

    cosine_greedy = CosineGreedy(tolerance=tolerance, mz_power=mz_power, intensity_power=intensity_power)
    score = cosine_greedy.pair(spectrum_1, spectrum_2)

    expected_score = compute_expected_score(mz_power, intensity_power, spectrum_1, spectrum_2, expected_matches)

    assert score["score"] == pytest.approx(expected_score, 0.0001), "Expected different cosine score."
    assert score["matches"] == len(expected_matches[0]), "Expected different number of matching peaks."


@pytest.mark.parametrize("symmetric", [[True], [False]])
def test_cosine_greedy_matrix(symmetric):
    builder = SpectrumBuilder()
    spectrum_1 = builder.with_mz(numpy.array([100, 200, 300], dtype="float")).with_intensities(
        numpy.array([0.1, 0.2, 1.0], dtype="float")).build()

    spectrum_2 = builder.with_mz(numpy.array([110, 190, 290], dtype="float")).with_intensities(
        numpy.array([0.5, 0.2, 1.0], dtype="float")).build()

    spectrums = [spectrum_1, spectrum_2]
    cosine_greedy = CosineGreedy()
    scores = cosine_greedy.matrix(spectrums, spectrums, is_symmetric=symmetric)

    assert scores[0][0][0] == pytest.approx(scores[1][1][0], 0.000001), "Expected different cosine score."
    assert scores[0][0]["score"] == pytest.approx(scores[1][1]["score"], 0.000001), \
        "Expected different cosine score."
    assert scores[0][1][0] == pytest.approx(scores[1][0][0], 0.000001), "Expected different cosine score."
    assert scores[0][1]["score"] == pytest.approx(scores[1][0]["score"], 0.000001), \
        "Expected different cosine score."
