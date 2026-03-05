import numpy as np
import pytest
from matchms import Spectrum
from matchms.filtering import normalize_intensities
from matchms.similarity import CosineHungarian, ModifiedCosineGreedy, ModifiedCosineHungarian
from ..builder_Spectrum import SpectrumBuilder


# ruff: noqa: E501

def compute_expected_score(spectrum_1, spectrum_2, matches):
    spec1 = spectrum_1.peaks.intensities
    spec2 = spectrum_2.peaks.intensities
    peak_pairs_multiplied = 0
    for match in matches:
        peak_pairs_multiplied += spec1[match[0]] * spec2[match[1]]
    return peak_pairs_multiplied / np.sqrt(np.sum(spec1 ** 2) * np.sum(spec2 ** 2))


def test_modified_cosine_hungarian_without_precursor_mz():
    """Test without precursor-m/z. Should raise assertion error."""
    mz = np.array([100, 150, 200, 300, 500, 510, 1100], dtype="float")
    intensities = np.array([700, 200, 100, 1000, 200, 5, 500], dtype="float")
    builder = SpectrumBuilder()
    spectrum_1 = builder.with_mz(mz).with_intensities(intensities).build()
    spectrum_2 = builder.with_mz(mz).with_intensities(intensities).build()

    norm_spectrum_1 = normalize_intensities(spectrum_1)
    norm_spectrum_2 = normalize_intensities(spectrum_2)
    modified_cosine = ModifiedCosineHungarian()

    with pytest.raises(AssertionError) as msg:
        modified_cosine.pair(norm_spectrum_1, norm_spectrum_2)

    expected_message = "Precursor_mz missing. Apply 'add_precursor_mz' filter first."
    assert str(msg.value) == expected_message


@pytest.mark.parametrize("peaks, tolerance, masses, expected_matches", [
    [
        [
            [np.array([100, 150, 200, 300, 500, 510, 1100], dtype="float"), np.array([700, 200, 100, 1000, 200, 5, 500], dtype="float")],
            [np.array([55, 105, 205, 304.5, 494.5, 515.5, 1045], dtype="float"), np.array([700, 200, 100, 1000, 200, 5, 500], dtype="float")]
        ],
        None, (1000.0, 1005.0), [(0, 1), (2, 2)]
    ], [
        [
            [np.array([100, 299, 300, 301, 500, 510], dtype="float"), np.array([10, 500, 100, 200, 20, 100], dtype="float")],
            [np.array([105, 305, 306, 505, 517], dtype="float"), np.array([10, 500, 100, 20, 100], dtype="float")],
        ],
        2.0, (1000.0, 1005.0), [(0, 0), (1, 1), (3, 2), (4, 3), (5, 4)]
    ], [
        [
            [np.array([100, 110, 200, 300, 400, 500, 600], dtype="float"), np.array([100, 50, 1, 80, 1, 1, 50], dtype="float")],
            [np.array([110, 200, 300, 310, 700, 800], dtype="float"), np.array([100, 1, 90, 90, 1, 100], dtype="float")],
        ],
        None, (1000.0, 1010.0), [(0, 0), (2, 1), (3, 3)]
    ], [
        [
            [np.array([100, 200, 300], dtype="float"), np.array([10, 10, 500], dtype="float")],
            [np.array([120, 220, 320], dtype="float"), np.array([10, 10, 500], dtype="float")],
        ],
        None, (1000.0, 1010.0), []
    ]
])
def test_modified_cosine_hungarian_with_mass_shift(peaks, tolerance, masses, expected_matches):
    """Test modified cosine Hungarian on two spectra with mass shift."""
    builder = SpectrumBuilder()
    spectrum_1 = builder.with_mz(peaks[0][0]).with_intensities(peaks[0][1]).with_metadata(metadata={"precursor_mz": masses[0]}).build()
    spectrum_2 = builder.with_mz(peaks[1][0]).with_intensities(peaks[1][1]).with_metadata(metadata={"precursor_mz": masses[1]}).build()

    norm_spectrum_1 = normalize_intensities(spectrum_1)
    norm_spectrum_2 = normalize_intensities(spectrum_2)
    if tolerance is None:
        modified_cosine = ModifiedCosineHungarian()
    else:
        modified_cosine = ModifiedCosineHungarian(tolerance=tolerance)

    score = modified_cosine.pair(norm_spectrum_1, norm_spectrum_2)
    expected_score = compute_expected_score(norm_spectrum_1, norm_spectrum_2, expected_matches)
    assert score["score"] == pytest.approx(expected_score, 0.0001), "Expected different cosine score."
    assert score["matches"] == len(expected_matches), "Expected differnt number of matching peaks."


def test_modified_cosine_hungarian_order_of_input_spectra():
    """Test modified cosine Hungarian on two spectra in changing order."""
    spectrum_1 = Spectrum(mz=np.array([100, 150, 200, 300, 500, 510, 1100], dtype="float"),
                          intensities=np.array([700, 200, 100, 1000, 200, 5, 500], dtype="float"),
                          metadata={"precursor_mz": 1000.0})

    spectrum_2 = Spectrum(mz=np.array([55, 105, 205, 304.5, 494.5, 515.5, 1045], dtype="float"),
                          intensities=np.array([700, 200, 100, 1000, 200, 5, 500], dtype="float"),
                          metadata={"precursor_mz": 1005.0})

    norm_spectrum_1 = normalize_intensities(spectrum_1)
    norm_spectrum_2 = normalize_intensities(spectrum_2)
    modified_cosine = ModifiedCosineHungarian(tolerance=2.0)
    score_1_2 = modified_cosine.pair(norm_spectrum_1, norm_spectrum_2)
    score_2_1 = modified_cosine.pair(norm_spectrum_2, norm_spectrum_1)

    assert score_1_2["score"] == score_2_1["score"], "Expected that the order of the arguments would not matter."
    assert score_1_2 == score_2_1, "Expected that the order of the arguments would not matter."


def test_modified_cosine_hungarian_precursor_mz_as_invalid_string():
    """Test modified cosine Hungarian on two spectra with precursor_mz given as string."""
    spectrum_1 = Spectrum(mz=np.array([100, 200, 300], dtype="float"),
                          intensities=np.array([10, 10, 500], dtype="float"),
                          metadata={"precursor_mz": 1000.0})

    spectrum_2 = Spectrum(mz=np.array([120, 220, 320], dtype="float"),
                          intensities=np.array([10, 10, 500], dtype="float"),
                          metadata={"precursor_mz": "mz 1005.0"})

    norm_spectrum_1 = normalize_intensities(spectrum_1)
    norm_spectrum_2 = normalize_intensities(spectrum_2)
    modified_cosine = ModifiedCosineHungarian(tolerance=1.0)
    with pytest.raises(AssertionError) as msg:
        _ = modified_cosine.pair(norm_spectrum_1, norm_spectrum_2)

    expected_message = "Precursor_mz missing. Apply 'add_precursor_mz' filter first."
    assert str(msg.value) == expected_message


def test_modified_cosine_hungarian_precursor_mz_as_string(caplog):
    """Test modified cosine Hungarian on two spectra with precursor_mz given as string."""
    spectrum_1 = Spectrum(mz=np.array([100, 200, 300], dtype="float"),
                          intensities=np.array([10, 10, 500], dtype="float"),
                          metadata={"precursor_mz": 1000.0},
                          metadata_harmonization=False)

    spectrum_2 = Spectrum(mz=np.array([120, 220, 320], dtype="float"),
                          intensities=np.array([10, 10, 500], dtype="float"),
                          metadata={"precursor_mz": "1005.0"},
                          metadata_harmonization=False)

    norm_spectrum_1 = normalize_intensities(spectrum_1)
    norm_spectrum_2 = normalize_intensities(spectrum_2)
    modified_cosine = ModifiedCosineHungarian(tolerance=1.0)
    score = modified_cosine.pair(norm_spectrum_1, norm_spectrum_2)

    assert score["score"] == pytest.approx(0.0, 1e-5), "Expected different modified cosine score."
    assert score["matches"] == 0, "Expected 0 matching peaks."
    expected_msg = "Precursor_mz must be int or float. Apply 'add_precursor_mz' filter first."
    assert expected_msg in caplog.text, "Expected different log message"


def test_modified_cosine_hungarian_reduces_to_cosine_hungarian_for_zero_shift():
    """With equal precursor m/z, exact modified cosine should equal exact cosine Hungarian."""
    spectrum_1 = Spectrum(mz=np.array([100.005, 100.016], dtype="float"),
                          intensities=np.array([1.0, 0.9], dtype="float"),
                          metadata={"precursor_mz": 500.0})

    spectrum_2 = Spectrum(mz=np.array([100.005, 100.01], dtype="float"),
                          intensities=np.array([0.9, 1.0], dtype="float"),
                          metadata={"precursor_mz": 500.0})

    modified_cosine_hungarian = ModifiedCosineHungarian(tolerance=0.01)
    cosine_hungarian = CosineHungarian(tolerance=0.01)

    modified_score = modified_cosine_hungarian.pair(spectrum_1, spectrum_2)
    cosine_score = cosine_hungarian.pair(spectrum_1, spectrum_2)

    assert modified_score["score"] == pytest.approx(cosine_score["score"], 1e-12)
    assert modified_score["matches"] == cosine_score["matches"]


def test_modified_cosine_hungarian_outperforms_approximation_on_conflicting_matches():
    """Exact matching should outperform greedy matching when candidate edges conflict."""
    spectrum_1 = Spectrum(mz=np.array([100.005, 100.016], dtype="float"),
                          intensities=np.array([1.0, 0.9], dtype="float"),
                          metadata={"precursor_mz": 500.0})

    spectrum_2 = Spectrum(mz=np.array([100.005, 100.01], dtype="float"),
                          intensities=np.array([0.9, 1.0], dtype="float"),
                          metadata={"precursor_mz": 500.0})

    exact = ModifiedCosineHungarian(tolerance=0.01).pair(spectrum_1, spectrum_2)
    approx = ModifiedCosineGreedy(tolerance=0.01).pair(spectrum_1, spectrum_2)

    assert exact["score"] == pytest.approx(0.994475, 0.0001)
    assert exact["matches"] == 2
    assert approx["matches"] == 1
    assert exact["score"] > approx["score"]
