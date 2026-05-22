import numpy as np
import pytest
from matchms import SpectraCollection
from matchms.filtering import require_minimum_number_of_high_peaks
from tests.builder_Spectrum import SpectrumBuilder
from tests.run_spectrum_and_collection import run_filter_as_spectrum_or_collection


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
@pytest.mark.parametrize(
    "peaks, no_peaks, intensity_percent, expected_is_none",
    [
        [
            [
                np.array([10, 20, 30, 40], dtype="float"),
                np.array([0, 1, 10, 100], dtype="float"),
            ],
            2,
            2,
            False,
        ],
        [
            [
                np.array([10, 20, 30, 40], dtype="float"),
                np.array([0, 1, 10, 100], dtype="float"),
            ],
            5,
            2,
            True,
        ],
        [
            [
                np.array([10, 20, 30, 40, 50, 60, 70], dtype="float"),
                np.array([0, 1, 10, 25, 50, 75, 100], dtype="float"),
            ],
            2,
            10,
            False,
        ],
    ],
)
def test_require_minimum_number_of_high_peaks(
    peaks,
    no_peaks,
    intensity_percent,
    expected_is_none,
    as_collection,
):
    spectrum_in = (
        SpectrumBuilder()
        .with_mz(peaks[0])
        .with_intensities(peaks[1])
        .build()
    )

    spectrum = run_filter_as_spectrum_or_collection(
        require_minimum_number_of_high_peaks,
        spectrum_in,
        as_collection,
        no_peaks=no_peaks,
        intensity_percent=intensity_percent,
    )

    if expected_is_none:
        assert spectrum is None
    else:
        assert spectrum is not None
        np.testing.assert_allclose(spectrum.peaks.mz, peaks[0], atol=1e-6)
        np.testing.assert_array_equal(spectrum.peaks.intensities, peaks[1])


def test_require_minimum_number_of_high_peaks_collection_drops_only_failing_rows():
    spectra = [
        SpectrumBuilder()
        .with_metadata({"id": "keep"})
        .with_mz(np.array([10, 20, 30, 40], dtype="float"))
        .with_intensities(np.array([0, 1, 10, 100], dtype="float"))
        .build(),
        SpectrumBuilder()
        .with_metadata({"id": "drop"})
        .with_mz(np.array([10, 20, 30, 40], dtype="float"))
        .with_intensities(np.array([0, 0.1, 0.2, 100], dtype="float"))
        .build(),
    ]
    collection = SpectraCollection(spectra)

    filtered = require_minimum_number_of_high_peaks(
        collection,
        no_peaks=2,
        intensity_percent=2,
    )

    assert isinstance(filtered, SpectraCollection)
    assert len(filtered) == 1
    assert filtered.metadata.loc[0, "id"] == "keep"


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_require_minimum_number_of_high_peaks_rejects_invalid_no_peaks(as_collection):
    spectrum_in = (
        SpectrumBuilder()
        .with_mz(np.array([10, 20, 30, 40], dtype="float"))
        .with_intensities(np.array([0, 1, 10, 100], dtype="float"))
        .build()
    )

    with pytest.raises(ValueError, match="no_peaks must be a positive nonzero integer"):
        run_filter_as_spectrum_or_collection(
            require_minimum_number_of_high_peaks,
            spectrum_in,
            as_collection,
            no_peaks=0,
        )


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
@pytest.mark.parametrize("intensity_percent", [-1, 101])
def test_require_minimum_number_of_high_peaks_rejects_invalid_intensity_percent(
    intensity_percent,
    as_collection,
):
    spectrum_in = (
        SpectrumBuilder()
        .with_mz(np.array([10, 20, 30, 40], dtype="float"))
        .with_intensities(np.array([0, 1, 10, 100], dtype="float"))
        .build()
    )

    with pytest.raises(ValueError, match="intensity_percent must be a scalar between 0-100"):
        run_filter_as_spectrum_or_collection(
            require_minimum_number_of_high_peaks,
            spectrum_in,
            as_collection,
            intensity_percent=intensity_percent,
        )


def test_if_spectrum_is_cloned():
    """Test if filter is correctly cloning the input spectrum."""
    mz = np.array([10, 20, 30, 40], dtype="float")
    intensities = np.array([0, 1, 10, 100], dtype="float")
    spectrum_in = SpectrumBuilder().with_mz(mz).with_intensities(intensities).build()

    spectrum = require_minimum_number_of_high_peaks(spectrum_in, no_peaks=2)
    spectrum.set("testfield", "test")

    assert not spectrum_in.get("testfield"), "Expected input spectrum to remain unchanged."


def test_with_input_none():
    assert require_minimum_number_of_high_peaks(None) is None
