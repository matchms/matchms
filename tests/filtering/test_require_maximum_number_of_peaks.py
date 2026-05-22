import numpy as np
import pytest
from matchms import SpectraCollection
from matchms.filtering.peak_processing.require_maximum_number_of_peaks import (
    require_maximum_number_of_peaks,
)
from matchms.Spectrum import Spectrum
from tests.run_spectrum_and_collection import run_filter_as_spectrum_or_collection


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_require_number_of_peaks_above_maximum_is_removed(as_collection):
    mz = np.array([10, 20, 30, 40], dtype="float")
    intensities = np.array([0, 1, 10, 100], dtype="float")
    spectrum_in = Spectrum(mz, intensities)

    spectrum = run_filter_as_spectrum_or_collection(
        require_maximum_number_of_peaks,
        spectrum_in,
        as_collection,
        maximum_number_of_fragments=3,
    )

    assert spectrum is None


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
def test_require_number_of_peaks_below_maximum_not_removed(as_collection):
    mz = np.array([10, 20, 30, 40], dtype="float")
    intensities = np.array([0, 1, 10, 100], dtype="float")
    spectrum_in = Spectrum(mz, intensities)

    spectrum = run_filter_as_spectrum_or_collection(
        require_maximum_number_of_peaks,
        spectrum_in,
        as_collection,
        maximum_number_of_fragments=10,
    )

    assert spectrum is not None
    assert len(spectrum.peaks) == 4
    np.testing.assert_allclose(spectrum.peaks.mz, mz, atol=1e-6)
    np.testing.assert_array_equal(spectrum.peaks.intensities, intensities)


def test_require_maximum_number_of_peaks_collection_drops_only_failing_rows():
    spectra = [
        Spectrum(
            mz=np.array([10, 20], dtype="float"),
            intensities=np.array([1, 2], dtype="float"),
            metadata={"id": "keep"},
        ),
        Spectrum(
            mz=np.array([10, 20, 30, 40], dtype="float"),
            intensities=np.array([1, 2, 3, 4], dtype="float"),
            metadata={"id": "drop"},
        ),
    ]
    collection = SpectraCollection(spectra)

    filtered = require_maximum_number_of_peaks(
        collection,
        maximum_number_of_fragments=2,
    )

    assert isinstance(filtered, SpectraCollection)
    assert len(filtered) == 1
    assert filtered.metadata.loc[0, "id"] == "keep"
    assert len(filtered[0].peaks) == 2


def test_require_maximum_number_of_peaks_none_input():
    assert require_maximum_number_of_peaks(None) is None
