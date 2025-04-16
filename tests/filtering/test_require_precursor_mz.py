import numpy as np
import pytest
from matchms.filtering.metadata_processing.require_precursor_mz import require_precursor_mz
from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize(
    "metadata, expected",
    [
        [{"precursor_mz": 60.0}, SpectrumBuilder().with_metadata({"precursor_mz": 60}).build()],
        [{"precursor_mz": 0.0}, None],
        [{"precursor_mz": -3.5}, None],
        [{}, None],
    ],
)
def test_require_precursor_mz(metadata, expected):
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    spectrum = require_precursor_mz(spectrum_in)

    assert spectrum == expected, "Expected no changes."


def test_if_spectrum_is_cloned():
    """Test if filter is correctly cloning the input spectrum."""
    spectrum_in = SpectrumBuilder().with_metadata({"precursor_mz": 100.0}).build()

    spectrum = require_precursor_mz(spectrum_in)
    spectrum.set("testfield", "test")

    assert not spectrum_in.get("testfield"), "Expected input spectrum to remain unchanged."


def test_require_precursor_mz_with_input_none():
    """Test if input spectrum is None."""
    spectrum_in = None
    spectrum = require_precursor_mz(spectrum_in)
    assert spectrum is None


@pytest.mark.parametrize("precursor_mz", [0, 9.0, -200])
def test_require_precursor_mz_fail_when_mz_too_small(precursor_mz):
    """Test if spectrum is None when precursor_mz <= minimum_accepted_mz"""
    spectrum_in = SpectrumBuilder().build()
    spectrum_in.set("precursor_mz", precursor_mz)

    spectrum = require_precursor_mz(spectrum_in, minimum_accepted_mz=10)
    assert spectrum is None, "Expected spectrum to be None."


def test_require_precursor_mz_fail_because_below_zero():
    """Test if spectrum is None when precursor_mz < 0"""
    mz = np.array([10, 20, 30, 40], dtype="float")
    intensities = np.array([0, 1, 10, 100], dtype="float")
    spectrum_in = SpectrumBuilder().with_mz(mz).with_intensities(intensities).build()
    spectrum_in.set("precursor_mz", -3.5)

    spectrum = require_precursor_mz(spectrum_in)

    assert spectrum is None, "Expected spectrum to be None."
