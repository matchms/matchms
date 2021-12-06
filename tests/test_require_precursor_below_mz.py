import numpy
import pytest
from matchms import Spectrum
from matchms.filtering import require_precursor_below_mz
from .builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize("metadata, max_mz, expected", [
    [{"precursor_mz": 60.}, 1000, SpectrumBuilder().with_metadata(
        {"precursor_mz": 60}).build()],
    [{"precursor_mz": 60.}, 50, None]
])
def test_require_precursor_below_mz(metadata, max_mz, expected):
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()
    spectrum = require_precursor_below_mz(spectrum_in, max_mz=max_mz)
    assert spectrum == expected


def test_if_spectrum_is_cloned():
    """Test if filter is correctly cloning the input spectrum."""
    spectrum_in = SpectrumBuilder().with_metadata({"precursor_mz": 1.}).build()

    spectrum = require_precursor_below_mz(spectrum_in)
    spectrum.set("testfield", "test")

    assert not spectrum_in.get("testfield"), "Expected input spectrum to remain unchanged."


def test_require_precursor_below_without_precursor_mz():
    """Test if correct assert error is raised for missing precursor-mz."""
    spectrum_in = SpectrumBuilder().build()

    with pytest.raises(AssertionError) as msg:
        _ = require_precursor_below_mz(spectrum_in)

    assert str(msg.value) == "Precursor mz absent.", "Expected different error message."


def test_require_precursor_below_with_wrong_precursor_mz():
    """Test if correct assert error is raised for precursor-mz as string."""
    spectrum_in = SpectrumBuilder().with_metadata({"precursor_mz": "445.0"}).build()

    with pytest.raises(AssertionError) as msg:
        _ = require_precursor_below_mz(spectrum_in)

    assert "Expected 'precursor_mz' to be a scalar number." in str(msg.value)


def test_with_input_none():
    """Test if input spectrum is None."""
    spectrum_in = None
    spectrum = require_precursor_below_mz(spectrum_in)
    assert spectrum is None
