import numpy
import pytest
from matchms import Spectrum
from .builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize("metadata", [
    {},
    {"precursor_mz": 150.5},
    {"precursor_mz": 150.5, "test": "someInfo"},
])
def test_spectrum_builder_only_metadata(metadata):
    spectrum_1 = SpectrumBuilder().with_metadata(metadata).build()

    spectrum_2 = Spectrum(numpy.array([], dtype="float"),
                          numpy.array([], dtype="float"),
                          metadata)
    assert spectrum_1 == spectrum_2, "Spectra should be identical!"


@pytest.mark.parametrize("mz, intensities, metadata", [
    [[10.1, 20.2], [0.5, 1], {}],
    [[10.1, 20.2], [0.5, 1], {"precursor_mz" : 150.5}],
])
def test_spectrum_builder_all(mz, intensities, metadata):
    spectrum_1 = SpectrumBuilder().with_mz(mz).with_intensities(intensities).with_metadata(metadata).build()

    spectrum_2 = Spectrum(numpy.array(mz, dtype="float"),
                          numpy.array(intensities, dtype="float"),
                          metadata)
    assert spectrum_1 == spectrum_2, "Spectra should be identical!"
