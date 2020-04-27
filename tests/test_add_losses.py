from matchms import Spectrum
from matchms.filtering import add_losses
import numpy


def test_add_losses():
    spectrum_in = Spectrum(mz=numpy.array([100, 150, 200, 300], dtype="float"),
                           intensities=numpy.array([700, 200, 100, 1000], dtype="float"),
                           metadata={"precursor_mz": 445.0})

    spectrum = add_losses(spectrum_in)

    assert numpy.allclose(spectrum.losses.mz, numpy.array([145, 245, 295, 345], "float"))
