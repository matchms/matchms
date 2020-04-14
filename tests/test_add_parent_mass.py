from matchms import Spectrum
from matchms.filtering import add_parent_mass
import numpy as np


def test_add_parent_mass():
    """Test if parent mass is correctly derived."""
    mz = np.array([10, 20, 30, 40], dtype='float')
    intensities = np.array([0, 1, 10, 100], dtype='float')
    metadata = {"pepmass": 444.0,
                "charge": -1}
    spectrum_in = Spectrum(mz=mz,
                           intensities=intensities,
                           metadata=metadata)

    spectrum = add_parent_mass(spectrum_in)

    assert np.abs(spectrum.metadata["parent_mass"] - 445.0) < .001, "Expected parent mass of about 445.0."


if __name__ == '__main__':
    test_add_parent_mass()
