import numpy
from matchms import Spectrum
from matchms.filtering import add_parent_mass


def test_add_parent_mass():
    """Test if parent mass is correctly derived."""
    mz = numpy.array([], dtype='float')
    intensities = numpy.array([], dtype='float')
    metadata = {"pepmass": (444.0, 10),
                "charge": -1}
    spectrum_in = Spectrum(mz=mz,
                           intensities=intensities,
                           metadata=metadata)

    spectrum = add_parent_mass(spectrum_in)

    assert numpy.abs(spectrum.get("parent_mass") - 445.0) < .01, "Expected parent mass of about 445.0."


def test_add_parent_mass_no_pepmass():
    """Test if parent mass is correctly derived if "pepmass" is not present."""
    mz = numpy.array([], dtype='float')
    intensities = numpy.array([], dtype='float')
    metadata = {"precursor_mz": 444.0,
                "charge": -1}
    spectrum_in = Spectrum(mz=mz,
                           intensities=intensities,
                           metadata=metadata)

    spectrum = add_parent_mass(spectrum_in)

    assert numpy.abs(spectrum.get("parent_mass") - 445.0) < .01, "Expected parent mass of about 445.0."


@pytest.mark.parametrize("adduct, expected", [("[M+2Na-H]+", 399.02884),
                                               ("[M+H+NH4]2+", 212.47945),
                                               ("[2M+FA-H]-", 843.001799)])
def test_add_parent_mass_using_adduct(adduct, expected):
    """Test if parent mass is correctly derived from adduct information."""
    mz = numpy.array([], dtype='float')
    intensities = numpy.array([], dtype='float')
    metadata = {"precursor_mz": 444.0,
                "adduct": adduct,
                "charge": +1}
    spectrum_in = Spectrum(mz=mz,
                           intensities=intensities,
                           metadata=metadata)

    spectrum = add_parent_mass(spectrum_in)

    assert numpy.allclose(spectrum.get("parent_mass"), expected, atol=1e-4), "Expected parent mass of about 339.0."


def test_empty_spectrum():
    spectrum_in = None
    spectrum = add_parent_mass(spectrum_in)

    assert spectrum is None, "Expected different handling of None spectrum."
