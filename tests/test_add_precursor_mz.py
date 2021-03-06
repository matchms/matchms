import numpy
from matchms import Spectrum
from matchms.filtering import add_precursor_mz


def test_add_precursor_mz():
    """Test if precursor_mz is correctly derived. Here nothing should change."""
    mz = numpy.array([], dtype='float')
    intensities = numpy.array([], dtype='float')
    metadata = {"precursor_mz": 444.0}
    spectrum_in = Spectrum(mz=mz,
                           intensities=intensities,
                           metadata=metadata)

    spectrum = add_precursor_mz(spectrum_in)

    assert spectrum.get("precursor_mz") == 444.0, "Expected different precursor_mz."


def test_add_precursor_mz_no_masses():
    """Test if no precursor_mz is handled correctly. Here nothing should change."""
    mz = numpy.array([], dtype='float')
    intensities = numpy.array([], dtype='float')
    metadata = {}
    spectrum_in = Spectrum(mz=mz,
                           intensities=intensities,
                           metadata=metadata)

    spectrum = add_precursor_mz(spectrum_in)

    assert spectrum.get("precursor_mz") is None, "Outcome should be None."


def test_add_precursor_mz_only_pepmass_present():
    """Test if precursor_mz is correctly derived if only pepmass is present."""
    mz = numpy.array([], dtype='float')
    intensities = numpy.array([], dtype='float')
    metadata = {"pepmass": (444.0, 10)}
    spectrum_in = Spectrum(mz=mz,
                           intensities=intensities,
                           metadata=metadata)

    spectrum = add_precursor_mz(spectrum_in)

    assert spectrum.get("precursor_mz") == 444.0, "Expected different precursor_mz."


def test_add_precursor_mz_no_precursor_mz():
    """Test if precursor_mz is correctly derived if "precursor_mz" is str."""
    mz = numpy.array([], dtype='float')
    intensities = numpy.array([], dtype='float')
    metadata = {"precursor_mz": "444.0"}
    spectrum_in = Spectrum(mz=mz,
                           intensities=intensities,
                           metadata=metadata)

    spectrum = add_precursor_mz(spectrum_in)

    assert spectrum.get("precursor_mz") == 444.0, "Expected different precursor_mz."


def test_empty_spectrum():
    spectrum_in = None
    spectrum = add_precursor_mz(spectrum_in)

    assert spectrum is None, "Expected different handling of None spectrum."
