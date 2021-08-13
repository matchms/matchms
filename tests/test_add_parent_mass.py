import numpy
import pytest
from matchms import Spectrum
from matchms.filtering import add_parent_mass


def test_add_parent_mass_pepmass_no_precursormz(capsys):
    """Test if correct exception is returned."""
    mz = numpy.array([], dtype='float')
    intensities = numpy.array([], dtype='float')
    metadata = {"pepmass": (444.0, 10),
                "charge": -1}
    spectrum_in = Spectrum(mz=mz,
                           intensities=intensities,
                           metadata=metadata)

    spectrum = add_parent_mass(spectrum_in)

    assert spectrum.get("parent_mass") is None, "Expected no parent mass"
    assert "Not sufficient spectrum metadata to derive parent mass." not in capsys.readouterr().out


def test_add_parent_mass_no_precursormz(capsys):
    """Test if correct exception is returned."""
    mz = numpy.array([], dtype='float')
    intensities = numpy.array([], dtype='float')
    metadata = {"charge": -1}
    spectrum_in = Spectrum(mz=mz,
                           intensities=intensities,
                           metadata=metadata)

    spectrum = add_parent_mass(spectrum_in)

    assert spectrum.get("parent_mass") is None, "Expected no parent mass"
    assert "Missing precursor m/z to derive parent mass." in capsys.readouterr().out


def test_add_parent_mass_precursormz_zero_charge(capsys):
    """Test if correct exception is returned."""
    mz = numpy.array([], dtype='float')
    intensities = numpy.array([], dtype='float')
    metadata = {"precursor_mz": 444.0,
                "charge": 0}
    spectrum_in = Spectrum(mz=mz,
                           intensities=intensities,
                           metadata=metadata)

    spectrum = add_parent_mass(spectrum_in)

    assert spectrum.get("parent_mass") is None, "Expected no parent mass"
    assert "Not sufficient spectrum metadata to derive parent mass." in capsys.readouterr().out


def test_add_parent_mass_precursormz(capsys):
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
    assert isinstance(spectrum.get("parent_mass"), float), "Expected parent mass to be float."
    assert "Not sufficient spectrum metadata to derive parent mass." not in capsys.readouterr().out


@pytest.mark.parametrize("adduct, expected", [("[M+2Na-H]+", 399.02884),
                                              ("[M+H+NH4]2+", 212.47945),
                                              ("[2M+FA-H]-", 843.001799),
                                              ("M+H", 442.992724),
                                              ("M+H-H2O", 461.003289)])
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

    assert numpy.allclose(spectrum.get("parent_mass"), expected, atol=1e-4), f"Expected parent mass of about {expected}."
    assert isinstance(spectrum.get("parent_mass"), float), "Expected parent mass to be float."


def test_add_parent_mass_overwrite():
    """Test if parent mass is replaced by newly calculated value."""
    mz = numpy.array([], dtype='float')
    intensities = numpy.array([], dtype='float')
    metadata = {"precursor_mz": 444.0,
                "parent_mass": 443.0,
                "adduct": "[M+H]+",
                "charge": +1}
    spectrum_in = Spectrum(mz=mz,
                           intensities=intensities,
                           metadata=metadata)

    spectrum = add_parent_mass(spectrum_in, overwrite_existing_entry=True)

    assert numpy.allclose(spectrum.get("parent_mass"), 442.992724, atol=1e-4), \
        "Expected parent mass to be replaced by new value."


def test_add_parent_mass_not_sufficient_data(capsys):
    """Test when there is not enough information to derive parent_mass."""
    mz = numpy.array([], dtype='float')
    intensities = numpy.array([], dtype='float')
    metadata = {"precursor_mz": 444.0}
    spectrum_in = Spectrum(mz=mz,
                           intensities=intensities,
                           metadata=metadata)

    spectrum = add_parent_mass(spectrum_in)

    assert spectrum.get("parent_mass") is None, "Expected no parent mass"
    assert "Not sufficient spectrum metadata to derive parent mass." in capsys.readouterr().out


def test_empty_spectrum():
    spectrum_in = None
    spectrum = add_parent_mass(spectrum_in)

    assert spectrum is None, "Expected different handling of None spectrum."
