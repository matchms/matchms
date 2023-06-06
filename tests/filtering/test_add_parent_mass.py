import numpy as np
import pytest
from matchms.filtering import add_parent_mass
from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize('metadata, expected', [
    [{"pepmass": (444.0, 10), "charge": -1}, "Missing precursor m/z to derive parent mass."],
    [{"charge": -1}, "Missing precursor m/z to derive parent mass."],
    [{"precursor_mz": 444.0, "charge": 0}, "Not sufficient spectrum metadata to derive parent mass."]
])
def test_add_parent_mass_exceptions(metadata, expected, caplog):
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()
    spectrum = add_parent_mass(spectrum_in)

    assert spectrum.get("parent_mass") is None, "Expected no parent mass"
    assert expected in caplog.text


def test_add_parent_mass_precursormz(caplog):
    """Test if parent mass is correctly derived if "pepmass" is not present."""
    metadata = {"precursor_mz": 444.0, "charge": -1}
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()
    spectrum = add_parent_mass(spectrum_in)

    assert np.abs(spectrum.get("parent_mass") - 445.0) < .01, "Expected parent mass of about 445.0."
    assert isinstance(spectrum.get("parent_mass"), float), "Expected parent mass to be float."
    assert "Not sufficient spectrum metadata to derive parent mass." not in caplog.text


@pytest.mark.parametrize("overwrite, expected", [(True, 442.992724),
                                                 (False, 443.0)])
def test_add_parent_mass_overwrite(overwrite, expected):
    """Test if parent mass is replaced by newly calculated value."""
    metadata = {"precursor_mz": 444.0,
                "parent_mass": 443.0,
                "adduct": "[M+H]+",
                "charge": +1}
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()
    spectrum = add_parent_mass(spectrum_in, overwrite_existing_entry=overwrite)

    assert np.allclose(spectrum.get("parent_mass"), expected, atol=1e-4), \
        "Expected parent mass to be replaced by new value."


@pytest.mark.parametrize("parent_mass_field, parent_mass, expected",
                         [("parent_mass", 400., 400.),
                          ("parent_mass", "400.", 400.),
                          ("exact_mass", 200, 200.),
                          ("parentmass", 200, 200.),
                          ("parent_mass", "n/a", 442.992724)])
def test_add_parent_mass_already_present(parent_mass_field, parent_mass, expected):
    """Test if parent mass is correctly derived from adduct information."""
    metadata = {"precursor_mz": 444.0,
                parent_mass_field: parent_mass,
                "charge": +1}
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()
    spectrum = add_parent_mass(spectrum_in)

    assert np.allclose(spectrum.get("parent_mass"), expected, atol=1e-4), f"Expected parent mass of about {expected}."
    assert isinstance(spectrum.get("parent_mass"), (float, int)), "Expected parent mass to be float."


def test_add_parent_mass_not_sufficient_data(caplog):
    """Test when there is not enough information to derive parent_mass."""
    metadata = {"precursor_mz": 444.0}
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()
    spectrum = add_parent_mass(spectrum_in)

    assert spectrum.get("parent_mass") is None, "Expected no parent mass"
    assert "Not sufficient spectrum metadata to derive parent mass." in caplog.text


def test_empty_spectrum():
    spectrum_in = None
    spectrum = add_parent_mass(spectrum_in)

    assert spectrum is None, "Expected different handling of None spectrum."
