import numpy as np
import pytest
from matchms.constants import PROTON_MASS
from matchms.filtering.filter_utils.derive_precursor_mz_and_parent_mass import derive_parent_mass_from_precursor_mz, derive_precursor_mz_from_parent_mass
from tests.builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize(
    "adduct, expected", [("[M+2Na-H]+", 399.02884), ("[M+H+NH4]2+", 868.9589), ("[2M+FA-H]-", 199.5008995), ("M+H", 442.992724), ("M+H-H2O", 461.003289)]
)
def test_add_parent_mass_using_adduct(adduct, expected):
    """Test if parent mass is correctly derived from adduct information."""
    metadata = {"precursor_mz": 444.0, "adduct": adduct, "charge": +1}
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()
    parent_mass = derive_parent_mass_from_precursor_mz(spectrum_in, True)

    assert np.allclose(parent_mass, expected, atol=1e-4), f"Expected parent mass of about {expected}."
    assert isinstance(parent_mass, float), "Expected parent mass to be float."

    spectrum_in.set("parent_mass", parent_mass)
    precursor_mz = derive_precursor_mz_from_parent_mass(spectrum_in)
    assert np.allclose(precursor_mz, 444.0, atol=1e-4), (
        "Expected derive_precursor_mz_from_parent_mass to be the opposite of derive_parent_mass_from_precursor_mz"
    )


@pytest.mark.parametrize("ionmode, expected", [("positive", 444.0 - PROTON_MASS), ("negative", 444.0 + PROTON_MASS)])
def test_use_of_ionmode(ionmode, expected):
    """Test when there is no charge given, than the ionmode
    is used to derive parent mass."""
    metadata = {"precursor_mz": 444.0, "ionmode": ionmode}
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    parent_mass = derive_parent_mass_from_precursor_mz(spectrum_in, True)

    assert parent_mass == expected, "Expected a different parent_mass"
    spectrum_in.set("parent_mass", parent_mass)
    precursor_mz = derive_precursor_mz_from_parent_mass(spectrum_in)
    assert np.allclose(precursor_mz, 444.0, atol=1e-4), (
        "Expected derive_precursor_mz_from_parent_mass to be the opposite of derive_parent_mass_from_precursor_mz"
    )
