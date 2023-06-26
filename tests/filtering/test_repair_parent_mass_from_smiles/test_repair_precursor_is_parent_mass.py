import pytest
import math
from matchms.filtering import repair_precursor_is_parent_mass
from tests.builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize("smiles, adduct, precursor_mz, expected_precursor_mz, expected_parent_mass",
                         [("CN1CCCC1C2=CN=CC=C2", "[M+H]+", 162.115698455, 163.122974447999988, 162.115698455),
                          ("CN1CCCC1C2=CN=CC=C2", "[M+2H]2+", 162.115698455, 82.065125224, 162.115698455),
                          # Should not be changed
                          ("CN1CCCC1C2=CN=CC=C2", "[M+H]+", 145.3, 145.3, 10.0),
                          ("CN1CCCC1C2=CN=CC=C2", "[M-H]-", 162.115698455, 161.108422448, 162.115698455),
                          ])
def test_repair_precursor_is_parent_mass(smiles, adduct, precursor_mz, expected_precursor_mz, expected_parent_mass):
    # CH4 is used as smiles, this has a mass of 16
    spectrum_in = SpectrumBuilder().with_metadata({"smiles": smiles,
                                                   "adduct": adduct,
                                                   "precursor_mz": precursor_mz,
                                                   "parent_mass": 10.0}).build()
    spectrum_out = repair_precursor_is_parent_mass(spectrum_in, mass_tolerance=0.1)
    assert math.isclose(spectrum_out.get("precursor_mz"), expected_precursor_mz)
    assert math.isclose(spectrum_out.get("parent_mass"), expected_parent_mass)


