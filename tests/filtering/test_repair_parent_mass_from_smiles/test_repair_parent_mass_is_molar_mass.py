import math
import pytest
from matchms.filtering import repair_parent_mass_is_molar_mass
from tests.builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize("smiles, adduct, precursor_mz, parent_mass, expected_precursor_mz, expected_parent_mass",
                         [("CN1CCCC1C2=CN=CC=C2", "[M+H]+", 163.23, 162.23, 163.23, 162.115698455),
                          ("CN1CCCC1C2=CN=CC=C2", "[M+2H]2+", 82.12, 162.23, 82.12, 162.115698455),
                          # Should not be changed, since it is not close to the molecular weight
                          ("CN1CCCC1C2=CN=CC=C2", "[M+H]+", 10.0, 100.0, 10.0, 100.0),
                          # negative
                          ("CN1CCCC1C2=CN=CC=C2", "[M-H]-", 161.23, 162.23, 161.23, 162.115698455),
                          # not matching precursor mz (should still be changed
                          ("CN1CCCC1C2=CN=CC=C2", "[M-H]-", 10, 162.23, 10, 162.115698455),
                          ])
def test_repair_parent_mass_is_molar_mass(smiles, adduct, precursor_mz, parent_mass, expected_precursor_mz, expected_parent_mass):
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    pytest.importorskip("rdkit")
    spectrum_in = SpectrumBuilder().with_metadata({"smiles": smiles,
                                                   "adduct": adduct,
                                                   "precursor_mz": precursor_mz,
                                                   "parent_mass": parent_mass}).build()
    spectrum_out = repair_parent_mass_is_molar_mass(spectrum_in, mass_tolerance=0.1)
    assert math.isclose(spectrum_out.get("precursor_mz"), expected_precursor_mz)
    assert math.isclose(spectrum_out.get("parent_mass"), expected_parent_mass)
