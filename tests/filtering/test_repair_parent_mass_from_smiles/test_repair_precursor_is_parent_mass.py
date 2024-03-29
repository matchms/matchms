import math
import pytest
from matchms.filtering import repair_precursor_is_parent_mass
from tests.builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize("smiles, adduct, parent_mass, precursor_mz, expected_precursor_mz, expected_parent_mass",
                         [("CN1CCCC1C2=CN=CC=C2", "[M+H]+", 161.108, 162.115698455, 163.122974447999988, 162.115698455),
                          ("CN1CCCC1C2=CN=CC=C2", "[M+2H]2+", 322.2174, 162.115698455, 82.065125224, 162.115698455),
                          ("CN1CCCC1C2=CN=CC=C2", "[M-H]-", 163.12297, 162.115698455, 161.108422448, 162.115698455),
                          # Should not be cleaned
                          ("CN1CCCC1C2=CN=CC=C2", "[M-H]-", 0.0, 162.115698455, 162.115698455, 0.0),
                          ("CN1CCCC1C2=CN=CC=C2", "[M-H]-", 1.007, 0.0, 0.0, 1.007),
                          ])
def test_repair_precursor_is_parent_mass(smiles, adduct, parent_mass, precursor_mz,
                                         expected_precursor_mz, expected_parent_mass):
    # pylint: disable=too-many-arguments
    pytest.importorskip("rdkit")
    spectrum_in = SpectrumBuilder().with_metadata({"smiles": smiles,
                                                   "adduct": adduct,
                                                   "precursor_mz": precursor_mz,
                                                   "parent_mass": parent_mass}).build()
    spectrum_out = repair_precursor_is_parent_mass(spectrum_in, mass_tolerance=0.1)
    assert math.isclose(spectrum_out.get("precursor_mz"), expected_precursor_mz)
    assert math.isclose(spectrum_out.get("parent_mass"), expected_parent_mass)
