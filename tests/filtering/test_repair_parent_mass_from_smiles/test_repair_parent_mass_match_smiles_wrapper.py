import math
import pytest
from matchms.filtering import repair_parent_mass_match_smiles_wrapper
from tests.builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize("smiles, parent_mass, precursor_mz, adduct, "
                         "expected_smiles, expected_parent_mass, expected_precursor_mz, expected_adduct",
                         # Test repair parent mass is mol wt
                         [("CN1CCCC1C2=CN=CC=C2", 162.23, 163.23, "[M+H]+",
                           "CN1CCCC1C2=CN=CC=C2", 162.115698455, 163.23, "[M+H]+"),
                          # When the precursor mz could be mistaken with the parent mass it should not be repaired.
                          ("CN1CCCC1C2=CN=CC=C2",  161.108, 162.115698455, "[M+H]+",
                           "CN1CCCC1C2=CN=CC=C2", 161.108, 162.115698455, "[M+H]+"),
                          # Test repair adduct based on smiles
                          ("C", 0.0, 17.5, "[M+H]+",
                           "C", 16.031300128, 17.5, "[M+H+NH4]2+")
                          ])
def test_repair_parent_mass_match_smiles_wrapper(smiles, parent_mass, precursor_mz, adduct,
                                                 expected_smiles, expected_parent_mass,
                                                 expected_precursor_mz, expected_adduct):
    # pylint: disable=too-many-arguments, too-many-positional-arguments
    pytest.importorskip("rdkit")
    spectrum_in = SpectrumBuilder().with_metadata({"smiles": smiles,
                                                   "precursor_mz": precursor_mz,
                                                   "parent_mass": parent_mass,
                                                   "adduct": adduct,
                                                   "ionmode": "positive"}).build()
    spectrum_out = repair_parent_mass_match_smiles_wrapper(spectrum_in, mass_tolerance=0.1)
    assert math.isclose(spectrum_out.get("precursor_mz"), expected_precursor_mz)
    assert math.isclose(spectrum_out.get("parent_mass"), expected_parent_mass)
    assert spectrum_out.get("smiles") == expected_smiles
    assert spectrum_out.get("adduct") == expected_adduct
