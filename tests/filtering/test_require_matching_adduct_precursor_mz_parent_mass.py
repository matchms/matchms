import pytest
from matchms.filtering.metadata_processing.require_matching_adduct_precursor_mz_parent_mass import require_matching_adduct_precursor_mz_parent_mass
from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize(
    "adduct, parent_mass, precursor_mz, should_be_removed",
    [
        ["[M+H]+", 100, 100, True],  # incorrect matches
        ["[M+H]+blabla", 100, 100, True],  # wrong adduct
        ["[M+H]+", "blabal", 100, True],  # wrong parent mass
        ["[M+H]+", 100, "bla", True],  # wrong precursor mz
        ["[M+H]+", 100, None, True],  # wrong precursor mz
        ["[M+H]+", 100.0, 101.0, False],  # wrong precursor mz
    ],
)
def test_require_matching_adduct_precursor_mz_parent_mass(adduct, parent_mass, precursor_mz, should_be_removed):
    spectrum_in = SpectrumBuilder().with_metadata({"adduct": adduct, "parent_mass": parent_mass, "precursor_mz": precursor_mz}).build()
    result = require_matching_adduct_precursor_mz_parent_mass(spectrum_in)
    if result is None:
        assert should_be_removed
    else:
        assert spectrum_in == result
