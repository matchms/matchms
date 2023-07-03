import pytest
from matchms.filtering.SpectrumProcessor import SpectrumProcessor
from matchms.filtering import require_correct_ionmode
from ..builder_Spectrum import SpectrumBuilder


def test_filter_sorting():
    processing = SpectrumProcessor("default")
    expected_filters = [
        'make_charge_int',
        'add_compound_name',
        'derive_adduct_from_name',
        'derive_formula_from_name',
        'clean_compound_name',
        'interpret_pepmass',
        'add_precursor_mz',
        'derive_ionmode',
        'correct_charge',
        'require_precursor_mz',
        'add_parent_mass',
        'harmonize_undefined_inchikey',
        'harmonize_undefined_inchi',
        'harmonize_undefined_smiles',
        'repair_inchi_inchikey_smiles',
        'repair_parent_mass_match_smiles_wrapper',
        'require_correct_ionmode',
        'normalize_intensities'
        ]
    actual_filters = [x.__name__ for x in processing.filters]
    assert actual_filters == expected_filters


@pytest.mark.parametrize("metadata, expected", [
    [{}, None],
    [{"ionmode": "positive"}, {"ionmode": "positive", "charge": 1}],
    [{"ionmode": "positive", "charge": 2}, {"ionmode": "positive", "charge": 2}],
])
def test_add_filter(metadata, expected):
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()
    processor = SpectrumProcessor("minimal")
    processor.add_filter(("require_correct_ionmode",
                          {"ion_mode_to_keep": "both"}))
    spectrum = processor.process_spectrum(spectrum_in)
    if expected is None:
        assert spectrum is None
    else:
        assert dict(spectrum.metadata) == expected
