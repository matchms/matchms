import pytest
import numpy as np
from matchms import SpectrumProcessor
from ..builder_Spectrum import SpectrumBuilder


@pytest.fixture
def spectrums():
    metadata1 = {"charge": "+1",
                 "pepmass": 100}
    metadata2 = {"charge": "-1",
                 "pepmass": 102}
    metadata3 = {"charge": -1,
                 "pepmass": 104}

    s1 = SpectrumBuilder().with_metadata(metadata1).build()
    s2 = SpectrumBuilder().with_metadata(metadata2).build()
    s3 = SpectrumBuilder().with_metadata(metadata3).build()
    return [s1, s2, s3]


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


def test_no_filters():
    spectrum_in = SpectrumBuilder().with_metadata({}).build()
    processor = SpectrumProcessor(predefined_pipeline=None)
    with pytest.raises(TypeError) as msg:
        _ = processor.process_spectrum(spectrum_in)
    assert str(msg.value) == "No filters to process"


def test_filter_spectrums(spectrums):
    processor = SpectrumProcessor("minimal")
    spectrums = processor.process_spectrums(spectrums)
    assert len(spectrums) == 3
    actual_masses = [s.get("precursor_mz") for s in spectrums]
    expected_masses = [100, 102, 104]
    assert actual_masses == expected_masses


def test_filter_spectrums_report(spectrums):
    processor = SpectrumProcessor("minimal")
    spectrums, report = processor.process_spectrums(spectrums, create_report=True)
    assert len(spectrums) == 3
    actual_masses = [s.get("precursor_mz") for s in spectrums]
    expected_masses = [100, 102, 104]
    assert actual_masses == expected_masses
    assert np.all(report.loc[["charge", "ionmode", "precursor_mz"]].values == np.array(
        [[2],
        [3],
        [3]]))
