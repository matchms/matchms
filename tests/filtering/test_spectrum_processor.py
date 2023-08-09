import numpy as np
import pytest
from matchms import SpectrumProcessor
from matchms.SpectrumProcessor import ProcessingReport
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


def test_filter_sorting_and_output():
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
    # 2nd way to access the filter names via processing_steps attribute:
    assert processing.processing_steps == expected_filters


def test_string_output():
    processing = SpectrumProcessor("minimal")
    expected_str = "SpectrumProcessor\nProcessing steps:\n - make_charge_int\n - interpret_pepmass"\
        "\n - derive_ionmode\n - correct_charge"
    assert processing.__str__() == expected_str


@pytest.mark.parametrize("metadata, expected", [
    [{}, None],
    [{"ionmode": "positive"}, {"ionmode": "positive", "charge": 1}],
    [{"ionmode": "positive", "charge": 2}, {"ionmode": "positive", "charge": 2}],
])
def test_add_matchms_filter(metadata, expected):
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()
    processor = SpectrumProcessor("minimal")
    processor.add_matchms_filter(("require_correct_ionmode",
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


def test_unknown_keyword():
    with pytest.raises(ValueError) as msg:
        _ = SpectrumProcessor(predefined_pipeline="something_wrong")
    assert "Unknown processing pipeline" in str(msg.value)


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
    assert report.counter_number_processed == 3
    assert report.counter_changed_field == {'make_charge_int': 2}
    assert report.counter_added_field == {'interpret_pepmass': 3, 'derive_ionmode': 3}
    report_df = report.to_dataframe()
    assert np.all(report_df.loc[["make_charge_int", "interpret_pepmass", "derive_ionmode"]].values == np.array(
        [[0, 2, 0],
         [0, 0, 3],
         [0, 0, 3]]))


def test_processing_report_class(spectrums):
    processing_report = ProcessingReport()
    for s in spectrums:
        spectrum_processed = s.clone()
        spectrum_processed.set("smiles", "test")
        processing_report.add_to_report(s, spectrum_processed, "test_filter")

    assert processing_report.counter_removed_spectrums == {}
    assert processing_report.counter_changed_field == {}
    assert processing_report.counter_added_field == {"test_filter": 3}


def test_adding_custom_filter(spectrums):
    def nonsense_inchikey(s):
        s.set("inchikey", "NONSENSE")
        return s

    processor = SpectrumProcessor("minimal")
    processor.add_custom_filter(nonsense_inchikey)
    filters = processor.filters
    assert filters[-1].__name__ == "nonsense_inchikey"
    spectrums, report = processor.process_spectrums(spectrums, create_report=True)
    assert report.counter_number_processed == 3
    assert report.counter_changed_field == {'make_charge_int': 2}
    assert report.counter_added_field == {'interpret_pepmass': 3, 'derive_ionmode': 3, 'nonsense_inchikey': 3}
    assert spectrums[0].get("inchikey") == "NONSENSE", "Custom filter not executed properly"


def test_adding_custom_filter_with_parameters(spectrums):
    def nonsense_inchikey_multiple(s, number):
        s.set("inchikey", number * "NONSENSE")
        return s

    processor = SpectrumProcessor("minimal")
    processor.add_custom_filter(nonsense_inchikey_multiple, {"number": 2})
    filters = processor.filters
    assert filters[-1].__name__ == "nonsense_inchikey_multiple"
    spectrums, report = processor.process_spectrums(spectrums, create_report=True)
    assert report.counter_number_processed == 3
    assert report.counter_changed_field == {'make_charge_int': 2}
    assert report.counter_added_field == {'interpret_pepmass': 3, 'derive_ionmode': 3, 'nonsense_inchikey_multiple': 3}
    assert spectrums[0].get("inchikey") == "NONSENSENONSENSE", "Custom filter not executed properly"


def test_add_filter_with_custom(spectrums):
    def nonsense_inchikey_multiple(s, number):
        s.set("inchikey", number * "NONSENSE")
        return s

    processor = SpectrumProcessor("minimal")
    processor.add_filter((nonsense_inchikey_multiple, {"number": 2}))
    filters = processor.filters
    assert filters[-1].__name__ == "nonsense_inchikey_multiple"
    spectrums, report = processor.process_spectrums(spectrums, create_report=True)
    assert spectrums[0].get("inchikey") == "NONSENSENONSENSE", "Custom filter not executed properly"


def test_add_filter_with_matchms_filter(spectrums):
    processor = SpectrumProcessor("minimal")
    processor.add_filter(("require_correct_ionmode",
                         {"ion_mode_to_keep": "both"}))
    filters = processor.filters
    assert filters[-1].__name__ == "require_correct_ionmode"
    spectrums, report = processor.process_spectrums(spectrums, create_report=True)
    assert spectrums == []
