import os
import tempfile
import numpy as np
import pytest
from matchms import SpectrumProcessor
from matchms.filtering.filter_order_and_default_pipelines import BASIC_FILTERS
from matchms.filtering.SpectrumProcessor import ProcessingReport
from matchms.importing.load_spectra import load_spectra
from ..builder_Spectrum import SpectrumBuilder


@pytest.fixture
def spectrums():
    s1 = SpectrumBuilder().\
        with_metadata({"charge": "+1", "pepmass": 100}).\
        with_mz([10, 20, 30]).with_intensities([0.1, 0.4, 10]).build()
    s2 = SpectrumBuilder().with_metadata({"charge": "-1",
                                          "pepmass": 102}).\
        with_mz([10, 20, 30]).with_intensities([0.1, 0.2, 1]).build()
    s3 = SpectrumBuilder().with_metadata({"charge": -1,
                                          "pepmass": 104}).\
        with_mz([10, ]).with_intensities([0.1, ]).build()
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
        'add_retention_time',
        'derive_ionmode',
        'correct_charge',
        'require_precursor_mz',
        'harmonize_undefined_inchikey',
        'harmonize_undefined_inchi',
        'harmonize_undefined_smiles',
        'repair_inchi_inchikey_smiles',
        'add_parent_mass',
        'normalize_intensities'
    ]

    actual_filters = [x.__name__ for x in processing.filters]
    assert actual_filters == expected_filters
    # 2nd way to access the filter names via processing_steps attribute:
    expected_filters = ['make_charge_int',
                        'add_compound_name',
                        ('derive_adduct_from_name', {'remove_adduct_from_name': True}),
                        ('derive_formula_from_name', {'remove_formula_from_name': True}),
                        'clean_compound_name',
                        'interpret_pepmass',
                        'add_precursor_mz',
                        'add_retention_time',
                        'derive_ionmode',
                        'correct_charge',
                        ('require_precursor_mz', {'minimum_accepted_mz': 10.0}),
                        ('harmonize_undefined_inchikey', {'aliases': None, 'undefined': ''}),
                        ('harmonize_undefined_inchi', {'aliases': None, 'undefined': ''}),
                        ('harmonize_undefined_smiles', {'aliases': None, 'undefined': ''}),
                        'repair_inchi_inchikey_smiles',
                        ('add_parent_mass', {'estimate_from_adduct': True, 'overwrite_existing_entry': False}),
                        'normalize_intensities']
    assert processing.processing_steps == expected_filters


@pytest.mark.parametrize("filter_step, expected", [
    [("add_parent_mass", {'estimate_from_adduct': False}),
     ('add_parent_mass', {'estimate_from_adduct': False, 'overwrite_existing_entry': False})],
    ["derive_adduct_from_name",
     ('derive_adduct_from_name', {'remove_adduct_from_name': True})],
    [("require_correct_ionmode", {"ion_mode_to_keep": "both"}),
     ("require_correct_ionmode", {"ion_mode_to_keep": "both"})],
])
def test_overwrite_default_settings(filter_step: str, expected):
    """Test if both default settings and set settings are returned in processing steps"""
    processor = SpectrumProcessor(None)
    processor.add_filter(filter_step)
    expected_filters = [expected]
    assert processor.processing_steps == expected_filters


def test_incomplete_parameters():
    """Test if an error is raised when running an incomplete command"""
    with pytest.raises(AssertionError):
        processor = SpectrumProcessor(None)
        processor.add_filter("require_correct_ionmode")


def test_string_output():
    processing = SpectrumProcessor("minimal")
    expected_str = "Processing steps:\n- make_charge_int\n- interpret_pepmass" \
                   "\n- derive_ionmode\n- correct_charge\n"
    assert str(processing) == expected_str


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
    spectrums, _ = processor.process_spectrums(spectrums)
    assert len(spectrums) == 3
    actual_masses = [s.get("precursor_mz") for s in spectrums]
    expected_masses = [100, 102, 104]
    assert actual_masses == expected_masses


def test_filter_spectrums_report(spectrums):
    processor = SpectrumProcessor("minimal")
    processor.add_filter(filter_function=("require_minimum_number_of_peaks", {"n_required": 2}))
    processor.add_filter(filter_function="add_losses")
    spectrums, report = processor.process_spectrums(spectrums)
    assert len(spectrums) == 2
    actual_masses = [s.get("precursor_mz") for s in spectrums]
    expected_masses = [100, 102]
    assert actual_masses == expected_masses
    assert report.counter_number_processed == 3
    assert report.counter_changed_metadata == {'make_charge_int': 2, 'interpret_pepmass': 3, 'derive_ionmode': 3}
    report_df = report.to_dataframe()
    assert np.all(report_df.loc[["require_minimum_number_of_peaks", "interpret_pepmass",
                                 "add_losses", "correct_charge"]].values == np.array(
        [[1, 0, 0],
         [0, 3, 0],
         [0, 0, 2],
         [0, 0, 0]]))


def test_processing_report_class(spectrums):
    processing_report = ProcessingReport()
    for s in spectrums:
        spectrum_processed = s.clone()
        spectrum_processed.set("smiles", "test")
        processing_report.add_to_report(s, spectrum_processed, "test_filter")

    assert not processing_report.counter_removed_spectrums
    assert processing_report.counter_changed_metadata == {"test_filter": 3}


def test_adding_custom_filter(spectrums):
    def nonsense_inchikey(s):
        s_in = s.clone()
        s_in.set("inchikey", "NONSENSE")
        return s_in

    processor = SpectrumProcessor("minimal")
    processor.add_custom_filter(nonsense_inchikey)
    filters = processor.filters
    assert filters[-1].__name__ == "nonsense_inchikey"
    spectrums, report = processor.process_spectrums(spectrums)
    assert report.counter_number_processed == 3
    assert report.counter_changed_metadata == {'make_charge_int': 2, 'interpret_pepmass': 3,
                                               'derive_ionmode': 3, 'nonsense_inchikey': 3}
    assert spectrums[0].get("inchikey") == "NONSENSE", "Custom filter not executed properly"


def test_adding_custom_filter_with_parameters(spectrums):
    def nonsense_inchikey_multiple(s, number):
        s_in = s.clone()
        s_in.set("inchikey", number * "NONSENSE")
        return s_in

    processor = SpectrumProcessor("minimal")
    processor.add_custom_filter(nonsense_inchikey_multiple, {"number": 2})
    filters = processor.filters
    assert filters[-1].__name__ == "nonsense_inchikey_multiple"
    spectrums, report = processor.process_spectrums(spectrums)
    assert report.counter_number_processed == 3
    assert report.counter_changed_metadata == {'make_charge_int': 2, 'interpret_pepmass': 3,
                                               'derive_ionmode': 3, 'nonsense_inchikey_multiple': 3}
    assert spectrums[0].get("inchikey") == "NONSENSENONSENSE", "Custom filter not executed properly"


@pytest.mark.parametrize("filter_position, expected", [
    [0, 0],
    [1, 1],
    [2, 2],
    [3, 3],
    [None, 4],
    [5, 4],
    [6, 4]
])
def test_add_custom_filter_in_position(filter_position, expected):
    def nonsense_inchikey_multiple(s, number):
        s.set("inchikey", number * "NONSENSE")
        return s

    processor = SpectrumProcessor("minimal")
    processor.add_custom_filter(nonsense_inchikey_multiple, {"number": 2},
                                filter_position=filter_position)
    filters = processor.filters

    assert filters[expected].__name__ == "nonsense_inchikey_multiple"


def test_add_filter_with_custom(spectrums):
    def nonsense_inchikey_multiple(s, number):
        s.set("inchikey", number * "NONSENSE")
        return s

    processor = SpectrumProcessor("minimal")
    processor.add_filter((nonsense_inchikey_multiple, {"number": 2}))
    filters = processor.filters

    assert filters[-1].__name__ == "nonsense_inchikey_multiple"
    spectrums, _ = processor.process_spectrums(spectrums)
    assert spectrums[0].get("inchikey") == "NONSENSENONSENSE", "Custom filter not executed properly"


def test_add_filter_with_matchms_filter(spectrums):
    processor = SpectrumProcessor("minimal")
    processor.add_filter(("require_correct_ionmode",
                          {"ion_mode_to_keep": "both"}))
    filters = processor.filters
    assert filters[-1].__name__ == "require_correct_ionmode"
    spectrums, _ = processor.process_spectrums(spectrums)
    assert not spectrums, "Expected to be empty list"


def test_add_duplicated_filter_to_existing_pipeline():
    """Tests if adding a filter that is already in the basic pipeline is overwritten and not duplicated"""
    processor = SpectrumProcessor("basic")
    duplicated_filter = ("derive_adduct_from_name", {"remove_adduct_from_name": False})
    processor.add_filter(duplicated_filter)
    assert len(processor.processing_steps) == len(BASIC_FILTERS), "The duplicated filter was not replaced"
    assert duplicated_filter in processor.processing_steps, "The new settings of the duplicated filter were not added"


def test_add_filter_twice():
    """Tests if adding a filter that is already in the basic pipeline is overwritten and not duplicated"""
    processor = SpectrumProcessor(None)
    processor.add_filter(("derive_adduct_from_name", {"remove_adduct_from_name": False}))
    processor.add_filter("derive_adduct_from_name")
    assert processor.processing_steps == [("derive_adduct_from_name", {"remove_adduct_from_name": True})]


def test_save_spectra_spectrum_processor(spectrums):
    processor = SpectrumProcessor("default")
    with tempfile.TemporaryDirectory() as temp_dir:
        filename = os.path.join(temp_dir, "spectra.msp")
        _, _ = processor.process_spectrums(spectrums, cleaned_spectra_file=filename)
        assert os.path.exists(filename)
        reloaded_spectra = list(load_spectra(filename))
    assert len(reloaded_spectra) == len(spectrums)
    for spectrum in reloaded_spectra:
        # to check that the processed spectra are stored instead of the unprocessed.
        assert spectrum.get("precursor_mz") is not None
