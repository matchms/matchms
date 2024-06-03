import os
import numpy as np
import pytest
from matchms import SpectrumProcessor
from matchms import filtering as msfilters
from matchms.filtering.default_pipelines import BASIC_FILTERS
from matchms.filtering.SpectrumProcessor import ProcessingReport
from matchms.importing.load_spectra import load_spectra
from ..builder_Spectrum import SpectrumBuilder


@pytest.fixture
def spectrums():
    s1 = SpectrumBuilder(). \
        with_metadata({"charge": "+1", "pepmass": 100}). \
        with_mz([10, 20, 30]).with_intensities([0.1, 0.4, 10]).build()
    s2 = SpectrumBuilder().with_metadata({"charge": "-1",
                                          "pepmass": 102}). \
        with_mz([10, 20, 30]).with_intensities([0.1, 0.2, 1]).build()
    s3 = SpectrumBuilder().with_metadata({"charge": -1,
                                          "pepmass": 104}). \
        with_mz([10, ]).with_intensities([0.1, ]).build()
    return [s1, s2, s3]


def test_filter_sorting_and_output():
    processing = SpectrumProcessor(filters=["make_charge_int",
                                            "derive_ionmode",
                                            "correct_charge",
                                            'derive_adduct_from_name',
                                            "interpret_pepmass",
                                            ])
    expected_filters = ['make_charge_int',
                        'derive_adduct_from_name',
                        'interpret_pepmass',
                        'derive_ionmode',
                        'correct_charge']

    actual_filters = [x.__name__ for x in processing.filters]
    assert actual_filters == expected_filters
    # 2nd way to access the filter names via processing_steps attribute:
    expected_filters = ['make_charge_int',
                        ('derive_adduct_from_name', {'remove_adduct_from_name': True}),
                        'interpret_pepmass',
                        'derive_ionmode',
                        'correct_charge']
    assert processing.processing_steps == expected_filters


@pytest.mark.parametrize("filter_step, expected", [
    [("add_parent_mass", {'estimate_from_adduct': False}),
     ('add_parent_mass', {'estimate_from_adduct': False, 'overwrite_existing_entry': False,
                          'estimate_from_charge': True})],
    ["derive_adduct_from_name",
     ('derive_adduct_from_name', {'remove_adduct_from_name': True})],
    [("require_correct_ionmode", {"ion_mode_to_keep": "both"}),
     ("require_correct_ionmode", {"ion_mode_to_keep": "both"})],
])
def test_overwrite_default_settings(filter_step: str, expected):
    """Test if both default settings and set settings are returned in processing steps"""
    processor = SpectrumProcessor(filters=())
    processor.parse_and_add_filter(filter_step)
    expected_filters = [expected]
    assert processor.processing_steps == expected_filters


def test_incomplete_parameters():
    """Test if an error is raised when running an incomplete command"""
    with pytest.raises(AssertionError):
        processor = SpectrumProcessor(filters=())
        processor.parse_and_add_filter("require_correct_ionmode")


def test_string_output():
    processing = SpectrumProcessor(filters=["make_charge_int",
                                            "interpret_pepmass",
                                            "derive_ionmode",
                                            "correct_charge",
                                            ])
    expected_str = "Processing steps:\n- make_charge_int\n- interpret_pepmass" \
                   "\n- derive_ionmode\n- correct_charge\n"
    assert str(processing) == expected_str


def test_no_filters():
    spectrum_in = SpectrumBuilder().with_metadata({}).build()
    processor = SpectrumProcessor(filters=())
    spectrum_out = processor.process_spectrum(spectrum_in)
    assert spectrum_out == spectrum_in


def test_filter_spectrums(spectrums):
    processor = SpectrumProcessor(filters=["make_charge_int",
                                           "interpret_pepmass",
                                           "derive_ionmode",
                                           "correct_charge",
                                           ])
    spectrums, _ = processor.process_spectrums(spectrums)

    assert len(spectrums) == 3
    actual_masses = [s.get("precursor_mz") for s in spectrums]
    expected_masses = [100, 102, 104]
    assert actual_masses == expected_masses


def test_filter_spectrums_report(spectrums):
    processor = SpectrumProcessor(filters=["make_charge_int",
                                           "interpret_pepmass",
                                           "derive_ionmode",
                                           "correct_charge",
                                           ])
    processor.parse_and_add_filter(filter_description=("require_minimum_number_of_peaks", {"n_required": 2}))
    processor.parse_and_add_filter(filter_description="add_losses")
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

    processor = SpectrumProcessor(filters=["make_charge_int",
                                           "interpret_pepmass",
                                           "derive_ionmode",
                                           "correct_charge",
                                           ])
    processor.parse_and_add_filter(nonsense_inchikey)
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

    processor = SpectrumProcessor(filters=["make_charge_int",
                                           "interpret_pepmass",
                                           "derive_ionmode",
                                           "correct_charge",
                                           ])
    processor.parse_and_add_filter((nonsense_inchikey_multiple, {"number": 2}))
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
def test_add_custom_filter_in_position(filter_position: int, expected):
    """Tests that a filter is added in the correct position"""

    def nonsense_inchikey_multiple(s, number):
        s.set("inchikey", number * "NONSENSE")
        return s

    processor = SpectrumProcessor(filters=["make_charge_int",
                                           "interpret_pepmass",
                                           "derive_ionmode",
                                           "correct_charge",
                                           ])
    processor.parse_and_add_filter((nonsense_inchikey_multiple, {"number": 2}),
                                   filter_position=filter_position)
    filters = processor.filters

    assert filters[expected].__name__ == "nonsense_inchikey_multiple"
    assert len(filters) == 5


def test_add_matchms_filter_in_position():
    processor = SpectrumProcessor(filters=["make_charge_int",
                                           "interpret_pepmass",
                                           "derive_ionmode",
                                           ])
    processor.parse_and_add_filter("correct_charge",
                                   filter_position=2)
    filters = processor.filters

    assert filters[2].__name__ == "correct_charge"
    assert len(filters) == 4


def test_add_custom_filter_with_parameters(spectrums):
    def nonsense_inchikey_multiple(s, number):
        s.set("inchikey", number * "NONSENSE")
        return s

    processor = SpectrumProcessor(filters=["make_charge_int",
                                           "interpret_pepmass",
                                           "derive_ionmode",
                                           "correct_charge",
                                           ])
    processor.parse_and_add_filter((nonsense_inchikey_multiple, {"number": 2}))
    filters = processor.filters

    assert filters[-1].__name__ == "nonsense_inchikey_multiple"
    spectrums, _ = processor.process_spectrums(spectrums)
    assert spectrums[0].get("inchikey") == "NONSENSENONSENSE", "Custom filter not executed properly"


@pytest.mark.parametrize("filter_description", [
    ("require_correct_ionmode", {"ion_mode_to_keep": "negative"}),
    (msfilters.require_correct_ionmode, {"ion_mode_to_keep": "negative"})

])
def test_add_matchms_filter(filter_description, spectrums):
    processor = SpectrumProcessor(filters=["make_charge_int",
                                           "interpret_pepmass",
                                           "derive_ionmode",
                                           "correct_charge",
                                           ])
    processor.parse_and_add_filter(filter_description)
    filters = processor.filters
    assert filters[-1].__name__ == "require_correct_ionmode"
    spectrums, _ = processor.process_spectrums(spectrums)
    assert len(spectrums) == 2


@pytest.mark.parametrize("filter_description", [
    ("derive_adduct_from_name", {"remove_adduct_from_name": False}),
])
def test_add_duplicated_filter_to_existing_pipeline(filter_description):
    """Tests if adding a filter that is already in the basic pipeline is overwritten and not duplicated"""
    processor = SpectrumProcessor(["derive_adduct_from_name",
                                   "interpret_pepmass",
                                   ])
    processor.parse_and_add_filter(filter_description)
    assert len(processor.processing_steps) == 2, "The duplicated filter was not replaced"
    assert filter_description in processor.processing_steps, "The new settings of the duplicated filter were not added"


def test_add_filter_twice():
    """Tests if adding a filter that is already in the basic pipeline is overwritten and not duplicated"""
    processor = SpectrumProcessor(filters=())
    processor.parse_and_add_filter(("derive_adduct_from_name", {"remove_adduct_from_name": False}))
    processor.parse_and_add_filter("derive_adduct_from_name")
    assert processor.processing_steps == [("derive_adduct_from_name", {"remove_adduct_from_name": True})]


def test_add_all_filter_types(spectrums):
    def nonsense_inchikey_multiple(s, number):
        s.set("inchikey", number * "NONSENSE")
        return s

    def nonsense_inchikey(s):
        s_in = s.clone()
        s_in.set("inchikey", "NONSENSE")
        return s_in

    processor = SpectrumProcessor(filters=["make_charge_int",
                                           msfilters.interpret_pepmass,
                                           nonsense_inchikey,
                                           (msfilters.derive_adduct_from_name, {"remove_adduct_from_name": False}),
                                           (nonsense_inchikey_multiple, {"number": 2}),
                                           ])
    filters = processor.filters
    assert [filter_func.__name__ for filter_func in filters] == ["make_charge_int",
                                                                 "derive_adduct_from_name",
                                                                 "interpret_pepmass",
                                                                 "nonsense_inchikey",
                                                                 "nonsense_inchikey_multiple",
                                                                 ]
    processor.process_spectrums(spectrums)
    spectrums, _ = processor.process_spectrums(spectrums)
    assert spectrums[0].get("inchikey") == "NONSENSENONSENSE", "Custom filter not executed properly"


def test_save_spectra_spectrum_processor(spectrums, tmp_path):
    processor = SpectrumProcessor(BASIC_FILTERS)
    filename = os.path.join(tmp_path, "spectra.msp")

    _, _ = processor.process_spectrums(spectrums, cleaned_spectra_file=str(filename))
    assert os.path.exists(filename)

    # Reload spectra and compare lengths
    reloaded_spectra = list(load_spectra(str(filename)))
    assert len(reloaded_spectra) == len(spectrums)

    # Check that the processed spectra are stored
    for spectrum in reloaded_spectra:
        assert spectrum.get("precursor_mz") is not None


def test_save_spectra_spectrum_processor_none_spectra(spectrums, tmp_path):
    processor = SpectrumProcessor(BASIC_FILTERS +
                                  [(msfilters.require_correct_ionmode, {"ion_mode_to_keep": "positive"})])
    filename = os.path.join(tmp_path, "spectra.msp")

    _, _ = processor.process_spectrums(spectrums, cleaned_spectra_file=str(filename))
    assert os.path.exists(filename)

    # Reload spectra and compare lengths
    reloaded_spectra = list(load_spectra(str(filename)))
    assert len(reloaded_spectra) == 1

    # Check that the processed spectra are stored
    for spectrum in reloaded_spectra:
        assert spectrum.get("precursor_mz") is not None


@pytest.mark.parametrize('ftype, should_work', [
    ['msp', True],
    ['mgf', True],
    ['json', False],
])
def test_save_partial_spectra_spectrum_processor(ftype, should_work, spectrums, tmp_path):
    processor = SpectrumProcessor(BASIC_FILTERS)

    # Deliberately introduce invalid spectrum
    spectrums[-2] = "Lorem Ipsum"

    filename = os.path.join(tmp_path, f"spectra.{ftype}")

    with pytest.raises(Exception):
        _, _ = processor.process_spectrums(spectrums, cleaned_spectra_file=str(filename))

    if should_work:
        assert os.path.exists(filename)

        # Reload spectra and compare lengths
        reloaded_spectra = list(load_spectra(str(filename)))

        # Make sure we are only missing last two "spectra"
        assert len(reloaded_spectra) == len(spectrums) - 2

        # Check that the processed spectra are stored
        for spectrum in reloaded_spectra:
            assert spectrum.get("precursor_mz") is not None
    else:
        assert not os.path.exists(filename)
