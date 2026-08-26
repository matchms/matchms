import pandas as pd
import pytest
from matchms import SpectraCollection
from matchms import filtering as msfilters
from matchms.filtering.spectra_processor import (
    ProcessingReport,
    SpectraProcessor,
    check_all_parameters_given,
    create_partial_function,
    get_parameter_settings,
)
from tests.builder_spectrum import SpectrumBuilder


@pytest.fixture
def spectra():
    s1 = (
        SpectrumBuilder()
        .with_metadata(
            {
                "charge": "+1",
                "pepmass": 100,
                "smiles": "n/a",
                "compound_name": "compound 1",
            }
        )
        .with_mz([10, 20, 30])
        .with_intensities([0.1, 0.4, 10])
        .build()
    )
    s2 = (
        SpectrumBuilder()
        .with_metadata(
            {
                "charge": "-1",
                "pepmass": 102,
                "smiles": "CCCO",
                "compound_name": "compound 2",
            }
        )
        .with_mz([10, 20, 30])
        .with_intensities([0.1, 0.2, 1])
        .build()
    )
    s3 = (
        SpectrumBuilder()
        .with_metadata(
            {
                "charge": -1,
                "pepmass": 104,
                "smiles": "no data",
                "compound_name": "compound 3",
            }
        )
        .with_mz([10])
        .with_intensities([0.1])
        .build()
    )
    return [s1, s2, s3]


@pytest.fixture
def collection(spectra):
    return SpectraCollection(spectra)


# -----------------------------------------------------------------------------
# Pipeline configuration
# -----------------------------------------------------------------------------


def test_filter_sorting_and_output():
    processor = SpectraProcessor(
        filters=[
            "make_charge_int",
            "derive_ionmode",
            "correct_charge",
            "derive_adduct_from_name",
            "interpret_pepmass",
        ]
    )

    assert [filter_func.__name__ for filter_func in processor.filters] == [
        "make_charge_int",
        "derive_adduct_from_name",
        "interpret_pepmass",
        "derive_ionmode",
        "correct_charge",
    ]

    assert processor.processing_steps == [
        ("make_charge_int", {"clone": True}),
        (
            "derive_adduct_from_name",
            {"remove_adduct_from_name": True, "clone": True},
        ),
        ("interpret_pepmass", {"clone": True}),
        ("derive_ionmode", {"clone": True}),
        ("correct_charge", {"clone": True}),
    ]


@pytest.mark.parametrize(
    "filter_step, expected",
    [
        [
            ("add_parent_mass", {"estimate_from_adduct": False, "clone": True}),
            (
                "add_parent_mass",
                {
                    "estimate_from_adduct": False,
                    "overwrite_existing_entry": False,
                    "estimate_from_charge": True,
                    "clone": True,
                },
            ),
        ],
        [
            "derive_adduct_from_name",
            (
                "derive_adduct_from_name",
                {"remove_adduct_from_name": True, "clone": True},
            ),
        ],
        [
            ("require_correct_ionmode", {"ion_mode_to_keep": "both"}),
            ("require_correct_ionmode", {"ion_mode_to_keep": "both"}),
        ],
        [
            ("select_by_relative_intensity", {"intensity_from": 0.01}),
            (
                "select_by_relative_intensity",
                {
                    "intensity_from": 0.01,
                    "intensity_to": 1.0,
                    "clone": True,
                },
            ),
        ],
        [
            ("harmonize_missing_entries", {"keys": ["smiles"], "undefined": ""}),
            (
                "harmonize_missing_entries",
                {
                    "keys": ["smiles"],
                    "undefined": "",
                    "aliases": None,
                    "clone": True,
                },
            ),
        ],
    ],
)
def test_overwrite_default_settings(filter_step, expected):
    processor = SpectraProcessor(filters=())
    processor.parse_and_add_filter(filter_step)

    assert processor.processing_steps == [expected]


def test_incomplete_parameters():
    processor = SpectraProcessor(filters=())

    with pytest.raises(AssertionError):
        processor.parse_and_add_filter("require_correct_ionmode")

    def custom_filter(spectrum, required_parameter, clone=True):
        return spectrum

    with pytest.raises(AssertionError):
        processor.parse_and_add_filter(custom_filter)

    with pytest.raises(ValueError):
        processor.parse_and_add_filter(
            ("add_parent_mass", {"estimate_from_adduct": False}, "unexpected")
        )


def test_invalid_filter_description_raises():
    processor = SpectraProcessor(filters=())

    with pytest.raises(TypeError, match="Expected callable filter function"):
        processor.parse_and_add_filter(123)

    with pytest.raises(TypeError, match="Expected a dictionary for filter parameters"):
        processor.parse_and_add_filter((msfilters.make_charge_int, "invalid"))


def test_string_output():
    processor = SpectraProcessor(
        filters=[
            "make_charge_int",
            "interpret_pepmass",
            "derive_ionmode",
            "correct_charge",
        ]
    )

    expected_str = (
        "Processing steps:\n- - make_charge_int\n  - clone: true\n"
        "- - interpret_pepmass\n  - clone: true\n"
        "- - derive_ionmode\n  - clone: true\n"
        "- - correct_charge\n  - clone: true\n"
    )

    assert str(processor) == expected_str


@pytest.mark.parametrize(
    "filter_position, expected",
    [[0, 0], [1, 1], [2, 2], [3, 3], [None, 4], [5, 4], [6, 4]],
)
def test_add_custom_filter_in_position(filter_position, expected):
    def custom_filter(spectrum, number):
        spectrum.set("inchikey", number * "NONSENSE")
        return spectrum

    processor = SpectraProcessor(
        filters=[
            "make_charge_int",
            "interpret_pepmass",
            "derive_ionmode",
            "correct_charge",
        ]
    )
    processor.parse_and_add_filter(
        (custom_filter, {"number": 2}),
        filter_position=filter_position,
    )

    assert processor.filters[expected].__name__ == "custom_filter"
    assert len(processor.filters) == 5


def test_add_matchms_filter_in_position():
    processor = SpectraProcessor(
        filters=[
            "harmonize_missing_entries",
            "select_by_intensity",
        ]
    )
    processor.parse_and_add_filter(
        "select_by_relative_intensity",
        filter_position=1,
    )

    assert processor.filters[1].__name__ == "select_by_relative_intensity"
    assert len(processor.filters) == 3


@pytest.mark.parametrize(
    "filter_description",
    [
        ("select_by_relative_intensity", {"intensity_from": 0.01}),
        (msfilters.select_by_relative_intensity, {"intensity_from": 0.01}),
    ],
)
def test_add_matchms_filter_for_collection(filter_description, collection):
    processor = SpectraProcessor(filters=["harmonize_missing_entries"])
    processor.parse_and_add_filter(filter_description)

    assert processor.filters[-1].__name__ == "select_by_relative_intensity"

    processed = processor.process_collection(collection)
    assert isinstance(processed, SpectraCollection)
    assert len(processed) == 3


@pytest.mark.parametrize(
    "filter_description",
    [
        ("require_correct_ionmode", {"ion_mode_to_keep": "negative"}),
        (msfilters.require_correct_ionmode, {"ion_mode_to_keep": "negative"}),
    ],
)
def test_add_matchms_filter_for_spectrum(filter_description, spectra):
    processor = SpectraProcessor(
        filters=[
            "make_charge_int",
            "interpret_pepmass",
            "derive_ionmode",
            "correct_charge",
        ]
    )
    processor.parse_and_add_filter(filter_description)

    assert processor.filters[-1].__name__ == "require_correct_ionmode"
    assert processor.process_spectrum(spectra[0]) is None
    assert processor.process_spectrum(spectra[1]) is not None


def test_add_duplicated_filter_to_existing_pipeline():
    processor = SpectraProcessor(
        [
            "harmonize_missing_entries",
            ("select_by_relative_intensity", {"intensity_from": 0.01}),
        ]
    )
    processor.parse_and_add_filter(
        ("select_by_relative_intensity", {"intensity_from": 0.1})
    )

    assert len(processor.processing_steps) == 2
    assert (
        "select_by_relative_intensity",
        {
            "intensity_from": 0.1,
            "intensity_to": 1.0,
            "clone": True,
        },
    ) in processor.processing_steps


def test_add_filter_twice_uses_last_settings():
    processor = SpectraProcessor(filters=())

    processor.parse_and_add_filter(
        ("select_by_relative_intensity", {"intensity_from": 0.01})
    )
    processor.parse_and_add_filter("select_by_relative_intensity")

    assert processor.processing_steps == [
        (
            "select_by_relative_intensity",
            {
                "intensity_from": 0.0,
                "intensity_to": 1.0,
                "clone": True,
            },
        )
    ]


def test_add_all_filter_types_for_spectrum(spectra):
    def add_inchikey(spectrum, clone=True):
        target = spectrum.clone() if clone else spectrum
        target.set("inchikey", "NONSENSE")
        return target

    def add_repeated_inchikey(spectrum, number, clone=True):
        target = spectrum.clone() if clone else spectrum
        target.set("inchikey", number * "NONSENSE")
        return target

    processor = SpectraProcessor(
        filters=[
            "make_charge_int",
            msfilters.interpret_pepmass,
            add_inchikey,
            (msfilters.derive_adduct_from_name, {"remove_adduct_from_name": False}),
            (add_repeated_inchikey, {"number": 2}),
        ]
    )

    assert [filter_func.__name__ for filter_func in processor.filters] == [
        "make_charge_int",
        "derive_adduct_from_name",
        "interpret_pepmass",
        "add_inchikey",
        "add_repeated_inchikey",
    ]

    processed = processor.process_spectrum(spectra[0])
    assert processed.get("inchikey") == "NONSENSENONSENSE"


def test_add_all_filter_types_for_collection(collection):
    def add_inchikey(collection_in, clone=True):
        target = collection_in.copy() if clone else collection_in
        target.add_metadata(["NONSENSE"] * len(target), col_name="inchikey")
        return target

    def add_repeated_inchikey(collection_in, number, clone=True):
        target = collection_in.copy() if clone else collection_in
        target.add_metadata(
            [number * "NONSENSE"] * len(target),
            col_name="inchikey",
            overwrite=True,
        )
        return target

    processor = SpectraProcessor(
        filters=[
            "harmonize_missing_entries",
            msfilters.select_by_intensity,
            add_inchikey,
            (msfilters.select_by_relative_intensity, {"intensity_from": 0.01}),
            (add_repeated_inchikey, {"number": 2}),
        ]
    )

    assert [filter_func.__name__ for filter_func in processor.filters] == [
        "harmonize_missing_entries",
        "select_by_intensity",
        "select_by_relative_intensity",
        "add_inchikey",
        "add_repeated_inchikey",
    ]

    processed = processor.process_collection(collection)
    assert processed.metadata["inchikey"].tolist() == [
        "NONSENSENONSENSE",
        "NONSENSENONSENSE",
        "NONSENSENONSENSE",
    ]


# -----------------------------------------------------------------------------
# Spectrum processing
# -----------------------------------------------------------------------------


def test_process_spectrum_no_filters_returns_clone(spectra):
    spectrum = spectra[0]
    processor = SpectraProcessor(filters=())

    processed = processor.process_spectrum(spectrum)

    assert processed == spectrum
    assert processed is not spectrum


def test_process_spectrum(spectra):
    processor = SpectraProcessor(
        filters=[
            "make_charge_int",
            "interpret_pepmass",
            "derive_ionmode",
            "correct_charge",
        ]
    )

    processed = processor.process_spectrum(spectra[0])

    assert processed is not spectra[0]
    assert processed.get("precursor_mz") == 100
    assert processed.get("charge") == 1
    assert processed.get("ionmode") == "positive"

    # Input is protected by the processor's one-time clone.
    assert spectra[0].get("precursor_mz") is None
    assert spectra[0].get("charge") == "+1"


def test_process_spectrum_rejects_non_spectrum():
    processor = SpectraProcessor(filters=())

    with pytest.raises(TypeError, match="process_spectrum expects a Spectrum"):
        processor.process_spectrum(["not", "a", "spectrum"])


def test_adding_custom_spectrum_filter(spectra):
    def add_inchikey(spectrum, clone=True):
        target = spectrum.clone() if clone else spectrum
        target.set("inchikey", "NONSENSE")
        return target

    processor = SpectraProcessor(filters=["make_charge_int"])
    processor.parse_and_add_filter(add_inchikey)

    processed = processor.process_spectrum(spectra[0])

    assert processor.filters[-1].__name__ == "add_inchikey"
    assert processed.get("inchikey") == "NONSENSE"
    assert spectra[0].get("inchikey") is None


def test_adding_custom_spectrum_filter_with_parameters(spectra):
    def add_repeated_inchikey(spectrum, number, clone=True):
        target = spectrum.clone() if clone else spectrum
        target.set("inchikey", number * "NONSENSE")
        return target

    processor = SpectraProcessor(filters=())
    processor.parse_and_add_filter((add_repeated_inchikey, {"number": 2}))

    processed = processor.process_spectrum(spectra[0])
    assert processed.get("inchikey") == "NONSENSENONSENSE"


def test_custom_spectrum_filter_without_clone_argument_is_supported(spectra):
    def add_inchikey(spectrum):
        spectrum.set("inchikey", "NONSENSE")
        return spectrum

    processor = SpectraProcessor(filters=[add_inchikey])
    report = processor.create_processing_report()

    processed = processor.process_spectrum(spectra[0], processing_report=report)

    assert processed.get("inchikey") == "NONSENSE"
    assert spectra[0].get("inchikey") is None
    assert report.to_dataframe().loc["add_inchikey", "changed metadata"] == 1


def test_process_spectrum_requirement_can_remove_spectrum(spectra):
    processor = SpectraProcessor(
        filters=[("require_minimum_number_of_peaks", {"n_required": 4})]
    )

    assert processor.process_spectrum(spectra[0]) is None


def test_process_spectrum_filter_returning_none_stops_processing(spectra):
    calls = []

    def drop_everything(spectrum, clone=True):
        calls.append("drop")

    def should_not_run(spectrum, clone=True):
        calls.append("after")
        raise AssertionError("This filter should not run after the spectrum was dropped.")

    processor = SpectraProcessor(filters=())
    processor.parse_and_add_filter(drop_everything)
    processor.parse_and_add_filter(should_not_run)

    assert processor.process_spectrum(spectra[0]) is None
    assert calls == ["drop"]


def test_process_spectrum_filter_returning_wrong_type_raises(spectra):
    def bad_filter(spectrum, clone=True):
        return ["not", "a", "spectrum"]

    processor = SpectraProcessor(filters=[bad_filter])

    with pytest.raises(TypeError, match="expected Spectrum or None"):
        processor.process_spectrum(spectra[0])


def test_process_spectrum_passes_clone_false_to_filters(spectra):
    def assert_clone_false(spectrum, clone=True):
        assert clone is False
        return spectrum

    processor = SpectraProcessor(filters=[assert_clone_false])

    processed = processor.process_spectrum(spectra[0])
    assert processed is not spectra[0]


# -----------------------------------------------------------------------------
# SpectraCollection processing
# -----------------------------------------------------------------------------


def test_process_collection_no_filters_returns_copy(collection):
    processor = SpectraProcessor(filters=())

    processed = processor.process_collection(collection)

    assert processed is not collection
    assert isinstance(processed, SpectraCollection)
    assert len(processed) == len(collection)
    assert processed.metadata.equals(collection.metadata)
    assert processed.fragment_hashes.tolist() == collection.fragment_hashes.tolist()


def test_process_collection(collection):
    processor = SpectraProcessor(
        filters=[
            "harmonize_missing_entries",
            ("select_by_relative_intensity", {"intensity_from": 0.02}),
        ]
    )

    processed = processor.process_collection(collection)

    assert isinstance(processed, SpectraCollection)
    assert len(processed) == 3

    assert pd.isna(processed.metadata.loc[0, "smiles"])
    assert processed.metadata.loc[1, "smiles"] == "CCCO"
    assert pd.isna(processed.metadata.loc[2, "smiles"])

    assert len(processed[0].peaks) == 2
    assert len(processed[1].peaks) == 3
    assert len(processed[2].peaks) == 1

    # Input is protected by the processor's one-time copy.
    assert collection.metadata.loc[0, "smiles"] == "n/a"
    assert len(collection[0].peaks) == 3


def test_process_collection_with_spectrum_style_metadata_filters(collection):
    processor = SpectraProcessor(
        filters=[
            "make_charge_int",
            "interpret_pepmass",
            "derive_ionmode",
            "correct_charge",
        ]
    )

    processed = processor.process_collection(collection)

    assert processed.metadata["precursor_mz"].tolist() == [100, 102, 104]
    assert processed.metadata["charge"].tolist() == [1, -1, -1]
    assert processed.metadata["ionmode"].tolist() == ["positive", "negative", "negative"]


def test_process_collection_rejects_non_collection(spectra):
    processor = SpectraProcessor(filters=())

    with pytest.raises(TypeError, match="process_collection expects a SpectraCollection"):
        processor.process_collection(spectra)


def test_adding_custom_collection_filter(collection):
    def add_inchikey(collection_in, clone=True):
        target = collection_in.copy() if clone else collection_in
        target.add_metadata(["NONSENSE"] * len(target), col_name="inchikey")
        return target

    processor = SpectraProcessor(filters=["harmonize_missing_entries"])
    processor.parse_and_add_filter(add_inchikey)

    processed = processor.process_collection(collection)

    assert processor.filters[-1].__name__ == "add_inchikey"
    assert processed.metadata["inchikey"].tolist() == [
        "NONSENSE",
        "NONSENSE",
        "NONSENSE",
    ]


def test_adding_custom_collection_filter_with_parameters(collection):
    def add_repeated_inchikey(collection_in, number, clone=True):
        target = collection_in.copy() if clone else collection_in
        target.add_metadata(
            [number * "NONSENSE"] * len(target),
            col_name="inchikey",
        )
        return target

    processor = SpectraProcessor(filters=())
    processor.parse_and_add_filter((add_repeated_inchikey, {"number": 2}))

    processed = processor.process_collection(collection)
    assert processed.metadata["inchikey"].tolist() == [
        "NONSENSENONSENSE",
        "NONSENSENONSENSE",
        "NONSENSENONSENSE",
    ]


def test_custom_collection_filter_without_clone_argument_is_supported(collection):
    def add_inchikey(collection_in):
        collection_in.add_metadata(["NONSENSE"] * len(collection_in), col_name="inchikey")
        return collection_in

    processor = SpectraProcessor(filters=[add_inchikey])
    report = processor.create_processing_report()

    processed = processor.process_collection(collection, processing_report=report)

    assert processed.metadata["inchikey"].tolist() == ["NONSENSE"] * 3
    assert "inchikey" not in collection.metadata.columns
    assert report.to_dataframe().loc["add_inchikey", "changed metadata"] == 3


def test_process_collection_filter_returning_none_stops_processing(collection):
    calls = []

    def drop_everything(collection_in, clone=True):
        calls.append("drop")

    def should_not_run(collection_in, clone=True):
        calls.append("after")
        raise AssertionError("This filter should not run after the collection was dropped.")

    processor = SpectraProcessor(filters=())
    processor.parse_and_add_filter(drop_everything)
    processor.parse_and_add_filter(should_not_run)

    assert processor.process_collection(collection) is None
    assert calls == ["drop"]


def test_process_collection_filter_returning_wrong_type_raises(collection):
    def bad_filter(collection_in, clone=True):
        return ["not", "a", "collection"]

    processor = SpectraProcessor(filters=[bad_filter])

    with pytest.raises(TypeError, match="expected SpectraCollection or None"):
        processor.process_collection(collection)


def test_process_collection_passes_clone_false_to_filters(collection):
    def assert_clone_false(collection_in, clone=True):
        assert clone is False
        return collection_in

    processor = SpectraProcessor(filters=[assert_clone_false])

    processed = processor.process_collection(collection)
    assert isinstance(processed, SpectraCollection)
    assert processed is not collection


def test_collection_requirement_filter_removes_rows(collection):
    processor = SpectraProcessor(
        filters=[("require_minimum_number_of_peaks", {"n_required": 2})]
    )

    processed = processor.process_collection(collection)

    assert len(processed) == 2
    assert processed.metadata["compound_name"].tolist() == ["compound 1", "compound 2"]


# -----------------------------------------------------------------------------
# ProcessingReport
# -----------------------------------------------------------------------------


def test_create_processing_report_contains_pipeline_steps_in_order():
    processor = SpectraProcessor(
        filters=[
            "make_charge_int",
            "interpret_pepmass",
            "derive_ionmode",
        ]
    )

    report = processor.create_processing_report()

    assert report.filter_names == [
        "make_charge_int",
        "interpret_pepmass",
        "derive_ionmode",
    ]
    assert report.to_dataframe().index.tolist() == report.filter_names


def test_no_filters_report_counts_processed_input(spectra, collection):
    processor = SpectraProcessor(filters=())
    report = processor.create_processing_report()

    processor.process_spectrum(spectra[0], processing_report=report)
    processor.process_collection(collection, processing_report=report)

    assert report.counter_number_processed == 4
    assert report.to_dataframe().empty


def test_process_spectrum_reports_metadata_and_fragment_changes(spectra):
    processor = SpectraProcessor(
        [
            "make_charge_int",
            "derive_ionmode",
            "normalize_intensities",
        ]
    )
    report = processor.create_processing_report()

    processed = processor.process_spectrum(spectra[0], processing_report=report)

    assert processed is not spectra[0]
    assert report.counter_number_processed == 1

    report_df = report.to_dataframe()

    assert report_df.loc["derive_ionmode", "input spectra"] == 1
    assert report_df.loc["derive_ionmode", "output spectra"] == 1
    assert report_df.loc["derive_ionmode", "removed spectra"] == 0
    assert report_df.loc["derive_ionmode", "changed metadata"] == 1
    assert report_df.loc["derive_ionmode", "changed fragments"] == 0

    assert report_df.loc["normalize_intensities", "changed metadata"] == 0
    assert report_df.loc["normalize_intensities", "changed fragments"] == 1


def test_processing_report_aggregates_multiple_process_spectrum_calls(spectra):
    processor = SpectraProcessor(
        filters=[
            "make_charge_int",
            "interpret_pepmass",
            "derive_ionmode",
            "correct_charge",
            ("require_minimum_number_of_peaks", {"n_required": 2}),
        ]
    )
    report = processor.create_processing_report()

    processed = [
        processor.process_spectrum(spectrum, processing_report=report)
        for spectrum in spectra
    ]
    processed = [spectrum for spectrum in processed if spectrum is not None]

    assert len(processed) == 2
    assert [spectrum.get("precursor_mz") for spectrum in processed] == [100, 102]
    assert report.counter_number_processed == 3

    report_df = report.to_dataframe()
    assert report_df.loc["make_charge_int", "changed metadata"] == 2
    assert report_df.loc["interpret_pepmass", "changed metadata"] == 3
    assert report_df.loc["derive_ionmode", "changed metadata"] == 3
    assert report_df.loc["correct_charge", "changed metadata"] == 0
    assert report_df.loc["require_minimum_number_of_peaks", "removed spectra"] == 1

    # The removal filter was reached for all three input spectra.
    assert report_df.loc["require_minimum_number_of_peaks", "input spectra"] == 3
    assert report_df.loc["require_minimum_number_of_peaks", "output spectra"] == 2


def test_process_collection_reports_metadata_fragment_and_remove_effects(collection):
    processor = SpectraProcessor(
        [
            "harmonize_missing_entries",
            ("require_minimum_number_of_peaks", {"n_required": 2}),
            "normalize_intensities",
        ]
    )
    report = processor.create_processing_report()

    processed = processor.process_collection(collection, processing_report=report)

    assert len(processed) == 2
    assert len(collection) == 3
    assert report.counter_number_processed == 3

    report_df = report.to_dataframe()

    assert report_df.loc["harmonize_missing_entries", "input spectra"] == 3
    assert report_df.loc["harmonize_missing_entries", "output spectra"] == 3
    assert report_df.loc["harmonize_missing_entries", "removed spectra"] == 0
    assert report_df.loc["harmonize_missing_entries", "changed metadata"] == 2
    assert report_df.loc["harmonize_missing_entries", "changed fragments"] == 0

    assert report_df.loc["require_minimum_number_of_peaks", "input spectra"] == 3
    assert report_df.loc["require_minimum_number_of_peaks", "output spectra"] == 2
    assert report_df.loc["require_minimum_number_of_peaks", "removed spectra"] == 1
    assert report_df.loc["require_minimum_number_of_peaks", "changed metadata"] == 0
    assert report_df.loc["require_minimum_number_of_peaks", "changed fragments"] == 0

    assert report_df.loc["normalize_intensities", "input spectra"] == 3
    assert report_df.loc["normalize_intensities", "output spectra"] == 3
    assert report_df.loc["normalize_intensities", "changed metadata"] == 0
    assert report_df.loc["normalize_intensities", "changed fragments"] == 2


def test_processing_report_aggregates_multiple_collection_calls(collection):
    processor = SpectraProcessor(filters=["harmonize_missing_entries"])
    report = processor.create_processing_report()

    processor.process_collection(collection, processing_report=report)
    processor.process_collection(collection, processing_report=report)

    report_df = report.to_dataframe()
    assert report.counter_number_processed == 6
    assert report_df.loc["harmonize_missing_entries", "input spectra"] == 6
    assert report_df.loc["harmonize_missing_entries", "output spectra"] == 6
    assert report_df.loc["harmonize_missing_entries", "changed metadata"] == 4


def test_remove_filter_report_stops_at_removed_spectrum(spectra):
    def should_not_run(spectrum, clone=True):
        raise AssertionError("This filter should not run after the spectrum was removed.")

    processor = SpectraProcessor(
        filters=[
            ("require_minimum_number_of_peaks", {"n_required": 4}),
            should_not_run,
        ]
    )
    report = processor.create_processing_report()

    processed = processor.process_spectrum(
        spectra[0],
        processing_report=report,
    )

    assert processed is None

    report_df = report.to_dataframe()

    assert report_df.loc[
        "require_minimum_number_of_peaks",
        "removed spectra",
    ] == 1

    assert report_df.loc["should_not_run", "input spectra"] == 0
    assert report_df.loc["should_not_run", "output spectra"] == 0


def test_unknown_custom_spectrum_filter_is_compared_with_both_hashes(spectra):
    def custom_metadata_update(spectrum):
        spectrum.set("custom_value", "changed")
        return spectrum

    processor = SpectraProcessor([custom_metadata_update])
    report = processor.create_processing_report()

    processor.process_spectrum(spectra[0], processing_report=report)

    report_df = report.to_dataframe()
    assert report_df.loc["custom_metadata_update", "changed metadata"] == 1
    assert report_df.loc["custom_metadata_update", "changed fragments"] == 0
    assert report_df.loc["custom_metadata_update", "removed spectra"] == 0


def test_unknown_custom_noop_filter_reports_no_changes(spectra):
    def custom_noop(spectrum):
        return spectrum

    processor = SpectraProcessor([custom_noop])
    report = processor.create_processing_report()

    processor.process_spectrum(spectra[0], processing_report=report)

    report_df = report.to_dataframe()
    assert report_df.loc["custom_noop", "changed metadata"] == 0
    assert report_df.loc["custom_noop", "changed fragments"] == 0


def test_unknown_custom_collection_filter_is_compared_when_rows_are_preserved(collection):
    def custom_metadata_update(collection_in):
        collection_in.add_metadata(["a", "b", "c"], col_name="custom_value")
        return collection_in

    processor = SpectraProcessor([custom_metadata_update])
    report = processor.create_processing_report()

    processor.process_collection(collection, processing_report=report)

    report_df = report.to_dataframe()
    assert report_df.loc["custom_metadata_update", "changed metadata"] == 3
    assert report_df.loc["custom_metadata_update", "changed fragments"] == 0
    assert report_df.loc["custom_metadata_update", "removed spectra"] == 0


def test_unknown_custom_collection_filter_with_row_removal_marks_changes_unknown(collection):
    def custom_drop_first(collection_in):
        return collection_in[1:]

    processor = SpectraProcessor([custom_drop_first])
    report = processor.create_processing_report()

    processed = processor.process_collection(collection, processing_report=report)

    assert len(processed) == 2

    report_df = report.to_dataframe()
    assert report_df.loc["custom_drop_first", "removed spectra"] == 1
    assert pd.isna(report_df.loc["custom_drop_first", "changed metadata"])
    assert pd.isna(report_df.loc["custom_drop_first", "changed fragments"])


def test_unknown_custom_filter_dropping_everything_has_no_retained_changes(collection):
    def custom_drop_everything(collection_in):
        return None

    processor = SpectraProcessor([custom_drop_everything])
    report = processor.create_processing_report()

    processed = processor.process_collection(collection, processing_report=report)

    assert processed is None

    report_df = report.to_dataframe()
    assert report_df.loc["custom_drop_everything", "removed spectra"] == 3
    assert report_df.loc["custom_drop_everything", "changed metadata"] == 0
    assert report_df.loc["custom_drop_everything", "changed fragments"] == 0


def test_processing_report_can_be_created_directly(spectra):
    def custom_metadata_update(spectrum):
        spectrum.set("custom_value", "changed")
        return spectrum

    processor = SpectraProcessor([custom_metadata_update])
    report = ProcessingReport()

    processor.process_spectrum(spectra[0], processing_report=report)

    assert report.filter_names == ["custom_metadata_update"]
    assert report.to_dataframe().loc["custom_metadata_update", "changed metadata"] == 1


def test_processing_report_dataframe_and_repr():
    report = ProcessingReport()

    assert list(report.to_dataframe().columns) == [
        "input spectra",
        "output spectra",
        "removed spectra",
        "changed metadata",
        "changed fragments",
    ]
    assert repr(report) == "ProcessingReport(n_processed=0, n_steps=0)"
    assert "Number of spectra processed: 0" in str(report)
    assert "Number of spectra removed: 0" in str(report)


# -----------------------------------------------------------------------------
# Helper functions retained by SpectraProcessor
# -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filter_params, expected_result, expected_exception",
    [
        ({"a": 2}, 5, None),
        (None, 5, None),
        ("invalid_param", None, TypeError),
    ],
)
def test_create_partial_filter(filter_params, expected_result, expected_exception):
    def sample_filter(a, b):
        return a + b

    if expected_exception:
        with pytest.raises(
            expected_exception,
            match="Expected a dictionary for filter parameters",
        ):
            create_partial_function(sample_filter, filter_params)
    else:
        partial_func = create_partial_function(sample_filter, filter_params)
        if filter_params:
            assert partial_func(b=3) == expected_result
        else:
            assert partial_func(a=2, b=3) == expected_result

        assert partial_func.__name__ == "sample_filter"


def test_check_all_parameters_given():
    def complete_filter(spectrum, optional=1):
        return spectrum

    def incomplete_filter(spectrum, required, optional=1):
        return spectrum

    check_all_parameters_given(complete_filter)

    with pytest.raises(AssertionError, match="More than one parameter"):
        check_all_parameters_given(incomplete_filter)


def test_get_parameter_settings():
    def filter_with_defaults(spectrum, a=1, b="x", clone=True):
        return spectrum

    def filter_without_defaults(spectrum):
        return spectrum

    assert get_parameter_settings(filter_with_defaults) == {
        "a": 1,
        "b": "x",
        "clone": True,
    }
    assert get_parameter_settings(filter_without_defaults) is None


def test_process_spectra(spectra):
    processor = SpectraProcessor(
        filters=[
            "make_charge_int",
            "interpret_pepmass",
            "derive_ionmode",
            "correct_charge",
        ]
    )

    processed = processor.process_spectra(
        spectra,
        progress_bar=False,
    )

    assert len(processed) == 3
    assert processed is not spectra

    actual_masses = [s.get("precursor_mz") for s in processed]
    expected_masses = [100, 102, 104]

    assert actual_masses == expected_masses

    # Input spectra should not be modified.
    assert [s.get("precursor_mz") for s in spectra] == [None, None, None]


def test_process_spectra_removes_filtered_spectra(spectra):
    processor = SpectraProcessor(
        filters=[
            ("require_minimum_number_of_peaks", {"n_required": 2}),
        ]
    )

    processed = processor.process_spectra(
        spectra,
        progress_bar=False,
    )

    assert len(processed) == 2

    assert len(processed[0].peaks) == 3
    assert len(processed[1].peaks) == 3


def test_process_spectra_skips_none(spectra):
    processor = SpectraProcessor(filters=())

    spectra_with_none = [
        spectra[0],
        None,
        spectra[1],
    ]

    processed = processor.process_spectra(
        spectra_with_none,
        progress_bar=False,
    )

    assert len(processed) == 2
    assert processed[0] == spectra[0]
    assert processed[1] == spectra[1]

    # Even with no filters, process_spectrum returns a copy.
    assert processed[0] is not spectra[0]
    assert processed[1] is not spectra[1]


def test_process_spectra_with_processing_report(spectra):
    processor = SpectraProcessor(
        filters=[
            "make_charge_int",
            "interpret_pepmass",
            ("require_minimum_number_of_peaks", {"n_required": 2}),
        ]
    )
    report = processor.create_processing_report()

    processed = processor.process_spectra(
        spectra,
        processing_report=report,
        progress_bar=False,
    )

    assert len(processed) == 2
    assert report.counter_number_processed == 3

    report_df = report.to_dataframe()

    assert report_df.loc["make_charge_int", "input spectra"] == 3
    assert report_df.loc["make_charge_int", "output spectra"] == 3
    assert report_df.loc["make_charge_int", "changed metadata"] == 2

    assert report_df.loc["interpret_pepmass", "input spectra"] == 3
    assert report_df.loc["interpret_pepmass", "output spectra"] == 3
    assert report_df.loc["interpret_pepmass", "changed metadata"] == 3

    assert (
        report_df.loc[
            "require_minimum_number_of_peaks",
            "input spectra",
        ]
        == 3
    )
    assert (
        report_df.loc[
            "require_minimum_number_of_peaks",
            "output spectra",
        ]
        == 2
    )
    assert (
        report_df.loc[
            "require_minimum_number_of_peaks",
            "removed spectra",
        ]
        == 1
    )


def test_process_spectra_accepts_generator(spectra):
    processor = SpectraProcessor(
        filters=["make_charge_int"]
    )

    spectra_generator = (spectrum for spectrum in spectra)

    processed = processor.process_spectra(
        spectra_generator,
        progress_bar=False,
    )

    assert len(processed) == 3
    assert [s.get("charge") for s in processed] == [1, -1, -1]