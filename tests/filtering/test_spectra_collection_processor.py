import os
import pandas as pd
import pytest
from matchms import SpectraCollection
from matchms import filtering as msfilters
from matchms.filtering.SpectraCollectionProcessor import SpectraCollectionProcessor
from matchms.importing.load_spectra import load_spectra
from tests.builder_Spectrum import SpectrumBuilder


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


def test_filter_sorting_and_output():
    processor = SpectraCollectionProcessor(
        filters=[
            "select_by_relative_intensity",
            "harmonize_missing_entries",
        ]
    )

    actual_filters = [x.__name__ for x in processor.filters]
    expected_filters = [
        "harmonize_missing_entries",
        "select_by_relative_intensity",
    ]
    assert actual_filters == expected_filters

    expected_steps = [
        ("harmonize_missing_entries", {"keys": None, "undefined": None, "aliases": None, "clone": True}),
        ("select_by_relative_intensity", {"intensity_from": 0.0, "intensity_to": 1.0, "clone": True}),
    ]
    assert processor.processing_steps == expected_steps


@pytest.mark.parametrize(
    "filter_step, expected",
    [
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
    processor = SpectraCollectionProcessor(filters=())
    processor.parse_and_add_filter(filter_step)

    assert processor.processing_steps == [expected]


def test_incomplete_parameters():
    def custom_filter(collection, required_parameter, clone=True):
        return collection

    processor = SpectraCollectionProcessor(filters=())

    with pytest.raises(AssertionError):
        processor.parse_and_add_filter(custom_filter)

    with pytest.raises(ValueError):
        processor.parse_and_add_filter(
            ("harmonize_missing_entries", {"keys": ["smiles"]}, "unexpected")
        )


def test_string_output():
    processor = SpectraCollectionProcessor(
        filters=[
            "harmonize_missing_entries",
            ("select_by_relative_intensity", {"intensity_from": 0.01}),
        ]
    )

    assert processor.processing_steps == [
        (
            "harmonize_missing_entries",
            {
                "keys": None,
                "undefined": None,
                "aliases": None,
                "clone": True,
            },
        ),
        (
            "select_by_relative_intensity",
            {
                "intensity_from": 0.01,
                "intensity_to": 1.0,
                "clone": True,
            },
        ),
    ]

    processor_str = str(processor)
    assert "Processing steps:" in processor_str
    assert "harmonize_missing_entries" in processor_str
    assert "select_by_relative_intensity" in processor_str


def test_no_filters(collection):
    processor = SpectraCollectionProcessor(filters=())

    processed = processor.process_collection(collection)

    assert processed is not collection
    assert isinstance(processed, SpectraCollection)
    assert len(processed) == len(collection)
    assert processed.metadata.equals(collection.metadata)


def test_process_collection(collection):
    processor = SpectraCollectionProcessor(
        filters=[
            "harmonize_missing_entries",
            ("select_by_relative_intensity", {"intensity_from": 0.02}),
        ]
    )

    processed = processor.process_collection(collection)

    assert isinstance(processed, SpectraCollection)
    assert len(processed) == 3

    assert processed.metadata.loc[0, "smiles"] is None
    assert processed.metadata.loc[1, "smiles"] == "CCCO"
    assert processed.metadata.loc[2, "smiles"] is None

    assert len(processed[0].peaks) == 2
    assert len(processed[1].peaks) == 3
    assert len(processed[2].peaks) == 1


def test_process_spectra_from_iterable(spectra):
    processor = SpectraCollectionProcessor(
        filters=[
            "harmonize_missing_entries",
            ("select_by_relative_intensity", {"intensity_from": 0.01}),
        ]
    )

    processed = processor.process_spectra(spectra)

    assert isinstance(processed, SpectraCollection)
    assert len(processed) == 3
    assert pd.isna(processed.metadata.loc[0, "smiles"])
    assert processed.metadata.loc[1, "smiles"] == "CCCO"
    assert pd.isna(processed.metadata.loc[2, "smiles"])


def test_process_spectra_from_collection(collection):
    processor = SpectraCollectionProcessor(
        filters=[
            "harmonize_missing_entries",
        ]
    )

    processed = processor.process_spectra(collection)

    assert isinstance(processed, SpectraCollection)
    assert len(processed) == len(collection)
    assert processed is not collection
    assert pd.isna(processed.metadata.loc[0, "smiles"])


def test_process_collection_rejects_non_collection(spectra):
    processor = SpectraCollectionProcessor(filters=())

    with pytest.raises(
        TypeError,
        match="process_collection expects a SpectraCollection",
    ):
        processor.process_collection(spectra)


def test_adding_custom_collection_filter(collection):
    def add_inchikey(collection, clone=True):
        target = collection.copy() if clone else collection
        target.add_metadata(["NONSENSE"] * len(target), col_name="inchikey")
        return target

    processor = SpectraCollectionProcessor(
        filters=[
            "harmonize_missing_entries",
        ]
    )
    processor.parse_and_add_filter(add_inchikey)

    assert processor.filters[-1].__name__ == "add_inchikey"

    processed = processor.process_collection(collection)

    assert processed.metadata["inchikey"].tolist() == ["NONSENSE", "NONSENSE", "NONSENSE"]


def test_adding_custom_collection_filter_with_parameters(collection):
    def add_repeated_inchikey(collection, number, clone=True):
        target = collection.copy() if clone else collection
        target.add_metadata([number * "NONSENSE"] * len(target), col_name="inchikey")
        return target

    processor = SpectraCollectionProcessor(
        filters=[
            "harmonize_missing_entries",
        ]
    )
    processor.parse_and_add_filter((add_repeated_inchikey, {"number": 2}))

    assert processor.filters[-1].__name__ == "add_repeated_inchikey"

    processed = processor.process_collection(collection)

    assert processed.metadata["inchikey"].tolist() == [
        "NONSENSENONSENSE",
        "NONSENSENONSENSE",
        "NONSENSENONSENSE",
    ]


@pytest.mark.parametrize(
    "filter_position, expected",
    [[0, 0], [1, 1], [None, 1], [5, 1]],
)
def test_add_custom_filter_in_position(filter_position, expected):
    def custom_collection_filter(collection):
        return collection

    processor = SpectraCollectionProcessor(
        filters=[
            "harmonize_missing_entries",
        ]
    )
    processor.parse_and_add_filter(
        custom_collection_filter,
        filter_position=filter_position,
    )

    filters = processor.filters

    assert filters[expected].__name__ == "custom_collection_filter"
    assert len(filters) == 2


def test_add_matchms_filter_in_position():
    processor = SpectraCollectionProcessor(
        filters=[
            "harmonize_missing_entries",
            "select_by_intensity",
        ]
    )
    processor.parse_and_add_filter("select_by_relative_intensity", filter_position=1)

    filters = processor.filters

    assert filters[1].__name__ == "select_by_relative_intensity"
    assert len(filters) == 3


@pytest.mark.parametrize(
    "filter_description",
    [
        ("select_by_relative_intensity", {"intensity_from": 0.01}),
        (msfilters.select_by_relative_intensity, {"intensity_from": 0.01}),
    ],
)
def test_add_matchms_filter(filter_description, collection):
    processor = SpectraCollectionProcessor(
        filters=[
            "harmonize_missing_entries",
        ]
    )
    processor.parse_and_add_filter(filter_description)

    filters = processor.filters
    assert filters[-1].__name__ == "select_by_relative_intensity"

    processed = processor.process_collection(collection)

    assert isinstance(processed, SpectraCollection)
    assert len(processed) == 3


def test_add_duplicated_filter_to_existing_pipeline():
    processor = SpectraCollectionProcessor(
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


def test_add_filter_twice():
    processor = SpectraCollectionProcessor(filters=())

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


def test_add_all_filter_types(collection):
    def add_inchikey(collection, clone=True):
        target = collection.copy() if clone else collection
        target.add_metadata(["NONSENSE"] * len(target), col_name="inchikey")
        return target

    def add_repeated_inchikey(collection, number, clone=True):
        target = collection.copy() if clone else collection
        target.add_metadata([number * "NONSENSE"] * len(target), col_name="inchikey", overwrite=True)
        return target

    processor = SpectraCollectionProcessor(
        filters=[
            "harmonize_missing_entries",
            msfilters.select_by_intensity,
            add_inchikey,
            (msfilters.select_by_relative_intensity, {"intensity_from": 0.01}),
            (add_repeated_inchikey, {"number": 2}),
        ]
    )

    filters = processor.filters

    assert [filter_func.__name__ for filter_func in filters] == [
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


def test_filter_returning_none_stops_processing(collection):
    def drop_everything(collection, clone=True):
        return None

    def should_not_run(collection, clone=True):
        raise AssertionError("This filter should not run after collection was dropped.")

    processor = SpectraCollectionProcessor(filters=())
    processor.parse_and_add_filter(drop_everything)
    processor.parse_and_add_filter(should_not_run)

    processed = processor.process_collection(collection)

    assert processed is None


def test_filter_returning_wrong_type_raises(collection):
    def bad_filter(collection, clone=True):
        return ["not", "a", "collection"]

    processor = SpectraCollectionProcessor(filters=())
    processor.parse_and_add_filter(bad_filter)

    with pytest.raises(
        TypeError,
        match="expected SpectraCollection or None",
    ):
        processor.process_collection(collection)


def test_processor_passes_clone_false_to_filters(collection):
    def assert_clone_false(collection, clone=True):
        assert clone is False
        return collection

    processor = SpectraCollectionProcessor(filters=())
    processor.parse_and_add_filter(assert_clone_false)

    processed = processor.process_collection(collection)

    assert isinstance(processed, SpectraCollection)


def test_save_spectra_collection_processor(spectra, tmp_path):
    processor = SpectraCollectionProcessor(
        filters=[
            "harmonize_missing_entries",
            "select_by_relative_intensity",
        ]
    )
    filename = os.path.join(tmp_path, "spectra.msp")

    processed = processor.process_spectra(spectra, cleaned_spectra_file=str(filename))

    assert isinstance(processed, SpectraCollection)
    assert os.path.exists(filename)

    reloaded_spectra = list(load_spectra(str(filename)))

    assert len(reloaded_spectra) == len(spectra)


def test_save_spectra_collection_processor_existing_file_raises(spectra, tmp_path):
    processor = SpectraCollectionProcessor(filters=())
    filename = os.path.join(tmp_path, "spectra.msp")

    with open(filename, "w", encoding="utf-8") as file:
        file.write("already exists")

    with pytest.raises(FileExistsError):
        processor.process_spectra(spectra, cleaned_spectra_file=str(filename))
