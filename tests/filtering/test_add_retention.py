import pandas as pd
import pytest
from matchms import SpectraCollection
from matchms.filtering import add_retention_index, add_retention_time
from tests.run_spectrum_and_collection import run_filter_as_spectrum_or_collection
from ..builder_Spectrum import SpectrumBuilder


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
@pytest.mark.parametrize(
    "metadata, expected",
    [
        [{"retention_time": 100.0}, 100.0],
        [{"retention_time": "NA"}, None],
        [{"retention_time": "100.0"}, 100.0],
        [{"retentiontime": 200}, 200.0],
        [{"retentiontime": -1}, None],
        [{"retentiontime": "-1"}, None],
        [{"rt": "4.810467 min"}, 288.62802],
        [{"rt": "no retention time in min available"}, None],
        [{"rt": 200}, 200.0],
        [{"RT": 200}, 200.0],
        [{"RT_Query": 200}, 200.0],
        [{"nothing": "200"}, None],
        [{"scan_start_time": 0.629566}, 0.629566],
        [{"scan_start_time": [0.629566]}, 0.629566],
        [{"rt": None, "retentiontime": 12.17}, 12.17],
        [{"retention_time": "100.0 sec"}, 100.0],
    ],
)
def test_add_retention_time(metadata, expected, as_collection):
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    spectrum = run_filter_as_spectrum_or_collection(
        add_retention_time,
        spectrum_in,
        as_collection,
    )
    actual = spectrum.get("retention_time")

    if expected is None:
        assert actual is None
    else:
        assert actual == pytest.approx(expected)
        assert isinstance(actual, (float, int))  # make less strict (SpectraCollection may convert to int)


@pytest.mark.parametrize("as_collection", [False, True], ids=["spectrum", "collection"])
@pytest.mark.parametrize(
    "metadata, expected",
    [
        [{"retention_index": 100.0}, 100.0],
        [{"retention_index": "NA"}, None],
        [{"retention_index": "100.0"}, 100.0],
        [{"retentionindex": 200}, 200.0],
        [{"retentionindex": -1}, None],
        [{"retentionindex": "-1"}, None],
        [{"ri": 200}, 200.0],
        [{"RI": 200}, 200.0],
        [{"nothing": "200"}, None],
    ],
)
def test_add_retention_index(metadata, expected, as_collection):
    spectrum_in = SpectrumBuilder().with_metadata(metadata).build()

    spectrum = run_filter_as_spectrum_or_collection(
        add_retention_index,
        spectrum_in,
        as_collection,
    )
    actual = spectrum.get("retention_index")

    if expected is None:
        assert actual is None
    else:
        assert actual == pytest.approx(expected)
        assert isinstance(actual, (float, int))  # make less strict (SpectraCollection may convert to int)


def test_add_retention_time_collection_multiple_rows():
    collection = SpectraCollection(
        [
            SpectrumBuilder().with_metadata({"rt": "4.810467 min"}).build(),
            SpectrumBuilder().with_metadata({"retentiontime": -1}).build(),
            SpectrumBuilder().with_metadata({"nothing": "200"}).build(),
        ]
    )

    processed = add_retention_time(collection)

    assert processed is not collection
    assert processed.metadata.loc[0, "retention_time"] == pytest.approx(288.62802)
    assert pd.isna(processed.metadata.loc[1, "retention_time"])
    assert "retention_time" not in processed.metadata.columns or pd.isna(processed.metadata.loc[2, "retention_time"])


def test_add_retention_index_collection_multiple_rows():
    collection = SpectraCollection(
        [
            SpectrumBuilder().with_metadata({"RI": 200}).build(),
            SpectrumBuilder().with_metadata({"retentionindex": "-1"}).build(),
            SpectrumBuilder().with_metadata({"nothing": "200"}).build(),
        ]
    )

    processed = add_retention_index(collection)

    assert processed is not collection
    assert processed.metadata.loc[0, "retention_index"] == pytest.approx(200.0)
    assert pd.isna(processed.metadata.loc[1, "retention_index"])
    assert "retention_index" not in processed.metadata.columns or pd.isna(processed.metadata.loc[2, "retention_index"])


def test_add_retention_time_collection_clone_false_modifies_input():
    collection = SpectraCollection(
        [
            SpectrumBuilder().with_metadata({"rt": 200}).build(),
        ]
    )

    processed = add_retention_time(collection, clone=False)

    assert processed is collection
    assert collection.metadata.loc[0, "retention_time"] == pytest.approx(200.0)


def test_add_retention_index_collection_clone_false_modifies_input():
    collection = SpectraCollection(
        [
            SpectrumBuilder().with_metadata({"ri": 200}).build(),
        ]
    )

    processed = add_retention_index(collection, clone=False)

    assert processed is collection
    assert collection.metadata.loc[0, "retention_index"] == pytest.approx(200.0)


def test_add_retention_empty_spectrum():
    assert add_retention_time(None) is None
    assert add_retention_index(None) is None