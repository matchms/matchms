import numpy as np
import pandas as pd
import pytest
from matchms import SpectraCollection
from matchms.MetadataCollection import (
    MetadataCollection,
    harmonize_metadata_collection_columns,
    harmonize_metadata_column_name,
)
from tests.builder_Spectrum import SpectrumBuilder


def test_harmonize_metadata_column_name_lowercase_and_regex():
    assert harmonize_metadata_column_name("Compound Name") == "compound_name"
    assert harmonize_metadata_column_name("Precursor MZ") == "precursor_mz"


def test_harmonize_metadata_collection_columns():
    metadata = pd.DataFrame(
        {
            "Compound Name": ["A", "B"],
            "Precursor MZ": [100.0, 200.0],
        }
    )

    result = harmonize_metadata_collection_columns(metadata)

    assert list(result.columns) == ["compound_name", "precursor_mz"]
    assert result.loc[0, "compound_name"] == "A"
    assert result.loc[1, "precursor_mz"] == 200.0


def test_harmonize_metadata_collection_columns_combines_duplicate_columns():
    metadata = pd.DataFrame(
        {
            "Compound Name": ["A", None],
            "compound_name": [None, "B"],
        }
    )

    result = harmonize_metadata_collection_columns(metadata)

    assert list(result.columns) == ["compound_name"]
    assert result["compound_name"].tolist() == ["A", "B"]


def test_harmonize_metadata_collection_columns_keeps_first_non_null_on_conflict():
    metadata = pd.DataFrame(
        {
            "Compound Name": ["A"],
            "compound_name": ["B"],
        }
    )

    result = harmonize_metadata_collection_columns(metadata)

    assert list(result.columns) == ["compound_name"]
    assert result.loc[0, "compound_name"] == "A"


def test_metadata_apply_to_rows_rejects_updates_for_unselected_rows():
    collection = SpectraCollection(
        [
            SpectrumBuilder().with_metadata({"id": "a", "value": 1}).build(),
            SpectrumBuilder().with_metadata({"id": "b", "value": 2}).build(),
        ]
    )

    def update_unselected_row(metadata):
        return pd.DataFrame({"value": [100]}, index=[1])

    with pytest.raises(ValueError, match="outside the selected input rows"):
        collection.metadata.apply_to_rows(
            update_unselected_row,
            row_mask=[True, False],
        )


def test_metadata_apply_to_rows_allows_sparse_updates():
    collection = SpectraCollection(
        [
            SpectrumBuilder().with_metadata({"id": "a", "value": 1}).build(),
            SpectrumBuilder().with_metadata({"id": "b", "value": 2}).build(),
            SpectrumBuilder().with_metadata({"id": "c", "value": 3}).build(),
        ]
    )

    def update_only_first_selected_row(metadata):
        return pd.DataFrame({"value": [100]}, index=[0])

    result = collection.metadata.apply_to_rows(
        update_only_first_selected_row,
        row_mask=[True, True, False],
    )

    assert result.loc[0, "value"] == 100
    assert result.loc[1, "value"] == 2
    assert result.loc[2, "value"] == 3


def test_metadata_apply_to_rows_preserves_fragment_alignment():
    collection = SpectraCollection(
        [
            SpectrumBuilder()
            .with_metadata({"id": "a"})
            .with_mz(np.array([10.0]))
            .with_intensities(np.array([1.0]))
            .build(),
            SpectrumBuilder()
            .with_metadata({"id": "b"})
            .with_mz(np.array([20.0]))
            .with_intensities(np.array([2.0]))
            .build(),
        ]
    )

    def update_selected(metadata):
        return pd.DataFrame({"new": ["updated"] * len(metadata)}, index=metadata.index)

    result = collection.apply_to_metadata_rows(
        update_selected,
        row_mask=[False, True],
    )

    assert result[0].metadata.get("id") == "a"
    assert result[0].peaks.mz[0] == pytest.approx(10.0)
    assert result[1].metadata.get("id") == "b"
    assert result[1].metadata.get("new") == "updated"
    assert result[1].peaks.mz[0] == pytest.approx(20.0)


def test_metadata_apply_to_rows_rejects_duplicate_update_indices():
    collection = SpectraCollection(
        [
            SpectrumBuilder().with_metadata({"id": "a", "value": 1}).build(),
            SpectrumBuilder().with_metadata({"id": "b", "value": 2}).build(),
        ]
    )

    def duplicate_updates(metadata):
        return pd.DataFrame({"value": [10, 20]}, index=[0, 0])

    with pytest.raises(ValueError, match="duplicate metadata row updates"):
        collection.metadata.apply_to_rows(
            duplicate_updates,
            row_mask=[True, True],
        )


def test_metadata_apply_to_rows_without_mask_updates_all_rows():
    collection = SpectraCollection(
        [
            SpectrumBuilder().with_metadata({"id": "a", "value": 1}).build(),
            SpectrumBuilder().with_metadata({"id": "b", "value": 2}).build(),
        ]
    )

    def update_all_rows(metadata):
        return pd.DataFrame({"value": [10, 20]}, index=metadata.index)

    result = collection.metadata.apply_to_rows(update_all_rows)

    assert result["value"].tolist() == [10, 20]


def test_row_to_dict_converts_missing_and_numpy_scalars():
    row = pd.Series(
        {
            "scan_number": np.int64(0),
            "precursor_mz": np.float64(123.4),
            "ionmode": np.nan,
            "compound_name": "test",
        }
    )

    metadata = MetadataCollection.row_to_dict(row)

    assert metadata == {
        "scan_number": 0,
        "precursor_mz": 123.4,
        "ionmode": None,
        "compound_name": "test",
    }
    assert isinstance(metadata["scan_number"], int)
    assert isinstance(metadata["precursor_mz"], float)