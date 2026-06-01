import numpy as np
import pandas as pd
import pytest
from matchms import SpectraCollection
from matchms.MetadataTable import (
    harmonize_metadata_column_name,
    harmonize_metadata_table_columns,
)
from tests.builder_Spectrum import SpectrumBuilder


def test_harmonize_metadata_column_name_lowercase_and_regex():
    assert harmonize_metadata_column_name("Compound Name") == "compound_name"
    assert harmonize_metadata_column_name("Precursor MZ") == "precursor_mz"


def test_harmonize_metadata_table_columns():
    metadata = pd.DataFrame(
        {
            "Compound Name": ["A", "B"],
            "Precursor MZ": [100.0, 200.0],
        }
    )

    result = harmonize_metadata_table_columns(metadata)

    assert list(result.columns) == ["compound_name", "precursor_mz"]
    assert result.loc[0, "compound_name"] == "A"
    assert result.loc[1, "precursor_mz"] == 200.0


def test_harmonize_metadata_table_columns_combines_duplicate_columns():
    metadata = pd.DataFrame(
        {
            "Compound Name": ["A", None],
            "compound_name": [None, "B"],
        }
    )

    result = harmonize_metadata_table_columns(metadata)

    assert list(result.columns) == ["compound_name"]
    assert result["compound_name"].tolist() == ["A", "B"]


def test_harmonize_metadata_table_columns_keeps_first_non_null_on_conflict():
    metadata = pd.DataFrame(
        {
            "Compound Name": ["A"],
            "compound_name": ["B"],
        }
    )

    result = harmonize_metadata_table_columns(metadata)

    assert list(result.columns) == ["compound_name"]
    assert result.loc[0, "compound_name"] == "A"


def test_metadata_apply_to_rows_rejects_row_reordering():
    collection = SpectraCollection(
        [
            SpectrumBuilder().with_metadata({"id": "a", "value": 2}).build(),
            SpectrumBuilder().with_metadata({"id": "b", "value": 1}).build(),
        ]
    )

    def sort_subset(metadata):
        return metadata.sort_values("value")

    with pytest.raises(ValueError, match="row index or row order"):
        collection.metadata.apply_to_rows(
            sort_subset,
            row_mask=[True, True],
        )


def test_metadata_apply_to_rows_rejects_row_dropping():
    collection = SpectraCollection(
        [
            SpectrumBuilder().with_metadata({"id": "a"}).build(),
            SpectrumBuilder().with_metadata({"id": "b"}).build(),
        ]
    )

    def drop_subset_row(metadata):
        return metadata.iloc[:1]

    with pytest.raises(ValueError, match="changed the number of metadata rows"):
        collection.metadata.apply_to_rows(
            drop_subset_row,
            row_mask=[True, True],
        )


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
        metadata = metadata.copy()
        metadata["new"] = "updated"
        return metadata

    result = collection.apply_to_metadata_rows(
        update_selected,
        row_mask=[False, True],
    )

    assert result[0].metadata.get("id") == "a"
    assert result[0].peaks.mz[0] == pytest.approx(10.0)
    assert result[1].metadata.get("id") == "b"
    assert result[1].metadata.get("new") == "updated"
    assert result[1].peaks.mz[0] == pytest.approx(20.0)
