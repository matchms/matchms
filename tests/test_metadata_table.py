import pandas as pd
from matchms.MetadataTable import (
    harmonize_metadata_column_name,
    harmonize_metadata_table_columns,
)


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
