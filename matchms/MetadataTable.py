import logging
import re
import pandas as pd
from .utils import load_known_key_conversions


logger = logging.getLogger("matchms")
_key_regex_replacements = {r"\s": "_", r"[!?.,;:]": ""}
_key_replacements = load_known_key_conversions()


def harmonize_metadata_column_name(column_name: str) -> str:
    """Return the matchms-style metadata column name."""
    column_name = column_name.lower()

    for regex_pattern, replacement in _key_regex_replacements.items():
        column_name = re.sub(regex_pattern, replacement, column_name)

    if column_name in _key_replacements:
        column_name = _key_replacements[column_name]

    return column_name


def harmonize_metadata_table_columns(metadata: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with harmonized metadata column names.

    If multiple columns map to the same harmonized column name, values are
    combined row-wise. Existing non-null values are kept, and missing values are
    filled from duplicate columns.
    """
    rename_map = {
        column: harmonize_metadata_column_name(str(column))
        for column in metadata.columns
    }

    harmonized = pd.DataFrame(index=metadata.index)

    for old_column, new_column in rename_map.items():
        values = metadata[old_column]

        if new_column not in harmonized.columns:
            harmonized[new_column] = values
            continue

        conflict_mask = (
            harmonized[new_column].notna()
            & values.notna()
            & (harmonized[new_column] != values)
        )

        if conflict_mask.any():
            logger.warning(
                "Metadata column '%s' maps to existing column '%s' with "
                "different non-null values. Keeping existing values and filling "
                "missing values only.",
                old_column,
                new_column,
            )

        harmonized[new_column] = harmonized[new_column].combine_first(values)

    return harmonized


class MetadataTable(pd.DataFrame):
    """
    Metadata proxy class.
    Used for filter directly on metadata and synchronize fragments.
    """

    _metadata = ["_collection"]

    def __init__(self, data, collection=None, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        object.__setattr__(self, "_collection", collection)

    @property
    def _constructor(self):
        def _c(*args, **kwargs):
            return MetadataTable(*args, collection=self._collection, **kwargs)

        return _c

    def sort_values(self, by, inplace=False, **kwargs):
        result = self._collection.sort(by=by, inplace=inplace, **kwargs)
        return None if inplace else result.metadata

    def harmonize_columns(self, inplace: bool = False):
        """Harmonize metadata columns to matchms key style."""
        harmonized = harmonize_metadata_table_columns(self)

        if inplace:
            self.drop(columns=list(self.columns), inplace=True)
            for column in harmonized.columns:
                self[column] = harmonized[column].values

            if self._collection is not None:
                self._collection._metadata = pd.DataFrame(self).reset_index(drop=True)
                self._collection._clear_cache(["metadata_hashes", "spectra_hashes"])

            return None

        return MetadataTable(harmonized, collection=self._collection)
