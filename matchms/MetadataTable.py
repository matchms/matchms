import logging
import re
import numpy as np
import pandas as pd
from .utils import load_known_key_conversions


logger = logging.getLogger("matchms")
_key_regex_replacements = {r"\s": "_", r"[!?.,;:]": ""}
_key_replacements = load_known_key_conversions()


def _needs_object_dtype(target_column: pd.Series, values: pd.Series) -> bool:
    """Return True if target column should be object before assigning values."""
    if pd.api.types.is_object_dtype(values.dtype):
        return True

    if pd.api.types.is_string_dtype(values.dtype):
        return True

    if pd.api.types.is_bool_dtype(target_column.dtype):
        return not pd.api.types.is_bool_dtype(values.dtype)

    if pd.api.types.is_numeric_dtype(target_column.dtype):
        return not pd.api.types.is_numeric_dtype(values.dtype)

    return True


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

    def apply_to_rows(
        self,
        func,
        *args,
        row_mask=None,
        inplace: bool = False,
        **kwargs,
    ):
        """Apply a metadata function to selected rows and merge the result back.
    
        The function may add or modify metadata columns, but it must preserve the
        selected rows exactly: same index, same order, same number of rows.

        Parameters
        ----------
        func
            Function that receives a ``MetadataTable`` or ``DataFrame`` subset as
            first argument and returns a ``DataFrame``/``MetadataTable`` or ``None``.
        *args
            Positional arguments passed to ``func``.
        row_mask
            Optional boolean mask selecting rows. If ``None``, all rows are passed
            directly to ``func``.
        inplace
            If True, update the bound collection metadata in place and return
            ``None``. If False, return a new ``MetadataTable``.
        **kwargs
            Keyword arguments passed to ``func``.

        Returns
        -------
        MetadataTable or None
            Updated metadata table if ``inplace=False``. Otherwise ``None``.

        Notes
        -----
        This method only updates metadata. It does not modify fragments.
        """
        if row_mask is not None:
            if isinstance(row_mask, pd.Series):
                row_mask = row_mask.values

            row_mask = np.asarray(row_mask, dtype=bool)

            if row_mask.shape[0] != len(self):
                raise ValueError(
                    f"Shape of row mask ({row_mask.shape[0]}) does not fit "
                    f"metadata table ({len(self)})."
                )
        else:
            row_mask = np.ones(len(self), dtype=bool)

        target = pd.DataFrame(self).copy()
        row_indices = target.index[row_mask]

        if len(row_indices) == 0:
            result = MetadataTable(target, collection=self._collection)
            return None if inplace else result

        subset = target.loc[row_indices].copy()

        processed_subset = func(subset, *args, **kwargs)

        if processed_subset is None:
            result = MetadataTable(target, collection=self._collection)
            return None if inplace else result

        processed_subset = pd.DataFrame(processed_subset)

        if len(processed_subset) != len(subset):
            raise ValueError(
                f"Function {getattr(func, '__name__', repr(func))} changed the number "
                f"of metadata rows from {len(subset)} to {len(processed_subset)}. "
                "MetadataTable.apply_to_rows only supports row-preserving transformations."
            )

        if not processed_subset.index.equals(subset.index):
            raise ValueError(
                f"Function {getattr(func, '__name__', repr(func))} changed the "
                "metadata row index or row order. MetadataTable.apply_to_rows only "
                "supports row-preserving metadata transformations."
            )

        for column in processed_subset.columns:
            if column not in target.columns:
                target[column] = pd.Series(index=target.index, dtype="object")

            values = processed_subset[column]

            if _needs_object_dtype(target[column], values):
                target[column] = target[column].astype("object")

            # Important: assign an aligned Series, not values.to_numpy(dtype=object).
            # This avoids pandas >= 3 failures for numeric columns.
            target.loc[values.index, column] = values

        if inplace:
            if self._collection is not None:
                self._collection._metadata = target.reset_index(drop=True)
                self._collection._clear_cache(["metadata_hashes", "spectra_hashes"])
            else:
                self.drop(columns=list(self.columns), inplace=True)
                for column in target.columns:
                    self[column] = target[column].values

            return None

        return MetadataTable(target.reset_index(drop=True), collection=self._collection)
