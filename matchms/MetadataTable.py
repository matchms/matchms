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
        drop_missing_updates: bool = True,
        **kwargs,
    ):
        """Apply a metadata function to selected rows and merge the result back.
    
        The function receives a pandas DataFrame containing either all metadata rows
        or the rows selected by ``row_mask``. It must return a DataFrame with metadata
        updates.

        The returned update DataFrame may contain fewer rows and fewer columns than
        the input subset. Its index must be a subset of the selected input rows.
        Missing values in the update DataFrame are treated as "no update" and do not
        overwrite existing metadata values.

        This method only updates metadata. It does not modify fragments.

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
        drop_missing_updates
            If True, missing values in the DataFrame returned by ``func`` are treated
            as "no update" and do not overwrite existing metadata values. If False,
            missing values are treated as explicit updates and will overwrite existing
            metadata values.
        **kwargs
            Keyword arguments passed to ``func``.

        Returns
        -------
        MetadataTable or None
            Updated metadata table if ``inplace=False``. Otherwise ``None``.
        """
        target = pd.DataFrame(self).copy()
        if row_mask is None:
            row_indices = target.index
        else:
            if isinstance(row_mask, pd.Series):
                row_mask = row_mask.values

            row_mask = np.asarray(row_mask, dtype=bool)

            if row_mask.shape[0] != len(self):
                raise ValueError(
                    f"Shape of row mask ({row_mask.shape[0]}) does not fit "
                    f"metadata table ({len(self)})."
                )

            row_indices = target.index[row_mask]

        if len(row_indices) == 0:
            return self._finalize_apply_to_rows(target, inplace=inplace)

        subset = target.loc[row_indices].copy()
        updates = func(subset, *args, **kwargs)

        if updates is None:
            return self._finalize_apply_to_rows(target, inplace=inplace)

        updates = pd.DataFrame(updates)

        self._validate_metadata_updates(
            selected_index=subset.index,
            updates=updates,
            func=func,
        )

        target = self._merge_metadata_updates(
            target,
            updates,
            drop_missing_updates=drop_missing_updates,
        )

        return self._finalize_apply_to_rows(target, inplace=inplace)


    def _validate_metadata_updates(
        self,
        selected_index: pd.Index,
        updates: pd.DataFrame,
        func,
    ) -> None:
        """Validate that metadata updates only refer to selected rows."""
        if updates.empty:
            return

        if updates.index.has_duplicates:
            raise ValueError(
                f"Function {getattr(func, '__name__', repr(func))} returned "
                "duplicate metadata row updates."
            )

        if not updates.index.isin(selected_index).all():
            raise ValueError(
                f"Function {getattr(func, '__name__', repr(func))} returned "
                "metadata updates for rows outside the selected input rows."
            )


    def _merge_metadata_updates(
        self,
        target: pd.DataFrame,
        updates: pd.DataFrame,
        *,
        drop_missing_updates: bool = True,
    ) -> pd.DataFrame:
        """Merge sparse metadata updates into target."""
        if updates.empty:
            return target

        for column in updates.columns:
            if column not in target.columns:
                target[column] = pd.Series(index=target.index, dtype="object")

            values = updates[column]

            if drop_missing_updates:
                values_to_assign = values.loc[values.notna()]
            else:
                values_to_assign = values

            if values_to_assign.empty:
                continue

            if not drop_missing_updates and values_to_assign.isna().any():
                target[column] = target[column].astype("object")
            elif _needs_object_dtype(target[column], values_to_assign):
                target[column] = target[column].astype("object")

            target.loc[values_to_assign.index, column] = values_to_assign

        return target


    def _finalize_apply_to_rows(
        self,
        target: pd.DataFrame,
        *,
        inplace: bool,
    ):
        """Write back or return metadata after apply_to_rows."""
        target = target.reset_index(drop=True)

        if inplace:
            if self._collection is not None:
                self._collection._metadata = target
                self._collection._clear_cache(["metadata_hashes", "spectra_hashes"])
            else:
                self.drop(columns=list(self.columns), inplace=True)
                for column in target.columns:
                    self[column] = target[column].values

            return None

        return MetadataTable(target, collection=self._collection)
