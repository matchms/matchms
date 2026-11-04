"""Internal helpers shared by matchms Parquet import and export."""

from __future__ import annotations
from collections.abc import Sequence
from typing import Any
import numpy as np


FORMAT_NAME = "matchms.spectra_collection"
FORMAT_VERSION = "1"
DEFAULT_MZ_COLUMN = "mz"
DEFAULT_INTENSITY_COLUMN = "intensities"
DEFAULT_MZ_PRECISION = 1e-6
DEFAULT_MZ_ROUNDING = "round"

SCHEMA_KEY_FORMAT = b"matchms_format"
SCHEMA_KEY_VERSION = b"matchms_format_version"
SCHEMA_KEY_MZ_COLUMN = b"matchms_mz_column"
SCHEMA_KEY_INTENSITY_COLUMN = b"matchms_intensity_column"
SCHEMA_KEY_MZ_PRECISION = b"matchms_mz_precision"
SCHEMA_KEY_MZ_ROUNDING = b"matchms_mz_rounding"

MZ_COLUMN_CANDIDATES = (
    DEFAULT_MZ_COLUMN,
    "mzs",
    "ms2_mzs",
    "fragment_mz",
    "fragment_mzs",
    "m/z array",
    "m/z_array",
)

INTENSITY_COLUMN_CANDIDATES = (
    DEFAULT_INTENSITY_COLUMN,
    "intensity",
    "ms2_intensities",
    "ms2_normalized_intensities",
    "fragment_intensities",
    "intensity array",
    "intensity_array",
)


def decode_schema_metadata(metadata: dict[bytes, bytes] | None) -> dict[str, str]:
    """Decode Parquet schema metadata written by matchms."""
    if not metadata:
        return {}

    decoded = {}
    for key, value in metadata.items():
        try:
            decoded[key.decode("utf-8")] = value.decode("utf-8")
        except (AttributeError, UnicodeDecodeError):
            continue
    return decoded


def resolve_peak_column(
    available_columns: Sequence[str],
    explicit_column: str | None,
    schema_column: str | None,
    candidates: Sequence[str],
    kind: str,
) -> str:
    """Resolve one peak column from explicit, schema, or well-known names."""
    available = list(available_columns)

    if explicit_column is not None:
        if explicit_column not in available:
            raise ValueError(
                f"Requested {kind} column {explicit_column!r} was not found. "
                f"Available columns are: {available}."
            )
        return explicit_column

    if schema_column is not None and schema_column in available:
        return schema_column

    matches = [candidate for candidate in candidates if candidate in available]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"Multiple possible {kind} columns were found: {matches}. "
            f"Pass `{kind}_column=...` explicitly."
        )

    raise ValueError(
        f"Could not identify the {kind} column. Pass `{kind}_column=...` explicitly. "
        f"Available columns are: {available}."
    )


def normalize_requested_columns(
    columns: Sequence[str] | None,
    mz_column: str,
    intensity_column: str,
) -> list[str] | None:
    """Ensure peak columns are read even when metadata columns are subsetted."""
    if columns is None:
        return None

    requested = list(dict.fromkeys(columns))
    for peak_column in (mz_column, intensity_column):
        if peak_column not in requested:
            requested.append(peak_column)
    return requested


def schema_metadata_for_collection(
    *,
    mz_column: str,
    intensity_column: str,
    mz_precision: float,
    mz_rounding: str,
) -> dict[bytes, bytes]:
    """Return matchms-specific Parquet schema metadata."""
    return {
        SCHEMA_KEY_FORMAT: FORMAT_NAME.encode("utf-8"),
        SCHEMA_KEY_VERSION: FORMAT_VERSION.encode("utf-8"),
        SCHEMA_KEY_MZ_COLUMN: mz_column.encode("utf-8"),
        SCHEMA_KEY_INTENSITY_COLUMN: intensity_column.encode("utf-8"),
        SCHEMA_KEY_MZ_PRECISION: repr(float(mz_precision)).encode("utf-8"),
        SCHEMA_KEY_MZ_ROUNDING: mz_rounding.encode("utf-8"),
    }


def metadata_value_to_arrow_compatible(value: Any) -> Any:
    """Convert common NumPy metadata values to Arrow-friendly Python values."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
