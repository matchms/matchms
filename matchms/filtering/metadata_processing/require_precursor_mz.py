import logging
import numpy as np
import pandas as pd
from matchms.filtering._dispatch import metadata_requirement_filter
from matchms.filtering.filter_utils.metadata_conversions import (
    as_float_or_none,
    is_missing_metadata_value,
)
from matchms.spectra_collection import SpectraCollection


logger = logging.getLogger("matchms")


_NUMERIC_PRECURSOR_TYPES = (
    int,
    float,
    np.integer,
    np.floating,
)


def _require_precursor_mz(
    metadata,
    minimum_accepted_mz: float | None = 10.0,
    maximum_mz: float | None = None,
) -> bool:
    """Require precursor m/z to be present and within optional bounds.

    Parameters
    ----------
    spectrum_in
        Input :class:`~matchms.spectrum.Spectrum` or :class:`~matchms.spectra_collection.SpectraCollection`.
    minimum_accepted_mz
        Minimum accepted precursor m/z. Default is ``10.0``. Use ``None`` to
        disable the lower bound.
    maximum_mz
        Maximum accepted precursor m/z. Default is ``None``.
    clone
        Optionally clone the input before applying the filter. If ``False``,
        the input object may be modified in place.

    Returns
    -------
    Spectrum, SpectraCollection, or None
        Spectrum input is returned unchanged if precursor m/z passes the checks,
        otherwise ``None``. SpectraCollection input is returned with failing rows
        removed.
    """
    precursor_mz = metadata.get("precursor_mz", None)

    if is_missing_metadata_value(precursor_mz):
        pepmass = metadata.get("pepmass", None)

        if not is_missing_metadata_value(pepmass):
            try:
                pepmass_mz = pepmass[0]
            except (TypeError, IndexError):
                pepmass_mz = None

            assert not isinstance(pepmass_mz, (float, int)), (
                "Found 'pepmass' but no 'precursor_mz'. Consider applying "
                "'add_precursor_mz' filter first."
            )

        return False

    precursor_mz_float = as_float_or_none(precursor_mz)

    if precursor_mz_float is None:
        logger.warning(
            "Precursor mz was not a number (%s) consider applying "
            "'add_precursor_mz' filter first",
            precursor_mz,
        )
        return False

    if (
        minimum_accepted_mz is not None
        and precursor_mz_float < minimum_accepted_mz
    ):
        logger.info(
            "Spectrum is removed since precursor mz (%s) was below minimum mz (%s)",
            precursor_mz_float,
            minimum_accepted_mz,
        )
        return False

    if (
        maximum_mz is not None
        and precursor_mz_float > maximum_mz
    ):
        logger.info(
            "Spectrum is removed since precursor mz (%s) was above maximum mz (%s)",
            precursor_mz_float,
            maximum_mz,
        )
        return False

    return True


def _require_precursor_mz_collection(
    spectrum_in: SpectraCollection,
    minimum_accepted_mz: float | None = 10.0,
    maximum_mz: float | None = None,
    clone: bool | None = True,
) -> SpectraCollection:
    """Apply precursor-m/z requirements using vectorized metadata operations."""
    metadata = spectrum_in.metadata
    n_spectra = len(spectrum_in)

    if n_spectra == 0:
        return (
            spectrum_in.copy()
            if clone
            else spectrum_in
        )

    # No precursor_mz column is an unusual/error-path situation.
    # Sticking to complete row-wise logic here, including the legacy pepmass check.
    if "precursor_mz" not in metadata.columns:
        if "pepmass" not in metadata.columns:
            keep_mask = np.zeros(
                n_spectra,
                dtype=bool,
            )
        else:
            keep_mask = np.fromiter(
                (
                    _require_precursor_mz(
                        metadata.iloc[row],
                        minimum_accepted_mz=minimum_accepted_mz,
                        maximum_mz=maximum_mz,
                    )
                    for row in range(n_spectra)
                ),
                dtype=bool,
                count=n_spectra,
            )

        return _filter_collection(
            spectrum_in,
            keep_mask,
            clone=clone,
        )

    precursor = metadata["precursor_mz"]

    # ---------------------------------------------------------
    # Fast path 1:
    # This should be the common SpectraCollection case after metadata harmonization.
    # ---------------------------------------------------------
    if (
        pd.api.types.is_integer_dtype(precursor.dtype)
        or pd.api.types.is_float_dtype(precursor.dtype)
    ):
        values = precursor.to_numpy(
            dtype=np.float64,
            na_value=np.nan,
        )

        keep_mask = ~np.isnan(values)

        if minimum_accepted_mz is not None:
            below_minimum = (
                keep_mask
                & (values < minimum_accepted_mz)
            )

            if np.any(below_minimum):
                logger.info(
                    "%d spectra removed because precursor m/z was below %s.",
                    int(below_minimum.sum()),
                    minimum_accepted_mz,
                )

            keep_mask[below_minimum] = False

        if maximum_mz is not None:
            above_maximum = (
                keep_mask
                & (values > maximum_mz)
            )

            if np.any(above_maximum):
                logger.info(
                    "%d spectra removed because precursor m/z was above %s.",
                    int(above_maximum.sum()),
                    maximum_mz,
                )

            keep_mask[above_maximum] = False

        return _filter_collection(
            spectrum_in,
            keep_mask,
            clone=clone,
        )

    # ---------------------------------------------------------
    # Fast path 2:
    # Mixed/object column. Process ordinary numeric scalar
    # entries together and use the full legacy logic only for unusual values.
    # ---------------------------------------------------------
    raw_values = precursor.to_numpy(
        dtype=object,
        copy=False,
    )

    numeric_mask = np.fromiter(
        (
            isinstance(
                value,
                _NUMERIC_PRECURSOR_TYPES,
            )
            for value in raw_values
        ),
        dtype=bool,
        count=n_spectra,
    )

    numeric_values = np.full(
        n_spectra,
        np.nan,
        dtype=np.float64,
    )

    if np.any(numeric_mask):
        numeric_values[numeric_mask] = np.asarray(
            raw_values[numeric_mask],
            dtype=np.float64,
        )

    # Numeric NaN has the same missing-value semantics as the scalar path.
    keep_mask = (
        numeric_mask
        & ~np.isnan(numeric_values)
    )

    if minimum_accepted_mz is not None:
        keep_mask &= (
            ~numeric_mask
            | (numeric_values >= minimum_accepted_mz)
        )

    if maximum_mz is not None:
        keep_mask &= (
            ~numeric_mask
            | (numeric_values <= maximum_mz)
        )

    # Only unusual object-valued entries need the complete scalar logic.
    fallback_rows = np.flatnonzero(
        ~numeric_mask
    )

    for row in fallback_rows:
        keep_mask[row] = _require_precursor_mz(
            metadata.iloc[row],
            minimum_accepted_mz=minimum_accepted_mz,
            maximum_mz=maximum_mz,
        )

    return _filter_collection(
        spectrum_in,
        keep_mask,
        clone=clone,
    )


def _filter_collection(
    collection: SpectraCollection,
    keep_mask: np.ndarray,
    *,
    clone: bool | None,
) -> SpectraCollection:
    """Apply a row mask while preserving collection clone semantics."""
    if clone:
        return collection.filter(
            keep_mask,
            inplace=False,
        )

    collection.filter(
        keep_mask,
        inplace=True,
    )
    return collection


require_precursor_mz = metadata_requirement_filter(
    _require_precursor_mz,
    collection_impl=_require_precursor_mz_collection,
)