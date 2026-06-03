import logging
from matchms.filtering._dispatch import metadata_requirement_filter
from matchms.filtering.filter_utils.metadata_conversions import (
    as_float_or_none,
    is_missing_metadata_value,
)


logger = logging.getLogger("matchms")


def _require_precursor_mz(
    metadata,
    minimum_accepted_mz: float | None = 10.0,
    maximum_mz: float | None = None,
) -> bool:
    """Require precursor m/z to be present and within optional bounds.

    Parameters
    ----------
    spectrum_in
        Input spectrum or spectra collection.
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

    if minimum_accepted_mz is not None and precursor_mz_float < minimum_accepted_mz:
        logger.info(
            "Spectrum is removed since precursor mz (%s) was below minimum mz (%s)",
            precursor_mz_float,
            minimum_accepted_mz,
        )
        return False

    if maximum_mz is not None and precursor_mz_float > maximum_mz:
        logger.info(
            "Spectrum is removed since precursor mz (%s) was above maximum mz (%s)",
            precursor_mz_float,
            maximum_mz,
        )
        return False

    return True


require_precursor_mz = metadata_requirement_filter(_require_precursor_mz)