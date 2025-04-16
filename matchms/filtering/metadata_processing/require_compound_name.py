import logging
from typing import Optional
from matchms.typing import SpectrumType


logger = logging.getLogger("matchms")


def require_compound_name(spectrum: SpectrumType) -> Optional[SpectrumType]:
    """Ensure that the compound name is present in the spectrum metadata."""
    if spectrum is None:
        return None

    compound_name = spectrum.get("compound_name", None)

    if compound_name:
        return spectrum

    logger.info("Spectrum does not contain a compound name.")
    return None
