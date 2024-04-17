import logging
from typing import Optional
from matchms.typing import SpectrumType


logger = logging.getLogger("matchms")


def require_retention_index(spectrum: SpectrumType) -> Optional[SpectrumType]:
    """Ensure that the retention index is present and is a positive number."""
    if spectrum is None:
        return None

    retention_index = spectrum.get('retention_index', None)
    if isinstance(retention_index, (int, float)) and retention_index > 0:
        return spectrum
    else:
        logger.info("Spectrum retention index is missing or not a positive number.")
        return None
