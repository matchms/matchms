import logging
from typing import Optional
from matchms.typing import SpectrumType


logger = logging.getLogger("matchms")


def require_retention_time(spectrum: SpectrumType) -> Optional[SpectrumType]:
    """Ensure that the retention time is present and is a positive number."""
    if spectrum is None:
        return None

    retention_time = spectrum.get('retention_time', None)
    if isinstance(retention_time, (int, float)) and retention_time > 0:
        return spectrum
    else:
        logger.info("Spectrum retention time is missing or not a positive number.")
        return None
