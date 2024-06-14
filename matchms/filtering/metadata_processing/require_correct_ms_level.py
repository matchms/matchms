from typing import Optional
from matchms.Spectrum import Spectrum


def require_correct_ms_level(spectrum: Spectrum, required_ms_level: int = 2) -> Optional[Spectrum]:
    if spectrum is None:
        return None
    if spectrum.get("ms_level") in (f"MS{required_ms_level}", str(required_ms_level)):
        return spectrum
    return None
