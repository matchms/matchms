from matchms.utils import get_first_common_element
from ..typing import SpectrumType


_accepted_keys = ["retention_time", "retentiontime", "rt", "scan_start_time"]


def add_retention_time(spectrum_in: SpectrumType) -> SpectrumType:
    """Add retention time information to the 'retention_time' key as float.

    Args:
        spectrum_in (SpectrumType): Spectrum with retention time information.

    Returns:
        SpectrumType: Spectrum with harmonized retention time information.
    """
    spectrum = spectrum_in.clone()

    rt_key = get_first_common_element(spectrum.metadata.keys(), _accepted_keys)
    rt = spectrum.get(rt_key)

    if rt is not None:
        spectrum.set("retention_time", float(rt))

    return spectrum
