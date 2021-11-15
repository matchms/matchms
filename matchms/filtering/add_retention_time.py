from matchms.utils import get_first_common_element
from ..typing import SpectrumType


_accepted_keys = ["retention_time", "retentiontime", "rt", "scan_start_time"]


def add_retention_time(spectrum_in: SpectrumType) -> SpectrumType:
    """Add retention time information to the 'retention_time' key as float.
    Negative values and those not convertible to a float result in 'retention_time'
    being 'None'.

    Args:
        spectrum_in (SpectrumType): Spectrum with retention time information.

    Returns:
        SpectrumType: Spectrum with harmonized retention time information.
    """
    spectrum = spectrum_in.clone()

    rt_key = get_first_common_element(spectrum.metadata.keys(), _accepted_keys)
    value = spectrum.get(rt_key)

    if value is not None:
        try:
            value = float(value)
            rt = value if value >= 0 else None
        except ValueError:
            print("%s can't be converted to float.", value)
            rt = None
        spectrum.set("retention_time", rt)

    return spectrum
