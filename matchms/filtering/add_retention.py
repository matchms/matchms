from matchms.utils import get_first_common_element
from ..typing import SpectrumType


_retention_time_keys = ["retention_time", "retentiontime", "rt", "scan_start_time"]
_retention_index_keys = ["retention_index", "retentionindex", "ri"]


def safe_store_value(spectrum: SpectrumType, value, target_key):
    if value is not None:   # one of accepted keys is present
        try:
            value = float(value)
            rt = value if value >= 0 else None  # discard negative RT values
        except ValueError:
            print("%s can't be converted to float.", value)
            rt = None
        spectrum.set(target_key, rt)
    return spectrum


def add_retention(spectrum, target_key, accepted_keys):
    rt_key = get_first_common_element(spectrum.metadata.keys(), accepted_keys)
    value = spectrum.get(rt_key)
    spectrum = safe_store_value(spectrum, value, target_key)
    return spectrum


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

    target_key = "retention_time"
    spectrum = add_retention(spectrum, target_key, _retention_time_keys)

    return spectrum


def add_retention_index(spectrum_in: SpectrumType) -> SpectrumType:
    spectrum = spectrum_in.clone()

    target_key = "retention_index"
    spectrum = add_retention(spectrum, target_key, _retention_index_keys)

    return spectrum
