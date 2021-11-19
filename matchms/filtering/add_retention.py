from typing import Any
from typing import List
from matchms.utils import get_first_common_element
from ..typing import SpectrumType


_retention_time_keys = ["retention_time", "retentiontime", "rt", "scan_start_time"]
_retention_index_keys = ["retention_index", "retentionindex", "ri"]


def safe_store_value(spectrum: SpectrumType, value: Any, target_key: str) -> SpectrumType:
    """Helper function to safely store a value in the target key without throwing an exception, but storing 'None' instead.

    Args:
        spectrum (SpectrumType): Spectrum to which to add 'value' in 'target_key'.
        value (Any): Value to parse into 'target_key'.
        target_key (str): Name of the key in which to store the value.

    Returns:
        SpectrumType: Spectrum with added key.
    """
    if value is not None:   # one of accepted keys is present
        try:
            value = float(value)
            rt = value if value >= 0 else None  # discard negative RT values
        except ValueError:
            print("%s can't be converted to float.", value)
            rt = None
        spectrum.set(target_key, rt)
    return spectrum


def _add_retention(spectrum: SpectrumType, target_key: str, accepted_keys: List[str]) -> SpectrumType:
    """Add value from one of accepted keys to target key.

    Args:
        spectrum (SpectrumType): Spectrum from which to read the values.
        target_key (str): Key under which to store the value.
        accepted_keys (List[str]): List of accepted keys from which a value will be read (in order).

    Returns:
        SpectrumType: Spectrum with value from first accepted key stored under target_key.
    """
    present_keys = spectrum.metadata.keys()
    rt_key = get_first_common_element(present_keys, accepted_keys)
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
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()


    target_key = "retention_time"
    spectrum = _add_retention(spectrum, target_key, _retention_time_keys)

    return spectrum


def add_retention_index(spectrum_in: SpectrumType) -> SpectrumType:
    """Add retention index into 'retention_index' key if present.


    Args:
        spectrum_in (SpectrumType): Spectrum with RI information.

    Returns:
        SpectrumType: Spectrum with RI info stored under 'retention_index'.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()


    target_key = "retention_index"
    spectrum = _add_retention(spectrum, target_key, _retention_index_keys)

    return spectrum
