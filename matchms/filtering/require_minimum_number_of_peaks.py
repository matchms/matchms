from typing import Union
from math import ceil
from matchms import Spectrum


def require_minimum_number_of_peaks(spectrum_in: Union[Spectrum, None], n_required=10, ratio_required=None) -> Union[Spectrum, None]:

    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    parent_mass = spectrum.get("parent_mass", None)
    if parent_mass:
        n_required_by_mass = int(ceil(ratio_required * parent_mass))
        threshold = max(n_required, n_required_by_mass)
    else:
        threshold = n_required

    if spectrum.intensities.size < threshold:
        return None

    return spectrum
