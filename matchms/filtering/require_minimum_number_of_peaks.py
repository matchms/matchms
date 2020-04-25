from math import ceil


def require_minimum_number_of_peaks(spectrum_in, n_required=10, ratio_required=None):

    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    parent_mass = spectrum.get("parent_mass", None)
    if parent_mass and ratio_required:
        n_required_by_mass = int(ceil(ratio_required * parent_mass))
        threshold = max(n_required, n_required_by_mass)
    else:
        threshold = n_required

    if spectrum.peaks.intensities.size < threshold:
        return None

    return spectrum
