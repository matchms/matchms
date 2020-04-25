from matchms import Spikes


def add_losses(spectrum_in):
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    precursor_mz = spectrum.get("precursor_mz")
    if precursor_mz is not None:
        peaks_mz, peaks_intensities = spectrum.peaks
        losses_mz = (precursor_mz - peaks_mz)[::-1]
        losses_intensities = peaks_intensities
        spectrum.losses = Spikes(mz=losses_mz, intensities=losses_intensities)

    return spectrum
