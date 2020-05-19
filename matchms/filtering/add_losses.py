from ..Spikes import Spikes


def add_losses(spectrum_in):
    """Derive losses based on precursor mass."""
    def precursor_mz_is_number():
        if isinstance(precursor_mz, int):
            return True
        if isinstance(precursor_mz, float):
            return True
        return False

    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    precursor_mz = spectrum.get("precursor_mz")
    if precursor_mz is not None:
        assert precursor_mz_is_number(), "Expected 'precursor_mz' to be a scalar number."
        peaks_mz, peaks_intensities = spectrum.peaks
        losses_mz = (precursor_mz - peaks_mz)[::-1]
        losses_intensities = peaks_intensities
        spectrum.losses = Spikes(mz=losses_mz, intensities=losses_intensities)

    return spectrum
