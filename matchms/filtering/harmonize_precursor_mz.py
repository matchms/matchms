from ..typing import SpectrumType


def harmonize_precursor_mz(spectrum_in: SpectrumType) -> SpectrumType:
    """Harmonize precursor_mz field."""
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    if spectrum.get("precursor_mz", None) is None:
        pepmass = spectrum.get("pepmass")
        if pepmass:
            spectrum.set("precursor_mz", pepmass[0])
        else:
            print("No precursor_mz found in metadata.")

    return spectrum
