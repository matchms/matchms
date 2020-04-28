from ..typing import SpectrumType


def set_ionmode_na_when_missing(spectrum_in: SpectrumType) -> SpectrumType:

    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    # Set ionmode to "n/a" when ionmode is missing from the metadata
    if spectrum.get("ionmode") is None:
        spectrum.set("ionmode", "n/a")

    return spectrum
