from ..typing import SpectrumType


def make_ionmode_lowercase(spectrum_in: SpectrumType) -> SpectrumType:
    """Convert entry in field "ionmode" to lower case."""
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    # if the ionmode key exists in the metadata, lowercase its value
    if spectrum.get("ionmode") is not None:
        spectrum.set("ionmode", spectrum.get("ionmode").lower())

    return spectrum
