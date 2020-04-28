from ..typing import SpectrumType


def make_charge_scalar(spectrum_in: SpectrumType) -> SpectrumType:

    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    # Avoid pyteomics ChargeList
    if isinstance(spectrum.get("charge", None), list):
        spectrum.set("charge", int(spectrum.get("charge")[0]))

    return spectrum
