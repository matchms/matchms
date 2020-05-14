from ..typing import SpectrumType


def make_charge_scalar(spectrum_in: SpectrumType) -> SpectrumType:
    """Convert charge field to scalar (if necessary)."""
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    # Avoid pyteomics ChargeList
    if isinstance(spectrum.get("charge", None), list):
        spectrum.set("charge", int(spectrum.get("charge")[0]))

    # Interpret charge when parsed as string
    if isinstance(spectrum.get("charge", None), str):
        charge = spectrum.get("charge").strip("+ ")
        if '-' in charge:
            charge = -int(charge.strip("-+"))
        else:
            charge = int(charge)
        spectrum.set("charge", charge)

    return spectrum
