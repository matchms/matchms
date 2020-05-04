from ..typing import SpectrumType


def add_adduct(spectrum_in: SpectrumType) -> SpectrumType:
    """Add adduct to metadata (if not present yet).

    Method to interpret the given compound name to find the adduct.
    """

    spectrum = spectrum_in.clone()

    if spectrum.get("adduct", None) is None:
        try:
            name = spectrum.get("name")
            adduct = name.split(' ')[-1]
            adduct = adduct.replace('\n', '') \
                           .replace(' ', '')  \
                           .replace('[', '')  \
                           .replace(']', '')  \
                           .replace('*', '')
            if adduct:
                spectrum.set("adduct", adduct)
        except KeyError:
            print("Spectrum's metadata does not have a 'name'.")

    return spectrum
