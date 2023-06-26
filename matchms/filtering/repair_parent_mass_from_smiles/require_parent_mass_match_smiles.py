from matchms import Spectrum
from matchms.filtering.filter_utils.get_neutral_mass_from_smiles import \
    get_monoisotopic_neutral_mass


def require_parent_mass_match_smiles(spectrum_in: Spectrum,
                                     mass_tolerance):
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()
    # Check if parent mass matches the smiles mass
    parent_mass = spectrum.get("parent_mass")
    smiles = spectrum.get("smiles")
    smiles_mass = get_monoisotopic_neutral_mass(smiles)
    mass_difference = parent_mass - smiles_mass
    if abs(mass_difference) < mass_tolerance:
        return spectrum
