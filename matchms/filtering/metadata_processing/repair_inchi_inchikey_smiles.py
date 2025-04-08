from typing import Optional
from matchms.filtering.SpeciesString import SpeciesString
from matchms.typing import SpectrumType


def repair_inchi_inchikey_smiles(spectrum_in: SpectrumType, clone: Optional[bool] = True) -> Optional[SpectrumType]:
    """Check if inchi, inchikey, and smiles entries seem correct. Detect and correct
    if any of those entries clearly belongs into one of the other two fields (e.g.
    inchikey found in inchi field).

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    clone:
        Optionally clone the Spectrum.

    Returns
    -------
    Spectrum or None
        Spectrum with repaired INCHI, INCHIKEY and SMILES, or `None` if not present.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone() if clone else spectrum_in

    # interpret available data and clean each
    inchi = spectrum.get("inchi", "")
    inchiaux = spectrum.get("inchiaux", "")
    inchikey = spectrum.get("inchikey", "")
    smiles = spectrum.get("smiles", "")

    cleaneds = [SpeciesString(s) for s in [inchi, inchiaux, inchikey, smiles]]

    # for each type, list what we have and pick one
    inchis = [c.cleaned for c in cleaneds if c.target == "inchi" and c.cleaned != ""]
    inchikeys = [c.cleaned for c in cleaneds if c.target == "inchikey" and c.cleaned != ""]
    smiles = [c.cleaned for c in cleaneds if c.target == "smiles" and c.cleaned != ""]

    spectrum.set("inchi", inchis[0] if len(inchis) > 0 else "")
    spectrum.set("inchikey", inchikeys[0] if len(inchikeys) > 0 else "")
    spectrum.set("smiles", smiles[0] if len(smiles) > 0 else "")

    return spectrum
