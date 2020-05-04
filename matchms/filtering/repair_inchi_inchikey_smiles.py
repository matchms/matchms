from .SpeciesString import SpeciesString
from ..typing import SpectrumType


def repair_inchi_inchikey_smiles(spectrum_in: SpectrumType):

    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    # interpret available data and clean each
    inchi = spectrum.get("inchi", "")
    inchiaux = spectrum.get("inchiaux", "")
    inchikey = spectrum.get("inchikey", "")
    smiles = spectrum.get("smiles", "")

    cleaneds = [SpeciesString(s) for s in [inchi, inchiaux, inchikey, smiles]]

    # for each type, list what we have and pick one
    # inchi
    inchis = [c.cleaned for c in cleaneds if c.target == "inchi" and c.cleaned != ""]
    inchis_no_issues = [x for x in inchis if not x.startswith("issue")]
    if len(inchis_no_issues) > 0:
        spectrum.set("inchi", inchis_no_issues[0])
    else:
        spectrum.set("inchi", inchis[0] if len(inchis) > 0 else "")

    # inchikey
    inchikeys = [c.cleaned for c in cleaneds if c.target == "inchikey" and c.cleaned != ""]
    inchikeys_no_issues = [x for x in inchikeys if not x.startswith("issue")]
    if len(inchikeys_no_issues) > 0:
        spectrum.set("inchikey", inchikeys_no_issues[0])
    else:
        spectrum.set("inchikey", inchikeys[0] if len(inchikeys) > 0 else "")

    # smiles
    smiles = [c.cleaned for c in cleaneds if c.target == "smiles" and c.cleaned != ""]
    smiles_no_issues = [x for x in smiles if not x.startswith("issue")]
    if len(smiles_no_issues) > 0:
        spectrum.set("smiles", smiles_no_issues[0])
    else:
        spectrum.set("smiles", smiles[0] if len(smiles) > 0 else "")

    return spectrum
