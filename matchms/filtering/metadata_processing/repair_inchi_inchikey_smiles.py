from matchms.typing import SpectrumType
from matchms.filtering.filters.repair_inchi_inchikey_smiles import RepairInchiInchikeySmiles


def repair_inchi_inchikey_smiles(spectrum_in: SpectrumType) -> SpectrumType:
    """Check if inchi, inchikey, and smiles entries seem correct. Detect and correct
    if any of those entries clearly belongs into one of the other two fields (e.g.
    inchikey found in inchi field).
    """

    spectrum = RepairInchiInchikeySmiles().process(spectrum_in)
    return spectrum
