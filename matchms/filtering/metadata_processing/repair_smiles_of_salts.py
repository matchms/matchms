from matchms.filtering.filters.repair_smiles_of_salts import RepairSmilesOfSalts


def repair_smiles_of_salts(spectrum_in,
                           mass_tolerance):
    """Repairs the smiles of a salt to match the parent mass.
    E.g. C1=NC2=NC=NC(=C2N1)N.Cl is converted to 1=NC2=NC=NC(=C2N1)N if this matches the parent mass
    Checks if parent mass matches one of the ions"""

    spectrum = RepairSmilesOfSalts(mass_tolerance).process(spectrum_in)
    return spectrum
