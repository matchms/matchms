from matchms import Spectrum
from matchms.filtering.filters.repair_smiles_from_compound_name import RepairSmilesFromCompoundName


def repair_smiles_from_compound_name(spectrum_in: Spectrum,
                                     annotated_compound_names_file,
                                     mass_tolerance=0.1):
    """Adds annotations (smiles, inchi, inchikey) based on compound name

    Based on a table of compound names and smiles matches (stored in a csv file) this function
    adds the new annotations to the input spectrums if the smiles seem consistent with the available
    spectrum metadata (e.g., parent mass).
    This function can be used with csv files that are returned by the pubchem_lookup.py
    from matchmextras.

    Parameters
    ----------
    spectrum_in:
        The input spectrum.
    annotated_compound_names_file: str
        A csv file containing the compound names and matching smiles, inchi, inchikey
        and monoisotopic_mass. This can be created using the the pubchem_lookup.py from matchmextras.
    mass_tolerance.
        Acceptable mass difference between query compound and pubchem result.
    """

    spectrum = RepairSmilesFromCompoundName(annotated_compound_names_file, mass_tolerance).process(spectrum_in)
    return spectrum
