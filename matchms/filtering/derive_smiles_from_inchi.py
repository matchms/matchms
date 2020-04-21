from matchms.utils import mol_converter


def derive_smiles_from_inchi(spectrum_in):
    """Find missing smiles and derive from Inchi where possible."""
    def has_inchi():
        """Return True if input is not an empty inchi."""
        empty_entry_types = ['N/A', 'n/a', 'n\a', 'NA', 0, '0', '""', '', 'nodata',
                             '"InChI=n/a"', '"InChI="', 'InChI=1S/N\n', '\t\r\n']
        return inchi is not None or inchi in empty_entry_types or len(inchi) < 12

    def has_smiles():
        """Return True if input is not an empty smiles."""
        empty_entry_types = ['N/A', 'n/a', 'n\a', 'NA', 0, '0', '""', '', 'nodata']
        return smiles is not None or smiles in empty_entry_types

    spectrum = spectrum_in.clone()

    inchi = spectrum.get("inchi")
    smiles = spectrum.get("smiles")

    if has_inchi() and not has_smiles():
        smiles = mol_converter(inchi, "inchi", "smi")
        if not smiles:
            print("Could not convert InChI", inchi, "to smiles.")
            smiles = 'n/a'
        smiles = smiles.replace('\n', '').replace('\t', '').replace('\r', '')
        spectrum.set("smiles", smiles)

    return spectrum
