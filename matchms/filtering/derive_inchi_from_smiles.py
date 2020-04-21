from matchms.utils import mol_converter


def derive_inchi_from_smiles(spectrum_in):
    """Find missing Inchi and derive from smiles where possible."""
    def inchi_is_empty():
        """Return True if input is an empty inchi."""
        empty_entry_types = ['N/A', 'n/a', 'n\a', 'NA', 0, '0', '""', '', 'nodata',
                             '"InChI=n/a"', '"InChI="', 'InChI=1S/N\n', '\t\r\n']
        if inchi:
            is_empty = inchi in empty_entry_types or len(inchi) < 12
        else:
            is_empty = True
        return is_empty

    def smiles_is_empty():
        """Return True if input is an empty "smiles"."""
        empty_entry_types = ['N/A', 'n/a', 'n\a', 'NA', 0, '0', '""', '', 'nodata']
        return smiles is None or smiles in empty_entry_types

    spectrum = spectrum_in.clone()

    inchi = spectrum.get("inchi")
    smiles = spectrum.get("smiles")

    if inchi_is_empty() and not smiles_is_empty():
        inchi = mol_converter(smiles, "smi", "inchi")
        if not inchi:
            # Second try: use smiley ("smy") parser
            inchi = mol_converter(smiles, "smy", "inchi")
        if not inchi:
            print("Could not convert smiles", smiles, "to InChI.")
            inchi = '"InChI=n/a"'
        inchi = inchi.replace('\n', '').replace('\t', '').replace('\r', '')
        spectrum.set("inchi", inchi)

    return spectrum
