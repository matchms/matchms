from matchms.utils import mol_converter


def derive_inchi_from_smiles(spectrum_in):
    """Find missing Inchi and derive from smiles where possible."""
    def has_inchi():
        """Return True if input is not an empty inchi."""
        empty_entry_types = ['N/A', 'n/a', 'n\a', 'NA', 0, '0', '""', '', 'nodata',
                             '"InChI=n/a"', '"InChI="', 'InChI=1S/N\n', '\t\r\n']
        return inchi is not None or inchi in empty_entry_types or len(inchi) < 12

    def has_smiles():
        """Return True if input is not an empty "smiles"."""
        empty_entry_types = ['N/A', 'n/a', 'n\a', 'NA', 0, '0', '""', '', 'nodata']
        return smiles is not None or inchi in empty_entry_types

    spectrum = spectrum_in.clone()

    inchi = spectrum.get("inchi")
    smiles = spectrum.get("smiles")

    if has_smiles() and not has_inchi():
        inchi = mol_converter(smiles, "smi", "inchi")
        if not inchi:
            # Second try: use smiley ("smy") parser
            inchi = mol_converter(smiles, "smy", "inchi")
        if not inchi:
            print("Could not convert smiles", smiles, "to InChI.")
            inchi = '"InChI=n/a"'
        # Clean inchi
        spectrum.set("inchi", inchi)

    return spectrum
