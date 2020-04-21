from matchms.utils import mol_converter


def derive_inchikey_from_inchi(spectrum_in):
    """Find missing InchiKey and derive from Inchi where possible."""
    def has_inchi():
        """Return True if input is not an empty inchi."""
        empty_entry_types = ['N/A', 'n/a', 'n\a', 'NA', 0, '0', '""', '', 'nodata',
                             '"InChI=n/a"', '"InChI="', 'InChI=1S/N\n', '\t\r\n']
        if inchi:
            return inchi not in empty_entry_types or len(inchi) < 12
        else:
            return False

    def has_inchikey():
        """Return True if input is not an empty inchikey."""
        empty_entry_types = ['N/A', 'n/a', 'n\a', 'NA', 0, '0', '""', '', 'nodata']
        return inchikey is not None or inchikey not in empty_entry_types

    spectrum = spectrum_in.clone()

    inchi = spectrum.get("inchi")
    inchikey = spectrum.get("inchikey")

    if has_inchi() and not has_inchikey():
        inchikey = mol_converter(inchi, "inchi", "inchikey")
        if not inchikey:
            print("Could not convert InChI", inchi, "to inchikey.")
            inchikey = 'n/a'
        spectrum.set("inchikey", inchikey)

    return spectrum
