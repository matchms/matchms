from matchms.utils import mol_converter


def derive_inchikey_from_inchi(spectrum_in):
    """Find missing InchiKey and derive from Inchi where possible."""
    def inchi_is_empty():
        """Return True if input is an empty inchi."""
        empty_entry_types = ['N/A', 'n/a', 'n\a', 'NA', 0, '0', '""', '', 'nodata',
                             '"InChI=n/a"', '"InChI="', 'InChI=1S/N\n', '\t\r\n']
        if inchi:
            is_empty = inchi in empty_entry_types or len(inchi) < 12
        else:
            is_empty = True
        return is_empty

    def inchikey_is_empty():
        """Return True if input is not an empty inchikey."""
        empty_entry_types = ['N/A', 'n/a', 'n\a', 'NA', 0, '0', '""', '', 'nodata']
        return inchikey is None or inchikey in empty_entry_types

    spectrum = spectrum_in.clone()

    inchi = spectrum.get("inchi")
    inchikey = spectrum.get("inchikey")

    if inchikey_is_empty() and not inchi_is_empty():
        inchikey = mol_converter(inchi, "inchi", "inchikey")
        if not inchikey:
            print("Could not convert InChI", inchi, "to inchikey.")
            inchikey = 'n/a'
        spectrum.set("inchikey", inchikey)

    return spectrum
