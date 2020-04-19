from matchms.utils import mol_converter


def complete_compound_annotation(spectrum_in):
    """Find missing Inchi, smiles, and Inchikey and fill is possible.

    1) If smiles but no Inchi --> Add Inchi by converting.from smiles.
    2) If Inchi but no smiles --> Add smiles by converting.from Inchi.
    3) If no Inchikey --> Add InchiKey by converting from Inchi.
    """

    spectrum = spectrum_in.clone()

    # Empirically found list of strings that represent empty entries
    empty_entry_types = ['N/A', 'n/a', 'NA', 0, '0', '""', '', 'nodata',
                         '"InChI=n/a"', '"InChI="', 'InChI=1S/N\n', '\t\r\n']
    inchi = spectrum.get("inchi")
    smiles = spectrum.get("smiles")
    inchikey = spectrum.get("inchikey")

    # 1) If smiles but no Inchi
    if inchi is None \
        or inchi in empty_entry_types \
        or len(inchi) < 12:
        if smiles and smiles not in empty_entry_types:
            inchi = mol_converter(smiles, "smi", "inchi")
            if not inchi:
                inchi = mol_converter(smiles, "smy", "inchi")  # test smiley parser
            if not inchi:
                print("Could not convert smiles", smiles, "to InChI.")
                inchi = '"InChI=n\a"'
            spectrum.set("inchi", inchi)

    # 2) If Inchi but no smiles
    if not smiles or smiles in empty_entry_types:
        if inchi and inchi not in empty_entry_types and len(inchi) > 12:
            smiles = mol_converter(inchi, "inchi", "smi")
            if not smiles:
                print("Could not convert InChI", inchi, "to smiles.")
                smiles = 'n\a'
            spectrum.set("smiles", smiles)

    # 3) If no Inchikey
    if not inchikey or inchikey in empty_entry_types:
        inchikey = mol_converter(inchi, "inchi", "inchikey")
        if not inchikey:
            print("Could not convert InChI", inchi, "to inchikey.")
            inchikey = 'n\a'
        spectrum.set("inchikey", inchikey)

    return spectrum
