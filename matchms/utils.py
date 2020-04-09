from openbabel import openbabel as ob


def mol_converter(mol_input, input_type, output_type):
    """Convert molecular representations using openbabel.

    Convert for instance from smiles to inchi or inchi to inchikey.

    Args:
    ----
    mol_input: str
        Input data, e.g. inchi or smiles.
    input_type: str
        Define input type (as named in openbabel). E.g. "smi"for smiles and "inchi" for inchi.
    output_type: str
        Define input type (as named in openbabel). E.g. "smi"for smiles and "inchi" for inchi.
    """
    conv = ob.OBConversion()
    conv.SetInAndOutFormats(input_type, output_type)
    mol = ob.OBMol()
    if conv.ReadString(mol, mol_input):
        mol_output = conv.WriteString(mol)
    else:
        print("Error when converting", mol_input)
        mol_output = None

    return mol_output
