import re
from typing import Optional


try:  # rdkit is not included in pip package
    from rdkit import Chem
except ImportError:
    _has_rdkit = False
    from collections import UserString

    class ChemMock(UserString):
        def __call__(self, *args, **kwargs):
            return self

        def __getattr__(self, key):
            return self

    Chem = ChemMock("")
else:
    _has_rdkit = True
rdkit_missing_message = "Conda package 'rdkit' is required for this functionality."


def convert_smiles_to_inchi(smiles: str) -> Optional[str]:
    """Convert smiles to inchi using rdkit."""
    return mol_converter(smiles, "smiles", "inchi")


def convert_inchi_to_smiles(inchi: str) -> Optional[str]:
    """Convert inchi to smiles using rdkit."""
    return mol_converter(inchi, "inchi", "smiles")


def convert_inchi_to_inchikey(inchi: str) -> Optional[str]:
    """Convert inchi to inchikey using rdkit."""
    return mol_converter(inchi, "inchi", "inchikey")


def mol_converter(mol_input: str, input_type: str, output_type: str) -> Optional[str]:
    """Convert molecular representations using rdkit.

    Convert from "smiles" or "inchi" to "inchi", "smiles", or "inchikey".
    Requires conda package *rdkit* to be installed.

    Parameters
    ----------
    mol_input
        Input data in "inchi" or "smiles" molecular representation.
    input_type
        Define input type: "smiles" for smiles and "inchi" for inchi.
    output_type
        Define output type: "smiles", "inchi", or "inchikey".

    Returns:
    --------

    Mol string in output type or None when conversion failure occurs.
    """
    if not _has_rdkit:
        raise ImportError(rdkit_missing_message)
    input_function = {"inchi": Chem.MolFromInchi,
                      "smiles": Chem.MolFromSmiles}
    output_function = {"inchi": Chem.MolToInchi,
                       "smiles": Chem.MolToSmiles,
                       "inchikey": Chem.MolToInchiKey}

    mol = input_function[input_type](mol_input.strip('"'))
    if mol is None:
        return None

    output = output_function[output_type](mol)
    if output:
        return output
    return None


def is_valid_inchi(inchi: str) -> bool:
    """Return True if input string is valid InChI.

    This functions test if string can be read by rdkit as InChI.
    Requires conda package *rdkit* to be installed.

    Parameters
    ----------
    inchi
        Input string to test if it has format of InChI.
    """
    if not _has_rdkit:
        raise ImportError(rdkit_missing_message)

    # First quick test to avoid excess in-depth testing
    if inchi is None:
        return False
    inchi = inchi.strip('"')
    regexp = r"(InChI=1|1)(S\/|\/)[0-9a-zA-Z\.]{2,}"
    if not re.search(regexp, inchi):
        return False
    # Proper chemical test
    mol = Chem.MolFromInchi(inchi)
    if mol:
        return True
    return False


def is_valid_smiles(smiles: str) -> bool:
    """Return True if input string is valid smiles.

    This functions test if string can be read by rdkit as smiles.
    Requires conda package *rdkit* to be installed.

    Parameters
    ----------
    smiles
        Input string to test if it can be imported as smiles.
    """
    if not _has_rdkit:
        raise ImportError(rdkit_missing_message)

    if smiles is None:
        return False

    regexp = r"^([^J][0-9ABCOHNMSPIFKiergalcons@+\-\[\]\(\)\\\/%=#$,.~&!]*)$"
    if not re.match(regexp, smiles):
        return False

    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return True
    return False


def is_valid_inchikey(inchikey: str) -> bool:
    """Return True if string has format of inchikey.

    Parameters
    ----------
    inchikey
        Input string to test if it format of an inchikey.
    """
    if inchikey is None:
        return False

    regexp = r"[A-Z]{14}-[A-Z]{10}-[A-Z]"
    if re.fullmatch(regexp, inchikey):
        return True
    return False
