import re
from typing import Optional
import numpy as np
from matchms.filtering.load_adducts import (load_adducts_dict,
                                            load_known_adduct_conversions)


try:  # rdkit is not included in pip package
    from rdkit import Chem
    from rdkit.Chem import AllChem
except ImportError:
    _has_rdkit = False
    from collections import UserString

    class ChemMock(UserString):
        def __call__(self, *args, **kwargs):
            return self

        def __getattr__(self, key):
            return self

    Chem = AllChem = ChemMock("")
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


def derive_fingerprint_from_smiles(smiles: str, fingerprint_type: str, nbits: int) -> np.ndarray:
    """Calculate molecule fingerprint based on given smiles or inchi (using rdkit).
    Requires conda package *rdkit* to be installed.

    Parameters
    ----------
    smiles
        Input smiles to derive fingerprint from.
    fingerprint_type
        Determine method for deriving molecular fingerprints. Supported choices are 'daylight',
        'morgan1', 'morgan2', 'morgan3'.
    nbits
        Dimension or number of bits of generated fingerprint.

    Returns
    -------
    fingerprint
        Molecular fingerprint.
    """
    if not _has_rdkit:
        raise ImportError(rdkit_missing_message)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return mol_to_fingerprint(mol, fingerprint_type, nbits)


def derive_fingerprint_from_inchi(inchi: str, fingerprint_type: str, nbits: int) -> np.ndarray:
    """Calculate molecule fingerprint based on given inchi (using rdkit).
    Requires conda package *rdkit* to be installed.

    Parameters
    ----------
    inchi
        Input InChI to derive fingerprint from.
    fingerprint_type
        Determine method for deriving molecular fingerprints. Supported choices are 'daylight',
        'morgan1', 'morgan2', 'morgan3'.
    nbits
        Dimension or number of bits of generated fingerprint.

    Returns
    -------
    fingerprint: np.array
        Molecular fingerprint.
    """
    if not _has_rdkit:
        raise ImportError(rdkit_missing_message)

    mol = Chem.MolFromInchi(inchi)
    if mol is None:
        return None
    return mol_to_fingerprint(mol, fingerprint_type, nbits)


def mol_to_fingerprint(mol: Chem.rdchem.Mol, fingerprint_type: str, nbits: int) -> np.ndarray:
    """Convert rdkit mol (molecule) to molecular fingerprint.
    Requires conda package *rdkit* to be installed.

    Parameters
    ----------
    mol
        Input rdkit molecule.
    fingerprint_type
        Determine method for deriving molecular fingerprints.
        Supported choices are 'daylight', 'morgan1', 'morgan2', 'morgan3'.
    nbits
        Dimension or number of bits of generated fingerprint.

    Returns
    -------
    fingerprint
        Molecular fingerprint.
    """
    if not _has_rdkit:
        raise ImportError(rdkit_missing_message)

    assert fingerprint_type in ["daylight", "morgan1", "morgan2", "morgan3"], "Unkown fingerprint type given."

    if fingerprint_type == "daylight":
        fp = Chem.RDKFingerprint(mol, fpSize=nbits)
    elif fingerprint_type == "morgan1":
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 1, nBits=nbits)
    elif fingerprint_type == "morgan2":
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=nbits)
    elif fingerprint_type == "morgan3":
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=nbits)

    if fp:
        return np.array(fp)
    return None


def looks_like_adduct(adduct):
    """Return True if input string has expected format of an adduct."""
    if not isinstance(adduct, str):
        return False
    # Clean adduct
    adduct = clean_adduct(adduct)
    # Load lists of default known adducts
    known_adducts = load_adducts_dict()
    if adduct in known_adducts:
        return True

    # Expect format like: "[2M-H]" or "[2M+Na]+"
    regexp1 = r"^\[(([0-4]M)|(M[0-9])|(M))((Br)|(Br81)|(Cl)|(Cl37)|(S)){0,}[+-][A-Z0-9\+\-\(\)aglire]{1,}[\]0-4+-]{1,4}"
    return re.search(regexp1, adduct) is not None


def clean_adduct(adduct: str) -> str:
    """Clean adduct and make it consistent in style.
    Will transform adduct strings of type 'M+H+' to '[M+H]+'.

    Parameters
    ----------
    adduct
        Input adduct string to be cleaned/edited.
    """
    def get_adduct_charge(adduct):
        regex_charges = r"[1-3]{0,1}[+,-]{1,2}$"
        match = re.search(regex_charges, adduct)
        if match:
            return match.group(0)
        return match

    def adduct_conversion(adduct):
        """Convert adduct if conversion rule is known"""
        adduct_conversions = load_known_adduct_conversions()
        if adduct in adduct_conversions:
            return adduct_conversions[adduct]
        return adduct

    if not isinstance(adduct, str):
        return adduct

    adduct = adduct.strip().replace("\n", "").replace("*", "")
    adduct = adduct.replace("++", "2+").replace("--", "2-")
    if adduct.startswith("["):
        return adduct_conversion(adduct)

    if adduct.endswith("]"):
        return adduct_conversion("[" + adduct)

    adduct_core = "[" + adduct
    # Remove parts that can confuse the charge extraction
    for mol_part in ["CH2", "CH3", "NH3", "NH4", "O2"]:
        if mol_part in adduct:
            adduct = adduct.split(mol_part)[-1]
    adduct_charge = get_adduct_charge(adduct)

    if adduct_charge is None:
        return adduct_conversion(adduct_core + "]")

    adduct_cleaned = adduct_core[:-len(adduct_charge)] + "]" + adduct_charge
    return adduct_conversion(adduct_cleaned)
