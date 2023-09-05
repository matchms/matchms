import logging
from typing import Optional
from matchms.constants import PROTON_MASS


logger = logging.getLogger("matchms")


try:  # rdkit is not included in pip package
    from rdkit import Chem
    from rdkit.Chem import Descriptors
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


def get_monoisotopic_neutral_mass(smiles: str) -> float:
    """Get the monoisotopic neutral mass from a SMILES string chemical
    identifier for the described molecule.

    Args:
        smiles (str): SMILES string defining the structure of the molecule.

    Raises:
        ImportError: If RDkit chem is not installed.

    Returns:
        float: Computed monoisotopic mass.
    """
    return _get_neutral_mass(smiles, True)


def get_molecular_weight_neutral_mass(smiles: str) -> float:
    """Get the (average) molecular weight for the isotopic distribution of the SMILES code.

    Args:
        smiles (str): SMILES string describing the structure of the molecule.

    Raises:
        ImportError: Raises import error if RDKitChem is not found.

    Returns:
        float: Average molecular mass.
    """
    return _get_neutral_mass(smiles, False)


def _get_neutral_mass(smiles: str, monoisotopic: bool) -> Optional[float]:
    """Get neutral mass of molecule, either average or most abundant monoisotopic mass.

    Args:
        smiles (str): SMILES describing the molecule.
        monoisotopic (bool): whether to compute the monoisotopic mass or average.

    Raises:
        ImportError: If RDKit Chem is not found

    Returns:
        float: Neutral mass.
    """
    if not _has_rdkit:
        raise ImportError(rdkit_missing_message)
    if smiles is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        logger.warning("No mass could be calculated for smiles: %s, since it is not a valid smiles.", smiles)
        return None
    mass = Descriptors.ExactMolWt(mol) if monoisotopic else Descriptors.MolWt(mol)
    charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    neutral_mass = mass + -charge * PROTON_MASS
    return neutral_mass
