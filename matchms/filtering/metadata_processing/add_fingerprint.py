import logging
import numpy as np
from rdkit import Chem

from matchms.typing import SpectrumType
from matchms.filtering.filter_utils.metadata_utils import (_has_rdkit,
                                                           rdkit_missing_message, Chem, AllChem)


logger = logging.getLogger("matchms")


def add_fingerprint(spectrum_in: SpectrumType, fingerprint_type: str = "daylight",
                    nbits: int = 2048) -> SpectrumType:
    """Add molecular finterprint to spectrum.

    If smiles or inchi present in metadata, derive a molecular finterprint and
    add it to the spectrum.

    Parameters
    ----------
    spectrum_in:
        Input spectrum.
    fingerprint_type:
        Determine method for deriving molecular fingerprints. Supported choices
        are "daylight", "morgan1", "morgan2", "morgan3". Default is "daylight".
    nbits:
        Dimension or number of bits of generated fingerprint. Default is 2048.
    """
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()

    # First try to get fingerprint from smiles
    if spectrum.get("smiles", None):
        fingerprint = derive_fingerprint_from_smiles(spectrum.get("smiles"),
                                                     fingerprint_type, nbits)
        if isinstance(fingerprint, np.ndarray) and fingerprint.sum() > 0:
            spectrum.set("fingerprint", fingerprint)
            return spectrum

    # Second try to get fingerprint from inchi
    if spectrum.get("inchi", None):
        fingerprint = derive_fingerprint_from_inchi(spectrum.get("inchi"),
                                                    fingerprint_type, nbits)
        if isinstance(fingerprint, np.ndarray) and fingerprint.sum() > 0:
            spectrum.set("fingerprint", fingerprint)
            return spectrum

    logger.info("No fingerprint was added (name: %s).", spectrum.get("compound_name"))
    return spectrum


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
