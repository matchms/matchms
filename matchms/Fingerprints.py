import json
import logging
from typing import List, Optional
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdFingerprintGenerator import (GetMorganGenerator,
                                               GetRDKitFPGenerator)
from matchms.filtering import (derive_inchi_from_smiles,
                               derive_inchikey_from_inchi)
from matchms.filtering.filter_utils.smile_inchi_inchikey_conversions import (
    is_valid_inchi, is_valid_inchikey)
from matchms.typing import SpectrumType


logger = logging.getLogger("matchms")


class Fingerprints:
    def __init__(self, fingerprint_algorithm: str = "daylight", fingerprint_method: str = "bit", nbits: int = 2048, min_path: int = 1, max_path: int = 7):
        self.inchikey_fingerprint_mapping = {}
        self.fingerprint_algorithm = fingerprint_algorithm
        self.fingerprint_method = fingerprint_method
        self.nbits = nbits
        self.minPath = min_path
        self.maxPath = max_path

    def __str__(self):
        return json.dumps(self.inchikey_fingerprint_mapping)

    def fingerprints_to_dataframe(self):
        return pd.DataFrame(self.inchikey_fingerprint_mapping)

    def get_fingerprint_by_inchikey(self, inchikey: str) -> Optional[np.ndarray]:
        if inchikey in self.inchikey_fingerprint_mapping:
            return self.inchikey_fingerprint_mapping[inchikey]

        logger.warning("Fingerprint is not present for given Spectrum/InchiKey. Use compute_fingerprint() first.")
        return None

    def get_fingerprint_by_spectrum(self, spectrum: SpectrumType) -> Optional[np.ndarray]:
        inchikey = spectrum.get("inchikey")

        return self.get_fingerprint_by_inchikey(inchikey)

    def compute_fingerprint(self, spectrum: SpectrumType) -> Optional[np.ndarray]:
        fingerprint = None
        if spectrum.get("smiles"):
            fingerprint = _derive_fingerprint_from_smiles(spectrum.get("smiles"), self.fingerprint_algorithm,
                                                          self.fingerprint_method, self.nbits)

        if fingerprint is None and spectrum.get("inchi"):
            fingerprint = _derive_fingerprint_from_inchi(spectrum.get("inchi"), self.fingerprint_algorithm,
                                                         self.fingerprint_method, self.nbits)

        return fingerprint

    def compute_fingerprints(self, spectra: List[SpectrumType]):
        for spectrum in spectra:
            spectrum = _require_inchikey(spectrum) # TODO: Add try catch

            # Fingerprint is in mapping dict -> skip iteration
            if spectrum.get("inchikey") in self.inchikey_fingerprint_mapping and self.inchikey_fingerprint_mapping[spectrum.get("inchikey")] is not None:
                continue

            fingerprint = self.compute_fingerprint(spectrum)

            # Incorrect fingerprints will not be added to list
            if isinstance(fingerprint, np.ndarray) and fingerprint.sum() > 0:
                self.inchikey_fingerprint_mapping[spectrum.get("inchikey")] = fingerprint
            else:
                logger.warning("Computed fingerprint is not a ndarray or invalid.")

def _require_inchikey(spectrum_in: SpectrumType) -> SpectrumType:
    spectrum = spectrum_in.clone()

    # If inchikey invalid
    if not is_valid_inchikey(spectrum.get("inchikey")):
        # If inchi invalid, derive from smiles
        if not is_valid_inchi(spectrum.get("inchi")):
            spectrum = derive_inchi_from_smiles(spectrum)

        # Derive Inchikey from Inchi
        spectrum = derive_inchikey_from_inchi(spectrum)

    # If Inchikey still missing, raise ValueError
    if not is_valid_inchikey(spectrum.get("inchikey")):
        raise ValueError("Inchikey is missing or invalid.")

    return spectrum


def _derive_fingerprint_from_smiles(smiles: str, fingerprint_algorithm: str, fingerprint_method: str, nbits: int) -> Optional[np.ndarray]:
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
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _mol_to_fingerprint(mol, fingerprint_algorithm, fingerprint_method, nbits)


def _derive_fingerprint_from_inchi(inchi: str, fingerprint_algorithm: str, fingerprint_method: str, nbits: int) -> Optional[np.ndarray]:
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
    mol = Chem.MolFromInchi(inchi)
    if mol is None:
        return None
    return _mol_to_fingerprint(mol, fingerprint_algorithm, fingerprint_method, nbits)


def _mol_to_fingerprint(mol: Mol, fingerprint_algorithm: str, fingerprint_method: str, nbits: int, minPath: int, maxPath: int) -> Optional[np.ndarray]:
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
    algorithms = {
        "daylight": lambda nbits, minPath, maxPath: GetRDKitFPGenerator(fpSize=nbits, minPath=minPath, maxPath=maxPath),
        "morgan1": lambda nbits: GetMorganGenerator(radius=1, fpSize=nbits),
        "morgan2": lambda nbits: GetMorganGenerator(radius=2, fpSize=nbits),
        "morgan3": lambda nbits: GetMorganGenerator(radius=3, fpSize=nbits)
    }

    if fingerprint_algorithm not in algorithms:
        raise ValueError(f"Unkown fingerprint algorithm given. Available algorithms: {*algorithms.keys(),}.")

    generator = algorithms[fingerprint_algorithm](nbits=nbits, minPath=minPath, maxPath=maxPath)

    methods = {
        "bit": lambda mol: generator.GetFingerprint(mol),
        #"bit": lambda mol: generator.GetFingerprintAsNumPy(mol),
        "sparse_bit": lambda mol: generator.GetSparseFingerprint(mol),
        "count": lambda mol: generator.GetCountFingerprint(mol),
        "sparse_count": lambda mol: generator.GetSparseCountFingerprint(mol)
    }
    if fingerprint_method not in methods:
        raise ValueError(f"Unkown fingerprint method given. Available methods: {*methods.keys(),}.")

    fingerprint = methods[fingerprint_method](mol)

    if fingerprint:
        return np.array(fingerprint)

    # TODO: use GetFingerprints to use multithreading

    return None
