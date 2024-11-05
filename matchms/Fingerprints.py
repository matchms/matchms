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
from .utils import to_camel_case


logger = logging.getLogger("matchms")


class Fingerprints:
    def __init__(self, fingerprint_algorithm: str = "daylight", fingerprint_method: str = "bit", nbits: int = 2048, **kwargs):
        self.inchikey_fingerprint_mapping = {}
        self.fingerprint_algorithm = fingerprint_algorithm
        self.fingerprint_method = fingerprint_method
        self.nbits = nbits
        self.kwargs = kwargs

    def __str__(self):
        return json.dumps({
            "config: ": self.config,
            "inchikey_fingerprint_mapping": self.inchikey_fingerprint_mapping
        })

    @property
    def config(self):
        return {
            "fingerprint_algorithm": self.fingerprint_algorithm,
            "fingerprint_method": self.fingerprint_method,
            "nbits": self.nbits,
            "additional_keyword_arguments": self.kwargs,
        }

    @property
    def fingerprints(self):
        return self.inchikey_fingerprint_mapping

    def fingerprints_to_dataframe(self):
        return pd.DataFrame(
            {'fingerprint': list(self.inchikey_fingerprint_mapping.values())},
            index=list(self.inchikey_fingerprint_mapping.keys())
        )

    def get_fingerprint_by_inchikey(self, inchikey: str) -> Optional[np.ndarray]:
        if inchikey in self.inchikey_fingerprint_mapping:
            return self.inchikey_fingerprint_mapping[inchikey]

        if not is_valid_inchikey(inchikey):
            logger.warning("The provided inchikey is not valid or may be the short form.")

        logger.warning("Fingerprint is not present for given Spectrum/InchiKey. Use compute_fingerprint() first.")
        return None

    def get_fingerprint_by_spectrum(self, spectrum: SpectrumType) -> Optional[np.ndarray]:
        inchikey = spectrum.get("inchikey")

        # Double check the form of the inchikey
        if not is_valid_inchikey(inchikey):
            spectrum = _require_inchikey(spectrum)
            inchikey = spectrum.get("inchikey")

        return self.get_fingerprint_by_inchikey(inchikey)

    def compute_fingerprint(self, spectrum: SpectrumType) -> Optional[np.ndarray]:
        fingerprint = None
        if spectrum.get("smiles"):
            fingerprint = _derive_fingerprint_from_smiles(spectrum.get("smiles"), self.fingerprint_algorithm,
                                                          self.fingerprint_method, self.nbits, **self.kwargs)

        if fingerprint is None and spectrum.get("inchi"):
            fingerprint = _derive_fingerprint_from_inchi(spectrum.get("inchi"), self.fingerprint_algorithm,
                                                         self.fingerprint_method, self.nbits, **self.kwargs)

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


def _derive_fingerprint_from_smiles(smiles: str, fingerprint_algorithm: str, fingerprint_method: str, nbits: int, **kwargs) -> Optional[np.ndarray]:
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
    return _mol_to_fingerprint(mol, fingerprint_algorithm, fingerprint_method, nbits, **kwargs)


def _derive_fingerprint_from_inchi(inchi: str, fingerprint_algorithm: str, fingerprint_method: str, nbits: int, **kwargs) -> Optional[np.ndarray]:
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
    return _mol_to_fingerprint(mol, fingerprint_algorithm, fingerprint_method, nbits, **kwargs)


def _mol_to_fingerprint(mol: Mol, fingerprint_algorithm: str, fingerprint_type: str, nbits: int, **kwargs) -> Optional[np.ndarray]:
    """Convert rdkit mol (molecule) to molecular fingerprint.
    Requires conda package *rdkit* to be installed.

    Parameters
    ----------
    mol
        Input rdkit molecule.
    fingerprint_algorithm
        Determine algorithm for deriving molecular fingerprints.
        Supported algorithms are 'daylight', 'morgan1', 'morgan2', 'morgan3'.
    fingerprint_type
        Determine type for deriving molecular fingerprints.
        Supported types are 'bit', 'sparse_bit', 'count', 'sparse_count'.
    nbits
        Dimension or number of bits of generated fingerprint.
    **kwargs
        Keyword arguments to pass additional parameters to FingerprintGenerator.
        The keywords should match the corresponding RDKit implementation (e.g., min_path/max_path for RDKitFPGenerator).
        See https://rdkit.org/docs/source/rdkit.Chem.rdFingerprintGenerator.html.

    Returns
    -------
    fingerprint
        Molecular fingerprint.
    """
    algorithms = {
        "daylight": lambda args: GetRDKitFPGenerator(**args),
        "morgan1": lambda args: GetMorganGenerator(**args, radius=1),
        "morgan2": lambda args: GetMorganGenerator(**args, radius=2),
        "morgan3": lambda args: GetMorganGenerator(**args, radius=3)
    }
    types = {
        "bit": "GetFingerprint",
        "sparse_bit": "GetSparseFingerprint",
        "count": "GetCountFingerprint",
        "sparse_count": "GetSparseCountFingerprint"
    }

    if fingerprint_algorithm not in algorithms:
        raise ValueError(f"Unkown fingerprint algorithm given. Available algorithms: {list(algorithms.keys())}.")
    if fingerprint_type not in types:
        raise ValueError(f"Unkown fingerprint type given. Available types: {list(types.keys())}.")

    args = {"fpSize": nbits, **{to_camel_case(k): v for k, v in kwargs.items()}}
    generator = algorithms[fingerprint_algorithm](args)

    fingerprint_func = getattr(generator, types[fingerprint_type])
    fingerprint = fingerprint_func(mol)

    return np.array(fingerprint) if fingerprint else None

    # TODO: use GetFingerprints to use multithreading
