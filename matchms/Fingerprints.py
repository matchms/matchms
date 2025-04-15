import json
import logging
from typing import Optional
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator, GetRDKitFPGenerator
from matchms.filtering.filter_utils.smile_inchi_inchikey_conversions import (
    is_valid_inchi,
    is_valid_inchikey,
    is_valid_smiles,
)
from matchms.typing import SpectrumType
from .utils import to_camel_case


logger = logging.getLogger("matchms")
FP_ALGORITHMS = {
    "daylight": lambda args: GetRDKitFPGenerator(**args),
    "morgan1": lambda args: GetMorganGenerator(**args, radius=1),
    "morgan2": lambda args: GetMorganGenerator(**args, radius=2),
    "morgan3": lambda args: GetMorganGenerator(**args, radius=3),
}


class Fingerprints:
    """
    Computes and stores inchikey-fingerprint mapping for a list of spectra,

    For example

    .. testcode::

        from matchms import Fingerprints
        from matchms import Spectrum
        import numpy as np

        spectrum_1 = Spectrum(mz=np.array([100, 150, 200.]),
                              intensities=np.array([0.7, 0.2, 0.1]),
                              metadata={"inchikey": "OTMSDBZUPAUEDD-UHFFFAOYSA-N", "smiles":"CC"})
        spectrum_2 = Spectrum(mz=np.array([100, 150, 200.]),
                              intensities=np.array([0.7, 0.2, 0.1]),
                              metadata={"inchikey": "UGFAIRIUMAVXCW-UHFFFAOYSA-N","smiles": "[C-]#[O+]"})
        spectra = [spectrum_1, spectrum_2]

        fpgen = Fingerprints()
        fpgen.compute_fingerprints(spectra)

        print(fpgen.fingerprint_count)
        print(type(fpgen.get_fingerprint_by_inchikey('OTMSDBZUPAUEDD-UHFFFAOYSA-N')))

    Should output

    .. testoutput::

        2
        <class 'numpy.ndarray'>

    Attributes
    ----------
    config:
        The configuration for the fingerprints e.g., used algorithm, nbits, ...
    fingerprints:
        The computed fingerprints. Use after compute_fingerprints().
    fingerprints_count
        The number of fingerprints computed.
    to_dataframe
        A DataFrame containing the inchikey and fingerprint

    """

    def __init__(
        self,
        fingerprint_algorithm: str = "daylight",
        fingerprint_method: str = "bit",
        nbits: int = 2048,
        ignore_stereochemistry: bool = False,
        **kwargs,
    ):
        """

        Parameters
        ----------
        fingerprint_algorithm
            The fingerprint algorithm to use. Available options: daylight, morgan1, morgan2, morgan3.
        fingerprint_method
            The fingerprint method to use. Available options: bit, sparse_bit, count, sparse_count.
        nbits
            The number of bits or fingerprint size. Defaults to 2048.
        ignore_stereochemistry
            Determines which inchikey version will be used. If set to true the first 14 chars of the inchikey are used.
        """
        self.inchikey_fingerprint_mapping = {}
        self.fingerprint_algorithm = fingerprint_algorithm
        self.fingerprint_method = fingerprint_method
        self.nbits = nbits
        self.ignore_stereochemistry = ignore_stereochemistry
        self.kwargs = kwargs

    def __str__(self):
        return json.dumps({"config: ": self.config, "inchikey_fingerprint_mapping": self.inchikey_fingerprint_mapping})

    @property
    def config(self) -> dict:
        return {
            "fingerprint_algorithm": self.fingerprint_algorithm,
            "fingerprint_method": self.fingerprint_method,
            "nbits": self.nbits,
            "ingore_stereochemistry": self.ignore_stereochemistry,
            "additional_keyword_arguments": self.kwargs,
        }

    @property
    def fingerprints(self):
        return self.inchikey_fingerprint_mapping

    @property
    def fingerprint_count(self) -> int:
        return len(self.inchikey_fingerprint_mapping)

    @property
    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            data={"fingerprint": list(self.inchikey_fingerprint_mapping.values())},
            index=list(self.inchikey_fingerprint_mapping.keys()),
        )

    def get_fingerprint_by_inchikey(self, inchikey: str) -> Optional[np.ndarray]:
        """
        Get fingerprint by inchikey.

        Parameters
        ----------
        inchikey
            Inchikey of a spectrum.

        Return:
        --------------
        Optional[np.ndarray]
            The corresponding fingerprint.
        """
        if inchikey in self.inchikey_fingerprint_mapping:
            return self.inchikey_fingerprint_mapping[inchikey]
        if (len(inchikey) == 14) and not self.ignore_stereochemistry:
            raise ValueError("Expected full 27 character InChIKey (or ignore_stereochemistry set to True)")

        if not is_valid_inchikey(inchikey):
            logger.warning("The provided inchikey is not valid or may be the short form.")

        logger.warning("Fingerprint is not present for given Spectrum/InchiKey. Use compute_fingerprint() first.")
        return None

    def get_fingerprint_by_spectrum(self, spectrum: SpectrumType) -> Optional[np.ndarray]:
        """
        Get fingerprint by spectrum.

        Parameters
        ----------
        spectrum
            Spectrum with a inchikey.

        Return:
        --------------
        Optional[np.ndarray]
            The corresponding fingerprint.
        """
        inchikey = spectrum.get("inchikey")

        return self.get_fingerprint_by_inchikey(inchikey)

    def compute_fingerprint(self, spectrum: SpectrumType) -> Optional[np.ndarray]:
        """
        Computes a single fingerprint for a given spectrum.

        Parameters
        ----------
        spectrum
            A spectrum for which a fingerprint is to be calculated.

        Return:
        --------------
        Optional[np.ndarray]
            The corresponding fingerprint.
        """
        fingerprint = None
        if spectrum.get("smiles"):
            fingerprint = _derive_fingerprint_from_smiles(
                spectrum.get("smiles"), self.fingerprint_algorithm, self.fingerprint_method, self.nbits, **self.kwargs
            )

        if fingerprint is None and spectrum.get("inchi"):
            fingerprint = _derive_fingerprint_from_inchi(
                spectrum.get("inchi"), self.fingerprint_algorithm, self.fingerprint_method, self.nbits, **self.kwargs
            )

        return fingerprint

    def compute_fingerprints(self, spectra: list[SpectrumType]):
        """
        Computes fingerprints for a list of spectra.

        This will first create a dict with unique spectra and then computes fingerprints for all mols.
        Only valid fingerprints will be added to the mapping.
        Query specific fingerprints by using get_fingerprint_by_spectrum() or get_fingerprint_by_inchikey()

        Parameters
        ----------
        spectra
            List of Spectrum
        """

        # Get/Set unique spectra via inchikey
        unique_spectra = {}
        for spectrum in spectra:
            try:
                # Validate metadata
                _validate_metadata(spectrum, self.ignore_stereochemistry)
                inchikey = spectrum.get("inchikey")

                # Add inchikey/spectrum to unique_spectra and ensure smiles or inchi
                if inchikey not in unique_spectra:
                    unique_spectra[inchikey] = spectrum
            except ValueError:
                logger.warning("%s doesn't have a inchikey. Skipping.", spectrum)

        # Get mols of unique spectra from smiles/inchi
        mols = [_get_mol(spectrum) for spectrum in unique_spectra.values()]

        # Get fingerprints of all mols
        fingerprints = _mols_to_fingerprints(
            mols, self.fingerprint_algorithm, self.fingerprint_method, self.nbits, **self.kwargs
        )

        # Map inchikey - fingerprint
        for inchikey, fp in zip(unique_spectra.keys(), fingerprints, strict=True):
            if isinstance(fp, np.ndarray) and fp.sum() > 0:
                self.inchikey_fingerprint_mapping[inchikey] = fp


def _get_mol(spectrum: SpectrumType) -> Optional[Mol]:
    """
    Get the molecule either from smiles or inchi.

    Parameter:
    ----------
    spectrum: SpectrumType
        Spectrum to get the mol from.

    Return:
    --------------
    Optional[Mol]
        RDKit Mol object or None if smiles and inchi missing or generation failed.
    """

    mol = None
    if spectrum.get("smiles"):
        return Chem.MolFromSmiles(spectrum.get("smiles"))

    if spectrum.get("inchi"):
        return Chem.MolFromInchi(spectrum.get("inchi"))

    return mol


def _validate_metadata(spectrum: SpectrumType, ignore_stereochemistry: bool):
    """
    Validates metadata for a given spectrum.

    Checks for a valid inchikey or if stereochemistry is ignored check for a inchikey of 14 chars.
    Checks if inchi or smiles are valid.

    Parameters
    ----------
    spectrum
        Spectrum to validate
    ignore_stereochemistry
        If true, a inchikey should contain 14 chars.
    """
    inchikey = spectrum.get("inchikey")
    if ignore_stereochemistry:
        if len(inchikey) > 14:
            spectrum.set("inchikey", inchikey[:14])
        elif len(inchikey) < 14:
            raise ValueError("Inchikey is missing or invalid.")
    elif not is_valid_inchikey(inchikey):
        raise ValueError("Inchikey is missing or invalid.")

    if not is_valid_inchi(spectrum.get("inchi")) and not is_valid_smiles(spectrum.get("smiles")):
        raise ValueError("Inchi or smiles is missing or invalid.")

    return spectrum


def _derive_fingerprint_from_smiles(
    smiles: str, fingerprint_algorithm: str, fingerprint_method: str, nbits: int, **kwargs
) -> Optional[np.ndarray]:
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


def _derive_fingerprint_from_inchi(
    inchi: str, fingerprint_algorithm: str, fingerprint_method: str, nbits: int, **kwargs
) -> Optional[np.ndarray]:
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


def _mol_to_fingerprint(
    mol: Mol, fingerprint_algorithm: str, fingerprint_type: str, nbits: int, **kwargs
) -> Optional[np.ndarray]:
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

    types = {
        "bit": "GetFingerprint",
        "sparse_bit": "GetSparseFingerprint",
        "count": "GetCountFingerprint",
        "sparse_count": "GetSparseCountFingerprint",
    }

    if fingerprint_algorithm not in FP_ALGORITHMS:
        raise ValueError(f"Unkown fingerprint algorithm given. Available algorithms: {list(FP_ALGORITHMS.keys())}.")
    if fingerprint_type not in types:
        raise ValueError(f"Unkown fingerprint type given. Available types: {list(types.keys())}.")

    args = {"fpSize": nbits, **{to_camel_case(k): v for k, v in kwargs.items()}}
    generator = FP_ALGORITHMS[fingerprint_algorithm](args)

    fingerprint_func = getattr(generator, types[fingerprint_type])
    fingerprint = fingerprint_func(mol)

    return np.array(fingerprint) if fingerprint else None


def _mols_to_fingerprints(
    mols: list[Mol], fingerprint_algorithm: str, fingerprint_type: str, nbits: int, **kwargs
) -> np.ndarray:
    """
    Computes a fingerprints for a list of molecules.

    Parameters
    ----------
    fingerprint_algorithm
        Specifies algorithm for deriving fingerprints.
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
    Return:
    --------------
    np.ndarray
        A np.ndarray of np.ndarrays, containing a fingerprints for each molecule.
        If the fingerprint for a molecule cannot be calculated, the corresponding fingerprint is an ndarray with zeros.
    """

    types = {
        "bit": "GetFingerprints",
        "sparse_bit": "GetSparseFingerprints",
        "count": "GetCountFingerprints",
        "sparse_count": "GetSparseCountFingerprints",
    }

    if fingerprint_algorithm not in FP_ALGORITHMS:
        raise ValueError(f"Unkown fingerprint algorithm given. Available algorithms: {list(FP_ALGORITHMS.keys())}.")
    if fingerprint_type not in types:
        raise ValueError(f"Unkown fingerprint type given. Available types: {list(types.keys())}.")

    args = {"fpSize": nbits, **{to_camel_case(k): v for k, v in kwargs.items()}}
    generator = FP_ALGORITHMS[fingerprint_algorithm](args)

    fingerprint_func = getattr(generator, types[fingerprint_type])
    fingerprints = fingerprint_func(mols, numThreads=-1)

    out = np.zeros((len(mols), nbits), dtype=np.int8)

    assert len(fingerprints) == len(mols)

    for i, fp in enumerate(fingerprints):
        if fp is not None:
            DataStructs.ConvertToNumpyArray(fp, out[i])

    return out
