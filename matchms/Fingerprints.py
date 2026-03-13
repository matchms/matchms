import json
import logging
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
import scipy.sparse as sp
from chemap import FingerprintConfig, compute_fingerprints
from rdkit import Chem
from matchms.filtering.filter_utils.smile_inchi_inchikey_conversions import (
    is_valid_inchi,
    is_valid_inchikey,
    is_valid_smiles,
)
from matchms.typing import SpectrumType


logger = logging.getLogger("matchms")


@dataclass(frozen=True)
class _FingerprintRecord:
    """Internal record linking one unique compound to structure metadata."""
    inchikey: str
    smiles: Optional[str]
    inchi: Optional[str]


class Fingerprints:
    """Compute and store an InChIKey-to-fingerprint mapping for a collection of spectra.

    This class is a container for molecular fingerprints keyed by InChIKey.
    Fingerprints are computed for unique compounds only and stored either as a
    dense NumPy array or as a SciPy CSR sparse matrix.

    Compared to the older implementation, this refactor is designed for larger
    scale use cases and delegates fingerprint computation to :mod:`chemap`.

    Example
    -------
    .. testcode::

        import numpy as np
        from rdkit.Chem import rdFingerprintGenerator
        from matchms import Fingerprints, Spectrum

        spectrum_1 = Spectrum(
            mz=np.array([100, 150, 200.]),
            intensities=np.array([0.7, 0.2, 0.1]),
            metadata={
                "inchikey": "OTMSDBZUPAUEDD-UHFFFAOYSA-N",
                "smiles": "CC",
                "precursor_mz": 150.0,
            },
        )
        spectrum_2 = Spectrum(
            mz=np.array([100, 150, 200.]),
            intensities=np.array([0.7, 0.2, 0.1]),
            metadata={
                "inchikey": "UGFAIRIUMAVXCW-UHFFFAOYSA-N",
                "smiles": "[C-]#[O+]",
                "precursor_mz": 150.0,
            },
        )

        spectra = [spectrum_1, spectrum_2]

        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=256)

        fpgen = Fingerprints(
            fingerprint_generator=generator,
            count=False,
            folded=True,
            return_csr=False,
        )
        fpgen.compute_fingerprints(spectra)

        print(fpgen.fingerprint_count)
        print(type(fpgen.get_fingerprint_by_inchikey("OTMSDBZUPAUEDD-UHFFFAOYSA-N")))

    Should output

    .. testoutput::

        2
        <class 'numpy.ndarray'>

    Attributes
    ----------
    fingerprints
        The computed fingerprints as either a NumPy array or SciPy CSR matrix.
    inchikeys
        Ordered list of unique InChIKeys corresponding to fingerprint rows.
    fingerprint_count
        Number of unique fingerprints currently stored.
    config
        Dictionary with configuration used for fingerprint computation.
    to_dataframe
        DataFrame containing InChIKeys and fingerprints.
    """

    def __init__(
        self,
        fingerprint_generator,
        *,
        ignore_stereochemistry: bool = False,
        count: bool = False,
        folded: bool = True,
        return_csr: bool = False,
        invalid_policy: str = "raise",
        **config_kwargs,
    ):
        """
        Parameters
        ----------
        fingerprint_generator
            A chemap-compatible fingerprint generator, for example an RDKit
            fingerprint generator or a scikit-fingerprints object.
        ignore_stereochemistry
            If True, the first 14 characters of the InChIKey are used.
        count
            Whether count fingerprints should be computed.
        folded
            Whether fingerprints should be folded.
        return_csr
            If True, fingerprints are stored as a SciPy CSR matrix.
            Otherwise they are stored as a dense NumPy array.
        invalid_policy
            Policy passed to chemap for invalid molecular inputs.
        **config_kwargs
            Additional keyword arguments passed into ``FingerprintConfig``.
        """
        self.fingerprint_generator = fingerprint_generator
        self.ignore_stereochemistry = ignore_stereochemistry

        self.count = count
        self.folded = folded
        self.return_csr = return_csr
        self.invalid_policy = invalid_policy
        self.config_kwargs = config_kwargs

        self._inchikeys: list[str] = []
        self._row_by_inchikey: dict[str, int] = {}
        self._records: list[_FingerprintRecord] = []
        self._fingerprints: Optional[np.ndarray | sp.csr_matrix] = None

    def __str__(self):
        return json.dumps(
            {
                "config": self.config,
                "inchikeys": self._inchikeys,
                "fingerprint_count": self.fingerprint_count,
                "is_sparse": self.is_sparse,
            }
        )

    @property
    def config(self) -> dict:
        """Return configuration used for fingerprint computation."""
        return {
            "ignore_stereochemistry": self.ignore_stereochemistry,
            "count": self.count,
            "folded": self.folded,
            "return_csr": self.return_csr,
            "invalid_policy": self.invalid_policy,
            "additional_keyword_arguments": self.config_kwargs,
        }

    @property
    def fingerprints(self) -> Optional[np.ndarray | sp.csr_matrix]:
        """Return the stored fingerprint matrix."""
        return self._fingerprints

    @property
    def inchikeys(self) -> list[str]:
        """Return ordered list of stored InChIKeys."""
        return list(self._inchikeys)

    @property
    def is_sparse(self) -> bool:
        """Return True if fingerprints are stored as CSR sparse matrix."""
        return self._fingerprints is not None and sp.issparse(self._fingerprints)

    @property
    def fingerprint_count(self) -> int:
        """Return the number of stored fingerprints."""
        return len(self._inchikeys)

    @property
    def to_dataframe(self) -> pd.DataFrame:
        """Return fingerprints as a pandas DataFrame indexed by InChIKey."""
        if self._fingerprints is None:
            return pd.DataFrame(index=[])

        if sp.issparse(self._fingerprints):
            fingerprints = list(self._fingerprints)
        else:
            fingerprints = list(self._fingerprints)

        return pd.DataFrame(
            data={"fingerprint": fingerprints},
            index=self._inchikeys,
        )

    def get_fingerprint_by_inchikey(self, inchikey: str):
        """Get fingerprint by InChIKey.

        Parameters
        ----------
        inchikey
            InChIKey of a compound.

        Returns
        -------
        Optional[np.ndarray | scipy.sparse.csr_matrix]
            The corresponding fingerprint row, or None if not present.
        """
        if inchikey is None:
            logger.warning("No InChIKey provided.")
            return None

        normalized = self._normalize_inchikey(inchikey, validate=False)
        if normalized in self._row_by_inchikey:
            row_idx = self._row_by_inchikey[normalized]
            return self._get_row(row_idx)

        if len(inchikey) == 14 and not self.ignore_stereochemistry:
            raise ValueError(
                "Expected full 27 character InChIKey (or set ignore_stereochemistry=True)."
            )

        if not is_valid_inchikey(inchikey):
            logger.warning("The provided InChIKey is not valid or may be the short form.")

        logger.warning("Fingerprint is not present for given Spectrum/InChIKey. Use compute_fingerprints() first.")
        return None

    def get_fingerprint_by_spectrum(self, spectrum: SpectrumType):
        """Get fingerprint by spectrum.

        Parameters
        ----------
        spectrum
            Spectrum with an InChIKey.

        Returns
        -------
        Optional[np.ndarray | scipy.sparse.csr_matrix]
            The corresponding fingerprint row, or None if not present.
        """
        inchikey = spectrum.get("inchikey")
        return self.get_fingerprint_by_inchikey(inchikey)

    def compute_fingerprint(self, spectrum: SpectrumType):
        """Compute one fingerprint for a given spectrum.

        This does not add the fingerprint to the internal storage. It only computes
        and returns the fingerprint.

        Parameters
        ----------
        spectrum
            A spectrum for which a fingerprint is to be calculated.

        Returns
        -------
        Optional[np.ndarray | scipy.sparse.csr_matrix]
            Fingerprint row, or None if fingerprint could not be computed.
        """
        record = self._record_from_spectrum(spectrum)
        if record is None:
            return None

        smiles = self._select_structure_string(record)
        fps = self._compute_from_smiles([smiles])
        if fps.shape[0] == 0:
            return None
        return self._extract_row(fps, 0)

    def compute_fingerprints(self, spectra: list[SpectrumType]):
        """Compute fingerprints for a list of spectra.

        Fingerprints are computed only for unique compounds, keyed by InChIKey.
        Existing stored fingerprints are replaced.

        Parameters
        ----------
        spectra
            List of spectra.
        """
        unique_records: dict[str, _FingerprintRecord] = {}

        for spectrum in spectra:
            record = self._record_from_spectrum(spectrum)
            if record is None:
                logger.warning("%s doesn't have valid fingerprint metadata. Skipping.", spectrum)
                continue

            if record.inchikey not in unique_records:
                unique_records[record.inchikey] = record

        self._records = list(unique_records.values())
        self._inchikeys = [record.inchikey for record in self._records]
        self._row_by_inchikey = {inchikey: i for i, inchikey in enumerate(self._inchikeys)}

        if len(self._records) == 0:
            self._fingerprints = None
            return

        smiles = [self._select_structure_string(record) for record in self._records]
        self._fingerprints = self._compute_from_smiles(smiles)

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _get_row(self, row_idx: int):
        """Return one fingerprint row from internal storage."""
        assert self._fingerprints is not None, "Fingerprints have not been computed yet."

        if sp.issparse(self._fingerprints):
            return self._fingerprints.getrow(row_idx)
        return self._fingerprints[row_idx]

    def _extract_row(self, matrix, row_idx: int):
        """Extract one row from a dense or sparse fingerprint result."""
        if sp.issparse(matrix):
            return matrix.getrow(row_idx)
        return matrix[row_idx]

    def _compute_from_smiles(self, smiles: list[str]):
        """Compute fingerprints from SMILES using chemap."""
        config = FingerprintConfig(
            count=self.count,
            folded=self.folded,
            return_csr=self.return_csr,
            invalid_policy=self.invalid_policy,
            **self.config_kwargs,
        )
        return compute_fingerprints(smiles, self.fingerprint_generator, config=config)

    def _record_from_spectrum(self, spectrum: SpectrumType) -> Optional[_FingerprintRecord]:
        """Build validated internal fingerprint record from a spectrum."""
        inchikey = spectrum.get("inchikey")
        smiles = spectrum.get("smiles")
        inchi = spectrum.get("inchi")

        if inchikey is None:
            return None

        try:
            normalized_inchikey = self._normalize_inchikey(inchikey, validate=True)
        except ValueError:
            return None

        if smiles and is_valid_smiles(smiles):
            canonical_smiles = self._smiles_from_smiles(smiles)
            if canonical_smiles is not None:
                return _FingerprintRecord(
                    inchikey=normalized_inchikey,
                    smiles=canonical_smiles,
                    inchi=None,
                )

        if inchi and is_valid_inchi(inchi):
            canonical_smiles = self._smiles_from_inchi(inchi)
            if canonical_smiles is not None:
                return _FingerprintRecord(
                    inchikey=normalized_inchikey,
                    smiles=canonical_smiles,
                    inchi=inchi,
                )

        return None

    def _normalize_inchikey(self, inchikey: str, validate: bool = True) -> str:
        """Normalize InChIKey depending on stereochemistry setting."""
        if inchikey is None:
            raise ValueError("InChIKey is missing or invalid.")

        if self.ignore_stereochemistry:
            if len(inchikey) < 14:
                raise ValueError("InChIKey is missing or invalid.")
            return inchikey[:14]

        if validate and not is_valid_inchikey(inchikey):
            raise ValueError("InChIKey is missing or invalid.")
        return inchikey

    @staticmethod
    def _smiles_from_smiles(smiles: str) -> Optional[str]:
        """Convert valid SMILES to canonical SMILES."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)

    @staticmethod
    def _smiles_from_inchi(inchi: str) -> Optional[str]:
        """Convert valid InChI to canonical SMILES."""
        mol = Chem.MolFromInchi(inchi)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)

    @staticmethod
    def _select_structure_string(record: _FingerprintRecord) -> str:
        """Return structure string used as chemap input."""
        assert record.smiles is not None, "Expected canonical SMILES to be present."
        return record.smiles
