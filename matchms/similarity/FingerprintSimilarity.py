from typing import Optional, Sequence, Union
import numpy as np
import scipy.sparse as sp
from chemap.metrics import (
    tanimoto_similarity_matrix,
)
from matchms.Fingerprints import Fingerprints
from matchms.Scores import Scores
from matchms.typing import SpectrumType
from .BaseSimilarity import BaseSimilarity
from .vector_similarity_functions import cosine_similarity_matrix


class FingerprintSimilarity(BaseSimilarity):
    """Calculate similarity between molecules based on molecular fingerprints.

    Fingerprints can either be provided explicitly as :class:`~matchms.Fingerprints`
    objects or computed internally from input spectra.

    This class no longer expects fingerprints to be stored directly in spectrum
    metadata. Instead, it uses a :class:`~matchms.Fingerprints` container.

    Currently supported similarity measures are:

    - ``"cosine"``
    - ``"tanimoto"``

    Notes
    -----
    - Tanimoto is used in its generalized form and therefore also works for
      count/weighted fingerprints.
    - Fingerprints may be stored densely (NumPy) or sparsely (CSR).
    """

    is_commutative = True
    score_datatype = np.float64
    score_fields = ("score",)

    def __init__(
        self,
        fingerprint_generator,
        similarity_measure: str = "tanimoto",
        set_empty_scores: Union[float, int, str] = "nan",
        ignore_stereochemistry: bool = False,
        count: bool = False,
        folded: bool = True,
        return_csr: bool = False,
        invalid_policy: str = "raise",
        **fingerprint_config_kwargs,
    ):
        """
        Parameters
        ----------
        fingerprint_generator
            A chemap-compatible fingerprint generator.
        similarity_measure
            Choose similarity measure from ``"cosine"`` or ``"tanimoto"``.
            The default is ``"tanimoto"``.
        set_empty_scores
            Define what should be returned instead of a similarity score in cases
            where fingerprints are missing. The default is ``"nan"``, which will
            return ``np.nan`` in such cases.
        ignore_stereochemistry
            Passed to internally created :class:`~matchms.Fingerprints` objects.
        count
            Passed to internally created :class:`~matchms.Fingerprints` objects.
        folded
            Passed to internally created :class:`~matchms.Fingerprints` objects.
        return_csr
            Passed to internally created :class:`~matchms.Fingerprints` objects.
        invalid_policy
            Passed to internally created :class:`~matchms.Fingerprints` objects.
        **fingerprint_config_kwargs
            Additional keyword arguments passed to internally created
            :class:`~matchms.Fingerprints` objects.
        """
        assert similarity_measure in ["cosine", "tanimoto"], "Unknown similarity measure."

        self.fingerprint_generator = fingerprint_generator
        self.similarity_measure = similarity_measure
        self.set_empty_scores = set_empty_scores

        self.ignore_stereochemistry = ignore_stereochemistry
        self.count = count
        self.folded = folded
        self.return_csr = return_csr
        self.invalid_policy = invalid_policy
        self.fingerprint_config_kwargs = fingerprint_config_kwargs

    def pair(self, spectrum_1: SpectrumType, spectrum_2: SpectrumType):
        """Pairwise fingerprint similarity is not supported in this API.

        FingerprintSimilarity works on precomputed Fingerprints containers or
        computes fingerprints internally for collections of spectra in `matrix()`.

        Use `matrix(...)` instead.
        """
        raise NotImplementedError(
            "FingerprintSimilarity.pair() is not supported. "
            "Use matrix(...) with spectra or Fingerprints objects instead."
        )

    def matrix(
        self,
        spectra_1: Optional[Sequence[SpectrumType]] = None,
        spectra_2: Optional[Sequence[SpectrumType]] = None,
        *,
        fingerprints_1: Optional[Fingerprints] = None,
        fingerprints_2: Optional[Fingerprints] = None,
        score_fields: Optional[Sequence[str]] = None,
        progress_bar: bool = True,
    ) -> Scores:
        """Calculate matrix of fingerprint-based similarity scores.

        Parameters
        ----------
        spectra_1
            First collection of spectra. Used only if `fingerprints_1` is not given.
        spectra_2
            Second collection of spectra. Used only if `fingerprints_2` is not given.
            If None and `fingerprints_2` is None, compare the first input against itself.
        fingerprints_1
            Optional precomputed Fingerprints object for the first input.
        fingerprints_2
            Optional precomputed Fingerprints object for the second input.
            If None, compare the first input against itself.
        score_fields
            Requested score fields. Only ``("score",)`` is supported.
        progress_bar
            Included for API compatibility. Not used here.

        Returns
        -------
        Scores
            Dense score matrix as a ``Scores`` object.
        """
        del progress_bar

        selected_fields = self._resolve_score_fields(score_fields)
        if selected_fields != ("score",):
            raise NotImplementedError(
                "FingerprintSimilarity.matrix() supports only score_fields=('score',)."
            )

        fingerprints_1, fingerprints_2, is_symmetric = self._prepare_fingerprint_inputs(
            spectra_1=spectra_1,
            spectra_2=spectra_2,
            fingerprints_1=fingerprints_1,
            fingerprints_2=fingerprints_2,
        )

        X1 = self._fingerprint_matrix(fingerprints_1)
        X2 = self._fingerprint_matrix(fingerprints_2)

        n_rows = fingerprints_1.fingerprint_count
        n_cols = fingerprints_2.fingerprint_count

        assert n_rows > 0 and n_cols > 0, (
            "Not enough molecular fingerprints.",
            "Provide valid spectra or precomputed Fingerprints first.",
        )

        similarity_matrix = self._compute_similarity_matrix(X1, X2)

        if is_symmetric and similarity_matrix.shape[0] == similarity_matrix.shape[1]:
            similarity_matrix = 0.5 * (similarity_matrix + similarity_matrix.T)

        return Scores({"score": similarity_matrix.astype(self.score_datatype, copy=False)})

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _prepare_fingerprint_inputs(
        self,
        spectra_1: Optional[Sequence[SpectrumType]],
        spectra_2: Optional[Sequence[SpectrumType]],
        fingerprints_1: Optional[Fingerprints],
        fingerprints_2: Optional[Fingerprints],
    ) -> tuple[Fingerprints, Fingerprints, bool]:
        """Normalize spectra / fingerprints inputs into two Fingerprints objects."""
        if fingerprints_1 is None and spectra_1 is None:
            raise ValueError("Either spectra_1 or fingerprints_1 must be provided.")

        if fingerprints_1 is not None and spectra_1 is not None:
            raise ValueError("Provide either spectra_1 or fingerprints_1, not both.")

        if fingerprints_2 is not None and spectra_2 is not None:
            raise ValueError("Provide either spectra_2 or fingerprints_2, not both.")

        if fingerprints_1 is None:
            fingerprints_1 = self._compute_fingerprints_from_spectra(spectra_1)

        if fingerprints_2 is None and spectra_2 is None:
            return fingerprints_1, fingerprints_1, True

        if fingerprints_2 is None:
            fingerprints_2 = self._compute_fingerprints_from_spectra(spectra_2)

        return fingerprints_1, fingerprints_2, False

    def _compute_fingerprints_from_spectra(self, spectra: Sequence[SpectrumType]) -> Fingerprints:
        """Compute fingerprints from spectra using the configured Fingerprints container."""
        fingerprints = Fingerprints(
            fingerprint_generator=self.fingerprint_generator,
            ignore_stereochemistry=self.ignore_stereochemistry,
            count=self.count,
            folded=self.folded,
            return_csr=self.return_csr,
            invalid_policy=self.invalid_policy,
            **self.fingerprint_config_kwargs,
        )
        fingerprints.compute_fingerprints(list(spectra))
        return fingerprints

    @staticmethod
    def _fingerprint_matrix(fingerprints: Fingerprints):
        """Return stored fingerprint matrix."""
        if fingerprints.fingerprints is None or fingerprints.fingerprint_count == 0:
            raise ValueError("Fingerprint container is empty.")
        return fingerprints.fingerprints

    def _compute_similarity_matrix(self, fingerprints_1, fingerprints_2) -> np.ndarray:
        """Compute similarity block between two fingerprint matrices."""
        if self.similarity_measure == "cosine":
            if sp.issparse(fingerprints_1):
                fingerprints_1 = fingerprints_1.toarray()
            if sp.issparse(fingerprints_2):
                fingerprints_2 = fingerprints_2.toarray()

            fingerprints_1 = np.asarray(fingerprints_1, dtype=np.float32)
            fingerprints_2 = np.asarray(fingerprints_2, dtype=np.float32)
            return cosine_similarity_matrix(fingerprints_1, fingerprints_2)

        if self.similarity_measure == "tanimoto":
            kind = "sparse" if sp.issparse(fingerprints_1) and sp.issparse(fingerprints_2) else "dense"

            if kind == "dense":
                if sp.issparse(fingerprints_1):
                    fingerprints_1 = fingerprints_1.toarray()
                if sp.issparse(fingerprints_2):
                    fingerprints_2 = fingerprints_2.toarray()

                fingerprints_1 = np.asarray(fingerprints_1, dtype=np.float32)
                fingerprints_2 = np.asarray(fingerprints_2, dtype=np.float32)

            return tanimoto_similarity_matrix(
                fingerprints_1,
                fingerprints_2,
                kind=kind,
            )

        raise NotImplementedError
