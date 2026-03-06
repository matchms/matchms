from abc import abstractmethod
from typing import Optional, Sequence
import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_array
from tqdm import tqdm
from matchms.typing import SpectrumType


class BaseSimilarity:
    """Similarity function base class.

    When building a custom similarity measure, inherit from this class and implement
    the desired methods.

    Attributes
    ----------
    is_commutative
        Whether the similarity function is commutative, meaning that the order of
        spectra does not matter: ``similarity(A, B) == similarity(B, A)``.
        Default is True.
    score_datatype
        NumPy dtype of a single score value.
        Examples are ``np.float64`` for scalar scores or a structured dtype such as
        ``np.dtype([("score", np.float64), ("matches", np.int64)])`` for multi-field scores.
    score_fields
        Names of the score fields. For scalar scores this should usually be
        ``("score",)``. For structured scores, this should match the dtype field names,
        for instance ``("score", "matches")``.
    """

    is_commutative = True
    score_datatype = np.float64
    score_fields = ("score",)

    @abstractmethod
    def pair(self, spectrum_1: SpectrumType, spectrum_2: SpectrumType):
        """Calculate the similarity for one pair of spectra.

        Parameters
        ----------
        spectrum_1
            First spectrum.
        spectrum_2
            Second spectrum.

        Returns
        -------
        score
            Similarity result for one pair. The returned value should be compatible with
            ``self.score_datatype``.

        Examples
        --------
        Scalar score:
            ``return np.asarray(score, dtype=self.score_datatype)``

        Structured score:
            ``return np.asarray((score, matches), dtype=self.score_datatype)``
        """
        raise NotImplementedError

    @property
    def is_structured_score(self) -> bool:
        """Return True if this similarity returns a structured score."""
        dtype = np.dtype(self.score_datatype)
        return dtype.names is not None

    def matrix(
        self,
        spectra_1: Sequence[SpectrumType],
        spectra_2: Optional[Sequence[SpectrumType]] = None,
        progress_bar: bool = True,
    ) -> np.ndarray:
        """Calculate a dense matrix of similarity scores.

        Parameters
        ----------
        spectra_1
            First collection of spectra.
        spectra_2
            Second collection of spectra. If None, compare ``spectra_1`` against itself.
            For commutative similarities this will automatically use the symmetric
            optimization.
        progress_bar
            When True, show a progress bar. Default is True.

        Returns
        -------
        np.ndarray
            Dense NumPy array of shape ``(len(spectra_1), len(spectra_2))``.
            The dtype is ``self.score_datatype``.

            For scalar scores, this will be a standard numeric array.
            For structured scores, this will be a structured NumPy array.
        """
        spectra_2, is_symmetric = self._validate_and_prepare_inputs(spectra_1, spectra_2)

        n_rows = len(spectra_1)
        n_cols = len(spectra_2)
        scores = np.zeros((n_rows, n_cols), dtype=self.score_datatype)

        iterator = tqdm(
            enumerate(spectra_1),
            total=n_rows,
            desc="Calculating similarities",
            disable=not progress_bar,
        )

        for i, spectrum_1 in iterator:
            if is_symmetric and self.is_commutative:
                start_j = i
                spectra_2_iter = enumerate(spectra_2[i:], start=i)
            else:
                start_j = 0
                spectra_2_iter = enumerate(spectra_2)

            for j, spectrum_2 in spectra_2_iter:
                score = self.pair(spectrum_1, spectrum_2)
                scores[i, j] = score

                if is_symmetric and self.is_commutative and j != i:
                    scores[j, i] = score

        return scores

    def sparse_matrix(
        self,
        spectra_1: Sequence[SpectrumType],
        spectra_2: Optional[Sequence[SpectrumType]] = None,
        idx_row: Optional[npt.ArrayLike] = None,
        idx_col: Optional[npt.ArrayLike] = None,
        progress_bar: bool = True,
    ):
        """Calculate sparse similarity results.

        Parameters
        ----------
        spectra_1
            First collection of spectra.
        spectra_2
            Second collection of spectra. If None, compare ``spectra_1`` against itself.
        idx_row
            Row indices of pairs to compute. If None and ``idx_col`` is also None, all
            pairwise comparisons are considered and only kept scores are stored.
        idx_col
            Column indices of pairs to compute. Must have the same shape as ``idx_row``.
        progress_bar
            When True, show a progress bar. Default is True.

        Returns
        -------
        scipy.sparse.coo_array or dict[str, scipy.sparse.coo_array]
            Sparse result.

            - For scalar scores, returns one ``coo_array``.
            - For structured scores, returns a dictionary that maps each score field
              name to one ``coo_array``.
        """
        spectra_2, is_symmetric = self._validate_and_prepare_inputs(spectra_1, spectra_2)

        if idx_row is None and idx_col is None:
            return self._sparse_from_all_pairs(
                spectra_1=spectra_1,
                spectra_2=spectra_2,
                is_symmetric=is_symmetric,
                progress_bar=progress_bar,
            )

        if idx_row is None or idx_col is None:
            raise ValueError("idx_row and idx_col must either both be given or both be None.")

        idx_row = np.asarray(idx_row, dtype=np.int_)
        idx_col = np.asarray(idx_col, dtype=np.int_)

        if idx_row.shape != idx_col.shape:
            raise ValueError("idx_row and idx_col must have the same shape.")

        return self._sparse_from_index_pairs(
            spectra_1=spectra_1,
            spectra_2=spectra_2,
            idx_row=idx_row,
            idx_col=idx_col,
            is_symmetric=is_symmetric,
            progress_bar=progress_bar,
        )

    def keep_score(self, score):
        """In the `.matrix` method scores will be collected in a sparse way.
        Overwrite this method here if values other than `False` or `0` should
        not be stored in the final collection.
        """
        if len(score.dtype) > 1:  # if structured array
            valuelike = True
            for dtype_name in score.dtype.names:
                valuelike = valuelike and (score[dtype_name] != 0)
            return valuelike
        return score != 0

    def to_dict(self) -> dict:
        """Return a dictionary representation of a similarity function."""
        return {
            "__Similarity__": self.__class__.__name__,
            **self.__dict__
            }

    def _validate_and_prepare_inputs(
        self,
        spectra_1: Sequence[SpectrumType],
        spectra_2: Optional[Sequence[SpectrumType]],
    ) -> tuple[Sequence[SpectrumType], bool]:
        """Validate inputs and determine whether symmetric optimization is possible."""
        if spectra_2 is None:
            spectra_2 = spectra_1
            is_symmetric = True
        else:
            is_symmetric = False

        return spectra_2, is_symmetric

    def _sparse_from_all_pairs(
        self,
        spectra_1: Sequence[SpectrumType],
        spectra_2: Sequence[SpectrumType],
        is_symmetric: bool,
        progress_bar: bool,
    ):
        """Compute sparse results by iterating over all pairs and only storing kept scores."""
        n_rows = len(spectra_1)
        n_cols = len(spectra_2)

        idx_row = []
        idx_col = []
        values = []

        iterator = tqdm(
            enumerate(spectra_1),
            total=n_rows,
            desc="Calculating sparse similarities",
            disable=not progress_bar,
        )

        for i, spectrum_1 in iterator:
            if is_symmetric and self.is_commutative:
                spectra_2_iter = enumerate(spectra_2[i:], start=i)
            else:
                spectra_2_iter = enumerate(spectra_2)

            for j, spectrum_2 in spectra_2_iter:
                score = np.asarray(self.pair(spectrum_1, spectrum_2), dtype=self.score_datatype)

                if not self.keep_score(score):
                    continue

                idx_row.append(i)
                idx_col.append(j)
                values.append(score)

                if is_symmetric and self.is_commutative and j != i:
                    idx_row.append(j)
                    idx_col.append(i)
                    values.append(score)

        return self._build_sparse_result(
            idx_row=np.asarray(idx_row, dtype=np.int_),
            idx_col=np.asarray(idx_col, dtype=np.int_),
            values=np.asarray(values, dtype=self.score_datatype),
            shape=(n_rows, n_cols),
        )

    def _sparse_from_index_pairs(
        self,
        spectra_1: Sequence[SpectrumType],
        spectra_2: Sequence[SpectrumType],
        idx_row: np.ndarray,
        idx_col: np.ndarray,
        is_symmetric: bool,
        progress_bar: bool,
    ):
        """Compute sparse results for explicitly given index pairs."""
        n_rows = len(spectra_1)
        n_cols = len(spectra_2)

        out_row = []
        out_col = []
        values = []

        iterator = tqdm(
            range(len(idx_row)),
            desc="Calculating sparse similarities",
            disable=not progress_bar,
        )

        for k in iterator:
            i = idx_row[k]
            j = idx_col[k]

            score = np.asarray(self.pair(spectra_1[i], spectra_2[j]), dtype=self.score_datatype)

            if not self.keep_score(score):
                continue

            out_row.append(i)
            out_col.append(j)
            values.append(score)

            if is_symmetric and self.is_commutative and i != j:
                out_row.append(j)
                out_col.append(i)
                values.append(score)

        return self._build_sparse_result(
            idx_row=np.asarray(out_row, dtype=np.int_),
            idx_col=np.asarray(out_col, dtype=np.int_),
            values=np.asarray(values, dtype=self.score_datatype),
            shape=(n_rows, n_cols),
        )

    def _build_sparse_result(
        self,
        idx_row: np.ndarray,
        idx_col: np.ndarray,
        values: np.ndarray,
        shape: tuple[int, int],
    ):
        """Build sparse result object.

        Returns
        -------
        scipy.sparse.coo_array or dict[str, scipy.sparse.coo_array]
            Scalar scores are returned as a single COO sparse array.
            Structured scores are returned as one COO sparse array per field.
        """
        if self.is_structured_score:
            result = {}
            for field in np.dtype(self.score_datatype).names:
                data = values[field]
                sparse = coo_array((data, (idx_row, idx_col)), shape=shape)
                sparse.eliminate_zeros()
                result[field] = sparse
            return result

        sparse = coo_array((values, (idx_row, idx_col)), shape=shape)
        sparse.eliminate_zeros()
        return sparse
