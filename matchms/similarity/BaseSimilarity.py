# Subclassing guide:
# - implement pair()
# - optionally define score_datatype and score_fields
# - optionally overwrite keep_score() for default sparse filtering
# - optionally overwrite matrix() for performance optimizations
# - for scores that also provide a sparse score compuation use BaseSimilarityWithSparse
#  and optionally overwrite sparse_matrix() for performance optimizations
# - users can also pass score_filter=... to sparse_matrix()

from abc import ABC, abstractmethod
from typing import Optional, Sequence
import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_array
from tqdm import tqdm
from matchms.Scores import Scores
from matchms.typing import ScoreFilter, SpectrumType


class BaseSimilarity(ABC):
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

    def matrix(
        self,
        spectra_1: Sequence[SpectrumType],
        spectra_2: Optional[Sequence[SpectrumType]] = None,
        score_fields: Optional[Sequence[str]] = None,
        progress_bar: bool = True,
    ):
        """Calculate a dense similarity matrix.

        Parameters
        ----------
        spectra_1
            First collection of spectra.
        spectra_2
            Second collection of spectra. If None, compare ``spectra_1`` against
            itself. For commutative similarities this automatically uses a
            symmetric optimization.
        score_fields
            Score fields to return.
            - ``None`` means return all available fields.
            - For scalar scores, only ``("score",)`` is valid.
            - For structured scores, this can be a subset such as ``("score",)``.
        progress_bar
            When True, show a progress bar. Default is True.

        Returns
        -------
        Scores
            Dense score result wrapped in a ``Scores`` container.
        """
        spectra_2, is_symmetric = self._prepare_inputs(spectra_1, spectra_2)
        selected_fields = self._resolve_score_fields(score_fields)

        n_rows = len(spectra_1)
        n_cols = len(spectra_2)
        result = self._create_dense_result(n_rows, n_cols, selected_fields)

        for i, spectrum_1 in tqdm(
            enumerate(spectra_1),
            total=n_rows,
            desc="Calculating similarities",
            disable=not progress_bar,
        ):
            if is_symmetric and self.is_commutative:
                pairs = enumerate(spectra_2[i:], start=i)
            else:
                pairs = enumerate(spectra_2)

            for j, spectrum_2 in pairs:
                score = self._as_score(self.pair(spectrum_1, spectrum_2))
                self._store_in_dense_result(result, i, j, score, selected_fields)

                if is_symmetric and self.is_commutative and i != j:
                    self._store_in_dense_result(result, j, i, score, selected_fields)

        return Scores(result)

    def sparse_matrix(
        self,
        spectra_1,
        spectra_2=None,
        idx_row=None,
        idx_col=None,
        score_fields=None,
        score_filter=None,
        progress_bar: bool = True,
    ):
        """Sparse score computation is not available for this similarity."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not implement sparse_matrix(). "
            "Use a similarity class derived from BaseSimilarityWithSparse "
            "or use matrix() instead."
        )

    def to_dict(self) -> dict:
        """Return a dictionary representation of the similarity function."""
        return {
            "__Similarity__": self.__class__.__name__,
            **self.__dict__,
        }

    @property
    def is_structured_score(self) -> bool:
        """Return True if this similarity uses a structured score dtype."""
        return np.dtype(self.score_datatype).names is not None

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _prepare_inputs(
        self,
        spectra_1: Sequence[SpectrumType],
        spectra_2: Optional[Sequence[SpectrumType]],
    ) -> tuple[Sequence[SpectrumType], bool]:
        """Prepare input collections and determine symmetry."""
        if spectra_2 is None:
            return spectra_1, True
        return spectra_2, False

    def _available_score_fields(self) -> tuple[str, ...]:
        """Return the available score fields and validate consistency."""
        dtype_names = np.dtype(self.score_datatype).names

        if dtype_names is None:
            if tuple(self.score_fields) != ("score",):
                raise ValueError("Scalar scores must define score_fields=('score',).")
            return ("score",)

        dtype_names = tuple(dtype_names)
        if tuple(self.score_fields) != dtype_names:
            raise ValueError(
                "score_fields does not match the field names in score_datatype. "
                f"Got score_fields={self.score_fields}, dtype names={dtype_names}."
            )
        return dtype_names

    def _resolve_score_fields(self, score_fields: Optional[Sequence[str]]) -> tuple[str, ...]:
        """Validate and resolve the requested score fields."""
        available_fields = self._available_score_fields()

        if score_fields is None:
            selected_fields = available_fields
        else:
            selected_fields = tuple(score_fields)

        if len(selected_fields) == 0:
            raise ValueError("score_fields must contain at least one field.")

        unknown = tuple(field for field in selected_fields if field not in available_fields)
        if unknown:
            raise ValueError(
                f"Unknown score field(s): {unknown}. Available fields are: {available_fields}."
            )

        return selected_fields

    def _as_score(self, score) -> np.ndarray:
        """Convert one score to the declared score dtype."""
        return np.asarray(score, dtype=self.score_datatype)

    def _create_dense_result(
        self,
        n_rows: int,
        n_cols: int,
        selected_fields: tuple[str, ...],
    ) -> dict[str, np.ndarray]:
        """Create an empty dense result container."""
        if not self.is_structured_score:
            return {"score": np.zeros((n_rows, n_cols), dtype=self.score_datatype)}

        return {
            field: np.zeros((n_rows, n_cols), dtype=np.dtype(self.score_datatype)[field])
            for field in selected_fields
        }

    def _store_in_dense_result(
        self,
        result: dict[str, np.ndarray],
        i: int,
        j: int,
        score: np.ndarray,
        selected_fields: tuple[str, ...],
    ) -> None:
        """Store one score in the dense result container."""
        if not self.is_structured_score:
            result["score"][i, j] = score
            return

        for field in selected_fields:
            result[field][i, j] = score[field]


class BaseSimilarityWithSparse(BaseSimilarity):
    """Base similarity class with a default sparse implementation.

    This class extends BaseSimilarity by providing a default implementation of
    sparse_matrix() that applies a score filter to the dense results.

    Subclasses can override keep_score() to define the default filtering behavior,
    and users can also pass a custom score_filter=... to sparse_matrix() for
    per-call control.
    """
   
    def sparse_matrix(
        self,
        spectra_1: Sequence[SpectrumType],
        spectra_2: Optional[Sequence[SpectrumType]] = None,
        idx_row: Optional[npt.ArrayLike] = None,
        idx_col: Optional[npt.ArrayLike] = None,
        score_fields: Optional[Sequence[str]] = None,
        score_filter: Optional[ScoreFilter] = None,
        progress_bar: bool = True,
    ):
        """Calculate sparse similarity results.

        Filtering is applied to the full score before score field projection.

        Parameters
        ----------
        spectra_1
            First collection of spectra.
        spectra_2
            Second collection of spectra. If None, compare ``spectra_1`` against
            itself.
        idx_row
            Row indices of pairs to compute. If None and ``idx_col`` is also None,
            all pairwise comparisons are considered and only retained scores are stored.
        idx_col
            Column indices of pairs to compute. Must have the same shape as ``idx_row``.
        score_fields
            Score fields to return.
            - ``None`` means return all available fields.
            - For scalar scores, only ``("score",)`` is valid.
            - For structured scores, this can be a subset such as ``("score",)``.
        score_filter
            Optional callable receiving the full score and returning whether it
            should be retained. If None, :meth:`keep_score` is used.
        progress_bar
            When True, show a progress bar.

        Returns
        -------
        Scores
            Sparse score result wrapped in a ``Scores`` container.
        """
        spectra_2, is_symmetric = self._prepare_inputs(spectra_1, spectra_2)
        selected_fields = self._resolve_score_fields(score_fields)

        # No explicit indices given, compute all pairwise comparisons and filter
        if idx_row is None and idx_col is None:
            sparse_result = self._sparse_matrix_from_all_pairs(
                spectra_1=spectra_1,
                spectra_2=spectra_2,
                is_symmetric=is_symmetric,
                selected_fields=selected_fields,
                score_filter=score_filter,
                progress_bar=progress_bar,
            )
            return Scores(sparse_result)

        # Both idx_row and idx_col must be given for explicit sparse computation
        if idx_row is None or idx_col is None:
            raise ValueError("idx_row and idx_col must either both be given or both be None.")

        # Explicit indices given, compute only those pairs and filter
        idx_row = np.asarray(idx_row, dtype=np.int_)
        idx_col = np.asarray(idx_col, dtype=np.int_)
        if idx_row.shape != idx_col.shape:
            raise ValueError("idx_row and idx_col must have the same shape.")

        # Avoid redundant computations for symmetric commutative similarities
        if is_symmetric and self.is_commutative:
            mask = idx_row <= idx_col
            idx_row = idx_row[mask]
            idx_col = idx_col[mask]

        sparse_result = self._sparse_matrix_from_explicit_indices(
            spectra_1=spectra_1,
            spectra_2=spectra_2,
            idx_row=idx_row,
            idx_col=idx_col,
            is_symmetric=is_symmetric,
            selected_fields=selected_fields,
            score_filter=score_filter,
            progress_bar=progress_bar,
        )
        return Scores(sparse_result)


    def _sparse_matrix_from_all_pairs(
        self,
        spectra_1: Sequence[SpectrumType],
        spectra_2: Sequence[SpectrumType],
        is_symmetric: bool,
        selected_fields: tuple[str, ...],
        score_filter: Optional[ScoreFilter],
        progress_bar: bool,
    ) -> dict[str, coo_array]:
        """Compute sparse scores directly from all pairwise comparisons.

        This implementation avoids Python list growth across the full matrix by
        collecting kept scores row-wise in NumPy arrays and storing trimmed row
        chunks. This is substantially more memory efficient than repeated append
        operations for large computations.
        """
        n_rows = len(spectra_1)
        n_cols = len(spectra_2)

        row_chunks = []
        col_chunks = []
        value_chunks = []

        for i, spectrum_1 in tqdm(
            enumerate(spectra_1),
            total=n_rows,
            desc="Calculating sparse similarities",
            disable=not progress_bar,
        ):
            if is_symmetric and self.is_commutative:
                start_j = i
                row_capacity = n_cols - i
            else:
                start_j = 0
                row_capacity = n_cols

            row_chunk = np.empty(row_capacity, dtype=np.int_)
            col_chunk = np.empty(row_capacity, dtype=np.int_)
            value_chunk = np.empty(row_capacity, dtype=self.score_datatype)

            fill = 0
            for j in range(start_j, n_cols):
                spectrum_2 = spectra_2[j]
                score = self._as_score(self.pair(spectrum_1, spectrum_2))

                if not self._should_keep(score, score_filter):
                    continue

                row_chunk[fill] = i
                col_chunk[fill] = j
                value_chunk[fill] = score
                fill += 1

            if fill > 0:
                kept_rows = row_chunk[:fill].copy()
                kept_cols = col_chunk[:fill].copy()
                kept_values = value_chunk[:fill].copy()

                row_chunks.append(kept_rows)
                col_chunks.append(kept_cols)
                value_chunks.append(kept_values)

                if is_symmetric and self.is_commutative:
                    offdiag = kept_rows != kept_cols
                    if np.any(offdiag):
                        row_chunks.append(kept_cols[offdiag].copy())
                        col_chunks.append(kept_rows[offdiag].copy())
                        value_chunks.append(kept_values[offdiag].copy())

        if not row_chunks:
            idx_row = np.array([], dtype=np.int_)
            idx_col = np.array([], dtype=np.int_)
            values = np.array([], dtype=self.score_datatype)
        else:
            idx_row = np.concatenate(row_chunks)
            idx_col = np.concatenate(col_chunks)
            values = np.concatenate(value_chunks)

        return self._build_sparse_result(
            idx_row=idx_row,
            idx_col=idx_col,
            values=values,
            shape=(n_rows, n_cols),
            selected_fields=selected_fields,
        )


    def _sparse_matrix_from_explicit_indices(
        self,
        spectra_1: Sequence[SpectrumType],
        spectra_2: Sequence[SpectrumType],
        idx_row: np.ndarray,
        idx_col: np.ndarray,
        is_symmetric: bool,
        selected_fields: tuple[str, ...],
        score_filter: Optional[ScoreFilter],
        progress_bar: bool,
    ) -> dict[str, coo_array]:
        """Compute sparse scores for explicitly given index pairs."""
        out_row = []
        out_col = []
        values = []

        for k in tqdm(
            range(len(idx_row)),
            desc="Calculating sparse similarities",
            disable=not progress_bar,
        ):
            i = idx_row[k]
            j = idx_col[k]

            score = self._as_score(self.pair(spectra_1[i], spectra_2[j]))

            if not self._should_keep(score, score_filter):
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
            shape=(len(spectra_1), len(spectra_2)),
            selected_fields=selected_fields,
        )


    def keep_score(self, score) -> bool:
        """Return whether a score should be retained in sparse outputs.

        This defines the default sparse retention behavior.
        Users can override it per call via ``score_filter=...``.

        Default behavior:
        - scalar score: keep if ``score != 0``
        - structured score: keep if all fields are non-zero
        """
        score = self._as_score(score)

        if self.is_structured_score:
            return all(score[field] != 0 for field in score.dtype.names)
        return bool(score != 0)

    def _should_keep(self, score: np.ndarray, score_filter: Optional[ScoreFilter]) -> bool:
        """Return whether a score should be kept in sparse output."""
        if score_filter is not None:
            return bool(score_filter(score))
        return self.keep_score(score)

    def _build_sparse_result(
        self,
        idx_row: np.ndarray,
        idx_col: np.ndarray,
        values: np.ndarray,
        shape: tuple[int, int],
        selected_fields: tuple[str, ...],
    ) -> dict[str, coo_array]:
        """Build sparse output from collected coordinates and values."""
        if not self.is_structured_score:
            sparse = coo_array((values, (idx_row, idx_col)), shape=shape)
            sparse.eliminate_zeros()
            return {"score": sparse}

        result = {}
        for field in selected_fields:
            sparse = coo_array((values[field], (idx_row, idx_col)), shape=shape)
            sparse.eliminate_zeros()
            result[field] = sparse
        return result
