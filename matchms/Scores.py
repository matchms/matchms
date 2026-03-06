from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
from scipy.sparse import coo_array


@dataclass(frozen=True)
class ScoresMask:
    """Boolean mask for Scores, stored either densely or as sparse coordinates.
    
    Parameters
    ----------
    shape
        Shape of the score matrix this mask applies to.
    dense_mask
        Optional dense boolean array of shape `shape`. If provided, `row` and `col` must be None.
    row
        Optional array of row indices for sparse representation.
    col
        Optional array of column indices for sparse representation. If provided, `dense_mask` must be None.
    """
    shape: tuple[int, int]
    dense_mask: Optional[np.ndarray] = None
    row: Optional[np.ndarray] = None
    col: Optional[np.ndarray] = None

    def __post_init__(self):
        has_dense = self.dense_mask is not None
        has_sparse = self.row is not None or self.col is not None

        if has_dense and has_sparse:
            raise ValueError("ScoresMask must be either dense or sparse, not both.")
        if not has_dense and not has_sparse:
            raise ValueError("ScoresMask requires either dense_mask or row/col.")
        if has_sparse:
            if self.row is None or self.col is None:
                raise ValueError("Sparse ScoresMask requires both row and col.")
            if self.row.shape != self.col.shape:
                raise ValueError("row and col must have the same shape.")

    @property
    def is_sparse(self) -> bool:
        return self.dense_mask is None

    def to_dense(self) -> np.ndarray:
        if self.dense_mask is not None:
            return self.dense_mask
        mask = np.zeros(self.shape, dtype=bool)
        mask[self.row, self.col] = True
        return mask

    def __and__(self, other: "ScoresMask") -> "ScoresMask":
        self._check_shape(other)
        if self.is_sparse and other.is_sparse:
            return self._from_coord_set(self._coord_set() & other._coord_set())
        return ScoresMask(shape=self.shape, dense_mask=self.to_dense() & other.to_dense())

    def __or__(self, other: "ScoresMask") -> "ScoresMask":
        self._check_shape(other)
        if self.is_sparse and other.is_sparse:
            return self._from_coord_set(self._coord_set() | other._coord_set())
        return ScoresMask(shape=self.shape, dense_mask=self.to_dense() | other.to_dense())

    def __invert__(self) -> "ScoresMask":
        return ScoresMask(shape=self.shape, dense_mask=~self.to_dense())

    def _check_shape(self, other: "ScoresMask") -> None:
        if self.shape != other.shape:
            raise ValueError(f"Incompatible mask shapes: {self.shape} and {other.shape}.")

    def _coord_set(self) -> set[tuple[int, int]]:
        return set(zip(self.row.tolist(), self.col.tolist()))

    def _from_coord_set(self, coords: set[tuple[int, int]]) -> "ScoresMask":
        if not coords:
            row = np.array([], dtype=np.int_)
            col = np.array([], dtype=np.int_)
        else:
            coords = sorted(coords)
            row = np.array([r for r, _ in coords], dtype=np.int_)
            col = np.array([c for _, c in coords], dtype=np.int_)
        return ScoresMask(shape=self.shape, row=row, col=col)


@dataclass(frozen=True)
class ScoresField:
    """View on one score field.
    
    Supports array-like access and comparison operations to create boolean masks.
    """
    _scores: "Scores"
    _field: str

    @property
    def shape(self) -> tuple[int, int]:
        return self._scores.shape

    @property
    def is_sparse(self) -> bool:
        return self._scores.is_sparse

    def to_array(self) -> np.ndarray:
        return self._scores.to_array(self._field)

    def to_coo(self) -> coo_array:
        return self._scores.to_coo(self._field)

    def __getitem__(self, key):
        return self.to_array()[key]

    def __gt__(self, other):
        return self._compare(other, np.greater, sparse_safe=True)

    def __ge__(self, other):
        return self._compare(other, np.greater_equal, sparse_safe=True)

    def __lt__(self, other):
        return self._compare(other, np.less, sparse_safe=False)

    def __le__(self, other):
        return self._compare(other, np.less_equal, sparse_safe=False)

    def __eq__(self, other):
        return self._compare(other, np.equal, sparse_safe=False)

    def __ne__(self, other):
        return self._compare(other, np.not_equal, sparse_safe=False)

    def _compare(self, other, op, sparse_safe: bool) -> ScoresMask:
        if not self.is_sparse:
            return ScoresMask(shape=self.shape, dense_mask=op(self.to_array(), other))

        if sparse_safe and other >= 0:
            coo = self.to_coo()
            keep = op(coo.data, other)
            return ScoresMask(
                shape=self.shape,
                row=coo.row[keep],
                col=coo.col[keep],
            )

        return ScoresMask(shape=self.shape, dense_mask=op(self.to_array(), other))


class Scores:
    """Container for computed matchms scores.
    
    Stores one or more score fields as either dense numpy arrays or sparse COO arrays.
    Provides methods for accessing score fields, filtering scores with boolean masks, and slicing.
    """

    def __init__(self, data: Dict[str, np.ndarray | coo_array]):
        if not data:
            raise ValueError("Scores requires at least one score field.")

        self._data = dict(data)
        self._score_fields = tuple(data.keys())

        first_value = next(iter(self._data.values()))
        self._is_sparse = isinstance(first_value, coo_array)
        self._shape = first_value.shape

        for field, value in self._data.items():
            if isinstance(value, coo_array) != self._is_sparse:
                raise ValueError("All score fields must be either dense or sparse.")
            if value.shape != self._shape:
                raise ValueError(
                    f"All score fields must have the same shape. "
                    f"Field {field!r} has shape {value.shape}, expected {self._shape}."
                )

    def __repr__(self) -> str:
        kind = "sparse" if self.is_sparse else "dense"
        return f"Scores(shape={self.shape}, score_fields={self.score_fields}, kind={kind})"

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def score_fields(self) -> tuple[str, ...]:
        return self._score_fields

    @property
    def is_sparse(self) -> bool:
        return self._is_sparse

    @property
    def is_scalar(self) -> bool:
        return len(self.score_fields) == 1

    def to_array(self, field: Optional[str] = None) -> np.ndarray:
        field = self._resolve_field(field)
        value = self._data[field]
        if self.is_sparse:
            return value.toarray()
        return value.copy()

    def to_coo(self, field: Optional[str] = None) -> coo_array:
        field = self._resolve_field(field)
        value = self._data[field]
        if self.is_sparse:
            return value
        row, col = np.nonzero(value)
        return coo_array((value[row, col], (row, col)), shape=value.shape)

    def filter(self, mask) -> "Scores":
        if isinstance(mask, ScoresMask):
            return self._filter_with_scores_mask(mask)

        mask = np.asarray(mask, dtype=bool)
        if mask.shape != self.shape:
            raise ValueError(f"Mask has shape {mask.shape}, expected {self.shape}.")
        return self._filter_with_dense_mask(mask)

    def __getitem__(self, key):
        if isinstance(key, str):
            return ScoresField(self, key)

        if isinstance(key, ScoresMask):
            return self.filter(key)

        if isinstance(key, np.ndarray):
            return self.filter(key)

        if self.is_scalar:
            field = self.score_fields[0]
            return self.to_array(field)[key]

        if isinstance(key, tuple):
            return {field: self.to_array(field)[key] for field in self.score_fields}

        sliced = {field: self.to_array(field)[key] for field in self.score_fields}
        normalized = {}
        for field, value in sliced.items():
            arr = np.asarray(value)
            if arr.ndim == 1:
                normalized[field] = arr.reshape(1, -1)
            else:
                normalized[field] = arr
        return Scores(normalized)

    def _filter_with_scores_mask(self, mask: ScoresMask) -> "Scores":
        if mask.shape != self.shape:
            raise ValueError(f"Mask has shape {mask.shape}, expected {self.shape}.")

        if self.is_sparse and mask.is_sparse:
            return self._filter_sparse_with_sparse_mask(mask)

        return self._filter_with_dense_mask(mask.to_dense())

    def _filter_sparse_with_sparse_mask(self, mask: ScoresMask) -> "Scores":
        mask_coords = set(zip(mask.row.tolist(), mask.col.tolist()))
        filtered = {}

        for field in self.score_fields:
            coo = self.to_coo(field)
            keep = np.array(
                [(r, c) in mask_coords for r, c in zip(coo.row.tolist(), coo.col.tolist())],
                dtype=bool,
            )
            filtered[field] = coo_array(
                (coo.data[keep], (coo.row[keep], coo.col[keep])),
                shape=self.shape,
            )

        return Scores(filtered)

    def _filter_with_dense_mask(self, mask: np.ndarray) -> "Scores":
        filtered = {}
        for field in self.score_fields:
            arr = self.to_array(field)
            arr = np.where(mask, arr, 0)
            if self.is_sparse:
                row, col = np.nonzero(arr)
                filtered[field] = coo_array((arr[row, col], (row, col)), shape=self.shape)
            else:
                filtered[field] = arr
        return Scores(filtered)

    def _resolve_field(self, field: Optional[str]) -> str:
        if field is None:
            if self.is_scalar:
                return self.score_fields[0]
            raise KeyError(f"Field name required. Available fields: {self.score_fields}.")
        if field not in self._data:
            raise KeyError(f"Unknown field {field!r}. Available fields: {self.score_fields}.")
        return field
    