import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import numpy as np
from scipy.sparse import coo_array
from matchms.typing import ScoresType


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


class Scores:
    """Container for computed matchms scores.
    
    The ``Scores`` class stores the output of one similarity computation and provides
    a small, intuitive API that works for both dense and sparse score matrices.

    A ``Scores`` instance can represent either:

    - a scalar score matrix with one field, usually ``"score"``
    - a multi-field score result, for example ``"score"`` and ``"matches"``
    - dense data stored as NumPy arrays
    - sparse data stored as SciPy COO arrays

    Parameters
    ----------
    data
        Dictionary mapping score field names to score data.
        Each value must be either a 2D NumPy array or a SciPy ``coo_array``.
        All fields must have the same shape and must all be either dense or sparse.

    Notes
    -----
    The class is designed to offer a consistent API independent of the underlying
    storage format.

    Field access
        Score fields can be accessed by name, for example ``scores["score"]`` or
        ``scores["matches"]``. Field selection returns another ``Scores`` object
        containing only the selected field.

    Scalar scores
        If only one field is present, direct comparisons are supported, for example
        ``scores > 0.5``. This is equivalent to ``scores["score"] > 0.5``.

    Masking
        Boolean masking returns a filtered ``Scores`` object with the same shape.
        For example, ``scores[scores["score"] > 0.5]`` keeps only entries where the
        condition is true.

    Slicing
        Basic slicing is supported, for example ``scores[3, 4]``, ``scores[3, :]``,
        or ``scores[:, 2]``.

    Conversion
        Use :meth:`to_array` to obtain a dense NumPy representation and
        :meth:`to_coo` to obtain a sparse COO representation.

    Examples
    --------
    Scalar dense scores:

    >>> scores = Scores({"score": np.array([[1.0, 0.0], [0.3, 0.8]])})
    >>> scores["score"].to_array()
    array([[1. , 0. ],
           [0.3, 0.8]])
    >>> filtered = scores[scores > 0.5]
    >>> filtered.to_array()
    array([[1. , 0. ],
           [0. , 0.8]])

    Multi-field scores:

    >>> scores = Scores({
    ...     "score": np.array([[1.0, 0.0], [0.3, 0.8]]),
    ...     "matches": np.array([[5, 0], [1, 4]])
    ... })
    >>> scores["score"].to_array()
    array([[1. , 0. ],
           [0.3, 0.8]])
    >>> scores["matches"].to_array()
    array([[5, 0],
           [1, 4]])
    >>> good = scores[(scores["score"] > 0.2) & (scores["matches"] >= 2)]
    >>> good.to_array("score")
    array([[1. , 0. ],
           [0. , 0.8]])
    """

    _FORMAT_NAME = "matchms.Scores"
    _FORMAT_VERSION = 1
    _METADATA_KEY = "__scores_metadata__"

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

    def filter(self, mask) -> ScoresType:
        if isinstance(mask, ScoresMask):
            return self._filter_with_scores_mask(mask)

        mask = np.asarray(mask, dtype=bool)
        if mask.shape != self.shape:
            raise ValueError(f"Mask has shape {mask.shape}, expected {self.shape}.")
        return self._filter_with_dense_mask(mask)

    def __getitem__(self, key):
        """Access fields, apply masks, or slice score data."""
        if isinstance(key, str):
            return Scores({key: self._data[self._resolve_field(key)]})

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

    def __gt__(self, other):
        """Element-wise comparison for scalar Scores."""
        return self._compare_scalar(other, np.greater, sparse_safe=True)

    def __ge__(self, other):
        """Element-wise comparison for scalar Scores."""
        return self._compare_scalar(other, np.greater_equal, sparse_safe=True)

    def __lt__(self, other):
        """Element-wise comparison for scalar Scores."""
        return self._compare_scalar(other, np.less, sparse_safe=False)

    def __le__(self, other):
        """Element-wise comparison for scalar Scores."""
        return self._compare_scalar(other, np.less_equal, sparse_safe=False)

    def __eq__(self, other):
        """Element-wise comparison for scalar Scores."""
        return self._compare_scalar(other, np.equal, sparse_safe=False)

    def __ne__(self, other):
        """Element-wise comparison for scalar Scores."""
        return self._compare_scalar(other, np.not_equal, sparse_safe=False)

    def _filter_with_scores_mask(self, mask: ScoresMask) -> ScoresType:
        if mask.shape != self.shape:
            raise ValueError(f"Mask has shape {mask.shape}, expected {self.shape}.")

        if self.is_sparse and mask.is_sparse:
            return self._filter_sparse_with_sparse_mask(mask)

        return self._filter_with_dense_mask(mask.to_dense())

    def _filter_sparse_with_sparse_mask(self, mask: ScoresMask) -> ScoresType:
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

    def _filter_with_dense_mask(self, mask: np.ndarray) -> ScoresType:
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

        if field in self._data:
            return field

        if field == "score" and self.is_scalar:
            return self.score_fields[0]

        raise KeyError(f"Unknown field {field!r}. Available fields: {self.score_fields}.")

    def _compare_scalar(self, other, op, sparse_safe: bool) -> ScoresMask:
        """Compare scalar Scores against a value and return a mask."""
        if not self.is_scalar:
            raise TypeError(
                "Direct comparisons are only supported for scalar Scores. "
                f"Available score fields: {self.score_fields}."
            )

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


    # File I/O methods for saving and loading Scores objects to/from .npz files
    # ---------------------------------------------------------------------------------
    def save(self, path: str | Path, compressed: bool = True) -> None:
        """Save the Scores object to a single `.npz` file.

        Parameters
        ----------
        path
            Output file path.
        compressed
            If True, use ``numpy.savez_compressed``. Default is True.
        """
        path = Path(path)

        metadata = {
            "format": self._FORMAT_NAME,
            "version": self._FORMAT_VERSION,
            "is_sparse": self.is_sparse,
            "score_fields": list(self.score_fields),
            "shape": list(self.shape),
        }

        payload = {
            self._METADATA_KEY: np.array(json.dumps(metadata)),
        }

        if self.is_sparse:
            for field in self.score_fields:
                coo = self.to_coo(field)
                payload[f"{field}__row"] = coo.row
                payload[f"{field}__col"] = coo.col
                payload[f"{field}__data"] = coo.data
        else:
            for field in self.score_fields:
                payload[field] = self._data[field]

        saver = np.savez_compressed if compressed else np.savez
        saver(path, **payload)

    @classmethod
    def load(cls, path: str | Path) -> "Scores":
        """Load a Scores object from a `.npz` file.

        Parameters
        ----------
        path
            Input file path.

        Returns
        -------
        Scores
            Reconstructed Scores object.
        """
        path = Path(path)

        with np.load(path, allow_pickle=False) as npz:
            if cls._METADATA_KEY not in npz:
                raise ValueError(
                    f"File {path} does not contain {cls._FORMAT_NAME} metadata."
                )

            metadata =  json.loads(str(npz[cls._METADATA_KEY]))
            cls._validate_metadata(metadata, path)

            is_sparse = bool(metadata["is_sparse"])
            score_fields = tuple(metadata["score_fields"])
            shape = tuple(metadata["shape"])

            data = {}
            if is_sparse:
                for field in score_fields:
                    row_key = f"{field}__row"
                    col_key = f"{field}__col"
                    data_key = f"{field}__data"

                    missing = [key for key in (row_key, col_key, data_key) if key not in npz]
                    if missing:
                        raise ValueError(
                            f"File {path} is missing sparse data for field {field!r}: {missing}"
                        )

                    row = npz[row_key]
                    col = npz[col_key]
                    values = npz[data_key]
                    data[field] = coo_array((values, (row, col)), shape=shape)
            else:
                for field in score_fields:
                    if field not in npz:
                        raise ValueError(
                            f"File {path} is missing dense data for field {field!r}."
                        )
                    data[field] = npz[field]

        return cls(data)


    @classmethod
    def _validate_metadata(cls, metadata: dict, path: Path) -> None:
        """Validate loaded metadata."""
        if metadata.get("format") != cls._FORMAT_NAME:
            raise ValueError(
                f"File {path} is not a {cls._FORMAT_NAME} file."
            )
        if metadata.get("version") != cls._FORMAT_VERSION:
            raise ValueError(
                f"Unsupported {cls._FORMAT_NAME} version {metadata.get('version')} in file {path}."
            )
        required_keys = {"format", "version", "is_sparse", "score_fields", "shape"}
        missing = required_keys.difference(metadata)
        if missing:
            raise ValueError(
                f"File {path} is missing metadata keys: {sorted(missing)}"
            )
