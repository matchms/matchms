"""Compact, ranked results from top-k spectral library retrieval."""

from __future__ import annotations
import operator
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class TopKResults:
    """Exact top-k matches for each query, without a dense similarity matrix.

    ``indices`` contains 0-based positions in the original reference index and
    ``scores`` holds the corresponding similarities. Both have shape
    ``(n_queries, k)`` and are ranked by descending score, breaking score ties
    by ascending library index. ``valid_counts[i]`` is the number of real hits
    for query ``i``; remaining positions are padded with index -1 and score 0.

    In top-k search, only scores *strictly greater* than ``min_score`` qualify.
    In particular, the default ``min_score=0`` never returns arbitrary zero-score
    library spectra. Omitted pairs must not be interpreted as computed zeroes.

    The arrays are presented as read-only views. Like other NumPy views, their
    backing data must not be mutated through external aliases.

    The result intentionally does not subclass ``matchms.scores.Scores`` because
    excluded entries may have positive but lower similarity scores.
    """

    indices: np.ndarray
    scores: np.ndarray
    valid_counts: np.ndarray
    n_references: int

    def __post_init__(self) -> None:
        if isinstance(self.n_references, (bool, np.bool_)):
            raise TypeError("n_references must be a nonnegative integer.")
        n_references = operator.index(self.n_references)
        if n_references < 0:
            raise ValueError("n_references must be nonnegative.")
        object.__setattr__(self, "n_references", n_references)

        indices, scores, counts = self.indices, self.scores, self.valid_counts
        if not all(isinstance(array, np.ndarray) for array in (indices, scores, counts)):
            raise TypeError("indices, scores, and valid_counts must be NumPy arrays.")
        if indices.ndim != 2 or scores.shape != indices.shape:
            raise ValueError("indices and scores must have matching two-dimensional shapes.")
        if indices.dtype.kind != "i" or scores.dtype.kind != "f":
            raise TypeError("indices must be signed integers and scores floating point.")
        if counts.shape != (indices.shape[0],) or counts.dtype.kind not in "iu":
            raise ValueError("valid_counts must be an integer vector with one entry per query.")
        if np.any(counts > indices.shape[1]) or np.any(counts < 0):
            raise ValueError("valid_counts must be between zero and the requested k.")

        for row, length in enumerate(counts):
            n = int(length)
            ids = indices[row, :n]
            values = scores[row, :n]
            if np.any(ids < 0) or np.any(ids >= n_references) or not np.all(np.isfinite(values)):
                raise ValueError("Valid hits require in-range library IDs and finite scores.")
            if np.any(indices[row, n:] != -1) or np.any(scores[row, n:] != 0):
                raise ValueError("Unused result entries must be padded by ID -1 and score 0.")
            if n > 1 and (
                np.any(values[1:] > values[:-1]) or np.any((values[1:] == values[:-1]) & (ids[1:] < ids[:-1]))
            ):
                raise ValueError("Hits must be ordered by descending score, then ascending ID.")

        for name in ("indices", "scores", "valid_counts"):
            read_only = getattr(self, name).view()
            read_only.flags.writeable = False
            object.__setattr__(self, name, read_only)

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of the padded ranked result arrays, ``(n_queries, k)``."""
        return self.indices.shape

    @property
    def n_queries(self) -> int:
        """Number of query spectra in the results."""
        return self.shape[0]

    @property
    def k(self) -> int:
        """Maximum number of hits per query."""
        return self.shape[1]

    def __len__(self) -> int:
        return self.n_queries

    def get_hits(self, query_index: int) -> tuple[np.ndarray, np.ndarray]:
        """Return library IDs and scores for one query, excluding padding."""
        n = int(self.valid_counts[query_index])
        return self.indices[query_index, :n], self.scores[query_index, :n]
