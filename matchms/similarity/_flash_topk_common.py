"""Reusable per-worker candidate state for exact indexed Flash top-k scoring.

The numerical assignment rules live in the individual entropy/cosine kernels.
This module only owns epoch-stamped candidate registration and compact hybrid
peak-use storage, shared by both kernels without dense output rows.
"""
from __future__ import annotations

import numpy as np
from numba import njit

_INITIAL_CAPACITY = 256
_MAX_EPOCH = int(np.iinfo(np.uint32).max)


class _FlashTopKWorkspace:
    """Common O(n_references) scratch plus growable touched/bitset storage.

    A workspace belongs to exactly one thread. ``scores`` accumulates float64
    values matching dense Flash kernels; ``selection_scores`` casts final values
    to the public dtype *before* thresholding and tie-breaking.
    """

    def __init__(self, n_references: int, n_peaks: int, dtype: np.dtype, hybrid: bool):
        dtype = np.dtype(dtype)
        index_dtype = np.int32 if n_references <= np.iinfo(np.int32).max else np.int64
        self.scores = np.empty(n_references, dtype=np.float64)
        self.selection_scores = (
            self.scores if dtype == np.dtype(np.float64)
            else np.empty(n_references, dtype=dtype)
        )
        self.epochs = np.zeros(n_references, dtype=np.uint32)
        capacity = min(n_references, _INITIAL_CAPACITY)
        self.touched = np.empty(capacity, dtype=index_dtype)
        self.slots = np.empty(n_references if hybrid else 0, dtype=index_dtype)
        self.query_bits = np.empty((capacity, 0), dtype=np.uint64)
        n_peak_words = (n_peaks + 63) // 64 if hybrid else 0
        self.peak_word_epochs = np.zeros(n_peak_words, dtype=np.uint32)
        self.peak_word_bits = np.empty(n_peak_words, dtype=np.uint64)
        self.epoch = 0

    def next_epoch(self) -> int:
        """Increment generation; clear generation stamps only at wraparound."""
        if self.epoch == _MAX_EPOCH:
            self.epochs.fill(0)
            self.peak_word_epochs.fill(0)
            self.epoch = 1
        else:
            self.epoch += 1
        return self.epoch

    def ensure_query_words(self, n_query_peaks: int) -> None:
        """Resize query-peak bitmap width outside Numba row kernels."""
        words = (n_query_peaks + 63) // 64
        if words > self.query_bits.shape[1]:
            self.query_bits = np.empty((self.touched.size, words), dtype=np.uint64)


@njit(cache=True, inline="always")
def _register_touched(epochs, slots, touched, query_bits, n_touched, epoch, column):
    """Register a reference on first encounter, preserving compact hybrid state.

    Return ``is_new`` so each scoring kernel can initialize its own matching
    cursors and scoring variables, without duplicating allocation bookkeeping.
    """
    is_new = epochs[column] != epoch
    if is_new:
        if n_touched == touched.size:
            new_size = min(epochs.size, max(1, touched.size * 2))
            grown = np.empty(new_size, dtype=touched.dtype)
            grown[:n_touched] = touched[:n_touched]
            touched = grown
            if query_bits.shape[1]:
                grown_bits = np.empty((new_size, query_bits.shape[1]), dtype=np.uint64)
                grown_bits[:n_touched, :] = query_bits[:n_touched, :]
                query_bits = grown_bits
            else:
                query_bits = np.empty((new_size, 0), dtype=np.uint64)
        epochs[column] = epoch
        touched[n_touched] = column
        if query_bits.shape[1]:
            slots[column] = n_touched
            query_bits[n_touched, :] = 0
        n_touched += 1
    return touched, query_bits, n_touched, is_new
