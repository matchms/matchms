"""Helper functions related to hashing."""

import hashlib
import json
import numpy as np
import pandas as pd
from scipy.sparse import csr_array
from .Fragments import Fragments


def spectrum_hash(peaks: Fragments, hash_length: int = 20, mz_precision: int = 5, intensity_precision: int = 2) -> str:
    """
    Compute hash from mz-intensity pairs of all peaks in spectrum.

    Parameters
    ----------
    peaks
        The Fragments object containing mz and intensities.
    hash_length
        The length of the hash to be computed.
    mz_precision
        The precision of the mz values.
    intensity_precision
        The precision of the intensities.

    Returns
    -------
    str
        The hash of the spectrum.
    """
    data = peaks.to_numpy
    return _compute_spectrum_hash(
        mz_array=data[:, 0],
        intensity_array=data[:, 1],
        hash_length=hash_length,
        mz_precision=mz_precision,
        intensity_precision=intensity_precision,
    )


def spectrum_hash_arrays(
    mz: np.ndarray, intensities: np.ndarray, hash_length: int = 20, mz_precision: int = 5, intensity_precision: int = 2
) -> str:
    """
    Compute hash from mz-intensity pairs of all peaks in spectrum.

    Parameters
    ----------
    mz
        mz values as ndarray.
    intensities
        intensities as ndarray.
    hash_length
        The length of the hash to be computed.
    mz_precision
        The precision of the mz values.
    intensity_precision
        The precision of the intensities.

    Returns
    -------
    str
        The hash of the spectrum.
    """
    return _compute_spectrum_hash(
        mz_array=mz,
        intensity_array=intensities,
        hash_length=hash_length,
        mz_precision=mz_precision,
        intensity_precision=intensity_precision,
    )


def _compute_spectrum_hash(
    mz_array: np.ndarray, intensity_array: np.ndarray, hash_length: int, mz_precision: int, intensity_precision: int
) -> str:
    """
    Compute hash from mz-intensity pairs of all peaks in spectrum.
    Method is inspired by SPLASH (doi:10.1038/nbt.3689).

    Parameters
    ----------
    mz_array
        mz values as ndarray.
    intensity_array
        intensities as ndarray.
    hash_length
        The length of the hash to be computed.
    mz_precision
        The precision of the mz values.
    intensity_precision
        The precision of the intensities.

    Returns
    -------
    str
        The hash of the spectrum.
    """
    mz_precision_factor = 10 ** mz_precision
    intensity_precision_factor = 10 ** intensity_precision

    mz_int = (mz_array * mz_precision_factor).astype(np.int64)
    int_int = (intensity_array * intensity_precision_factor).astype(np.int64)

    # Sort by increasing m/z and then by decreasing intensity
    order = np.lexsort((-int_int, mz_int))

    peak_strings = [f"{m}:{i}" for m, i in zip(mz_int[order], int_int[order])]
    encoded = " ".join(peak_strings).encode("utf-8")

    return hashlib.sha256(encoded).hexdigest()[:hash_length]


def spectra_hashes(fragments: csr_array, bin_to_mz_func, hash_length: int = 20, **kwargs) -> np.ndarray:
    """
    Compute hashes for a collection of spectra stored in a sparse (CSR) matrix.

    Parameters
    ----------
    fragments : csr_array
        A Scipy sparse matrix in CSR format where each row represents a spectrum
        and each column represents an m/z bin. Cell values are intensities.
    bin_to_mz_func : callable
        A function or method that accepts an array of bin indices (column indices)
        and returns an array of corresponding m/z values (floats).
        This is part of SpectraCollection.
    hash_length : int, optional
        The desired length of the resulting hash strings. Defaults to 20.
    **kwargs
        Additional parameters passed to `spectrum_hash_arrays`,
        such as `mz_precision` and `intensity_precision`.

    Returns
    -------
    np.ndarray
        A NumPy array of type 'U<hash_length>' containing the calculated
        hashes for all rows in the input matrix in their original order.
    """
    data = fragments.data
    indices = fragments.indices
    indptr = fragments.indptr

    n_spectra = fragments.shape[0]
    hashes = np.empty(n_spectra, dtype="U" + str(hash_length))

    for i in range(n_spectra):
        start, end = indptr[i], indptr[i + 1]

        row_mz = bin_to_mz_func(indices[start:end])
        row_int = data[start:end]

        hashes[i] = spectrum_hash_arrays(row_mz, row_int, hash_length, **kwargs)

    return hashes


def metadata_hash(metadata: dict, hash_length: int = 20):
    """Compute hash from metadata dictionary."""
    encoded = json.dumps(metadata, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()[:hash_length]


def compute_combined_hashes(fragment_hashes: list[str], metadata_hashes: list[str]) -> list[int]:
    s_frag = pd.Series(fragment_hashes, dtype=str)
    s_meta = pd.Series(metadata_hashes, dtype=str)

    combined = s_frag + s_meta

    return combined.map(
        lambda x: int(hashlib.sha1(x.encode()).hexdigest(), 16)
    ).tolist()
