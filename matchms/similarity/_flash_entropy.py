"""One-to-one spectral entropy accumulation from globally indexed peaks.

Fragment coordinates are traversed in ascending order; neutral-loss coordinates
are traversed in ascending loss order. Matching state is maintained per library
spectrum, so overlapping tolerance windows cannot reuse a physical peak.
"""
import numpy as np
from numba import njit
from ._flash_tolerance import identity_allowed, window, within


@njit(cache=True, inline="always")
def xlog2(value: float) -> float:
    """Return x log2(x), extending the function continuously to zero."""
    return value * np.log2(value) if value > 0.0 else 0.0


@njit(cache=True, nogil=True)
def entropy_rows(
    out: np.ndarray,
    q_offsets: np.ndarray,
    q_mz: np.ndarray,
    q_int: np.ndarray,
    q_pmz: np.ndarray,
    fragment: tuple,
    neutral_loss: tuple,
    lib_pmz: np.ndarray,
    n_lib_peaks: int,
    tolerance: float,
    use_ppm: bool,
    mode: int,
    identity_tol: float,
    identity_ppm: bool,
    start_row: int,
    stop_row: int,
) -> None:
    """Write entropy score rows into caller-owned output arrays.

    ``fragment`` and ``neutral_loss`` each hold coordinates, intensities, cached
    x-log-x terms, spectrum IDs, and physical peak IDs. Coordinate ties follow
    the physical-peak order of their respective sweep. Query peaks are packed
    spectrum-by-spectrum with offsets into ``q_mz`` and ``q_int``.

    Modes are 0 (fragment), 1 (neutral loss), and 2 (fragment-first hybrid).
    Hybrid records consumed physical peaks and query peaks for each library
    spectrum before the loss pass. Each call owns its scratch arrays, allowing
    concurrent invocations to write disjoint query rows safely.
    """
    n_library = out.shape[1]
    fragment_mz, fragment_int, fragment_terms, fragment_spec, fragment_pid = fragment
    loss_mz, loss_int, loss_terms, loss_spec, loss_pid = neutral_loss
    score = np.empty(n_library, dtype=np.float64)
    last_peak = np.empty(n_library, dtype=np.int64)
    last_query = np.empty(n_library, dtype=np.int64)

    # Epochs avoid clearing a library-peak-sized bitmap for every query. The
    # query bitmap is per library spectrum because one query is matched against
    # each spectrum independently. Multiple words support more than 64 peaks.
    peak_epochs = np.zeros(n_lib_peaks if mode == 2 else 0, dtype=np.int64)
    max_query_peaks = 0
    for row in range(start_row, stop_row):
        max_query_peaks = max(max_query_peaks, q_offsets[row + 1] - q_offsets[row])
    query_bits = np.zeros(
        (n_library if mode == 2 else 0, (max_query_peaks + 63) // 64), dtype=np.uint64,
    )

    for row in range(start_row, stop_row):
        epoch = row - start_row + 1
        score[:] = 0.0
        last_peak[:] = -1
        last_query[:] = -1
        if mode == 2:
            query_bits[:, :] = 0
        start, stop = q_offsets[row], q_offsets[row + 1]
        has_precursor = np.isfinite(q_pmz[row])

        if mode != 1:
            for query_index in range(stop - start):
                coordinate = np.float64(q_mz[start + query_index])
                intensity = np.float64(q_int[start + query_index])
                if intensity <= 0.0:
                    continue
                query_term = xlog2(intensity)
                left, right = window(fragment_mz, coordinate, tolerance, use_ppm)
                for posting in range(left, right):
                    column = fragment_spec[posting]
                    peak_id = fragment_pid[posting]
                    # Each query peak and each library peak may be used only
                    # once for a particular query/library spectrum pair.
                    if last_query[column] == query_index or peak_id <= last_peak[column]:
                        continue
                    library_intensity = np.float64(fragment_int[posting])
                    if library_intensity <= 0.0:
                        continue
                    if not within(coordinate, np.float64(fragment_mz[posting]), tolerance, use_ppm):
                        continue
                    score[column] += (
                        xlog2(intensity + library_intensity) - query_term - fragment_terms[posting]
                    )
                    last_query[column] = query_index
                    last_peak[column] = peak_id
                    if mode == 2:
                        peak_epochs[peak_id] = epoch
                        query_bits[column, query_index // 64] |= np.uint64(1) << np.uint64(query_index % 64)

        if mode != 0 and has_precursor:
            last_peak[:] = n_lib_peaks
            last_query[:] = -1
            # Descending fragment coordinates give ascending loss coordinates.
            for query_index in range(stop - start - 1, -1, -1):
                intensity = np.float64(q_int[start + query_index])
                if intensity <= 0.0:
                    continue
                coordinate = np.float64(q_pmz[row]) - np.float64(q_mz[start + query_index])
                query_term = xlog2(intensity)
                left, right = window(loss_mz, coordinate, tolerance, use_ppm)
                for posting in range(left, right):
                    column = loss_spec[posting]
                    peak_id = loss_pid[posting]
                    if last_query[column] == query_index or peak_id >= last_peak[column]:
                        continue
                    if mode == 2:
                        if peak_epochs[peak_id] == epoch:
                            continue
                        if query_bits[column, query_index // 64] & (np.uint64(1) << np.uint64(query_index % 64)):
                            continue
                    library_intensity = np.float64(loss_int[posting])
                    if library_intensity <= 0.0:
                        continue
                    if not within(coordinate, loss_mz[posting], tolerance, use_ppm):
                        continue
                    score[column] += (
                        xlog2(intensity + library_intensity) - query_term - loss_terms[posting]
                    )
                    last_query[column] = query_index
                    last_peak[column] = peak_id

        for column in range(n_library):
            value = score[column]
            if not identity_allowed(q_pmz[row], lib_pmz[column], identity_tol, identity_ppm):
                value = 0.0
            out[row, column] = value
