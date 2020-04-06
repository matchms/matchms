#
# Spec2Vec
#
# Copyright 2019 Netherlands eScience Center
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numba
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy import spatial

# Add multi core parallelization
from concurrent.futures import ThreadPoolExecutor #, as_completed
# TODO better use joblib ? or dask?


def mol_sim_matrix(fingerprints1,
                   fingerprints2,
                   method='cosine',
                   filename=None,
                   max_size=1000,
                   print_progress=True):
    """Create Matrix of all molecular similarities (based on molecular fingerprints).

    If filename is not None, the result will be saved as npy.
    To create molecular fingerprints see mol_fingerprints() function from MS_functions.

    Args:
    ----
    fingerprints1: list
        List of molecular fingerprints (numpy arrays).
    fingerprints2: list
        List of molecular fingerprints (numpy arrays).
    method: str
        Method to compare molecular fingerprints. Can be 'cosine', 'dice' etc.
        (see scipy.spatial.distance.cdist).
    filename: str
        Filename to save results to. OR: If file already exists it will be
        loaded instead.
    max_size: int
        Maximum size of (sub) all-vs-all matrix to handle in one go. Will split
        up larger matrices into
        max_size x max_size matrices.
    print_progress: bool, optional
        If True, print phase of the run to indicate progress. Default = True.
    """

    if filename is not None:
        try:
            molecular_similarities = np.load(filename)
            print("Molecular similarity scores found and loaded.")
            collect_new_data = False

        except FileNotFoundError:
            print("Could not find file ", filename)
            print("Molecular scores will be calculated from scratch.")
            collect_new_data = True
    else:
        collect_new_data = True

    if collect_new_data:
        # Create array of all finterprints
        fingerprints_arr1 = np.array(fingerprints1)
        fingerprints_arr2 = np.array(fingerprints2)

        # Calculate all-vs-all similarity matrix (similarity here= 1-distance )
        matrix_size = (fingerprints_arr1.shape[0], fingerprints_arr2.shape[0])

        molecular_similarities = np.zeros(matrix_size)

        # Split large matrices up into smaller ones to track progress
        splits = int(np.ceil(matrix_size[0]/max_size) * np.ceil(matrix_size[1]/max_size))
        count_splits = 0

        for i in range(int(np.ceil(matrix_size[0]/max_size))):
            low1 = i * max_size
            high1 = min((i + 1) * max_size, matrix_size[0])
            for j in range(int(np.ceil(matrix_size[1]/max_size))):
                low2 = j * max_size
                high2 = min((j + 1) * max_size, matrix_size[1])

                molecular_similarities[low1:high1, low2:high2] = 1 - spatial.distance.cdist(
                    fingerprints_arr1[low1:high1],
                    fingerprints_arr2[low2:high2],
                    method
                    )
                # Track progress:
                count_splits += 1
                if print_progress:
                    print('\r',
                          "Calculated submatrix {} out of {}".format(count_splits, splits),
                          end="")

        if print_progress:
            print(20 * '--')
            print("Succesfully calculated matrix with all-vs-all molecular similarity values.")
        if filename is not None:
            np.save(filename, molecular_similarities)
            print("Matrix was saved under:", filename)

    return molecular_similarities


# --------------------------------------------------------------------------------------------------
# ---------------------------- classical spectra similarity measures -------------------------------
# --------------------------------------------------------------------------------------------------


def cosine_score_greedy(spec1,
                        spec2,
                        mass_shift,
                        tol,
                        min_intens=0,
                        use_numba=True):
    """Calculate cosine score between spectrum1 and spectrum2.

    If mass_shifted = True it will shift the spectra with respect to each other
    by difference in their parentmasses.

    Args:
    ----
    spec1: Spectrum peaks and intensities as numpy array.
    spec2: Spectrum peaks and intensities as numpy array.
    tol: float
        Tolerance value to define how far two peaks can be apart to still count as match.
    min_intens: float
        Minimum intensity (relative to max.intensity peak in spectrum). Peaks with lower
        intensity will be ignored --> higher min_intens is faster, but less precise.
    """

    if spec1.shape[0] == 0 or spec2.shape[0] == 0:
        return 0.0, []

    # normalize intensities:
    spec1[:, 1] = spec1[:, 1]/max(spec1[:, 1])
    spec2[:, 1] = spec2[:, 1]/max(spec2[:, 1])

    # filter, if wanted:
    spec1 = spec1[spec1[:, 1] > min_intens, :]
    spec2 = spec2[spec2[:, 1] > min_intens, :]

    if use_numba:
        zero_pairs = find_pairs_numba(spec1, spec2, tol, shift=0.0)
    else:
        zero_pairs = find_pairs(spec1, spec2, tol, shift=0.0)
    if mass_shift is not None \
    and mass_shift != 0.0:
        if use_numba:
            nonzero_pairs = find_pairs_numba(spec1, spec2, tol, shift=mass_shift)
        else:
            nonzero_pairs = find_pairs(spec1, spec2, tol, shift=mass_shift)
        matching_pairs = zero_pairs + nonzero_pairs
    else:
        matching_pairs = zero_pairs
    matching_pairs = sorted(matching_pairs, key=lambda x: x[2], reverse=True)

    used1 = set()
    used2 = set()
    score = 0.0
    used_matches = []
    for m in matching_pairs:
        if not m[0] in used1 and not m[1] in used2:
            score += m[2]
            used1.add(m[0])
            used2.add(m[1])
            used_matches.append(m)

    # Normalize score:
    score = score/max(np.sum(spec1[:, 1]**2), np.sum(spec2[:, 1]**2))

    return score, used_matches


def one_hot_spectrum(spec,
                     tol,
                     max_mz,
                     shift=0,
                     min_mz=0,
                     method='max'):
    """Convert spectrum peaks into on-hot-vector

    method: str
        'max' take highest intensity peak within every bin.
        'sum' take sum of all peaks within every bin.
    """
    dim_vector = int((max_mz - min_mz)/tol)
    one_hot_spec = np.zeros((dim_vector))
    idx = ((spec[:, 0] + shift)*1/tol).astype(int)
    idx[idx >= dim_vector] = 0
    idx[idx < 0] = 0
    if method == 'max':
        for id1 in set(idx):
            one_hot_spec[id1] = np.max(spec[(idx == id1), 1])
    elif method == 'sum':
        for id1 in set(idx):
            one_hot_spec[id1] = np.sum(spec[(idx == id1), 1])
    else:
        print("Method not known...")
    return one_hot_spec


@numba.njit
def find_pairs_numba(spec1, spec2, tol, shift=0):
    """Find matching pairs between two spectra.

    Args
    ----
    spec1 : list of tuples
        List of (mz, intensity) tuples.
    spec2 : list of tuples
        List of (mz, intensity) tuples.
    tol : float
        Tolerance. Peaks will be considered a match when < tol appart.
    shift : float, optional
        Shift spectra peaks by shift. The default is 0.

    Returns
    -------
    matching_pairs : list
        List of found matching peaks.

    """
    matching_pairs = []

    for idx in range(len(spec1)):
        intensity = spec1[idx, 1]
        matches = np.where((np.abs(spec2[:, 0] - spec1[idx, 0] + shift) <= tol))[0]
        for match in matches:
            matching_pairs.append((idx, match, intensity*spec2[match][1]))

    return matching_pairs


def find_pairs(spec1, spec2, tol, shift=0):
    """Find matching pairs between two spectra.

    Args
    ----
    spec1 : list of tuples
        List of (mz, intensity) tuples.
    spec2 : list of tuples
        List of (mz, intensity) tuples.
    tol : float
        Tolerance. Peaks will be considered a match when < tol appart.
    shift : float, optional
        Shift spectra peaks by shift. The default is 0.

    Returns
    -------
    matching_pairs : list
        List of found matching peaks.

    """
    # Sort peaks and losses by m/z
    spec1 = spec1[np.lexsort((spec1[:, 1], spec1[:, 0])), :]
    spec2 = spec2[np.lexsort((spec2[:, 1], spec2[:, 0])), :]

    matching_pairs = []
    spec2lowpos = 0
    spec2length = len(spec2)

    for idx in range(len(spec1)):
        mz = spec1[idx, 0]
        intensity = spec1[idx, 1]
        # Do we need to increase the lower idx?
        while spec2lowpos < spec2length and spec2[spec2lowpos][0] + shift < mz - tol:
            spec2lowpos += 1
        if spec2lowpos == spec2length:
            break
        spec2pos = spec2lowpos
        while(spec2pos < spec2length and spec2[spec2pos][0] + shift < mz + tol):
            matching_pairs.append((idx, spec2pos, intensity * spec2[spec2pos][1]))
            spec2pos += 1

    return matching_pairs
