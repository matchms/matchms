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

import numpy as np
from scipy.optimize import linear_sum_assignment

from rdkit import DataStructs

# Add multi core parallelization
from concurrent.futures import ThreadPoolExecutor, as_completed

## --------------------------------------------------------------------------------------------------
## ---------------------------- classical spectra similarity measures -------------------------------
## --------------------------------------------------------------------------------------------------


def fast_cosine(spectrum1, 
                spectrum2, 
                tol, 
                min_intens = 0, 
                mass_shifted = False):
    """ Calculate cosine score between spectrum1 and spectrum2. 
    If mass_shifted = True it will shift the spectra with respect to each other 
    by difference in their parentmasses.
    
    Args:
    --------
    spectrum1: Spectrum object    
    spectrum2: Spectrum object
    tol: float
        Tolerance value to define how far two peaks can be apart to still count as match.
    min_intens: float
        Minimum intensity (relative to max.intensity peak in spectrum). Peaks with lower
        intensity will be ignored --> higher min_intens is faster, but less precise.
    """
    if len(spectrum1.peaks) == 0 or len(spectrum2.peaks) == 0:
        return 0.0,[]

    spec1 = np.array(spectrum1.peaks, dtype=float)
    spec2 = np.array(spectrum2.peaks, dtype=float)
    
    # normalize intensities:
    spec1[:,1] = spec1[:,1]/max(spec1[:,1])
    spec2[:,1] = spec2[:,1]/max(spec2[:,1])
    
    # filter, if wanted:
    spec1 = spec1[spec1[:,1] > min_intens,:]
    spec2 = spec2[spec2[:,1] > min_intens,:]
    
    zero_pairs = find_pairs(spec1, spec2, tol, shift=0.0)

    if mass_shifted:
        shift = spectrum1.parent_mz - spectrum2.parent_mz
    else:
        shift = 0
    nonzero_pairs = find_pairs(spec1, spec2, tol, shift = shift)
    matching_pairs = zero_pairs + nonzero_pairs
    matching_pairs = sorted(matching_pairs,key = lambda x: x[2], reverse = True)

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
    score = score/max(np.sum(spec1[:,1]**2), np.sum(spec2[:,1]**2))
    
    return score, used_matches

   
def fast_cosine_shift_hungarian(spectrum1, 
                                spectrum2, 
                                tol, 
                                min_intens=0):
    """ Taking full care of weighted bipartite matching problem:
        Use Hungarian algorithm (slow...)
    
    Args:
    --------
    spectrum1: Spectrum object    
    spectrum2: Spectrum object
    tol: float
        Tolerance value to define how far two peaks can be apart to still count as match.
    min_intens: float
        Minimum intensity (relative to max.intensity peak in spectrum). Peaks with lower
        intensity will be ignored --> higher min_intens is faster, but less precise.
    """
    if len(spectrum1.peaks) == 0 or len(spectrum2.peaks) == 0:
        return 0.0,[]

    spec1 = np.array(spectrum1.peaks, dtype=float)
    spec2 = np.array(spectrum2.peaks, dtype=float)
    
    # Normalize intensities:
    spec1[:,1] = spec1[:,1]/max(spec1[:,1])
    spec2[:,1] = spec2[:,1]/max(spec2[:,1])
    
    # Filter, if wanted:
    spec1 = spec1[spec1[:,1] > min_intens,:]
    spec2 = spec2[spec2[:,1] > min_intens,:]
    zero_pairs = find_pairs(spec1, spec2, tol, shift=0.0)

    shift = spectrum1.parent_mz - spectrum2.parent_mz

    nonzero_pairs = find_pairs(spec1, spec2, tol, shift = shift)

    matching_pairs = zero_pairs + nonzero_pairs

    # Use Hungarian_algorithm:
    set1 = set()
    set2 = set()
    for m in matching_pairs:
        set1.add(m[0])
        set2.add(m[1])
    
    list1 = list(set1)
    list2 = list(set2)
    matrix_size = max(len(set1), len(set2))    
    matrix = np.ones((matrix_size, matrix_size))

    if len(matching_pairs) > 0:
        for m in matching_pairs:
            matrix[list1.index(m[0]),list2.index(m[1])] = 1 - m[2]
    
        row_ind, col_ind = linear_sum_assignment(matrix)
        score = matrix.shape[0] - matrix[row_ind, col_ind].sum()
        
        """# TODO: Add min_match criteria!
        if np.sum(matrix[row_ind, col_ind] != 1) < min_match:
            score = 0.0
        else:      
            # normalize score:
            score = score/max(np.sum(spec1[:,1]**2), np.sum(spec2[:,1]**2))
        """
        score = score/max(np.sum(spec1[:,1]**2), np.sum(spec2[:,1]**2))
    else:
        score = 0.0
    
    return score


def cosine_matrix_fast(spectra,
                       tol,
                       max_mz, 
                       min_mz = 0):
    """
    Be careful! Binning is here done by creating one-hot vectors.
    It is hence really actual "bining" and different from the tolerance-based 
    approach used for the cosine_matrix or molnet_matrix!
    
    Also: tol here is about tol/2 when compared to cosine_matrix or molnet_matrix...
    """
    
    from scipy import spatial
    
    for i, spectrum in enumerate(spectra):
        spec = np.array(spectrum.peaks.copy(), dtype=float)

        # Normalize intensities:
        spec[:,1] = spec[:,1]/np.max(spec[:,1])
        
        if i == 0:
            vector = one_hot_spectrum(spec, tol, max_mz, shift = 0, min_mz = min_mz, method='max')
            spec_vectors = np.zeros((len(spectra), vector.shape[0]))
            spec_vectors[0,:] = vector
        else:
            spec_vectors[i,:] = one_hot_spectrum(spec, tol, max_mz, shift = 0, min_mz = min_mz, method='max')
    
    Cdist = spatial.distance.cdist(spec_vectors, spec_vectors, 'cosine')
    
    return 1 - Cdist


def cosine_matrix(spectra, 
                  tol, 
                  max_mz, 
                  min_mz = 0, 
#                  min_match = 2, 
                  min_intens = 0.01,
                  filename = None,
                  num_workers = 4):
    """ Create Matrix of all cosine similarities.
    
    spectra: list
        List of spectra (of Spectrum class)
    tol: float
        Tolerance to still count peaks a match (mz +- tolerance).
    max_mz: float
        Maxium m-z mass to take into account
    min_mz: float 
        Minimum m-z mass to take into account
#    min_match: int
#        Minimum numbe of peaks that need to be matches. Otherwise score will be set to 0
    min_intens: float
        Sets the minimum relative intensity peaks must have to be looked at for potential matches.
    filename: str/ None
        Filename to look for existing npy-file with molent matrix. Or, if not found, to 
        use to save the newly calculated matrix.
    num_workers: int
        Number of threads to use for calculation.
    """  
    if filename is not None:
        try: 
            cosine_sim = np.load(filename)
            cosine_matches = np.load(filename[:-4]+ "_matches.npy")
            # Check if matrix was calculated to the end:
            diagonal = cosine_sim.diagonal()
            if np.min(diagonal) == 0:
                print("Uncomplete cosine similarity scores found and loaded.")
                missing_scores = np.where(diagonal == 0)[0].astype(int)     
                print("Missing cosine scores will be calculated.")
                counter_total = int((len(spectra)**2)/2)
                counter_init = counter_total - np.sum(len(spectra) - missing_scores)

                print("About ", 100*(counter_init/counter_total),"% of the values already completed.")
                collect_new_data = True
            else:    
                print("Complete cosine similarity scores found and loaded.")
                missing_scores = []
                counter_init = 0
                collect_new_data = False
                
        except FileNotFoundError: 
            print("Could not find file ", filename) 
            print("Cosine scores will be calculated from scratch.")
            collect_new_data = True
            missing_scores = np.arange(0,len(spectra))
            counter_init = 0
    else:
        print("No filename given.")    
        print("Cosine scores will be calculated from scratch.")
        collect_new_data = True
        counter_init = 0
    
    if collect_new_data == True:  
        if counter_init == 0:
            cosine_sim = np.zeros((len(spectra), len(spectra)))
            cosine_matches = np.zeros((len(spectra), len(spectra)))

        counter = counter_init
        print("Calculate pairwise cosine scores by ", num_workers, "number of workers.")
        for i in missing_scores: #range(n_start, len(spectra)):
            parameter_collection = []    
            for j in range(i,len(spectra)):
                parameter_collection.append([spectra[i], spectra[j], i, j, tol, min_intens, counter])
                counter += 1

            # Create a pool of processes. For instance one for each core in your machine.
            cosine_pairs = []
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(cosine_pair, X, len(spectra)) for X in parameter_collection]
                cosine_pairs.append(futures)
             
            for m, future in enumerate(cosine_pairs[0]):
                spec_i, spec_j, ind_i, ind_j, _, _, counting = parameter_collection[m]
                cosine_sim[ind_i,ind_j] = future.result()[0]
                cosine_matches[ind_i,ind_j] = future.result()[1]

        # Symmetric matrix --> fill        
        for i in range(1,len(spectra)):
            for j in range(i):  
                cosine_sim[i,j] = cosine_sim[j,i]      
                cosine_matches[i,j] = cosine_matches[j,i]
    
        if filename is not None:
            np.save(filename, cosine_sim)
            np.save(filename[:-4]+ "_matches.npy", cosine_matches)
            
    return cosine_sim, cosine_matches


def molnet_matrix(spectra, 
                  tol, 
                  max_mz, 
                  min_mz = 0, 
                  min_intens = 0.01,
                  filename = None,
                  method='fast',
                  num_workers = 4,
                  safety_points = 50):
    """ Create Matrix of all mol.networking similarities.
    Takes some time to calculate, so better only do it once and save as npy.
    Now implemented: parallelization of code using concurrent.futures.
    
    spectra: list
        List of spectra (of Spectrum class)
    tol: float
        Tolerance to still count peaks a match (mz +- tolerance).
    max_mz: float
        Maxium m-z mass to take into account
    min_mz: float 
        Minimum m-z mass to take into account
#    min_match: int
#        Minimum numbe of peaks that need to be matches. Otherwise score will be set to 0
    min_intens: float
        Sets the minimum relative intensity peaks must have to be looked at for potential matches.
    filename: str/ None
        Filename to look for existing npy-file with molent matrix. Or, if not found, to 
        use to save the newly calculated matrix.
    method: 'fast' | 'hungarian'
        "Fast" will use Simon's molnet scoring which is much faster, but not 100% accurate
        regarding the weighted bipartite matching problem.
        "hungarian" will use the Hungarian algorithm, which is slower but more accurate.
    num_workers: int
        Number of threads to use for calculation. 
    safety_points: int
        Number of safety points, i.e. number of times the molnet-matrix is saved during process.
    """  
    if filename is not None:
        try: 
            molnet_sim = np.load(filename)
            molnet_matches = np.load(filename[:-4]+ "_matches.npy")
            # Check if matrix was calculated to the end:
            diagonal = molnet_sim.diagonal()
            if np.min(diagonal) == 0:
                print("Uncomplete MolNet similarity scores found and loaded.")
                missing_scores = np.where(diagonal == 0)[0].astype(int)     
                print("Missing MolNet scores will be calculated.")
                counter_total = int((len(spectra)**2)/2)
                counter_init = counter_total - np.sum(len(spectra) - missing_scores)
                print("About ", 100*(counter_init/counter_total),"% of the values already completed.")
                collect_new_data = True
            else:    
                print("Complete MolNet similarity scores found and loaded.")
                missing_scores = []
                counter_init = 0
                collect_new_data = False
                
        except FileNotFoundError: 
            print("Could not find file ", filename) 
            print("MolNet scores will be calculated from scratch.")
            collect_new_data = True
            missing_scores = np.arange(0,len(spectra))
            counter_init = 0
    else:
        collect_new_data = True
        missing_scores = np.arange(0,len(spectra))
        counter_init = 0
    
    if collect_new_data == True:  
        if counter_init == 0:
            molnet_sim = np.zeros((len(spectra), len(spectra)))
            molnet_matches = np.zeros((len(spectra), len(spectra)))

        counter = counter_init
        safety_save = int(((len(spectra)**2)/2)/safety_points)  # Save molnet-matrix along process
        print("Calculate pairwise MolNet scores by ", num_workers, "number of workers.")
        for i in missing_scores: #range(n_start, len(spectra)):
            parameter_collection = []    
            for j in range(i,len(spectra)):
                parameter_collection.append([spectra[i], spectra[j], i, j, tol, min_intens, method, counter])
                counter += 1

            # Create a pool of processes. For instance one for each CPU in your machine.
            molnet_pairs = []
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(molnet_pair, X, len(spectra)) for X in parameter_collection]
                molnet_pairs.append(futures)
             
            for m, future in enumerate(molnet_pairs[0]):
                spec_i, spec_j, ind_i, ind_j, _, _, _, counting = parameter_collection[m]
                molnet_sim[ind_i,ind_j] = future.result()[0]
                molnet_matches[ind_i,ind_j] = future.result()[1]
                if filename is not None:
                    if (counting+1) % safety_save == 0:
                        np.save(filename[:-4]+ str(i), molnet_sim)
                        np.save(filename[:-4]+ "_matches.npy" + str(i), molnet_matches)

        # Symmetric matrix --> fill        
        for i in range(1,len(spectra)):
            for j in range(i):  
                molnet_sim[i,j] = molnet_sim[j,i]    
                molnet_matches[i,j] = molnet_matches[j,i] 
    
        if filename is not None:
            np.save(filename, molnet_sim)
            np.save(filename[:-4]+ "_matches.npy", molnet_matches)
            
    return molnet_sim, molnet_matches


def cosine_pair(X, len_spectra):
    """ Single molnet pair calculation
    """ 
    spectra_i, spectra_j, i, j, tol, min_intens, counter = X
    cosine_pair, used_matches = cosine_score(spectra_i, spectra_j, tol, min_intens = min_intens)


    if (counter+1) % 1000 == 0 or counter == len_spectra-1:  
        print('\r', ' Calculated cosine for pair ', i, '--', j, '. ( ', np.round(200*(counter+1)/len_spectra**2, 2), ' % done).', end="")

    return cosine_pair, len(used_matches)


def molnet_pair(X, len_spectra):
    """ Single molnet pair calculation
    """ 
    spectra_i, spectra_j, i, j, tol, min_intens, method, counter = X
    if method == 'fast':
        molnet_pair, used_matches = fast_cosine(spectra_i, spectra_j, tol, min_intens = min_intens, mass_shifted = True)
    elif method == 'hungarian':
        molnet_pair = fast_cosine_shift_hungarian(spectra_i, spectra_j, tol, 0, min_intens = min_intens)
        used_matches = [] # TODO find way to get match number
    else:
        print("Given method does not exist...")

    if (counter+1) % 1000 == 0 or counter == len_spectra-1:  
        print('\r', ' Calculated MolNet for pair ', i, '--', j, '. ( ', np.round(200*(counter+1)/len_spectra**2, 2), ' % done).', end="")

    return molnet_pair, len(used_matches)


def mol_sim_matrix_symmetric(spectra, 
                  fingerprints,
                  filename = None):
    """ Create Matrix of all molecular similarities (based on annotated SMILES).
    Takes some time to calculate, so better only do it once and save as npy.
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
    
    if collect_new_data == True:      
        
        # Check type of fingerprints given as input:
        try: 
            DataStructs.FingerprintSimilarity(fingerprints[0], fingerprints[0])
            fingerprint_type = "daylight" # at least assumed here
        
        except AttributeError:
            fingerprint_type = "morgan" # at least assumed here
        
        molecular_similarities = np.zeros((len(spectra), len(spectra)))
        for i in range(len(spectra)):
            # Show progress
            if (i+1) % 10 == 0 or i == len(spectra)-1:  
                print('\r', ' Molecular similarity for spectrum ', i+1, ' of ', len(spectra), ' spectra.', end="")
            if fingerprints[i] != 0:
                for j in range(i,len(spectra)):
                    if fingerprints[j] != 0: 
                        if fingerprint_type == "daylight":
                            molecular_similarities[i,j] = DataStructs.FingerprintSimilarity(fingerprints[i], fingerprints[j])
                        elif fingerprint_type == "morgan":
                            molecular_similarities[i,j] = DataStructs.DiceSimilarity(fingerprints[i], fingerprints[j])
        
        # Symmetric matrix --> fill        
        for i in range(1,len(spectra)):
            for j in range(i):  
                molecular_similarities[i,j] = molecular_similarities[j,i]   
    
        if filename is not None:
            np.save(filename, molecular_similarities)

    return molecular_similarities


def mol_sim_matrix(fingerprints1,
                  fingerprints2,
                  filename = None):
    """ Create Matrix of all molecular similarities (based on annotated SMILES or INCHI).
    Takes some time to calculate, so better only do it once and save as npy.
    Here: comparing two different sets of molecular fingerprints!
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
    
    if collect_new_data == True:      
        
        # Check type of fingerprints given as input:
        try: 
            DataStructs.FingerprintSimilarity(fingerprints1[0], fingerprints2[0])
            fingerprint_type = "daylight" # at least assumed here
        
        except AttributeError:
            fingerprint_type = "morgan" # at least assumed here
        
        molecular_similarities = np.zeros((len(fingerprints1), len(fingerprints2)))
        for i in range(len(fingerprints1)):
            # Show progress
            if (i+1) % 10 == 0 or i == len(fingerprints1)-1:  
                print('\r', ' Molecular similarity for spectrum ', i+1, ' of ', len(fingerprints1), ' fingerprints-1.', end="")
            if fingerprints1[i] != 0:
                for j in range(len(fingerprints2)):
                    if fingerprints2[j] != 0: 
                        if fingerprint_type == "daylight":
                            molecular_similarities[i,j] = DataStructs.FingerprintSimilarity(fingerprints1[i], fingerprints2[j])
                        elif fingerprint_type == "morgan":
                            molecular_similarities[i,j] = DataStructs.DiceSimilarity(fingerprints1[i], fingerprints2[j])      
    
        if filename is not None:
            np.save(filename, molecular_similarities)

    return molecular_similarities



def one_hot_spectrum(spec, 
                     tol, 
                     max_mz, 
                     shift = 0, 
                     min_mz = 0,
                     method = 'max'):
    """ Convert spectrum peaks into on-hot-vector
    
    method: str
        'max' take highest intensity peak within every bin. 
        'sum' take sum of all peaks within every bin.
    """
    dim_vector = int((max_mz - min_mz)/tol)
    one_hot_spec = np.zeros((dim_vector))
    idx = ((spec[:,0] + shift)*1/tol).astype(int)
    idx[idx>=dim_vector] = 0
    idx[idx<0] = 0
    if method == 'max':
        for id1 in set(idx):
            one_hot_spec[id1] = np.max(spec[(idx==id1),1])
    elif method == 'sum':
        for id1 in set(idx):
            one_hot_spec[id1] = np.sum(spec[(idx==id1),1])
    else:
        print("Method not known...")
    return one_hot_spec
    

def find_pairs(spec1, spec2, tol, shift=0):
    matching_pairs = []
    spec2lowpos = 0
    spec2length = len(spec2)
    
    for idx in range(len(spec1)):
        mz = spec1[idx,0]
        intensity = spec1[idx,1]
        # Do we need to increase the lower idx?
        while spec2lowpos < spec2length and spec2[spec2lowpos][0] + shift < mz - tol:
            spec2lowpos += 1
        if spec2lowpos == spec2length:
            break
        spec2pos = spec2lowpos
        while(spec2pos < spec2length and spec2[spec2pos][0] + shift < mz + tol):
            matching_pairs.append((idx, spec2pos, intensity*spec2[spec2pos][1]))
            spec2pos += 1
        
    return matching_pairs    