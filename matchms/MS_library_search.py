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
#

import numpy as np
import pandas as pd
from scipy import spatial
from gensim import corpora

from .MS_functions import create_MS_documents
from . import MS_similarity_classical as MS_sim_classic

## --------------------------------------------------------------------------------------------------
## --------------------- Functions for MS library search (using Spec2Vec) ---------------------------
## --------------------------------------------------------------------------------------------------


def vectorize_spectra(spectra,
                      MS_library,
                      num_decimals = 2,
                      min_loss = 5.0,
                      max_loss = 500.0,
                      peak_loss_words = ['peak_', 'loss_'],
                      weighting_power = 0.5):
    """ Calculate Spec2Vec vectors for all given spectra (independent of whether
    they also a part of the MS_library).

    Args:
    --------
    spectra: list of spectrum objects
        Spectra (as spectrum objects) for which Spec2Vec vectors should be derived.
    MS_library: SimilarityMeasures() object
        Spectral library in form of SimilarityMeasures() object (see similarity_measure.py).
    num_decimals: int
        Number decimals to take into account for making words from peaks. Default = 2.
    min_loss: float
        Lower limit of losses to take into account (Default = 50.0).
    max_loss: float
        Upper limit of losses to take into account (Default = 500.0).
    peak_loss_words = ['peak_', 'loss_'],
    weighting_power: float
        If weights are present (self.corpus_weights), than those weights will be
        used to the power of 'weighting_power'.
        Set to 0 to ignore.
    """

    # Make document of spectrum
    MS_documents, MS_documents_intensity, _ = create_MS_documents(spectra,
                                                                 num_decimals,
                                                                 peak_loss_words,
                                                                 min_loss, max_loss)

    corpus = [[word.lower() for word in document] for document in MS_documents]

    # Check if all words are included in trained word2vec model
    dictionary = corpora.Dictionary(corpus)

    dictionary_lst = [dictionary[x] for x in dictionary]
    test_vocab = []
    for i, word in enumerate(dictionary_lst):
        if word not in MS_library.model_word2vec.wv.vocab:
            test_vocab.append((i, word))

    if len(test_vocab) > 0:
        print('\n', 20 * '--')
        print("Not all 'words' of the given documents are present in the trained word2vec model!")
        print(len(test_vocab), " out of ", len(dictionary), " 'words' were not found in the word2vec model.")
        print("'Words'missing in the pretrained word2vec model will be ignored.")

        _, missing_vocab = zip(*test_vocab)
        print("Removing missing 'words' from corpus...")
        # Update corpus and BOW-corpus
        corpus = [[word for word in document if word not in missing_vocab] for document in corpus]
        bow_corpus = [dictionary.doc2bow(text) for text in corpus]

    vector_size = MS_library.model_word2vec.wv.vector_size
    vectors_centroid = []

    print("----- Deriving Spec2Vec spectra vectors -----")
    for i in range(len(bow_corpus)):
        if (i+1) % 10 == 0 or i == len(bow_corpus)-1:  # show progress
            print('\r', ' Calculated Spec2Vec spectra vectors for ', i+1, ' of ', len(bow_corpus), ' documents.', end="")

        document = [dictionary[x[0]] for x in bow_corpus[i]]
        if weighting_power != 0 and MS_documents_intensity is not None:
            document_weight = [MS_documents_intensity[i][MS_documents[i].index(dictionary[x[0]])] for x in bow_corpus[i]]
            document_weight = np.array(document_weight)/np.max(document_weight)  # normalize
        else:
            document_weight = np.ones((len(document)))
        if len(document) > 0:
            term1 = MS_library.model_word2vec.wv[document]
            #if tfidf_weighted:
                #term2 = np.array(list(zip(*MS_library.tfidf[self.bow_corpus[i]]))[1])
            #else:
            term2 = np.ones((len(document)))

            term1 = term1 * np.tile(document_weight, (vector_size,1)).T
            weighted_docvector = np.sum((term1.T * term2).T, axis=0)
        else:
            weighted_docvector = np.zeros((MS_library.model_word2vec.vector_size))
        vectors_centroid.append(weighted_docvector)

    return np.array(vectors_centroid)


def library_matching(spectra_query,
                     spectra_library,
                     library_spectra_metadata,
                     MS_library,
                     top_n = 10,
                     mz_ppm = 10,
                     spectra_vectors = None,
                     ignore_non_annotated = True,
                     cosine_tol = 0.005):
    """
    Function to select potential spectra matches.
    Suitable candidates will be selected by 1) top_n Spec2Vec similarity, and 2) same precursor mass
    (within given mz_ppm tolerance(s)).
    For later matching routines, additional scores (cosine, modified cosine) are added as well.

    Args:
    --------
    spectra_query: list of spectrum objects
        List containing all spectrum objects that should be queried against the library.
    spectra_library: list of spectrum objects
        List containing all library spectrum objects.
    library_spectra_metadata: pandas DataFrame
        Metadata of all library spectra in form of a pandas DataFrame, as given by the load_MGF_data() function
        from MS_functions.py.
    MS_library: SimilarityMeasures() object
        Spec2Vec SimilarityMeasures() object built on library spectra, including trained model and spectra vectors.
    top_n: int, optional
        Number of entries witht the top_n highest Spec2Vec scores to keep as found matches. Default = 10.
    mz_ppm: int / list of int, optional
        Single int value or list of int values that determine the tolerance for precursor-mz matching. Masses are
        considered to be a match if they lie within +- 1e-6 * mz_ppm.
        If list of ppm values is given, all matches for the respective ppm values will be added to the final table.
        Default = 10.
    spectra_vectors: numpy array, optional
        Numpy array made from all Spec2Vec spectra vectors of the query spectra. Default = None.
    ignore_non_annotated: bool, optional
        If True, only annotated spectra will be considered for matching. Default = True.
    cosine_tol: float, optional
        Set tolerance for the cosine and modified cosine score. Default = 0.005
    """

    # Check input data
    if len(spectra_library) != library_spectra_metadata.shape[0]:
        print("Warning! Library spectra metadata input does not match given library spectra.")
    if len(spectra_library) != len(MS_library.corpus):
        print("Warning! Library spectra input does not match dimension of given MS_library object.")
    if MS_library.vectors_centroid.shape[0] != library_spectra_metadata.shape[0]:
        print("Warning! Number of found Spec2Vec spectral vectors does not agree with library metadata dimension.")
    if spectra_vectors is None:
        print("No Spec2Vec spectra vectors found for query data. Please do so using the vectorize_spectra() function")


    # Initializations
    found_matches = []
    S2V_matches = []

    if ignore_non_annotated:
        # Get array of all IDs for spectra with smiles
        annotated_spectra_IDs = np.where(library_spectra_metadata['smiles'].isna().values == False)[0]

    # --------------------------------------------------------------------------
    # 1. Search for top-n Spec2Vec matches -------------------------------------
    # --------------------------------------------------------------------------

    #Check if Spec2Vec vectors are present for library
    if len(MS_library.vectors_centroid) == 0:
        print("Apparently Spec2Vec spectra vectors have not yet been derived for library data.")
        print("Spec2Vec spectra vectors will be calculated using default parameters.")
        MS_library.get_vectors_centroid(method = 'ignore',
                                         extra_weights = None,
                                         tfidf_weighted = False,
                                         weight_method = 'sqrt',
                                         tfidf_model = None,
                                         extra_epochs = 1)
    else:
        print("Spec2Vec spectra vectors found for library data.")

    if ignore_non_annotated:
        library_vectors = MS_library.vectors_centroid[annotated_spectra_IDs]
    else:
        library_vectors = MS_library.vectors_centroid
    M_spec2vec_similairies = 1 - spatial.distance.cdist(library_vectors, spectra_vectors, 'cosine')

    # Select top_n similarity values:
    Top_n = np.argpartition(M_spec2vec_similairies, -top_n, axis=0)[-top_n:,:]

    # Sort selected values by order
    for i in range(len(spectra_query)):
        Top_n_sorted = Top_n[:,i][np.argsort(M_spec2vec_similairies[Top_n[:,i],i])][::-1]
        if ignore_non_annotated:
            Top_n_corrected_IDs = annotated_spectra_IDs[Top_n_sorted]
        else:
            Top_n_corrected_IDs = Top_n_sorted

        S2V_match = list(zip(Top_n_corrected_IDs, M_spec2vec_similairies[Top_n_sorted,i]))
        S2V_matches.append(S2V_match)

    # --------------------------------------------------------------------------
    # 2. Search for precursor mz based matches --------------------------------
    # --------------------------------------------------------------------------
    if ignore_non_annotated:
        library_masses = library_spectra_metadata['precursor_mz'].values[annotated_spectra_IDs]
    else:
        library_masses = library_spectra_metadata['precursor_mz'].values

    if not isinstance(mz_ppm, list):
        mz_ppm = [mz_ppm]
    mz_ppm = sorted(mz_ppm)[::-1]

    mass_matches_ppms = []
    for ppm in mz_ppm:
        library_masses_tol = library_masses * ppm/1e6
        mass_matches = []

        for spec in spectra_query:
            mass = spec.precursor_mz
            mass_match = np.where(((library_masses + library_masses_tol) > mass) &
                                  ((library_masses - library_masses_tol) < mass))[0]
            if ignore_non_annotated:
                mass_matches.append(annotated_spectra_IDs[mass_match])
            else:
                mass_matches.append(mass_match)
        mass_matches_ppms.append(mass_matches)

    # --------------------------------------------------------------------------
    # 3. Combine found matches -------------------------------------------------
    # --------------------------------------------------------------------------
    for i in range(len(spectra_query)):
        IDs = list(mass_matches_ppms[0][i])
        mass_match_lst = len(mass_matches_ppms[0][i]) * [1]
        s2v_match_lst = len(mass_matches_ppms[0][i]) * [0]

        # Add Spec2Vec top_n matches to list
        a, _ = list(zip(*S2V_matches[i]))
        for match in a:
            if match in IDs:
                s2v_match_lst[IDs.index(match)] = 1
            else:
                IDs.append(match)
                mass_match_lst.append(0)
                s2v_match_lst.append(1)

        # Add Spec2Vec scores for all entries
        # And calculate if found inchikey is a match
        inchikey_match = []
        inchikey_copies = []
        s2v_score_lst = []
        for m, idx in enumerate(IDs):
            #if s2v_match_lst[m] == 0:
            if ignore_non_annotated:
                idx_uncorrected = int(np.where(annotated_spectra_IDs == idx)[0])
                s2v_score_lst.append(M_spec2vec_similairies[idx_uncorrected,i])
            else:
                s2v_score_lst.append(M_spec2vec_similairies[idx,i])

            inchikey = spectra_library[idx].inchikey[:14]
            inchikey_match.append(1 * (inchikey == spectra_query[i].inchikey[:14]))
            inchikey_copies.append(np.where(library_spectra_metadata["inchikey"].str[:14] == inchikey)[0].shape[0])
        matches_df = pd.DataFrame(list(zip(IDs, inchikey_copies, mass_match_lst, s2v_match_lst, s2v_score_lst, inchikey_match)),
                                       columns = ['spectra_ID', 'inchikey_copies',
                                                  'mass_match_'+ str(int(mz_ppm[0])) + 'ppm',
                                                  'S2V_top_n', 'S2V_similarity', 'inchikey_match'])
        # Add mass matches from other ppms (if given)
        for m, ppm in enumerate(mz_ppm[1:]):
            col_name = 'mass_match_' + str(int(ppm)) + 'ppm'
            matches_df[col_name] = 0
            matches_df.loc[matches_df['spectra_ID'].isin(mass_matches_ppms[m+1][i]), col_name] = 1

        found_matches.append(matches_df)

    # --------------------------------------------------------------------------
    # 4. Add additonal similarity measures (cosine + mod.cosine) ---------------
    # --------------------------------------------------------------------------
    for i in range(len(spectra_query)):
        cosine_scores = []
        for j in found_matches[i]['spectra_ID']:
            cosine_score = cosine_check(spectra_query[i],
                                        spectra_library[j],
                                        tol = cosine_tol,
                                        mod_cosine = False)
            cosine_scores.append(cosine_score)
        a, b = list(zip(*cosine_scores))
        found_matches[i]['cosine_score'] = a
        found_matches[i]['cosine_matches'] = b

    for i in range(len(spectra_query)):
        cosine_scores = []
        for j in found_matches[i]['spectra_ID']:
            cosine_score = cosine_check(spectra_query[i],
                                        spectra_library[j],
                                        tol = cosine_tol,
                                        mod_cosine = True)
            cosine_scores.append(cosine_score)
        a, b = list(zip(*cosine_scores))
        found_matches[i]['modcosine_score'] = a
        found_matches[i]['modcosine_matches'] = b

    return found_matches



def cosine_check(spectra1,
                 spectra2,
                 tol = 0.005,
                 mod_cosine = False):
    """ Small helper function to calculate cosine or modified cosine score for pair of spectra.

    Args:
    --------
    spectra1: spectrum() object
    spectra2: spectrum() object
    tol: float
        Set tolerance for the cosine and modified cosine score. Default = 0.005
    mod_cosine: bool
        Calculates cosine score if set to False, and modified cosine score if True.
    """
    if mod_cosine:
        mass_shift = spectra1.parent_mz - spectra2.parent_mz
    else:
        mass_shift = None
    spec1 = np.array(spectra1.peaks, dtype=float)
    spec2 = np.array(spectra2.peaks, dtype=float)
    cosine_score = MS_sim_classic.cosine_score_greedy(spec1,
                                                    spec2,
                                                    mass_shift = mass_shift,
                                                    tol = tol,
                                                    min_intens = 0,
                                                    use_numba = True)
    return (cosine_score[0], len(cosine_score[1]))
