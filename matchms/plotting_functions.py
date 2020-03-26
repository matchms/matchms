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
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter
from rdkit import Chem
from rdkit.Chem import Draw
from IPython.display import SVG, display
from scour import scour

from .MS_functions import create_ms_documents
from . import MS_similarity_classical as MS_sim_classic

# ----------------------------------------------------------------------------------------
# ---------------------------- Plotting functions ----------------------------------------
# ----------------------------------------------------------------------------------------


def plot_precentile(arr_sim, arr_ref, num_bins=1000, show_top_percentile=1.0):
    """ Plot top percentile (as specified by show_top_percentile) of best restults
    in arr_sim and compare against reference values in arr_ref.

    Args:
    -------
    arr_sim: numpy array
        Array of similarity values to evaluate.
    arr_ref: numpy array
        Array of reference values to evaluate the quality of arr_sim.
    num_bins: int
        Number of bins to divide data (default = 1000)
    show_top_percentile
        Choose which part to plot. Will plot the top 'show_top_percentile' part of
        all similarity values given in arr_sim. Default = 1.0
    """
    start = int(arr_sim.shape[0] * show_top_percentile / 100)
    idx = np.argpartition(arr_sim, -start)
    starting_point = arr_sim[idx[-start]]
    if starting_point == 0:
        print("not enough datapoints != 0 above given top-precentile")

    # Remove all data below show_top_percentile
    low_as = np.where(arr_sim < starting_point)[0]

    length_selected = arr_sim.shape[0] - low_as.shape[0]  # start+1

    data = np.zeros((2, length_selected))
    data[0, :] = np.delete(arr_sim, low_as)
    data[1, :] = np.delete(arr_ref, low_as)
    data = data[:, np.lexsort((data[1, :], data[0, :]))]

    ref_score_cum = []

    for i in range(num_bins):
        low = int(i * length_selected / num_bins)
        # high = int((i+1) * length_selected/num_bins)
        ref_score_cum.append(np.mean(data[1, low:]))
    ref_score_cum = np.array(ref_score_cum)

    fig, ax = plt.subplots(figsize=(6, 6))
    plt.plot(
        (show_top_percentile / num_bins * (1 + np.arange(num_bins)))[::-1],
        ref_score_cum,
        color='black')
    plt.xlabel("Top percentile of spectral similarity score g(s,s')")
    plt.ylabel("Mean molecular similarity (f(t,t') within that percentile)")

    return ref_score_cum


def get_spaced_colors_hex(n):
    """ Create set of 'n' well-distinguishable colors
    """
    spaced_colors = [
        "FF0000", "00FF00", "0000FF", "FFFF00", "FF00FF", "00FFFF", "000000",
        "800000", "008000", "000080", "808000", "800080", "008080", "808080",
        "C00000", "00C000", "0000C0", "C0C000", "C000C0", "00C0C0", "C0C0C0",
        "400000", "004000", "000040", "404000", "400040", "004040", "404040",
        "200000", "002000", "000020", "202000", "200020", "002020", "202020",
        "600000", "006000", "000060", "606000", "600060", "006060", "606060",
        "A00000", "00A000", "0000A0", "A0A000", "A000A0", "00A0A0", "A0A0A0",
        "E00000", "00E000", "0000E0", "E0E000", "E000E0", "00E0E0", "E0E0E0"
    ]

    return ["#" + x for x in spaced_colors[:n]]


def plot_spectra(spectra, compare_ids, min_mz=50, max_mz=500):
    """ Plot different spectra together to compare.
    """
    plt.figure(figsize=(10, 10))

    peak_number = []
    rgb_colors = get_spaced_colors_hex(len(compare_ids))
    for i, id in enumerate(compare_ids):
        peaks = np.array(spectra[id].peaks.copy())
        peak_number.append(len(peaks))
        peaks[:, 1] = peaks[:, 1] / np.max(peaks[:, 1])

        markerline, stemlines, baseline = plt.stem(peaks[:, 0],
                                                   peaks[:, 1],
                                                   linefmt='-',
                                                   markerfmt='.',
                                                   basefmt='r-')
        plt.setp(stemlines, 'color', rgb_colors[i])

    plt.xlim((min_mz, max_mz))
    plt.grid(True)
    plt.title('Spectrum')
    plt.xlabel('m/z')
    plt.ylabel('peak intensity')

    plt.show()

    print("Number of peaks: ", peak_number)


def plot_losses(spectra, compare_ids, min_loss=0, max_loss=500):
    """ Plot different spectra together to compare.
    """
    plt.figure(figsize=(10, 10))

    losses_number = []
    rgb_colors = get_spaced_colors_hex(len(compare_ids) + 5)
    for i, id in enumerate(compare_ids):
        losses = np.array(spectra[id].losses.copy())
        losses_number.append(len(losses))
        losses[:, 1] = losses[:, 1] / np.max(losses[:, 1])

        markerline, stemlines, baseline = plt.stem(losses[:, 0],
                                                   losses[:, 1],
                                                   linefmt='-',
                                                   markerfmt='.',
                                                   basefmt='r-')
        plt.setp(stemlines, 'color', rgb_colors[i])

    plt.xlim((min_loss, max_loss))
    plt.grid(True)
    plt.title('Spectrum')
    plt.xlabel('m/z')
    plt.ylabel('peak intensity')

    plt.show()

    print("Number of peaks: ", losses_number)


def plot_spectra_comparison(ms_measure,
                            spectra,
                            num_decimals,
                            id1,
                            id2,
                            min_mz=5,
                            max_mz=500,
                            threshold=0.01,
                            method='cosine',
                            wordsim_cutoff=0.5,
                            circle_size=5,
                            circle_scaling='wordsim',
                            padding=10,
                            display_molecules=False,
                            figsize=(12, 12),
                            filename=None):
    """ In-depth visual comparison of spectral similarity scores,
    calculated based on cosine/mod.cosine and Spev2Vec.

    Args:
    --------
    method: str
        'cosine' or 'modcos' (modified cosine score)
    circle_scaling: str
        Scale circles based on 'wordsim' or 'peak_product'
    """
    plt.style.use('ggplot')

    from scipy import spatial
    plot_colors = ['darkcyan', 'purple']

    # Definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005

    rect_wordsim = [left, bottom, width, height]
    rect_specx = [left, bottom + height + spacing, width, 0.2]
    rect_specy = [left + width, bottom, 0.2, height]

    peaks1 = np.array(spectra[id1].peaks.copy())
    peaks2 = np.array(spectra[id2].peaks.copy())
    peaks1[:, 1] = peaks1[:, 1] / np.max(peaks1[:, 1])
    peaks2[:, 1] = peaks2[:, 1] / np.max(peaks2[:, 1])

    # Sort by m/z to be in line with word order in MS_documents
    peaks1 = peaks1[np.lexsort((peaks1[:, 1], peaks1[:, 0])), :]
    peaks2 = peaks2[np.lexsort((peaks2[:, 1], peaks2[:, 0])), :]

    # Remove peaks lower than threshold
    select1 = np.where((peaks1[:, 1] > threshold) & (peaks1[:, 0] <= max_mz)
                       & (peaks1[:, 0] >= min_mz))[0]
    select2 = np.where((peaks2[:, 1] > threshold) & (peaks2[:, 0] <= max_mz)
                       & (peaks2[:, 0] >= min_mz))[0]

    # TODO: only include sub-function to create documents...
    dictionary = [ms_measure.dictionary[x] for x in ms_measure.dictionary]
    ms_documents, ms_documents_intensity, _ = create_ms_documents(
        [spectra[x] for x in [id1, id2]],
        num_decimals=num_decimals,
        peak_loss_words=['peak_', 'loss_'],
        min_loss=0,
        max_loss=max_mz,
        ignore_losses=True)

    # Remove words/peaks that are not in dictionary
    select1 = np.array(
        [x for x in select1 if ms_documents[0][x] in dictionary])
    select2 = np.array(
        [x for x in select2 if ms_documents[1][x] in dictionary])

    peaks1 = peaks1[select1, :]
    peaks2 = peaks2[select2, :]
    min_peaks1 = np.min(peaks1[:, 0])
    min_peaks2 = np.min(peaks2[:, 0])
    max_peaks1 = np.max(peaks1[:, 0])
    max_peaks2 = np.max(peaks2[:, 0])

    word_vectors1 = ms_measure.model_word2vec.wv[[
        ms_documents[0][x] for x in select1
    ]]
    word_vectors2 = ms_measure.model_word2vec.wv[[
        ms_documents[1][x] for x in select2
    ]]

    csim_words = 1 - spatial.distance.cdist(word_vectors1, word_vectors2,
                                            'cosine')
    csim_words[csim_words < wordsim_cutoff] = 0  # Remove values below cutoff
    print(np.min(csim_words))

    # Plot spectra
    # -------------------------------------------------------------------------
    fig = plt.figure(figsize=figsize)
    # Word similariy plot (central)
    ax_wordsim = plt.axes(rect_wordsim)
    ax_wordsim.tick_params(direction='in', top=True, right=True)
    # Spectra plot (top)
    ax_specx = plt.axes(rect_specx)
    ax_specx.tick_params(direction='in', labelbottom=False)
    # Spectra plo 2 (right)
    ax_specy = plt.axes(rect_specy)
    ax_specy.tick_params(direction='in', labelleft=False)

    # Spec2Vec similarity plot:
    # -------------------------------------------------------------------------
    data_x = []
    data_y = []
    data_z = []
    data_peak_product = []
    for i in range(len(select1)):
        for j in range(len(select2)):
            data_x.append(peaks1[i, 0])
            data_y.append(peaks2[j, 0])
            data_z.append(csim_words[i, j])
            data_peak_product.append(peaks1[i, 1] * peaks2[j, 1])

    # Sort by word similarity
    data_x = np.array(data_x)
    data_y = np.array(data_y)
    data_z = np.array(data_z)
    data_peak_product = np.array(data_peak_product)
    idx = np.lexsort((data_x, data_y, data_z))

    cm = plt.cm.get_cmap('RdYlBu_r')  # 'YlOrRd') #'RdBu_r')

    # Plot word similarities
    if circle_scaling == 'peak_product':
        wordsimplot = ax_wordsim.scatter(data_x[idx],
                                         data_y[idx],
                                         s=100 * circle_size *
                                         (0.01 + data_peak_product[idx]**2),
                                         c=data_z[idx],
                                         cmap=cm,
                                         alpha=0.6)
    elif circle_scaling == 'wordsim':
        wordsimplot = ax_wordsim.scatter(data_x[idx],
                                         data_y[idx],
                                         s=100 * circle_size *
                                         (0.01 + data_z[idx]**2),
                                         c=data_z[idx],
                                         cmap=cm,
                                         alpha=0.6)

    # (modified) Cosine similarity plot:
    # -------------------------------------------------------------------------
    if method == 'cosine':
        shift = 0
    elif method == 'modcos':
        shift = spectra[id1].parent_mz - spectra[id2].parent_mz
    else:
        print("Given method unkown.")

    _, used_matches = MS_sim_classic.cosine_score_greedy(peaks1,
                                                         peaks2,
                                                         mass_shift=shift,
                                                         tol=0.005,
                                                         min_intens=0,
                                                         use_numba=True)

    idx1, idx2, _ = zip(*used_matches)
    cosine_x = []
    cosine_y = []
    for i in range(len(idx1)):
        cosine_x.append(peaks1[idx1[i], 0])
        cosine_y.append(peaks2[idx2[i], 0])

    # Plot (mod.) cosine similarities
    ax_wordsim.scatter(cosine_x, cosine_y, s=100, c='black', marker=(5, 2))
    ax_wordsim.set_xlim(min_peaks1 - padding, max_peaks1 + padding)
    ax_wordsim.set_ylim(min_peaks2 - padding, max_peaks2 + padding)
    ax_wordsim.set_xlabel('spectra 1 - fragment mz', fontsize=16)
    ax_wordsim.set_ylabel('spectra 2 - fragment mz', fontsize=16)
    ax_wordsim.tick_params(labelsize=13)

    # Plot spectra 1
    ax_specx.vlines(peaks1[:, 0], [0], peaks1[:, 1], color=plot_colors[0])
    ax_specx.plot(peaks1[:, 0], peaks1[:, 1], '.')  # Stem ends
    ax_specx.plot([peaks1[:, 0].max(), peaks1[:, 0].min()], [0, 0],
                  '--')  # Middle bar
    ax_specx.set_xlim(min_peaks1 - padding, max_peaks1 + padding)
    ax_specx.set_ylabel('peak intensity (relative)', fontsize=16)
    ax_specx.tick_params(labelsize=13)

    # Plot spectra 2
    ax_specy.hlines(peaks2[:, 0], [0], peaks2[:, 1], color=plot_colors[1])
    ax_specy.plot(peaks2[:, 1], peaks2[:, 0], '.')  # Stem ends
    ax_specy.plot([0, 0], [peaks2[:, 0].min(), peaks2[:, 0].max()],
                  '--')  # Middle bar
    ax_specy.set_ylim(min_peaks2 - padding, max_peaks2 + padding)
    ax_specy.set_xlabel('peak intensity (relative)', fontsize=16)
    ax_specy.tick_params(labelsize=13)

    fig.colorbar(wordsimplot, ax=ax_specy)
    if filename is not None:
        plt.savefig(filename)
    plt.show()

    # Plot molecules
    # -------------------------------------------------------------------------
    if display_molecules:
        smiles = []
        for i, candidate_id in enumerate([id1, id2]):
            smiles.append(spectra[candidate_id].metadata["smiles"])

        if filename is not None:
            plot_molecules(smiles, filename=filename[:-4] + '_molecules')
        else:
            plot_molecules(smiles)

    return csim_words


def scour(target, source, env=[]):
    """ Use scour to clean an svg file.

    """
    options = scour.generateDefaultOptions()

    # override defaults for max cleansing
    options.enable_viewboxing = True
    options.strip_comments = True
    options.strip_ids = True
    options.remove_metadata = True
    options.indent_type = None
    options.shorten_ids = True

    if 'SCOUR_OPTIONS' in env:
        options.__dict__.update(env['SCOUR_OPTIONS'])

    if False:
        from pprint import pprint
        print("\nUsing Scour options:\n")
        pprint(options.__dict__)

    instream = open(source, 'rb')
    outstream = open(target, 'wb')

    scour.start(options, instream, outstream)


def plot_molecules(smiles_lst, filename=None):
    """ Plot molecule from smile(s).
    Uses Scour to clean rdkit svg.

    filename: str
        If filename is given, molecules will be saved to filename.
    """
    if not isinstance(smiles_lst, list):
        smiles_lst = [smiles_lst]
    for i, smiles in enumerate(smiles_lst):
        temp_file = "draw_mols_temp.svg"
        mol = Chem.MolFromSmiles(smiles)
        Draw.MolToFile(mol, temp_file)

        # Clean svg using scour
        if filename is not None:
            file = filename.split('.svg')[0] + str(i) + '.svg'
        else:
            file = "draw_mols_temp_corr.svg"
        scour(file, temp_file, [])

        # Display cleaned svg
        display(SVG(filename=temp_file))


def plot_smiles(query_id,
                spectra,
                ms_measure,
                num_candidates=10,
                similarity_method="centroid",
                plot_type="single",
                molnet_sim=None):
    """ Plot molecules for closest candidates

    """

    # Select chosen similarity methods
    if similarity_method == "centroid":
        candidates_idx = ms_measure.list_similars_ctr_idx[
            query_id, :num_candidates]
        candidates_sim = ms_measure.list_similars_ctr[
            query_id, :num_candidates]
    elif similarity_method == "pca":
        candidates_idx = ms_measure.list_similars_pca_idx[
            query_id, :num_candidates]
        candidates_sim = ms_measure.list_similars_pca[
            query_id, :num_candidates]
    elif similarity_method == "autoencoder":
        candidates_idx = ms_measure.list_similars_ae_idx[
            query_id, :num_candidates]
        candidates_sim = ms_measure.list_similars_ae[query_id, :num_candidates]
    elif similarity_method == "lda":
        candidates_idx = ms_measure.list_similars_lda_idx[
            query_id, :num_candidates]
        candidates_sim = ms_measure.list_similars_lda[
            query_id, :num_candidates]
    elif similarity_method == "lsi":
        candidates_idx = ms_measure.list_similars_lsi_idx[
            query_id, :num_candidates]
        candidates_sim = ms_measure.list_similars_lsi[
            query_id, :num_candidates]
    elif similarity_method == "doc2vec":
        candidates_idx = ms_measure.list_similars_d2v_idx[
            query_id, :num_candidates]
        candidates_sim = ms_measure.list_similars_d2v[
            query_id, :num_candidates]
    elif similarity_method == "molnet":
        if molnet_sim is None:
            print(
                "If 'molnet' is chosen as similarity measure, molnet-matrix needs to be provided."
            )
            print("Use molnet_matrix function.")
        else:
            candidates_idx = molnet_sim[
                query_id, :].argsort()[-num_candidates:][::-1]
            candidates_sim = molnet_sim[query_id, candidates_idx]
    else:
        print("Chosen similarity measuring method not found.")

    size = (200, 200)  # Smaller figures than the default

    if isinstance(spectra, dict):
        # If spectra is given as a dictionary
        keys = []
        for key, value in spectra.items():
            keys.append(key)

        smiles = []
        molecules = []

        for i, candidate_id in enumerate(candidates_idx):
            key = keys[candidate_id]
            smiles.append(spectra[key]["smiles"])
            mol = Chem.MolFromSmiles(smiles[i])
            if mol != None:
                mol.SetProp('_Name', smiles[i])
                if plot_type == 'single':
                    Draw.MolToMPL(mol, size=size)

        if plot_type != "single":  # this will only work if there's no conflict with rdkit and pillow...
            Draw.MolsToGridImage(
                molecules, legends=[mol.GetProp('_Name') for mol in molecules])

    elif isinstance(spectra, list):
        # Assume that it is then a list of Spectrum objects

        smiles = []
        for i, candidate_id in enumerate(candidates_idx):
            smiles.append(spectra[candidate_id].metadata["smiles"])
            mol = Chem.MolFromSmiles(smiles[i])
            if plot_type == 'single':
                Draw.MolToMPL(mol, size=size)

        if plot_type != "single":  # this will only work if there's no conflict with rdkit and pillow...
            Draw.MolsToGridImage(
                molecules, legends=[mol.GetProp('_Name') for mol in molecules])


def top_score_histogram(spec_sim,
                        mol_sim,
                        score_threshold,
                        num_candidates,
                        num_bins=50,
                        filename=None):
    """ Plot histogram of Tanimoto scores (mol_sim) of top selected candidates based on
    spectrum similarity scores (spec_sim).

    spec_sim, mol_sim : to be calculated with evaluate_measure function.

    filename: str
        If not none: save figure to file with given name.
    """

    fig, ax = plt.subplots(figsize=(10, 10))

    selection = np.where(spec_sim[:, 1:] > score_threshold)
    x = mol_sim[selection[0], selection[1] + 1].reshape(len(selection[0]))
    n, bins, patches = plt.hist(x,
                                num_bins,
                                weights=np.ones(len(x)) / len(x),
                                facecolor='blue',
                                edgecolor='white',
                                alpha=0.9)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.title("Tanimoto scores of TOP " + str(num_candidates - 1) +
              " candidates with score > " + str(score_threshold))
    plt.xlabel("Tanimoto score (based on spectra annotated SMILES)")
    plt.ylabel("Percentage")

    test = spec_sim[:, 1:].reshape(spec_sim.shape[0] * (spec_sim.shape[1] - 1))
    test.sort()
    text1 = "Mean Tanimoto similarity is " + str(
        np.round(np.mean(mol_sim[selection[0], selection[1] + 1]), 4))
    text2 = "Spectrum similarity score for TOP " + str(
        num_candidates - 1) + ", top 20% is " + str(
            np.round(test[int(len(test) * 0.8)], 4))
    text3 = ""
    plt.text(0, 0.96 * np.max(n), text1, fontsize=12, backgroundcolor="white")
    plt.text(0, 0.91 * np.max(n), text2, fontsize=12, backgroundcolor="white")
    plt.text(0, 0.86 * np.max(n), text3, fontsize=12, backgroundcolor="white")

    if filename is not None:
        plt.savefig(filename, dpi=600)

    plt.show()


def similarity_histogram(m_sim,
                         m_sim_ref,
                         score_threshold,
                         num_bins=50,
                         exclude_ids=None,
                         filename=None,
                         exclude_diagonal=True):
    """ Plot histogram of Reference scores (from matrix m_sim_ref) for all pairs
    with similarity score >= score_threshold.

    m_sim: numpy array
        Matrix with similarities between pairs.
    m_sim_ref: numpy array
        Matrix with reference scores/similarity values between pairs.

    filename: str
        If not none: save figure to file with given name.
    """
    fig, ax = plt.subplots(figsize=(10, 10))

    if exclude_ids is not None:
        # Remove elements in exclude_ids array
        ids = np.arange(0, m_sim.shape[0])
        m_sim = np.delete(m_sim, ids[exclude_ids], axis=0)
        m_sim = np.delete(m_sim, ids[exclude_ids], axis=1)
        m_sim_ref = np.delete(m_sim_ref, ids[exclude_ids], axis=0)
        m_sim_ref = np.delete(m_sim_ref, ids[exclude_ids], axis=1)

        ids = np.delete(ids, ids[exclude_ids])

    if exclude_diagonal:
        # Exclude diagonal
        m_sim[np.arange(0, m_sim.shape[0]),
              np.arange(0, m_sim.shape[0])] = score_threshold - 1

    selection = np.where(m_sim[:, :] >= score_threshold)
    x = m_sim_ref[selection].reshape(len(selection[0]))
    n, bins, patches = plt.hist(x,
                                num_bins,
                                weights=np.ones(len(x)) / len(x),
                                facecolor='blue',
                                edgecolor='white',
                                alpha=0.9)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.title(
        "Total reference scores for all candidates with similarity score > " +
        str(score_threshold),
        fontsize=16)
    #    plt.set_xticklabels(rotation=0, fontsize=12)
    ax.tick_params(labelsize=12)
    plt.xlabel("Reference score.", fontsize=14)
    plt.ylabel("Percentage", fontsize=14)

    if filename is not None:
        plt.savefig(filename, dpi=600)

    plt.show()

    return n, bins


def compare_best_results(spectra_dict,
                         spectra,
                         ms_measure,
                         tanimoto_sim,
                         molnet_sim,
                         num_candidates=25,
                         similarity_method=["centroid"]):
    """ Compare spectra-based similarity with smile-based similarity and mol.networking.

    Args:
    -------
    spectra_dict: dict
        Dictionary containing all spectra peaks, losses, metadata.
    ms_measure: object
        Similariy object containing the model and distance matrices.
    tanimoto_sim: numpy array
        Matrix of Tanimoto similarities between SMILES of spectra.
    molnet_sim: numpy array
        Matrix of mol. networking similarities of spectra.
    num_candidates: int
        Number of candidates to list (default = 25) .
    similarity_method: str
        Define method to use (default = "centroid").
    """
    num_spectra = len(spectra)

    spec_best = np.zeros((num_spectra, num_candidates, len(similarity_method)))
    #    spec_best_idx = np.zeros((num_spectra, num_candidates))
    mol_best = np.zeros((num_spectra, num_candidates))
    tanimoto_best = np.zeros((num_spectra, num_candidates))

    candidates_idx = np.zeros(num_candidates, dtype=int)
    candidates_sim = np.zeros(num_candidates)
    for k, method in enumerate(similarity_method):
        for i in range(num_spectra):
            # Select chosen similarity methods
            if method == "centroid":
                candidates_idx = ms_measure.list_similars_ctr_idx[
                    i, :num_candidates]
            elif method == "pca":
                candidates_idx = ms_measure.list_similars_pca_idx[
                    i, :num_candidates]
            elif method == "autoencoder":
                candidates_idx = ms_measure.list_similars_ae_idx[
                    i, :num_candidates]
            elif method == "lda":
                candidates_idx = ms_measure.list_similars_lda_idx[
                    i, :num_candidates]
            elif method == "lsi":
                candidates_idx = ms_measure.list_similars_lsi_idx[
                    i, :num_candidates]
            elif method == "doc2vec":
                candidates_idx = ms_measure.list_similars_d2v_idx[
                    i, :num_candidates]
            else:
                print("Chosen similarity measuring method not found.")

            candidates_sim = tanimoto_sim[i, candidates_idx]
            spec_best[i, :, k] = candidates_sim

    for i in range(num_spectra):
        # Compare to molecular networking score
        molnet_candidates_idx = molnet_sim[
            i, :].argsort()[-num_candidates:][::-1]
        molnet_candidates_sim = tanimoto_sim[i, molnet_candidates_idx]

        # Compare to maximum possible Tanimoto score
        tanimoto_candidates_idx = tanimoto_sim[
            i, :].argsort()[-num_candidates:][::-1]
        tanimoto_candidates_sim = tanimoto_sim[i, tanimoto_candidates_idx]

        mol_best[i, :] = molnet_candidates_sim
        tanimoto_best[i, :] = tanimoto_candidates_sim

    labels = []
    avg_best_scores = []
    labels.append('Tanimoto (best)')
    avg_best_scores.append(np.mean(tanimoto_best, axis=0))
    labels.append('Mol.networking score')
    avg_best_scores.append(np.mean(mol_best, axis=0))

    for k, method in enumerate(similarity_method):
        labels.append('Spectrum similarity (' + method + ')')
        avg_best_scores.append(np.mean(spec_best[:, :, k], axis=0))

    return avg_best_scores, labels


def plot_best_results(avg_best_scores, labels, tanimoto_sim, filename=None):
    """ Plot best candidate average results.
    """

    num_candidates = len(avg_best_scores[0])

    # These are the colors that will be used in the plot
    color_sequence = [
        '#003f5c', '#882556', '#D65113', '#ffa600', '#58508d', '#bc5090',
        '#2651d1', '#2f4b7c', '#ff6361', '#a05195', '#d45087'
    ]
    markers = ['^', 'v', 'o']  # , 'v']

    fig, ax = plt.subplots(figsize=(10, 16))
    plt.subplot(211)
    for i, label in enumerate(labels):
        plt.plot(np.arange(0, num_candidates),
                 avg_best_scores[i],
                 label=label,
                 linewidth=1,
                 markersize=12,
                 marker=markers[min(i,
                                    len(markers) - 1)],
                 linestyle=':',
                 color=color_sequence[i])

    # Add mean Tanimoto baseline
    plt.plot(np.arange(0, num_candidates),
             np.mean(tanimoto_sim) * np.ones(num_candidates),
             label='Average Tanimoto similarity',
             linewidth=2,
             color='black')

    plt.legend(fontsize=12)
    plt.xticks(range(0, num_candidates), fontsize=12)
    plt.xlabel("Top 'x' candidates")
    plt.ylabel("Average Tanimoto score.")

    #    fig, ax = plt.subplots(figsize=(10,8))
    plt.subplot(212)
    for i, label in enumerate(labels[1:], start=1):
        plt.plot(np.arange(1, num_candidates),
                 avg_best_scores[i][1:] / avg_best_scores[0][1:],
                 label=label + '/Tanimoto max',
                 linewidth=1,
                 markersize=12,
                 marker=markers[min(i,
                                    len(markers) - 1)],
                 linestyle=':',
                 color=color_sequence[i])

    # Add mean Tanimoto baseline
    plt.plot(np.arange(1, num_candidates),
             np.mean(tanimoto_sim) * np.ones(
                 (num_candidates - 1)) / avg_best_scores[0][1:],
             label='Baseline: random candidate selection',
             linewidth=2,
             color='black')

    plt.legend(fontsize=12)
    plt.xticks(range(1, num_candidates), fontsize=12)
    plt.xlabel("Top 'x' candidates")
    plt.ylabel("Fraction of max. possible average Tanimoto score")

    if filename is not None:
        plt.savefig(filename, dpi=600)


def ms_similarity_network(ms_measure,
                          similarity_method="centroid",
                          link_method="single",
                          filename="MS_Spec2Vec_graph.graphml",
                          cutoff=0.7,
                          max_links=10,
                          extern_matrix=None):
    """ Built network from closest connections found.
        Using networkx

    Args:
    -------
    ms_measure: SimilarityMeasures object
    method: str
        Determine similarity method (default = "centroid").
    filename: str
        Filename to save network to (as graphml file).
    cutoff: float
        Define cutoff. Only consider edges for similarities > cutoff. Default = 0.7.
    max_links: int
        Maximum number of similar candidates to add to edges. Default = 10.
    """

    if similarity_method == "centroid":
        list_similars_idx = ms_measure.list_similars_ctr_idx
        list_similars = ms_measure.list_similars_ctr
    elif similarity_method == "lda":
        list_similars_idx = ms_measure.list_similars_lda_idx
        list_similars = ms_measure.list_similars_lda
    elif similarity_method == "lsi":
        list_similars_idx = ms_measure.list_similars_lsi_idx
        list_similars = ms_measure.list_similars_lsi
    elif similarity_method == "extern":
        num_candidates = ms_measure.list_similars_ctr_idx.shape[1]
        list_similars = np.zeros(ms_measure.list_similars_ctr_idx.shape)
        list_similars_idx = np.zeros(
            ms_measure.list_similars_ctr_idx.shape).astype(int)

        if extern_matrix is None:
            print("Need externally derived similarity matrix to proceed.")
        else:
            if extern_matrix.shape[0] == extern_matrix.shape[
                    1] == list_similars.shape[0]:
                for i in range(0, list_similars.shape[0]):
                    list_similars_idx[i, :] = (-extern_matrix[i]).argsort(
                    )[:num_candidates].astype(int)
                    list_similars[i, :] = extern_matrix[i, list_similars_idx[
                        i, :]]
            else:
                print(
                    "External matrix with similarity scores does not have the right dimensions."
                )
    else:
        print("Wrong method given. Or method not yet implemented in function.")

    if max_links > (list_similars_idx.shape[1] - 1):
        print(
            "Maximum number of candidate links exceeds dimension of 'list_similars'-array."
        )

    dimension = list_similars_idx.shape[0]

    # Initialize network graph
    import networkx as nx
    MSnet = nx.Graph()
    MSnet.add_nodes_from(np.arange(0, dimension))

    for i in range(0, dimension):
        idx = np.where(list_similars[i, :] > cutoff)[0][:max_links]
        if link_method == "single":
            new_edges = [(i, int(list_similars_idx[i, x]),
                          float(list_similars[i, x])) for x in idx
                         if list_similars_idx[i, x] != i]
        elif link_method == "mutual":
            new_edges = [(i, int(list_similars_idx[i, x]),
                          float(list_similars[i, x])) for x in idx
                         if list_similars_idx[i, x] != i
                         if i in list_similars_idx[x, :]]
        else:
            print("Link method not kown")
        MSnet.add_weighted_edges_from(new_edges)

    # Export graph for drawing (e.g. using Cytoscape)
    nx.write_graphml(MSnet, filename)
    print("Network stored as graphml file under: ", filename)
