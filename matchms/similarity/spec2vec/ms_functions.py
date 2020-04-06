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
"""Functions specific to MS data
(e.g. importing and data processing functions)
"""

import os
import operator
import fnmatch
import copy

import numpy as np
from scipy.optimize import curve_fit
import pandas as pd

from pyteomics import mgf
from openbabel import openbabel as ob
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import AllChem

from . import helper_functions as functions


# -----------------------------------------------------------------------------
# ---------------------- Functions to analyse MS data -------------------------
# -----------------------------------------------------------------------------


def create_ms_documents(spectra,
                        num_decimals,
                        peak_loss_words=['peak_', 'loss_'],
                        min_loss=5.0,
                        max_loss=500.0,
                        ignore_losses=False):
    """Create documents from peaks and losses.

    Every peak and every loss will be transformed into a WORD.
    Words then look like this: "peak_100.038" or "loss_59.240".

    Args:
    --------
    spectra: list
        List of all spectrum class elements = all spectra to be in corpus
    num_decimals: int
        Number of decimals to take into account
    min_loss: float
        Lower limit of losses to take into account (Default = 5.0).
    max_loss: float
        Upper limit of losses to take into account (Default = 500.0).
    ignore_losses: bool
        True: Ignore losses, False: make words from losses and peaks.
    """
    ms_documents = []
    ms_documents_intensity = []

    # Collect spectra metadata
    metadata_lst = []

    for spec_id, spectrum in enumerate(spectra):
        doc = []
        doc_intensity = []
        if not ignore_losses:
            losses = np.array(spectrum.losses)
            if len(losses) > 0:
                keep_idx = np.where((losses[:, 0] > min_loss)
                                    & (losses[:, 0] < max_loss))[0]
                losses = losses[keep_idx, :]
            #else:
                #print("No losses detected for: ", spec_id, spectrum.id)

        peaks = np.array(spectrum.peaks)

        # Sort peaks and losses by m/z
        peaks = peaks[np.lexsort((peaks[:, 1], peaks[:, 0])), :]
        if not ignore_losses:
            if len(losses) > 0:
                losses = losses[np.lexsort((losses[:, 1], losses[:, 0])), :]

        if (spec_id+1) % 100 == 0 or spec_id == len(spectra)-1:  # show progress
            print('\r',
                  ' Created documents for {} of {} spectra'.format(spec_id+1, len(spectra)),
                  end="")

        for i in range(len(peaks)):
            doc.append(peak_loss_words[0] +
                       "{:.{}f}".format(peaks[i, 0], num_decimals))
            doc_intensity.append(int(peaks[i, 1]))
        if not ignore_losses:
            for i in range(len(losses)):
                doc.append(peak_loss_words[1] +
                           "{:.{}f}".format(losses[i, 0], num_decimals))
                doc_intensity.append(int(losses[i, 1]))

        ms_documents.append(doc)
        ms_documents_intensity.append(doc_intensity)

        if 'spectrumid' in spectrum.metadata:
            gnps_id = spectrum.metadata['spectrumid']
        else:
            gnps_id = 'n/a'
        if 'name' in spectrum.metadata:
            spec_name = spectrum.metadata['name']
        else:
            spec_name = 'n/a'
        if 'title' in spectrum.metadata:
            spec_title = spectrum.metadata['title']
        else:
            spec_title = 'n/a'
        metadata_lst += [[
            spec_id, gnps_id, spec_name, spec_title, spectrum.precursor_mz,
            len(ms_documents[spec_id]), spectrum.inchi, spectrum.inchikey,
            spectrum.smiles, spectrum.metadata['charge']]]

    # Transfer metadata to pandas dataframe
    spectra_metadata = pd.DataFrame(metadata_lst,
                                    columns=[
                                        'ID', 'gnps_id', 'name', 'title',
                                        'precursor_mz', 'num_peaks_losses',
                                        'inchi', 'inchikey', 'smiles', 'charge'
                                    ])

    return ms_documents, ms_documents_intensity, spectra_metadata



def get_mol_fingerprints(spectra,
                         method="daylight",
                         nbits=1024,
                         print_progress=True):
    """Calculate molecule fingerprints based on given smiles.
    (using RDkit)

    Output: exclude_ids list with spectra that had no smiles or problems when deriving fingerprint

    Args:
    --------
    spectra: list of spectrum objects
        List containing all spectrum objects which also includes peaks, losses, metadata.
    method: str
        Determine method for deriving molecular fingerprints. Supported choices are 'daylight',
        'morgan1', 'morgan2', 'morgan3'.
    nbits: int
        Dimension or number of bits of generated fingerprint. Default is nbits = 1024.
    print_progress: bool, optional
        If True, print phase of the run to indicate progress. Default = True.
    """

    if print_progress:
        print("---- (1) Generating RDkit molecules from inchi or smiles...")
    exclude_ids = []
    molecules = []

    if not isinstance(spectra, list):
        spectra = [spectra]
    if not isinstance(spectra[0], Spectrum):
        print("Wrong input: spectra must be list of Spectrum objects.")

    for i, spec in enumerate(spectra):
        mol = None
        if spec.inchi is not None:
            mol = Chem.MolFromInchi(spec.inchi.replace('"', ''),
                                    sanitize=True,
                                    removeHs=True,
                                    logLevel=None,
                                    treatWarningAsError=False)
        if mol is None or mol.GetNumAtoms() < 3:
            if spec.smiles is not None:  # Smiles but no InChikey OR too small fingerprint
                mol = Chem.MolFromSmiles(spec.smiles)
        if mol is None or mol.GetNumAtoms() < 3:
            print("No proper molecule generated for spectrum", i)
            mol = None
            exclude_ids.append(i)

        molecules.append(mol)

    if print_progress:
        print("---- (2) Generating fingerprints from molecules...")
    fingerprints = []
    for i in range(len(molecules)):
        if molecules[i] is None:
            print("Problem with molecule from spectrum", i)
            fp = np.zeros((nbits)).astype(int)
        else:
            if method == "daylight":
                fp = Chem.RDKFingerprint(molecules[i], fpSize=nbits)
            elif method == "morgan1":
                fp = AllChem.GetMorganFingerprintAsBitVect(molecules[i],
                                                           1,
                                                           nBits=nbits)
            elif method == "morgan2":
                fp = AllChem.GetMorganFingerprintAsBitVect(molecules[i],
                                                           2,
                                                           nBits=nbits)
            elif method == "morgan3":
                fp = AllChem.GetMorganFingerprintAsBitVect(molecules[i],
                                                           3,
                                                           nBits=nbits)
            else:
                print("Unkown fingerprint method given...")
            fp = np.array(fp)
        fingerprints.append(fp)

    return fingerprints, exclude_ids
