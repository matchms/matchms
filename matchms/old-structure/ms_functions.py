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
""" Functions specific to MS data
(e.g. importing and data processing functions)
"""

import os
import operator

from . import helper_functions as functions

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



# --------------------------------------------------------------------------------------------------
# ---------------------------- Spectrum class ------------------------------------------------------
# --------------------------------------------------------------------------------------------------

class Spectrum(object):
    """ Spectrum class to store key information

    Functions include:
        - Import data from mass spec files (protoype so far, works with only few formats)
        - Calculate losses from peaks.
        - Process / filter peaks

    Args:
    -------
    min_frag: float
        Lower limit of m/z to take into account (Default = 0.0).
    max_frag: float
        Upper limit of m/z to take into account (Default = 1000.0).
    min_loss: float
        Lower limit of losses to take into account (Default = 10.0).
    max_loss: float
        Upper limit of losses to take into account (Default = 200.0).
    min_intensity_perc: float
        Filter out peaks with intensities lower than the min_intensity_perc percentage
        of the highest peak intensity (Default = 0.0, essentially meaning: OFF).
    exp_intensity_filter: float
        Filter out peaks by applying an exponential fit to the intensity histogram.
        Intensity threshold will be set at where the exponential function will have dropped
        to exp_intensity_filter (Default = 0.01).
    min_peaks: int
        Minimum number of peaks to keep, unless less are present from the start (Default = 10).
    merge_energies: bool
        Merge close peaks or not (False | True, Default is True).
    merge_ppm: int
        Merge peaks if their m/z is <= 1e6*merge_ppm (Default = 10).
    replace: 'max' or None
        If peaks are merged, either the heighest intensity of both is taken ('max'),
        or their intensitites are added (None).
    """
    def __init__(self,
                 min_frag = 0.0,
                 max_frag = 1000.0,
                 min_loss = 10.0,
                 max_loss = 200.0,
                 min_intensity_perc = 0.0,
                 exp_intensity_filter = 0.01,
                 min_peaks = 10,
                 max_peaks = None,
                 aim_min_peak = None,
                 merge_energies = True,
                 merge_ppm = 1,
                 replace = 'max'):

        self.id = []
        self.filename = []
        self.peaks = []
        self.precursor_mz = None
        self.parent_mz = None
        # Annotations of spectrum:
        self.metadata = {}
        self.family = None
        self.annotations = []
        self.smiles = None
        self.inchi = None
        self.inchikey = None
        self.losses = None
        self.n_peaks = None
        self.intensity = None

        self.PROTON_MASS = 1.00727645199076

        self.min_frag = min_frag
        self.max_frag = max_frag
        self.min_loss = min_loss
        self.max_loss = max_loss
        self.min_intensity_perc = min_intensity_perc
        if exp_intensity_filter == 0:
            self.exp_intensity_filter = None
        else:
            self.exp_intensity_filter = exp_intensity_filter
        self.aim_min_peak = aim_min_peak
        self.min_peaks = min_peaks
        self.max_peaks = max_peaks
        self.merge_energies = merge_energies
        self.merge_ppm = merge_ppm
        self.replace = replace

    def ion_masses(self, precursormass, int_charge):
        """
        Compute the parent masses. Single charge version is used for
        loss computation.
        """
        mul = abs(int_charge)
        parent_mass = precursormass * mul
        parent_mass -= int_charge * self.PROTON_MASS
        single_charge_precursor_mass = precursormass*mul
        if int_charge > 0:
            single_charge_precursor_mass -= (int_charge - 1) * self.PROTON_MASS
        elif int_charge < 0:
            single_charge_precursor_mass += (mul - 1) * self.PROTON_MASS
        else:
            # charge = zero - leave them all the same
            parent_mass = precursormass
            single_charge_precursor_mass = precursormass
        return parent_mass, single_charge_precursor_mass

    def interpret_charge(self, charge):
        """
        Method to interpret the ever variable charge field in the different
        formats. Should never fail now.
        """
        if not charge:
            return 1
        try:
            if not isinstance(charge, str):
                charge = str(charge)

            # Try removing any + signs
            charge = charge.replace("+", "")

            # Remove trailing minus signs
            if charge.endswith('-'):
                charge = charge[:-1]
                if not charge.startswith('-'):
                    charge = '-' + charge
            # Turn into int
            int_charge = int(charge)
            return int_charge
        except:
            int_charge = 1
        return int_charge

    def read_spectrum(self, path, file, id):
        """ Read .ms file and extract most relevant information
        """

        with open(os.path.join(path, file), 'r') as f:
            temp_mass = []
            temp_intensity = []
            doc_name = file.split('/')[-1]
            self.filename = doc_name
            self.id = id
            for line in f:
                rline = line.rstrip()
                if len(rline) > 0:
                    if rline.startswith('>') or rline.startswith('#'):
                        keyval = rline[1:].split(' ')[0]
                        valval = rline[len(keyval) + 2:]
                        if not keyval == 'ms2peaks':
                            self.metadata[keyval] = valval
                        if keyval == 'compound':
                            self.annotation = valval
                        if keyval == 'precursormass':
                            self.precursor_mz = float(valval)
                        if keyval == 'Precursor_MZ':
                            self.precursor_mz = float(valval)
                        if keyval == 'parentmass':
                            self.parent_mz = float(valval)
                        if keyval == 'intensity':
                            self.intensity = float(valval)
                        if keyval == 'inchi':
                            self.inchi = valval
                        if keyval == 'inchikey':
                            self.inchikey = valval
                        if keyval == 'smiles':
                            self.smiles = valval
                    else:
                        # If it gets here, its a fragment peak (MS2 level peak)
                        sr = rline.split(' ')
                        mass = float(sr[0])
                        intensity = float(sr[1])
                        if self.merge_energies and len(temp_mass)>0:
                            # Compare to other peaks
                            errs = 1e6*np.abs(mass-np.array(temp_mass)) / mass
                            if errs.min() < self.merge_ppm:
                                # Don't add, but merge the intensity
                                min_pos = errs.argmin()
                                if self.replace == 'max':
                                    temp_intensity[min_pos] = max(intensity,temp_intensity[min_pos])
                                else:
                                    temp_intensity[min_pos] += intensity
                            else:
                                temp_mass.append(mass)
                                temp_intensity.append(intensity)
                        else:
                            temp_mass.append(mass)
                            temp_intensity.append(intensity)

        peaks = list(zip(temp_mass, temp_intensity))
        peaks = process_peaks(peaks,
                              self.min_frag,
                              self.max_frag,
                              self.min_intensity_perc,
                              self.exp_intensity_filter,
                              self.min_peaks,
                              self.max_peaks,
                              self.aim_min_peak)
        self.peaks = peaks
        self.n_peaks = len(peaks)

    def read_spectrum_mgf(self, spectrum_mgf, id):
        """ Translate spectrum dictionary as created by pyteomics package
        into metabolomics.py spectrum object.
        """
        self.id = id
        self.metadata = spectrum_mgf['params']
        if 'charge' in spectrum_mgf['params']:
            self.metadata['charge'] = spectrum_mgf['params']['charge'][0]
        else:
            self.metadata['charge'] = 1
        self.metadata['precursormass'] = spectrum_mgf['params']['pepmass'][0]
        self.metadata['parentintensity'] = spectrum_mgf['params']['pepmass'][1]

        # Following corrects parentmass according to charge if charge is known.
        # This should lead to better computation of neutral losses
        single_charge_precursor_mass = self.metadata['precursormass']
        precursor_mass = self.metadata['precursormass']
        parent_mass = self.metadata['precursormass']

        str_charge = self.metadata['charge']
        int_charge = self.interpret_charge(str_charge)

        parent_mass, single_charge_precursor_mass = self.ion_masses(
            precursor_mass, int_charge)

        self.metadata['parentmass'] = parent_mass
        self.metadata[
            'singlechargeprecursormass'] = single_charge_precursor_mass
        self.metadata['charge'] = int_charge

        # Get precursor mass (later used to calculate losses!)
        self.precursor_mz = float(self.metadata['precursormass'])
        self.parent_mz = float(self.metadata['parentmass'])

        if 'smiles' in self.metadata:
            self.smiles = self.metadata['smiles']
        if 'inchi' in self.metadata:
            self.inchi = self.metadata['inchi']
        if 'inchikey' in self.metadata:
            self.inchikey = self.metadata['inchikey']

        peaks = list(
            zip(spectrum_mgf['m/z array'], spectrum_mgf['intensity array']))
        if len(peaks) >= self.min_peaks:
            peaks = process_peaks(peaks,
                                  self.min_frag,
                                  self.max_frag,
                                  self.min_intensity_perc,
                                  self.exp_intensity_filter,
                                  self.min_peaks,
                                  self.max_peaks,
                                  self.aim_min_peak)

        self.peaks = peaks
        self.n_peaks = len(peaks)

    def get_losses(self):
        """ Use spectrum class and extract peaks and losses
        Losses are here the differences between the spectrum precursor mz and the MS2 level peaks.

        Remove losses outside window min_loss <-> max_loss.
        """
        ms1_peak = self.precursor_mz
        losses = np.array(self.peaks.copy())
        losses[:,0] = ms1_peak - losses[:,0]
        keep_idx = np.where((losses[:,0] > self.min_loss)
                            & (losses[:,0] < self.max_loss))[0]

        # TODO: now array is tranfered back to list (to be able to store as json later). Seems weird.
        losses_list = [(x[0], x[1]) for x in losses[keep_idx, :]]
        self.losses = losses_list


# --------------------------------------------------------------------------------------------------
# ---------------------------- Spectrum processing functions ---------------------------------------
# --------------------------------------------------------------------------------------------------


def dict_to_spectrum(spectra_dict):
    """ Create spectrum object from spectra_dict.
    Spectra_dict is a python dictionary that stores all information for a set of spectra
    that is needed to re-create spectrum objects for all spectra.
    This includes peaks and intensities as well as annotations and method metadata.

    Outputs a list of spectrum objects.
    """
    spectra = []
    keys = []
    for key, value in spectra_dict.items():
        keys.append(key)

        if "max_peaks" not in value:
            value["max_peaks"] = None

        spectrum = Spectrum(min_frag = value["min_frag"],
                            max_frag = value["max_frag"],
                            min_loss = value["min_loss"],
                            max_loss = value["max_loss"],
                            min_intensity_perc = 0,
                            exp_intensity_filter = value["exp_intensity_filter"],
                            min_peaks = value["min_peaks"],
                            max_peaks = value["max_peaks"])

        for key2, value2 in value.items():
            setattr(spectrum, key2, value2)

        spectrum.peaks = [(x[0],x[1]) for x in spectrum.peaks]  # convert to tuples

        # Collect in form of list of spectrum objects
        spectra.append(spectrum)

    return spectra


def process_peaks(peaks,
                  min_frag = 0.0,
                  max_frag = 1000.0,
                  min_intensity_perc = 0,
                  exp_intensity_filter = 0.01,
                  min_peaks = 10,
                  max_peaks = None,
                  aim_min_peaks = None):
    """ Processes peaks.
    Remove peaks outside window min_frag <-> max_frag.
    Remove peaks with intensities < min_intensity_perc/100*max(intensities)

    Uses exponential fit to intensity histogram. Threshold for maximum allowed peak
    intensity will be set where the exponential fit reaches exp_intensity_filter.

    Args:
    -------
    min_frag: float
        Lower limit of m/z to take into account (Default = 0.0).
    max_frag: float
        Upper limit of m/z to take into account (Default = 1000.0).
    min_intensity_perc: float
        Filter out peaks with intensities lower than the min_intensity_perc percentage
        of the highest peak intensity (Default = 0.0, essentially meaning: OFF).
    exp_intensity_filter: float
        Filter out peaks by applying an exponential fit to the intensity histogram.
        Intensity threshold will be set at where the exponential function will have dropped
        to exp_intensity_filter (Default = 0.01).
    min_peaks: int
        Minimum number of peaks to keep, unless less are present from the start (Default = 10).
    max_peaks: int
        Maximum number of peaks to keep. Set to 'None' to ignore  (Default = 'None').
    aim_min_peaks: int
        Minium number of peaks to keep (if present) during exponential filtering.
    """
    # Fixed parameters:
    num_bins = 100  # number of bins for histogram
    min_peaks_for_exp_fit = 25  # With less peaks exponential fit doesn't make any sense.

    if aim_min_peaks is None:  # aim_min_peaks is not given
        aim_min_peaks = min_peaks

    if isinstance(peaks, list):
        peaks = np.array(peaks)
        if peaks.shape[1] != 2:
            print("Peaks were given in unexpected format...")

    # Remove peaks outside min_frag <-> max_frag window:
    keep_idx = np.where((peaks[:,0] > min_frag) & (peaks[:,0] < max_frag))[0]
    peaks = peaks[keep_idx,:]

    # Remove peaks based on relative intensity below min_intensity_perc/100 * max_intensity
    if min_intensity_perc > 0:
        intensity_thres = np.max(peaks[:, 1]) * min_intensity_perc/100
        keep_idx = np.where((peaks[:, 0] > min_frag) & (peaks[:, 0] < max_frag)
                            & (peaks[:, 1] > intensity_thres))[0]
        if len(keep_idx) > min_peaks:
            peaks = peaks[keep_idx,:]

    # Fit exponential to peak intensity distribution
    if (exp_intensity_filter is not None) and len(peaks) >= min_peaks_for_exp_fit:

        peaks = exponential_peak_filter(peaks,
                                        exp_intensity_filter,
                                        aim_min_peaks,
                                        num_bins)

        # Sort by peak intensity
        peaks = peaks[np.lexsort((peaks[:, 0], peaks[:, 1])),:]
        if max_peaks is not None:
            return [(x[0], x[1]) for x in peaks[-max_peaks:,:]]  # TODO: now array is transfered back to list (to be able to store as json later). Seems weird.

        else:
            return [(x[0], x[1]) for x in peaks]
    else:
        # Sort by peak intensity
        peaks = peaks[np.lexsort((peaks[:,0], peaks[:,1])),:]
        if max_peaks is not None:
            return [(x[0], x[1]) for x in peaks[-max_peaks:,:]]
        else:
            return [(x[0], x[1]) for x in peaks]


def exponential_peak_filter(peaks,
                            exp_intensity_filter,
                            aim_min_peaks,
                            num_bins):
    """ Fit exponential to peak intensity distribution and

    Args:
    -------
    peaks: list of tuples
        List of tuples containing (m/z, intensity) pairs.
    exp_intensity_filter: float
        Intensity threshold will be set where exponential fit to intensity
        histogram drops below 1 - exp_intensity_filter.
    aim_min_peaks: int
        Desired minimum number of peaks. Filtering step will stop removing peaks
        when it reaches aim_min_peaks.
    num_bins: int
        Number of bins for histogram (to fit exponential to).

    Returns
    -------
    Filtered List of tuples containing (m/z, intensity) pairs.
    """
    def exponential_func(x, a, b):
        return a*np.exp(-b*x)

    # Ignore highest peak for further analysis
    peaks2 = peaks.copy()
    peaks2[np.where(peaks2[:, 1] == np.max(peaks2[:, 1])),:] = 0

    # Create histogram
    hist, bins = np.histogram(peaks2[:, 1], bins=num_bins)
    offset = np.where(hist == np.max(hist))[0][0]  # Take maximum intensity bin as starting point
    last = int(num_bins/2)
    x = bins[offset:last]
    y = hist[offset:last]
    # Try exponential fit:
    try:
        popt, _ = curve_fit(exponential_func, x , y, p0=(peaks.shape[0], 1e-4))
        lower_guess_offset = bins[max(0,offset-1)]
        threshold = lower_guess_offset -np.log(1 - exp_intensity_filter)/popt[1]
    except RuntimeError:
        print("RuntimeError for ", len(peaks), " peaks. Use 1/2 mean intensity as threshold.")
        threshold = np.mean(peaks2[:,1])/2
    except TypeError:
        print("Unclear TypeError for ", len(peaks), " peaks. Use 1/2 mean intensity as threshold.")
        print(x, "and y: ", y)
        threshold = np.mean(peaks2[:,1])/2

    keep_idx = np.where(peaks[:, 1] > threshold)[0]
    if len(keep_idx) < aim_min_peaks:
        peaks = peaks[np.lexsort((peaks[:, 0], peaks[:, 1])),:][-aim_min_peaks:]
    else:
        peaks = peaks[keep_idx, :]

    return peaks


# ----------------------------------------------------------------------------
# -------------------------- Functions to load MS data------------------------
# ----------------------------------------------------------------------------

def load_MS_data(path_data,
                 path_json,
                 filefilter="*.*",
                 results_file = None,
                 num_decimals = 3,
                 min_frag = 0.0, max_frag = 1000.0,
                 min_loss = 5.0, max_loss = 500.0,
                 min_intensity_perc = 0.0,
                 exp_intensity_filter = 0.01,
                 min_keep_peaks_0 = 10,
                 min_keep_peaks_per_mz = 20/200,
                 min_peaks = 10,
                 max_peaks = None,
                 aim_min_peak = None,
                 #merge_energies = False,
                 #merge_ppm = 10,
                 #replace = 'max',
                 peak_loss_words = ['peak_', 'loss_']):
    """ Collect spectra from set of files
    Partly taken from ms2ldaviz.
    Prototype. Needs to be replaces by more versatile parser, accepting more MS data formats.

    # TODO: add documentation.
    # TODO: consider removing this function alltogether and only allow for MGF input.
    """

    spectra = []
    spectra_dict = {}
    ms_documents = []
    ms_documents_intensity = []

    dirs = os.listdir(path_data)
    spectra_files = fnmatch.filter(dirs, filefilter)

    if results_file is not None:
        try:
            spectra_dict = functions.json_to_dict(path_json + results_file)
            spectra_metadata = pd.read_csv(path_json + results_file[:-5] + "_metadata.csv")
            print("Spectra json file found and loaded.")
            spectra = dict_to_spectrum(spectra_dict)
            collect_new_data = False

            with open(path_json + results_file[:-4] + "txt", "r") as f:
                for line in f:
                    line = line.replace('"', '').replace("'", "").replace("[", "").replace("]", "").replace("\n", "")
                    ms_documents.append(line.split(", "))

            with open(path_json + results_file[:-5] + "_intensity.txt", "r") as f:
                for line in f:
                    line = line.replace("[", "").replace("]", "")
                    ms_documents_intensity.append([int(x) for x in line.split(", ")])

        except FileNotFoundError:
            print("Could not find file ", path_json, results_file)
            print("New data from ", path_data, " will be imported.")
            collect_new_data = True

    # Read data from files if no pre-stored data is found:
    if spectra_dict == {} or results_file is None:

        # Run over all spectrum files:
        for i, filename in enumerate(spectra_files):

            # Show progress
            if (i+1) % 10 == 0 or i == len(spectra_files)-1:
                print('\r', ' Load spectrum ', i+1, ' of ', len(spectra_files), ' spectra.', end="")

            if min_keep_peaks_per_mz != 0\
            and min_keep_peaks_0 > min_peaks:
                # TODO: remove following BAD BAD hack:
                # Import first (acutally only needed is PRECURSOR MASS)
                spec = Spectrum(min_frag = min_frag,
                                max_frag = max_frag,
                                min_loss = min_loss,
                                max_loss = max_loss,
                                min_intensity_perc = min_intensity_perc,
                                exp_intensity_filter = exp_intensity_filter,
                                min_peaks = min_peaks,
                                max_peaks = max_peaks,
                                aim_min_peak = aim_min_peak)

                # Load spectrum data from file:
                spec.read_spectrum(path_data, filename, i)

                # Scale the min_peak filter
                def min_peak_scaling(x, A, B):
                    return int(A + B * x)

                min_peaks_scaled = min_peak_scaling(spec.precursor_mz, min_keep_peaks_0, min_keep_peaks_per_mz)
            else:
                min_peaks_scaled = min_peaks

            spectrum = Spectrum(min_frag = min_frag,
                                max_frag = max_frag,
                                min_loss = min_loss,
                                max_loss = max_loss,
                                min_intensity_perc = min_intensity_perc,
                                exp_intensity_filter = exp_intensity_filter,
                                min_peaks = min_peaks,
                                max_peaks = max_peaks,
                                aim_min_peak = min_peaks_scaled)

            # Load spectrum data from file:
            spectrum.read_spectrum(path_data, filename, i)

            # Get precursor mass (later used to calculate losses!)
            if spec.precursor_mz is not None:
                if 'Precursor_MZ' in spec.metadata:
                    spec.precursor_mz = float(spec.metadata['Precursor_MZ'])
                else:
                    spec.precursor_mz = spec.parent_mz

            # Calculate losses:
            spectrum.get_losses()

            # Collect in form of list of spectrum objects, and as dictionary
            spectra.append(spectrum)
            spectra_dict[filename] = spectrum.__dict__

        ms_documents, ms_documents_intensity, spectra_metadata = create_ms_documents(spectra,
                                                                                     num_decimals,
                                                                                     peak_loss_words,
                                                                                     min_loss, max_loss)
        # Add filenames to metadata
        filenames = []
        for spectrum in spectra:
            filenames.append(spectrum.filename)
        spectra_metadata["filename"] = filenames

        # Save collected data
        if collect_new_data == True:
            spectra_metadata.to_csv(path_json + results_file[:-5] + "_metadata.csv", index=False)

            functions.dict_to_json(spectra_dict, path_json + results_file)
            # Store documents
            with open(path_json + results_file[:-4] + "txt", "w") as f:
                for s in ms_documents:
                    f.write(str(s) +"\n")

            with open(path_json + results_file[:-5] + "_intensity.txt", "w") as f:
                for s in ms_documents_intensity:
                    f.write(str(s) +"\n")

    return spectra, spectra_dict, ms_documents, ms_documents_intensity, spectra_metadata


def load_MGF_data(file_mgf,
                 file_json = None,
                 num_decimals = 2,
                 min_frag = 0.0, max_frag = 1000.0,
                 min_loss = 10.0, max_loss = 200.0,
                 min_intensity_perc = 0.0,
                 exp_intensity_filter = 0.01,
                 min_keep_peaks_0 = 10,
                 min_keep_peaks_per_mz = 20/200,
                 min_peaks = 10,
                 max_peaks = None,
                 peak_loss_words = ['peak_', 'loss_'],
                 ignore_losses = False,
                 create_docs = True):
    """ Collect spectra from MGF file
    1) Importing MGF file - based on pyteomics parser.
    2) Filter spectra: can be based on mininum relative intensity or  based on
    and exponential intenstiy distribution.
    3) Create documents with peaks (and losses) as words. Words are constructed
    from peak mz values and restricted to 'num_decimals' decimals.

    Args:
    -------
    file_mgf: str
        MGF file that should be imported.
    file_json: str
        File under which already processed data is stored. If not None and if it
        exists, data will simply be imported from that file.
        Otherwise data will be imported from file_mgf and final results are stored
        under file_json.(default= None).
    num_decimals: int
        Number of decimals to keep from each peak-position for creating words.
    min_frag: float
        Lower limit of m/z to take into account (Default = 0.0).
    max_frag: float
        Upper limit of m/z to take into account (Default = 1000.0).
    min_loss: float
        Lower limit of losses to take into account (Default = 10.0).
    max_loss: float
        Upper limit of losses to take into account (Default = 200.0).
    min_intensity_perc: float
        Filter out peaks with intensities lower than the min_intensity_perc percentage
        of the highest peak intensity. min_intensity_perc = 1.0 will lead to removal of
        all peaks with intensities below 1% of the maximum intensity.
        (Default = 0.0, essentially meaning: OFF).
    exp_intensity_filter: float
        Filter out peaks by applying an exponential fit to the intensity histogram.
        Intensity threshold will be set at where the exponential function will have dropped
        to exp_intensity_filter (Default = 0.01).
    min_keep_peaks_0: float
        Factor to describe constant of mininum peaks per spectrum with increasing
        parentmass. Formula is: int(min_keep_peaks_0 + min_keep_peaks_per_mz * parentmass).
    min_keep_peaks_per_mz: float
        Factor to describe linear increase of mininum peaks per spectrum with increasing
        parentmass. Formula is: int(min_keep_peaks_0 + min_keep_peaks_per_mz * parentmass).
    min_peaks: int
        Minimum number of peaks to keep, unless less are present from the start (Default = 10).
    max_peaks: int
        Maximum number of peaks to keep (Default = None).
    ignore_losses: bool
        If False: Calculate losses and create documents from both peaks and losses.
    create_docs: bool.
        If True create documents for all spectra (or load if present). Otherwise skip
        this step.
    """

    spectra = []
    spectra_dict = {}
    ms_documents = []
    ms_documents_intensity = []
    spectra_metadata = []
    collect_new_data = True

    if file_json is not None:
        try:
            spectra_dict = functions.json_to_dict(file_json)
            spectra_metadata = pd.read_csv(file_json[:-5] + "_metadata.csv")
            spectra = dict_to_spectrum(spectra_dict)
            if len(spectra) > 0:
                print("Spectra json file found and loaded.")
                collect_new_data = False
            else:
                print("Found json file empty or not expected format.")

            if create_docs:
                with open(file_json[:-4] + "txt", "r") as f:
                    for line in f:
                        line = line.replace('"', '').replace("'", "").replace(
                            "[", "").replace("]", "").replace("\n", "")
                        ms_documents.append(line.split(", "))

                with open(file_json[:-5] + "_intensity.txt", "r") as f:
                    for line in f:
                        line = line.replace("[", "").replace("]", "")
                        ms_documents_intensity.append(
                            [int(x) for x in line.split(", ")])

        except FileNotFoundError:
            print(20 * '--')
            print("Could not find file ", file_json)
            print(20 * '--')

    if len(spectra) == 0: # No data was loaded.
        if os.path.isfile(file_mgf):
            print("Data will be imported from ", file_mgf)
        else:
            print("No data was imported. Could not find MGF file", file_mgf)

    # Read data from files if no pre-stored data is found:
    if len(spectra) == 0:

        # Scale the min_peak filter
        def min_peak_scaling(x, a, b):
            return int(a + b * x)

        with mgf.MGF(file_mgf) as reader:
            for i, spec in enumerate(reader):

                # Make conform with spectrum class as defined in MS_functions.py
                # --------------------------------------------------------------------

                # Peaks will only be removed if they do not bring the total number of peaks
                # below min_peaks_scaled.
                if spec is not None:
                    min_peaks_scaled = min_peak_scaling(
                        spec['params']['pepmass'][0], min_keep_peaks_0,
                        min_keep_peaks_per_mz)

                    spectrum = Spectrum(
                        min_frag = min_frag,
                        max_frag = max_frag,
                        min_loss = min_loss,
                        max_loss = max_loss,
                        min_intensity_perc = min_intensity_perc,
                        exp_intensity_filter = exp_intensity_filter,
                        min_peaks = min_peaks,
                        max_peaks = max_peaks,
                        aim_min_peak = min_peaks_scaled)

                    id = i  #spec.spectrum_id
                    spectrum.read_spectrum_mgf(spec, id)
                    # spectrum.get_losses

                    # Calculate losses:
                    if len(spectrum.peaks) >= min_peaks \
                    and not ignore_losses:
                        spectrum.get_losses()

                    # Collect in form of list of spectrum objects
                    spectra.append(spectrum)

                else:
                    print("Found empty spectra for ID: ", i)

        # Filter out spectra with few peaks -----------------------------------------------------
        print(20 * '--')
        min_peaks_absolute = min_peaks
        num_spectra_initial = len(spectra)
        spectra = [copy.deepcopy(x) for x in spectra if len(x.peaks) >= min_peaks_absolute]
        print("Take", len(spectra), "spectra out of", num_spectra_initial)

        # Check spectrum IDs
        ids = []
        for spec in spectra:
            ids.append(spec.id)
        if len(list(set(ids))) < len(spectra):
            print("Non-unique spectrum IDs found. Resetting all IDs.")
            for i, spec in enumerate(spectra):
                spectra[i].id = i

        # Collect dictionary
        for spec in spectra:
            id = spec.id
            spectra_dict[id] = spec.__dict__

        if create_docs:
            # Create documents from peaks (and losses)
            ms_documents, ms_documents_intensity, spectra_metadata = create_ms_documents(
                spectra,
                num_decimals,
                peak_loss_words,
                min_loss,
                max_loss,
                ignore_losses = ignore_losses)

        # Save collected data ----------------------------------------------------------------------
        print()
        if collect_new_data == True and file_json is not None:
            # Store spectra
            print(20 * '--')
            print("Saving spectra...")
            if create_docs:
                spectra_metadata.to_csv(file_json[:-5] + "_metadata.csv",
                                        index=False)
            functions.dict_to_json(spectra_dict, file_json)

            if create_docs:
                # Store documents
                print(20 * '--')
                print("Saving documents...")
                with open(file_json[:-4] + "txt", "w") as f:
                    for s in ms_documents:
                        f.write(str(s) +"\n")

                with open(file_json[:-5] + "_intensity.txt", "w") as f:
                    for s in ms_documents_intensity:
                        f.write(str(s) +"\n")

    return spectra, spectra_dict, ms_documents, ms_documents_intensity, spectra_metadata




# --------------------------------------------------------------------------------------------------
# ---------------------- Functions to analyse MS data ----------------------------------------------
# --------------------------------------------------------------------------------------------------


def create_ms_documents(spectra,
                        num_decimals,
                        peak_loss_words = ['peak_', 'loss_'],
                        min_loss = 5.0,
                        max_loss = 500.0,
                        ignore_losses = False):
    """ Create documents from peaks and losses.

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
                keep_idx = np.where((losses[:,0] > min_loss)
                                    & (losses[:,0] < max_loss))[0]
                losses = losses[keep_idx,:]
            #else:
                #print("No losses detected for: ", spec_id, spectrum.id)

        peaks = np.array(spectrum.peaks)

        # Sort peaks and losses by m/z
        peaks = peaks[np.lexsort((peaks[:,1], peaks[:,0])),:]
        if not ignore_losses:
            if len(losses) > 0:
                losses = losses[np.lexsort((losses[:,1], losses[:,0])),:]

        if (spec_id+1) % 100 == 0 or spec_id == len(spectra)-1:  # show progress
                print('\r',
                      ' Created documents for {} of {} spectra'.format(spec_id+1, len(spectra)),
                      end="")

        for i in range(len(peaks)):
            doc.append(peak_loss_words[0] +
                       "{:.{}f}".format(peaks[i,0], num_decimals))
            doc_intensity.append(int(peaks[i,1]))
        if not ignore_losses:
            for i in range(len(losses)):
                doc.append(peak_loss_words[1] +
                           "{:.{}f}".format(losses[i,0], num_decimals))
                doc_intensity.append(int(losses[i,1]))

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


def mol_converter(mol_input, input_type, output_type, method='openbabel'):
    """ Convert molecular representations using openbabel. E.g. smiles to inchi,
    or inchi to inchikey.

    Args:
    --------
    mol_input: str
        Input data, e.g. inchi or smiles.
    input_type: str
        Define input type (as named in openbabel). E.g. "smi"for smiles and "inchi" for inchi.
    output_type: str
        Define input type (as named in openbabel). E.g. "smi"for smiles and "inchi" for inchi.
    method: str
        Default is making use of 'openbabel'. Alternative option could be 'RDkit'. Not supported yet.
        TODO: add RDkit as alternative ?
    """
    if method == 'openbabel':
        conv = ob.OBConversion()
        conv.SetInAndOutFormats(input_type, output_type)
        mol = ob.OBMol()
        try:
            conv.ReadString(mol, mol_input)
            mol_output = conv.WriteString(mol)
        except:
            print("error when converting...")
            mol_output = None

        return mol_output


def likely_inchi_match(inchi_1, inchi_2, min_agreement=3):
    """ Try to match defective inchi to non-defective ones.
    Compares inchi parts seperately. Match is found if at least the first 'min_agreement' parts
    are a good enough match.
    The main 'defects' this method accounts for are missing '-' in the inchi.
    In addition differences between '-', '+', and '?'will be ignored.

    Args:
    --------
    inchi_1: str
        inchi of molecule.
    inchi_2: str
        inchi of molecule.
    min_agreement: int
        Minimum number of first parts that MUST be a match between both input inchi to finally consider
        it a match. Default is min_agreement=3.
    """

    if min_agreement < 2:
        print("Warning! 'min_agreement' < 2 has no discriminative power. Should be => 2.")
    if min_agreement == 2:
        print("Warning! 'min_agreement' == 2 has little discriminative power",
              "(only looking at structure formula. Better use > 2.")
    agreement = 0

    # Remove spaces and '"' to account for different notations.
    # And remove all we assume is of minor importance only.
    ignore_lst = ['"', ' ', '-', '+', '?']
    for ignore in ignore_lst:
        inchi_1 = inchi_1.replace(ignore, '')
        inchi_2 = inchi_2.replace(ignore, '')

    # Split inchi in parts. And ignore '-' to account for defective inchi.
    inchi_1_parts = inchi_1.split('/')
    inchi_2_parts = inchi_2.split('/')

    # Check if both inchi have sufficient parts (seperated by '/')
    if len(inchi_1_parts) >= min_agreement and len(
            inchi_2_parts) >= min_agreement:
        # Count how many parts mostly agree
        for i in range(min_agreement):
            agreement += (inchi_1_parts[i] == inchi_2_parts[i])

    if agreement == min_agreement:
        return True
    else:
        return False


def likely_inchikey_match(inchikey_1, inchikey_2, min_agreement=2):
    """ Try to match inchikeys.
    Compares inchikey parts seperately. Match is found if at least the first 'min_agreement' parts
    are a good enough match.

    Args:
    --------
    inchikey_1: str
        inchikey of molecule.
    inchikey_2: str
        inchikey of molecule.
    min_agreement: int
        Minimum number of first parts that MUST be a match between both input inchikey to finally consider
        it a match. Default is min_agreement=2.
    """

    if min_agreement not in [1, 2, 3]:
        print("Warning! 'min_agreement' should be 1, 2, or 3.")
    agreement = 0

    # Make sure all letters are capitalized. Remove spaces and '"' to account for different notations
    inchikey_1 = inchikey_1.upper().replace('"', '').replace(' ', '')
    inchikey_2 = inchikey_2.upper().replace('"', '').replace(' ', '')

    # Split inchikey in parts.
    inchikey_1_parts = inchikey_1.split('-')
    inchikey_2_parts = inchikey_2.split('-')

    # Check if both inchikey have sufficient parts (seperated by '/')
    if len(inchikey_1_parts) >= min_agreement and len(
            inchikey_2_parts) >= min_agreement:
        # Count how many parts mostly agree
        for i in range(min_agreement):
            agreement += (inchikey_1_parts[i] == inchikey_2_parts[i])

    return agreement == min_agreement


def find_pubchem_match(compound_name,
                       inchi,
                       inchikey = None,
                       mode = 'and',
                       min_inchi_match = 3,
                       min_inchikey_match = 1,
                       name_search_depth = 10,
                       formula_search = False,
                       formula_search_depth = 25):
    """ Searches pubmed for compounds based on name.
    Then check if inchi and/or inchikey can be matched to (defective) input inchi and/or inchikey.

    In case no matches are found: For formula_search = True, the search will continue based on the
    formula extracted from the inchi.

    Outputs found inchi and found inchikey (will be None if none is found).

    Args:
    -------
    compound_name: str
        Name of compound to search for on Pubchem.
    inchi: str
        Inchi (correct, or defective...). Set to None to ignore.
    inchikey: str
        Inchikey (correct, or defective...). Set to None to ignore.
    mode: str
        Determines the final matching criteria (can be se to 'and' or 'or').
        For 'and' and given inchi AND inchikey, a match has to be a match with inchi AND inchikey.
        For 'or' it will be sufficient to find a good enough match with either inchi OR inchikey.
    min_inchi_match: int
        Minimum number of first parts that MUST be a match between both input inchi to finally consider
        it a match. Default is min_inchi_match=3.
    min_inchikey_match: int
        Minimum parts of inchikey that must be equal to be considered a match. Can be 1, 2, or 3.
    name_search_depth: int
        How many of the most relevant name matches to explore deeper. Default = 10.
    formula_search: bool
        If True an additional search using the chemical formula is done if the name did not
        already give a good match. Makes the search considerable slower.
    formula_search_depth: int
        How many of the most relevant formula matches to explore deeper. Default = 25.
    """

    if inchi is None:
        match_inchi = True
        mode = 'and'  # Do not allow 'or' in that case.
    else:
        match_inchi = False

    if inchikey is None:
        match_inchikey = True
        mode = 'and'  # Do not allow 'or' in that case.
    else:
        match_inchikey = False

    if mode == 'and':
        operate = operator.and_
    elif mode == 'or':
        operate = operator.or_
    else:
        print("Wrong mode was given!")

    inchi_pubchem = None
    inchikey_pubchem = None

    # Search pubmed for compound name:
    results_pubchem = pcp.get_compounds(compound_name,
                                        'name',
                                        listkey_count = name_search_depth)
    print("Found at least", len(results_pubchem), "compounds of that name on pubchem.")


    # Loop through first 'name_search_depth' results found on pubchem. Stop once first match is found.
    for result in results_pubchem:
        inchi_pubchem = '"' + result.inchi + '"'
        inchikey_pubchem = result.inchikey

        if inchi is not None:
            match_inchi = likely_inchi_match(
                inchi,
                inchi_pubchem,
                min_agreement=min_inchi_match)
        if inchikey is not None:
            match_inchikey = likely_inchikey_match(
                inchikey,
                inchikey_pubchem,
                min_agreement=min_inchikey_match)

        if operate(
                match_inchi, match_inchikey
                ):  # Found match for inchi and/or inchikey (depends on mode = 'and'/'or')
            print("--> FOUND MATCHING COMPOUND ON PUBCHEM.")
            if inchi is not None:
                print("Inchi ( input ): " + inchi)
                print("Inchi (pubchem): " + inchi_pubchem + "\n")
            if inchikey is not None:
                print("Inchikey ( input ): " + inchikey)
                print("Inchikey (pubchem): " + inchikey_pubchem + "\n")
            break

    if not operate(match_inchi, match_inchikey):
        if inchi is not None and formula_search:
            # Do additional search on Pubchem with the formula

            # Get formula from inchi
            inchi_parts = inchi.split('InChI=')[1].split('/')
            if len(inchi_parts) >= min_inchi_match:
                compound_formula = inchi_parts[1]

                # Search formula on Pubchem
                sids_pubchem = pcp.get_sids(compound_formula,
                                            'formula',
                                            listkey_count=formula_search_depth)
                print("Found at least", len(sids_pubchem),
                      "compounds with formula", compound_formula,
                      "on pubchem.")

                results_pubchem = []
                for sid in sids_pubchem:
                    result = pcp.Compound.from_cid(sid['CID'])
                    results_pubchem.append(result)

                for result in results_pubchem:
                    inchi_pubchem = '"' + result.inchi + '"'
                    inchikey_pubchem = result.inchikey

                    if inchi is not None:
                        match_inchi = likely_inchi_match(
                            inchi,
                            inchi_pubchem,
                            min_agreement=min_inchi_match)
                    if inchikey is not None:
                        match_inchikey = likely_inchikey_match(
                            inchikey,
                            inchikey_pubchem,
                            min_agreement=min_inchikey_match)

                    if operate(
                            match_inchi, match_inchikey
                            ):  # Found match for inchi and/or inchikey (depends on mode = 'and'/'or')
                        print("--> FOUND MATCHING COMPOUND ON PUBCHEM.")
                        if inchi is not None:
                            print("Inchi ( input ): " + inchi)
                            print("Inchi (pubchem): " + inchi_pubchem + "\n")
                        if inchikey is not None:
                            print("Inchikey ( input ): " + inchikey)
                            print("Inchikey (pubchem): " + inchikey_pubchem +
                                  "\n")
                        break

    if not operate(match_inchi, match_inchikey):
        inchi_pubchem = None
        inchikey_pubchem = None

        if inchi is not None and inchikey is not None:
            print("No matches found for inchi", inchi, mode, " inchikey",
                  inchikey, "\n")
        elif inchikey is None:
            print("No matches found for inchi", inchi, "\n")
        else:
            print("No matches found for inchikey", inchikey, "\n")

    return inchi_pubchem, inchikey_pubchem



def get_mol_fingerprints(spectra,
                         method = "daylight",
                         nbits = 1024,
                         print_progress = True):
    """ Calculate molecule fingerprints based on given smiles.
    (using RDkit)

    Output: exclude_IDs list with spectra that had no smiles or problems when deriving fingerprint

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
    exclude_IDs = []
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
            exclude_IDs.append(i)

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
                fp = Chem.RDKFingerprint(molecules[i], fpSize = nbits)
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

    return fingerprints, exclude_IDs
