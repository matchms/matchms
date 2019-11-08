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

import helper_functions as functions
import MS_similarity_classical as MS_sim_classic

import fnmatch
import copy
import numpy as np
from scipy.optimize import curve_fit
import random
import pandas as pd

from pyteomics import mgf

from openbabel import openbabel as ob
from openbabel import pybel

import pubchempy as pcp

from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import AllChem




## --------------------------------------------------------------------------------------------------
## ---------------------------- Spectrum class ------------------------------------------------------
## --------------------------------------------------------------------------------------------------

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
    def __init__(self, min_frag = 0.0, 
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
        self.fingerprint = None
        self.fingerprint_type = None
        
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
            single_charge_precursor_mass -= (int_charge-1) * self.PROTON_MASS
        elif int_charge < 0:
            single_charge_precursor_mass += (mul-1) * self.PROTON_MASS
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
            if not type(charge) == str:
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
    
        with open(os.path.join(path, file),'r') as f:            
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
                        valval = rline[len(keyval)+2:]
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
                            errs = 1e6*np.abs(mass-np.array(temp_mass))/mass
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
        #  This should lead to better computation of neutral losses
        single_charge_precursor_mass = self.metadata['precursormass']
        precursor_mass = self.metadata['precursormass']
        parent_mass = self.metadata['precursormass']

        str_charge = self.metadata['charge']
        int_charge = self.interpret_charge(str_charge)

        parent_mass, single_charge_precursor_mass = self.ion_masses(precursor_mass, int_charge)

        self.metadata['parentmass'] = parent_mass
        self.metadata['singlechargeprecursormass'] = single_charge_precursor_mass
        self.metadata['charge'] = int_charge
        
        # Get precursor mass (later used to calculate losses!)
        self.precursor_mz = float(self.metadata['precursormass'])
        self.parent_mz = float(self.metadata['parentmass'])
        
        if 'smiles' in self.metadata:
            self.smiles = self.metadata['smiles']
        if 'inchi' in self.metadata:
            self.inchi = self.metadata['inchi']
        if 'inchikey' in self.metadata:
            self.inchikey = self.metadata['inchi']

        peaks = list(zip(spectrum_mgf['m/z array'], spectrum_mgf['intensity array']))
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
        
        MS1_peak = self.precursor_mz
        losses = np.array(self.peaks.copy())
        losses[:,0] = MS1_peak - losses[:,0]
        keep_idx = np.where((losses[:,0] > self.min_loss) & (losses[:,0] < self.max_loss))[0]
        
        # TODO: now array is tranfered back to list (to be able to store as json later). Seems weird.
        losses_list = [(x[0], x[1]) for x in losses[keep_idx,:]]
        self.losses = losses_list      
        
        
    def get_fingerprint(self, type = "ecfp6"):
        """ Calculate molecule fingerprint for spectrum object based on given inchi or smiles (using openbabel).
        
        Output: exclude_IDs list with spectra that had no inchi or smiles or problems when deriving fingerprint
        
        Args:
        --------
        type: str
            Determine type of molecular fingerprint to be calculated. Supports choices from openbabel, e.g:
            'ecfp0', 'ecfp10', 'ecfp2', 'ecfp4', 'ecfp6', 'ecfp8', 'fp2', 'fp3', 'fp4', 'maccs'. (see "pybel.fps").
            Default is = "ecfp6".
        """
        self.fingerprint_type = type
        
        if self.inchi is not None \
        or self.smiles is not None:   
            self.fingerprint = get_mol_fingerprint(self.inchi, self.smiles, type = type)
             
        else: 
            self.fingerprint = None
            print("Spectrum object contains no inchi or smiles --> No molecular fingerprint can be derived.")



## --------------------------------------------------------------------------------------------------
## ---------------------------- Spectrum processing functions ---------------------------------------
## --------------------------------------------------------------------------------------------------

        
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
    min_peaks_for_exp_fit = 25 # With less peaks exponential fit doesn't make any sense.
    
    if aim_min_peaks is None: # aim_min_peaks is not given
        aim_min_peaks = min_peaks
            
    def exponential_func(x, a, b):
        return a*np.exp(-b*x)
   
    if isinstance(peaks, list):
        peaks = np.array(peaks)
        if peaks.shape[1] != 2:
            print("Peaks were given in unexpected format...")
    
    # Remove peaks outside min_frag <-> max_frag window:
    keep_idx = np.where((peaks[:,0] > min_frag) & (peaks[:,0] < max_frag))[0]
    peaks = peaks[keep_idx,:]
    
    # Remove peaks based on relative intensity below min_intensity_perc/100 * max_intensity
    if min_intensity_perc > 0:
        intensity_thres = np.max(peaks[:,1]) * min_intensity_perc/100
        keep_idx = np.where((peaks[:,0] > min_frag) & (peaks[:,0] < max_frag) & (peaks[:,1] > intensity_thres))[0]
        if len(keep_idx) > min_peaks: 
            peaks = peaks[keep_idx,:]

    # Fit exponential to peak intensity distribution
    if (exp_intensity_filter is not None) and len(peaks) >= min_peaks_for_exp_fit: 

        # Ignore highest peak for further analysis 
        peaks2 = peaks.copy()
        peaks2[np.where(peaks2[:,1] == np.max(peaks2[:,1])),:] = 0
        
        # Create histogram
        hist, bins = np.histogram(peaks2[:,1], bins=num_bins)
        offset = np.where(hist == np.max(hist))[0][0]  # Take maximum intensity bin as starting point
        last = int(num_bins/2)
        x = bins[offset:last]
        y = hist[offset:last]
        # Try exponential fit:
        try:
            popt, pcov = curve_fit(exponential_func, x , y, p0=(peaks.shape[0], 1e-4)) 
            lower_guess_offset = bins[max(0,offset-1)]
            threshold = lower_guess_offset -np.log(1 - exp_intensity_filter)/popt[1]
        except RuntimeError:
            print("RuntimeError for ", len(peaks), " peaks. Use 1/2 mean intensity as threshold.")
            threshold = np.mean(peaks2[:,1])/2
        except TypeError:
            print("Unclear TypeError for ", len(peaks), " peaks. Use 1/2 mean intensity as threshold.")
            print(x, "and y: ", y)
            threshold = np.mean(peaks2[:,1])/2

        keep_idx = np.where(peaks[:,1] > threshold)[0]
        if len(keep_idx) < aim_min_peaks:
            peaks = peaks[np.lexsort((peaks[:,0], peaks[:,1])),:][-aim_min_peaks:]
        else:
            peaks = peaks[keep_idx, :]
                  
        # Sort by peak intensity
        peaks = peaks[np.lexsort((peaks[:,0], peaks[:,1])),:]
        if max_peaks is not None:
            return [(x[0], x[1]) for x in peaks[-max_peaks:,:]] # TODO: now array is transfered back to list (to be able to store as json later). Seems weird.

        else:
            return [(x[0], x[1]) for x in peaks]   
    else:
        # Sort by peak intensity
        peaks = peaks[np.lexsort((peaks[:,0], peaks[:,1])),:]
        if max_peaks is not None:
            return [(x[0], x[1]) for x in peaks[-max_peaks:,:]]
        else:
            return [(x[0], x[1]) for x in peaks]


## ----------------------------------------------------------------------------
## -------------------------- Functions to load MS data------------------------
## ----------------------------------------------------------------------------

def load_MS_data(path_data, path_json,
                 filefilter="*.*", 
                 results_file = None,
                 num_decimals = 3,
                 min_frag = 0.0, max_frag = 1000.0,
                 min_loss = 10.0, max_loss = 200.0,
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
    """

    spectra = []
    spectra_dict = {}
    MS_documents = []
    MS_documents_intensity = []

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
                    MS_documents.append(line.split(", "))
                    
            with open(path_json + results_file[:-5] + "_intensity.txt", "r") as f:
                for line in f:
                    line = line.replace("[", "").replace("]", "")
                    MS_documents_intensity.append([int(x) for x in line.split(", ")])
                
        except FileNotFoundError: 
            print("Could not find file ", path_json,  results_file) 
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

        MS_documents, MS_documents_intensity, spectra_metadata = create_MS_documents(spectra, 
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
                for s in MS_documents:
                    f.write(str(s) +"\n")
                    
            with open(path_json + results_file[:-5] + "_intensity.txt", "w") as f:
                for s in MS_documents_intensity:
                    f.write(str(s) +"\n")

    return spectra, spectra_dict, MS_documents, MS_documents_intensity, spectra_metadata


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
    MS_documents = []
    MS_documents_intensity = []
    spectra_metadata = []
    collect_new_data = True
        
    if file_json is not None:
        try: 
            spectra_dict = functions.json_to_dict(file_json)
            spectra_metadata = pd.read_csv(file_json[:-5] + "_metadata.csv")
            print("Spectra json file found and loaded.")
            spectra = dict_to_spectrum(spectra_dict)
            collect_new_data = False
            
            if create_docs:
                with open(file_json[:-4] + "txt", "r") as f:
                    for line in f:
                        line = line.replace('"', '').replace("'", "").replace("[", "").replace("]", "").replace("\n", "")
                        MS_documents.append(line.split(", "))
                        
                with open(file_json[:-5] + "_intensity.txt", "r") as f:
                    for line in f:
                        line = line.replace("[", "").replace("]", "")
                        MS_documents_intensity.append([int(x) for x in line.split(", ")])
                
        except FileNotFoundError: 
            print(20 * '--')
            print("Could not find file ", file_json) 
            print(20 * '--')
            print("Data will be imported from ", file_mgf)

    # Read data from files if no pre-stored data is found:
    if spectra_dict == {} or file_json is None:
        
        # Scale the min_peak filter
        def min_peak_scaling(x, A, B):
            return int(A + B * x)

        with mgf.MGF(file_mgf) as reader:
            for i, spec in enumerate(reader):
        
                # Make conform with spectrum class as defined in MS_functions.py
                #--------------------------------------------------------------------

                # Peaks will only be removed if they do not bring the total number of peaks
                # below min_peaks_scaled.
                if spec is not None:
                    min_peaks_scaled = min_peak_scaling(spec['params']['pepmass'][0], min_keep_peaks_0, min_keep_peaks_per_mz)   
                
                    spectrum = Spectrum(min_frag = min_frag, 
                                        max_frag = max_frag,
                                        min_loss = min_loss, 
                                        max_loss = max_loss,
                                        min_intensity_perc = min_intensity_perc,
                                        exp_intensity_filter = exp_intensity_filter,
                                        min_peaks = min_peaks,
                                        max_peaks = max_peaks,
                                        aim_min_peak = min_peaks_scaled)
                    
                    id = i #spec.spectrum_id
                    spectrum.read_spectrum_mgf(spec, id)
                    #spectrum.get_losses
        
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
            MS_documents, MS_documents_intensity, spectra_metadata = create_MS_documents(spectra, 
                                                                                         num_decimals, 
                                                                                         peak_loss_words, 
                                                                                         min_loss, 
                                                                                         max_loss,
                                                                                         ignore_losses = ignore_losses)

        # Save collected data ----------------------------------------------------------------------
        print()
        if collect_new_data == True:
            # Store spectra
            print(20 * '--')
            print("Saving spectra...")
            if create_docs:
                spectra_metadata.to_csv(file_json[:-5] + "_metadata.csv", index=False)           
            functions.dict_to_json(spectra_dict, file_json) 
            
            if create_docs:
                # Store documents
                print(20 * '--')
                print("Saving documents...")
                with open(file_json[:-4] + "txt", "w") as f:
                    for s in MS_documents:
                        f.write(str(s) +"\n")
                        
                with open(file_json[:-5] + "_intensity.txt", "w") as f:
                    for s in MS_documents_intensity:
                        f.write(str(s) +"\n")
                    
    return spectra, spectra_dict, MS_documents, MS_documents_intensity, spectra_metadata




## --------------------------------------------------------------------------------------------------
## ---------------------- Functions to analyse MS data ----------------------------------------------
## --------------------------------------------------------------------------------------------------


def create_MS_documents(spectra, 
                        num_decimals, 
                        peak_loss_words = ['peak_', 'loss_'],
                        min_loss = 10.0, 
                        max_loss = 200.0, 
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
        Lower limit of losses to take into account (Default = 10.0).
    max_loss: float
        Upper limit of losses to take into account (Default = 200.0).
    ignore_losses: bool
        True: Ignore losses, False: make words from losses and peaks.
    """
    
    MS_documents = []
    MS_documents_intensity = []
    spectra_metadata = pd.DataFrame(columns=['doc_ID', 'spectrum_ID', 'sub_ID', 'precursor_mz', 'parent_intensity', 'no_peaks_losses'])
    
    for spec_id, spectrum in enumerate(spectra):
        doc = []
        doc_intensity = []
        if not ignore_losses:
            losses = np.array(spectrum.losses)
            if len(losses) > 0: 
                keep_idx = np.where((losses[:,0] > min_loss) & (losses[:,0] < max_loss))[0]
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
                print('\r', ' Created documents for ', spec_id+1, ' of ', len(spectra), ' spectra.', end="")
                
        for i in range(len(peaks)):
            doc.append(peak_loss_words[0] + "{:.{}f}".format(peaks[i,0], num_decimals))
            doc_intensity.append(int(peaks[i,1]))
        if not ignore_losses:    
            for i in range(len(losses)):
                doc.append(peak_loss_words[1]  + "{:.{}f}".format(losses[i,0], num_decimals))
                doc_intensity.append(int(losses[i,1]))

        MS_documents.append(doc)
        MS_documents_intensity.append(doc_intensity)
        spectra_metadata.loc[spec_id] = [spec_id, int(spectrum.id), 0, spectrum.precursor_mz, 1, len(doc)]
         
    return MS_documents, MS_documents_intensity, spectra_metadata


def mol_converter(mol_input, input_type, output_type, method = 'openbabel'):
    """ Convert molecular representations using openbabel (or RDkit). E.g. smiles to inchi,
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


def likely_inchi_match(inchi_1, inchi_2, min_agreement = 3):
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
    
    agreement = 0
    
    # Remove spaces and '"' to account for different notations. And remove all we assume is of minor importance only.    
    ignore_lst = ['"', ' ', '-', '+', '?']
    for ignore in ignore_lst:
        inchi_1 = inchi_1.replace(ignore, '')
        inchi_2 = inchi_2.replace(ignore, '')
    
    # Split inchi in parts. And ignore '-' to account for defective inchi.
    inchi_1_parts = inchi_1.split('/')
    inchi_2_parts = inchi_2.split('/')
    
    # Check if both inchi have sufficient parts (seperated by '/')
    if len(inchi_1_parts) >= min_agreement and len(inchi_2_parts) >= min_agreement:
        # Count how many parts mostly agree 
        for i in range(min_agreement):
            agreement += (inchi_1_parts[i] == inchi_2_parts[i])

    if agreement == min_agreement:
        return True
    else:
        return False


def likely_inchikey_match(inchikey_1, inchikey_2, min_agreement = 2):
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
    if len(inchikey_1_parts) >= min_agreement and len(inchikey_2_parts) >= min_agreement:
        # Count how many parts mostly agree 
        for i in range(min_agreement):
            agreement += (inchikey_1_parts[i] == inchikey_2_parts[i])

    if agreement == min_agreement:
        return True
    else:
        return False



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
        mode = 'and' # do not allow 'or' in that case.
    else:
        match_inchi = False
        
    if inchikey is None:
        match_inchikey = True
        mode = 'and' # do not allow 'or' in that case.
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
    results_pubchem = pcp.get_compounds(compound_name, 'name', listkey_count = name_search_depth)
    print("Found at least", len(results_pubchem), "compounds of that name on pubchem.")
    
    
    # Loop through first 'name_search_depth' results found on pubchem. Stop once first match is found.
    for result in results_pubchem:
        inchi_pubchem = '"' + result.inchi + '"'
        inchikey_pubchem = result.inchikey

        if inchi is not None:
            match_inchi = likely_inchi_match(inchi, inchi_pubchem, min_agreement = min_inchi_match)
        if inchikey is not None:
            match_inchikey = likely_inchikey_match(inchikey, inchikey_pubchem, min_agreement = min_inchikey_match)
                     
        if operate(match_inchi, match_inchikey): # found match for inchi and/or inchikey (depends on mode = 'and'/'or')
            print("--> FOUND MATCHING COMPOUND ON PUBCHEM.")
            if inchi is not None:
                print("Inchi ( input ): " + inchi)
                print("Inchi (pubchem): " + inchi_pubchem + "\n")
            if inchikey is not None:
                print("Inchikey ( input ): " + inchikey)
                print("Inchikey (pubchem): " + inchikey_pubchem + "\n")
            break
            
    if not operate(match_inchi, match_inchikey):
        if inchi is not None \
        and formula_search:
            # Do additional search on Pubchem with the formula
            
            # Get formula from inchi
            inchi_parts = inchi.split('InChI=')[1].split('/')
            if len(inchi_parts) >= min_inchi_match:
                compound_formula = inchi_parts[1]
                
                # Search formula on Pubchem
                sids_pubchem = pcp.get_sids(compound_formula, 'formula', listkey_count = formula_search_depth)
                print("Found at least", len(sids_pubchem), "compounds with formula", compound_formula,"on pubchem.")
    
                results_pubchem = []
                for sid in sids_pubchem:
                    result = pcp.Compound.from_cid(sid['CID'])
                    results_pubchem.append(result)
                    
                for result in results_pubchem:
                    inchi_pubchem = '"' + result.inchi + '"'
                    inchikey_pubchem = result.inchikey
                    
                    if inchi is not None:
                        match_inchi = likely_inchi_match(inchi, inchi_pubchem, min_agreement = min_inchi_match)
                    if inchikey is not None:
                        match_inchikey = likely_inchikey_match(inchikey, inchikey_pubchem, min_agreement = min_inchikey_match)
                    
                    if operate(match_inchi, match_inchikey): # found match for inchi and/or inchikey (depends on mode = 'and'/'or')
                        print("--> FOUND MATCHING COMPOUND ON PUBCHEM.")
                        if inchi is not None:
                            print("Inchi ( input ): " + inchi)
                            print("Inchi (pubchem): " + inchi_pubchem + "\n")
                        if inchikey is not None:
                            print("Inchikey ( input ): " + inchikey)
                            print("Inchikey (pubchem): " + inchikey_pubchem + "\n")
                        break
    
    if not operate(match_inchi, match_inchikey):    
        inchi_pubchem = None
        inchikey_pubchem = None
        
        if inchi is not None and inchikey is not None:
            print("No matches found for inchi", inchi, mode, " inchikey", inchikey, "\n")
        elif inchikey is None:
            print("No matches found for inchi", inchi, "\n")
        else:
            print("No matches found for inchikey", inchikey, "\n")
    
    return inchi_pubchem, inchikey_pubchem


def get_mol_fingerprint(inchi, smiles, 
                     type = "ecfp6"):
    """ Calculate molecule fingerprints based on given inchi or smiles (using openbabel).
    Preference will be given to fingerprint from inchi. Only if that won't work, smiles are used.
    
    Output: derived fingerprint.
    
    Args:
    --------
    inchi: str
        Inchi. Set to None to ignore.
    smiles: str
        Smiles. Set to None to ignore.
    type: str
        Determine type of molecular fingerprint to be calculated. Supports choices from openbabel, e.g:
        'ecfp0', 'ecfp10', 'ecfp2', 'ecfp4', 'ecfp6', 'ecfp8', 'fp2', 'fp3', 'fp4', 'maccs'. (see "pybel.fps").
        Default is = "ecfp6".
    """
     
    mol = None
    if inchi is not None:
        if len(inchi) > 12 \
        and inchi.split('InChI=')[-1][0] == '1': # try to sort out empty and defective inchis
            #mol = 1
            try:
                mol = pybel.readstring("inchi", inchi) 
            except:
                print('Error while handling inchi:', inchi)
                mol = None

    if smiles is not None and mol is None:  # Smiles but no InChikey or inchi handling failed
        if len(smiles) > 5: # try to sort out empty and defective smiles
            try:
                mol = pybel.readstring("smi", smiles)
                if len(mol.atoms)>2:
                    print('Molecule found using smiles:', smiles)
            except:
                print('Error while handling smiles:', smiles)
                mol = None
        
    if mol is None \
    or mol == '':
        print("Problem with molecule.")
        fingerprint = None
    else:
        try:
            fingerprint = mol.calcfp(type)
        except:
            print("Problem deriving molecular fingerprint.")
            fingerprint = None
    
    return fingerprint



# TODO: this function is not needed anymore? 
def compare_molecule_selection(query_id, spectra_dict, MS_measure, 
                               fingerprints,
                               num_candidates = 25, 
                               similarity_method = "centroid"):
    """ Compare spectra-based similarity with smile-based similarity.
    
    Args:
    -------
    query_id: int
        Number of spectra to use as query.
    spectra_dict: dict
        Dictionary containing all spectra peaks, losses, metadata.
    MS_measure: object
        Similariy object containing the model and distance matrices.
    fingerprints: object
        Fingerprint objects for all molecules (if smiles exist for the spectra).
    num_candidates: int
        Number of candidates to list (default = 25) .
    similarity_method: str
        Define method to use (default = "centroid").
    """
    
    # Select chosen similarity methods
    if similarity_method == "centroid":
        candidates_idx = MS_measure.list_similars_ctr_idx[query_id, :num_candidates]
        candidates_dist = MS_measure.list_similars_ctr[query_id, :num_candidates]
    elif similarity_method == "pca":
        candidates_idx = MS_measure.list_similars_pca_idx[query_id, :num_candidates]
        candidates_dist = MS_measure.list_similars_pca[query_id, :num_candidates]
    elif similarity_method == "autoencoder":
        candidates_idx = MS_measure.list_similars_ae_idx[query_id, :num_candidates]
        candidates_dist = MS_measure.list_similars_ae[query_id, :num_candidates]
    elif similarity_method == "lda":
        candidates_idx = MS_measure.list_similars_lda_idx[query_id, :num_candidates]
        candidates_dist = MS_measure.list_similars_lda[query_id, :num_candidates]
    elif similarity_method == "lsi":
        candidates_idx = MS_measure.list_similars_lsi_idx[query_id, :num_candidates]
        candidates_dist = MS_measure.list_similars_lsi[query_id, :num_candidates]
    elif similarity_method == "doc2vec":
        candidates_idx = MS_measure.list_similars_d2v_idx[query_id, :num_candidates]
        candidates_dist = MS_measure.list_similars_d2v[query_id, :num_candidates]
    else:
        print("Chosen similarity measuring method not found.")
        
    mol_sim = np.zeros((len(fingerprints)))
    if fingerprints[query_id] != 0:
        for j in range(len(fingerprints)):
            if fingerprints[j] != 0:     
                mol_sim[j] = DataStructs.FingerprintSimilarity(fingerprints[query_id], fingerprints[j])
                
    smiles_similarity = np.array([np.arange(0, len(mol_sim)), mol_sim]).T
    smiles_similarity = smiles_similarity[np.lexsort((smiles_similarity[:,0], smiles_similarity[:,1])),:]
    
    print("Selected candidates based on spectrum: ")
    print(candidates_idx)
    print("Selected candidates based on smiles: ")
    print(smiles_similarity[:num_candidates,0])
    print("Selected candidates based on spectrum: ")
    for i in range(num_candidates):
        print("id: "+ str(candidates_idx[i]) + " (similarity: " +  str(candidates_dist[i]) + " | Tanimoto: " + str(mol_sim[candidates_idx[i]]) +")")


def evaluate_measure(spectra_dict, 
                     spectra,
                     MS_measure, 
                     fingerprints,
                     num_candidates = 25,
                     num_of_molecules = "all", 
                     similarity_method = "centroid",
                     molnet_sim = None,
                     reference_list = None):
    """ Compare spectra-based similarity with smile-based similarity.
    
    Output:
    -------
    mol_sim: matrix with molecule similarity scores for TOP 'num_candidates' for 'num_of_molecules'.
    spec_sim: matrix with spectra similarity for TOP 'num_candidates' for 'num_of_molecules' (using 'similarity_method').
    spec_idx: matrix with spectra IDs corresponding to spec_sim values.
    reference_list: list of selected 'num_of_molecules'. Will contain all IDs if num_of_molecules = "all".
        
    Args:
    -------
    spectra_dict: dict
        Dictionary containing all spectra peaks, losses, metadata.
    MS_measure: object
        Similariy object containing the model and distance matrices.
    fingerprints: object
        Fingerprint objects for all molecules (if smiles exist for the spectra).
    num_candidates: int
        Number of candidates to list (default = 25) .
    num_of_molecules: int
        Number of molecules to test method on (default= 100)
    similarity_method: str
        Define method to use (default = "centroid").
    """
    num_spectra = len(MS_measure.corpus)
    
    # Create reference list if not given as args:
    if reference_list is None:
        if num_of_molecules == "all":
            reference_list = np.arange(num_spectra)
        elif isinstance(num_of_molecules, int): 
            reference_list = np.array(random.sample(list(np.arange(len(fingerprints))),k=num_of_molecules))
        else:
            print("num_of_molecules needs to be integer or 'all'.")
        
    mol_sim = np.zeros((len(reference_list), num_candidates))
    spec_sim = np.zeros((len(reference_list), num_candidates))
    spec_idx = np.zeros((len(reference_list), num_candidates))
    
    candidates_idx = np.zeros((num_candidates), dtype=int)
    candidates_sim = np.zeros((num_candidates))
    
    for i, query_id in enumerate(reference_list):
        # Show progress:
        
        if (i+1) % 10 == 0 or i == len(reference_list)-1:  
                print('\r', ' Evaluate spectrum ', i+1, ' of ', len(reference_list), ' spectra.', end="")

        # Select chosen similarity methods
        if similarity_method == "centroid":
            candidates_idx = MS_measure.list_similars_ctr_idx[query_id, :num_candidates]
            candidates_sim = MS_measure.list_similars_ctr[query_id, :num_candidates]
        elif similarity_method == "pca":
            candidates_idx = MS_measure.list_similars_pca_idx[query_id, :num_candidates]
            candidates_sim = MS_measure.list_similars_pca[query_id, :num_candidates]
        elif similarity_method == "autoencoder":
            candidates_idx = MS_measure.list_similars_ae_idx[query_id, :num_candidates]
            candidates_sim = MS_measure.list_similars_ae[query_id, :num_candidates]
        elif similarity_method == "lda":
            candidates_idx = MS_measure.list_similars_lda_idx[query_id, :num_candidates]
            candidates_sim = MS_measure.list_similars_lda[query_id, :num_candidates]
        elif similarity_method == "lsi":
            candidates_idx = MS_measure.list_similars_lsi_idx[query_id, :num_candidates]
            candidates_sim = MS_measure.list_similars_lsi[query_id, :num_candidates]
        elif similarity_method == "doc2vec":
            candidates_idx = MS_measure.list_similars_d2v_idx[query_id, :num_candidates]
            candidates_sim = MS_measure.list_similars_d2v[query_id, :num_candidates]
            
        elif similarity_method == "molnet":      
            candidates_idx = molnet_sim[i,:].argsort()[-num_candidates:][::-1]
            candidates_sim = molnet_sim[i, candidates_idx]
                         
        else:
            print("Chosen similarity measuring method not found.")

        # Check type of fingerprints given as input:
        try: 
            DataStructs.FingerprintSimilarity(fingerprints[0], fingerprints[0])
            fingerprint_type = "daylight" # at least assumed here
        
        except AttributeError:
            fingerprint_type = "morgan" # at least assumed here

        # Calculate Tanimoto similarity for selected candidates
        if fingerprints[query_id] != 0:
            for j, cand_id in enumerate(candidates_idx): 
                if fingerprints[cand_id] != 0:     
                    if fingerprint_type == "daylight":
                        mol_sim[i, j] = DataStructs.FingerprintSimilarity(fingerprints[query_id], fingerprints[cand_id])
                    elif fingerprint_type == "morgan":
                        mol_sim[i, j] = DataStructs.DiceSimilarity(fingerprints[query_id], fingerprints[cand_id])

        spec_sim[i,:] = candidates_sim
        spec_idx[i,:] = candidates_idx

    return mol_sim, spec_sim, spec_idx, reference_list







## --------------------------------------------------------------------------------------------------
## ---------------------------- Plotting functions --------------------------------------------------
## --------------------------------------------------------------------------------------------------
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter


def get_spaced_colors_hex(n):
    """ Create set of 'n' well-distinguishable colors
    """
    spaced_colors = ["FF0000", "00FF00", "0000FF", "FFFF00", "FF00FF", "00FFFF", "000000", 
        "800000", "008000", "000080", "808000", "800080", "008080", "808080", 
        "C00000", "00C000", "0000C0", "C0C000", "C000C0", "00C0C0", "C0C0C0", 
        "400000", "004000", "000040", "404000", "400040", "004040", "404040", 
        "200000", "002000", "000020", "202000", "200020", "002020", "202020", 
        "600000", "006000", "000060", "606000", "600060", "006060", "606060", 
        "A00000", "00A000", "0000A0", "A0A000", "A000A0", "00A0A0", "A0A0A0", 
        "E00000", "00E000", "0000E0", "E0E000", "E000E0", "00E0E0", "E0E0E0"]
    
    RGB_colors = ["#"+x for x in spaced_colors[:n] ]

    return RGB_colors


def plot_spectra(spectra, compare_ids, min_mz = 50, max_mz = 500):
    """ Plot different spectra together to compare.
    """
    plt.figure(figsize=(10,10))

    peak_number = []
    RGB_colors = get_spaced_colors_hex(len(compare_ids))
    for i, id in enumerate(compare_ids):
        peaks = np.array(spectra[id].peaks.copy())
        peak_number.append(len(peaks))
        peaks[:,1] = peaks[:,1]/np.max(peaks[:,1]); 

        markerline, stemlines, baseline = plt.stem(peaks[:,0], peaks[:,1], linefmt='-', markerfmt='.', basefmt='r-')
        plt.setp(stemlines, 'color', RGB_colors[i])
    
    plt.xlim((min_mz, max_mz))
    plt.grid(True)
    plt.title('Spectrum')
    plt.xlabel('m/z')
    plt.ylabel('peak intensity')
    
    plt.show()
    
    print("Number of peaks: ", peak_number)


def plot_losses(spectra, compare_ids, min_loss = 0, max_loss = 500):
    """ Plot different spectra together to compare.
    """
    plt.figure(figsize=(10,10))

    losses_number = []
    RGB_colors = get_spaced_colors_hex(len(compare_ids)+5)
    for i, id in enumerate(compare_ids):
        losses = np.array(spectra[id].losses.copy())
        losses_number.append(len(losses))
        losses[:,1] = losses[:,1]/np.max(losses[:,1]); 

        markerline, stemlines, baseline = plt.stem(losses[:,0], losses[:,1], linefmt='-', markerfmt='.', basefmt='r-')
        plt.setp(stemlines, 'color', RGB_colors[i])
    
    plt.xlim((min_loss, max_loss))
    plt.grid(True)
    plt.title('Spectrum')
    plt.xlabel('m/z')
    plt.ylabel('peak intensity')
    
    plt.show()
    
    print("Number of peaks: ", losses_number)


def plot_spectra_comparison(MS_measure,
                            spectra,
                            num_decimals,
                            ID1, ID2, 
                            min_mz = 5, 
                            max_mz = 500,
                            threshold = 0.01,
                            tol = 0.5,
                            method = 'cosine', #'molnet'
                            wordsim_cutoff = 0.5,
                            plot_molecules = False):
    """

    """
    from scipy import spatial
    #import matplotlib
    plot_colors = ['darkcyan', 'purple']#['seagreen', 'steelblue']#['darkcyan', 'firebrick']
    
    
    # Definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    spacing = 0.005
#    cbar_space = 0.1

    rect_wordsim = [left, bottom, width, height]
    rect_specx = [left, bottom + height + spacing, width, 0.2]
    rect_specy = [left + width, bottom, 0.2, height]
#    rect_cbar = [left, bottom, width, cbar_space]
    
    
    peaks1 = np.array(spectra[ID1].peaks.copy())
    peaks2 = np.array(spectra[ID2].peaks.copy())
#    peak_number.append(len(peaks))
#    max_intens = max(np.max(peaks1[:,1]), np.max(peaks2[:,1])) 
    peaks1[:,1] = peaks1[:,1]/np.max(peaks1[:,1])
    peaks2[:,1] = peaks2[:,1]/np.max(peaks2[:,1])
    
    # Remove peaks lower than threshold
    dictionary = [MS_measure.dictionary[x] for x in MS_measure.dictionary]
    select1 = np.where((peaks1[:,1] > threshold) & (peaks1[:,0] <= max_mz) & (peaks1[:,0] >= min_mz))[0]
    select2 = np.where((peaks2[:,1] > threshold) & (peaks2[:,0] <= max_mz) & (peaks2[:,0] >= min_mz))[0]
    
    # TODO: only include sub-function to create documents...
    MS_documents, MS_documents_intensity, _ = create_MS_documents([spectra[x] for x in [ID1,ID2]], 
                                                                 num_decimals = num_decimals, 
                                                                 peak_loss_words = ['peak_', 'loss_'],
                                                                 min_loss = 0, 
                                                                 max_loss = max_mz,
                                                                 ignore_losses = True)
    
    # Remove words/peaks that are not in dictionary
    select1 = np.array([x for x in select1 if MS_documents[0][x] in dictionary])    
    select2 = np.array([x for x in select2 if MS_documents[1][x] in dictionary])    
    
    peaks1 = peaks1[select1, :]
    peaks2 = peaks2[select2, :] 

    word_vectors1 = MS_measure.model_word2vec.wv[[MS_documents[0][x] for x in select1]]
    word_vectors2 = MS_measure.model_word2vec.wv[[MS_documents[1][x] for x in select2]]
    
    Csim_words = 1 - spatial.distance.cdist(word_vectors1, word_vectors2, 'cosine')
    Csim_words[Csim_words < wordsim_cutoff] = 0  # Remove values below cutoff
    

    # Plot spectra
    # -------------------------------------------------------------------------
    plt.figure(figsize=(12, 12))
    
    ax_wordsim = plt.axes(rect_wordsim)
    ax_wordsim.tick_params(direction='in', top=True, right=True)
    ax_specx = plt.axes(rect_specx)
    ax_specx.tick_params(direction='in', labelbottom=False)
    ax_specy = plt.axes(rect_specy)
    ax_specy.tick_params(direction='in', labelleft=False)
#    ax_cbar= fig.add_axes(rect_cbar)
    
    # Word similarity plot:
    # -------------------------------------------------------------------------
    data_x = []
    data_y = []
    data_z = []
    for i in range(len(select1)):
        for j in range(len(select2)):
            data_x.append(peaks1[i,0])
            data_y.append(peaks2[j,0])
            data_z.append(Csim_words[i,j])


    cm = plt.cm.get_cmap('PuRd') #PuRdYlGn('RdYlBu')
    
    ax_wordsim.scatter(data_x, data_y, s = 500*np.array(data_z)**2, c= data_z, cmap=cm, alpha=0.4) #s = 10000*np.array(data_z)**2 

    zero_pairs = MS_sim_classic.find_pairs(peaks1, peaks2, tol=tol, shift=0.0)
    
    if method == 'cosine':
        matching_pairs = zero_pairs
    elif method == 'molnet':
        shift = spectra[ID1].parent_mz - spectra[ID2].parent_mz
        nonzero_pairs = MS_sim_classic.find_pairs(peaks1, peaks2, tol=tol, shift=shift)
        matching_pairs = zero_pairs + nonzero_pairs
    else:
        print("Given method inkown.")
        
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
       
#    zero_pairs = find_pairs(peaks1, peaks2, tol=tol, shift=0.0)
#    zero_pairs = sorted(zero_pairs, key = lambda x: x[2], reverse = True)
#    idx1, idx2, _ = zip(*zero_pairs)
    idx1, idx2, _ = zip(*used_matches)
    cosine_x = []
    cosine_y = []
    for i in range(len(idx1)):
        cosine_x.append(peaks1[idx1[i],0])
        cosine_y.append(peaks2[idx2[i],0])
    ax_wordsim.scatter(cosine_x, cosine_y, s= 50, c = 'black')    

    ax_specx.vlines(peaks1[:,0], [0], peaks1[:,1], color=plot_colors[0])
    ax_specx.plot(peaks1[:,0], peaks1[:,1], '.')  # Stem ends
    ax_specx.plot([peaks1[:,0].max(), peaks1[:,0].min()], [0, 0],  '--')  # Middle bar
#    plt.title('Spectrum 1')
    
    ax_specy.hlines(peaks2[:,0], [0], peaks2[:,1], color=plot_colors[1])
    ax_specy.plot(peaks2[:,1], peaks2[:,0], '.')  # Stem ends
    ax_specy.plot([0, 0], [peaks2[:,0].min(), peaks2[:,0].max()], '--')  # Middle bar
#    plt.title('Spectrum 2')

   
    plt.show()
    
    # Plot molecules
    # -------------------------------------------------------------------------
    if plot_molecules:
        size = (200, 200)
        smiles = []  
        for i, candidate_id in enumerate([ID1, ID2]):
            smiles.append(spectra[candidate_id].metadata["smiles"])
            mol = Chem.MolFromSmiles(smiles[i])
            Draw.MolToMPL(mol, size=size, kekulize=True, wedgeBonds=True, imageType=None, fitImage=True)
            plt.xlim((0, 2.5))
            plt.ylim((0, 2.5))
    
    return Csim_words


def plot_smiles(query_id, spectra, MS_measure, num_candidates = 10,
                   sharex=True, labels=False, similarity_method = "centroid",
                   plot_type = "single", molnet_sim = None):
    """ Plot molecules for closest candidates
    
    """

    # Select chosen similarity methods
    if similarity_method == "centroid":
        candidates_idx = MS_measure.list_similars_ctr_idx[query_id, :num_candidates]
        candidates_sim = MS_measure.list_similars_ctr[query_id, :num_candidates]
    elif similarity_method == "pca":
        candidates_idx = MS_measure.list_similars_pca_idx[query_id, :num_candidates]
        candidates_sim = MS_measure.list_similars_pca[query_id, :num_candidates]
    elif similarity_method == "autoencoder":
        candidates_idx = MS_measure.list_similars_ae_idx[query_id, :num_candidates]
        candidates_sim = MS_measure.list_similars_ae[query_id, :num_candidates]
    elif similarity_method == "lda":
        candidates_idx = MS_measure.list_similars_lda_idx[query_id, :num_candidates]
        candidates_sim = MS_measure.list_similars_lda[query_id, :num_candidates]
    elif similarity_method == "lsi":
        candidates_idx = MS_measure.list_similars_lsi_idx[query_id, :num_candidates]
        candidates_sim = MS_measure.list_similars_lsi[query_id, :num_candidates]
    elif similarity_method == "doc2vec":
        candidates_idx = MS_measure.list_similars_d2v_idx[query_id, :num_candidates]
        candidates_sim = MS_measure.list_similars_d2v[query_id, :num_candidates]
    elif similarity_method == "molnet":
        if molnet_sim is None:
            print("If 'molnet' is chosen as similarity measure, molnet-matrix needs to be provided.")
            print("Use molnet_matrix function.")
        else:
            candidates_idx = molnet_sim[query_id,:].argsort()[-num_candidates:][::-1]
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
        
        if plot_type != "single":    # this will only work if there's no conflict with rdkit and pillow...       
            Chem.Draw.MolsToGridImage(molecules,legends=[mol.GetProp('_Name') for mol in molecules])
            
    elif isinstance(spectra, list):
        # Assume that it is then a list of Spectrum objects
        
        smiles = []  
        for i, candidate_id in enumerate(candidates_idx):
            smiles.append(spectra[candidate_id].metadata["smiles"])
            mol = Chem.MolFromSmiles(smiles[i])
#            mol.SetProp('_Name', smiles[i])
            if plot_type == 'single':
                Draw.MolToMPL(mol, size=size)
        
        if plot_type != "single":    # this will only work if there's no conflict with rdkit and pillow...       
            Chem.Draw.MolsToGridImage(molecules,legends=[mol.GetProp('_Name') for mol in molecules])


def top_score_histogram(spec_sim, mol_sim, 
                        score_threshold, 
                        num_candidates, 
                        num_bins = 50, 
                        filename = None):
    """ Plot histogram of Tanimoto scores (mol_sim) of top selected candidates based on 
    spectrum similarity scores (spec_sim). 
    
    spec_sim, mol_sim : to be calculated with evaluate_measure function.
    
    filename: str
        If not none: save figure to file with given name.
    """
    
    fig, ax = plt.subplots(figsize=(10,10))

    selection = np.where(spec_sim[:,1:] > score_threshold)
    X = mol_sim[selection[0], selection[1]+1].reshape(len(selection[0]))
    n, bins, patches = plt.hist(X, num_bins, weights=np.ones(len(X))/len(X), facecolor='blue', edgecolor='white', alpha=0.9)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.title("Tanimoto scores of TOP " + str(num_candidates-1) + " candidates with score > " + str(score_threshold))
    plt.xlabel("Tanimoto score (based on spectra annotated SMILES)")
    plt.ylabel("Percentage")

    test = spec_sim[:,1:].reshape(spec_sim.shape[0]*(spec_sim.shape[1]-1))
    test.sort()
    text1 = "Mean Tanimoto similarity is " + str(np.round(np.mean(mol_sim[selection[0], selection[1]+1]), 4))
    text2 = "Spectrum similarity score for TOP " + str(num_candidates-1) + ", top 20% is " + str(np.round(test[int(len(test)*0.8)], 4))
    text3 = ""
    plt.text(0, 0.96*np.max(n), text1, fontsize=12, backgroundcolor = "white")
    plt.text(0, 0.91*np.max(n), text2, fontsize=12, backgroundcolor = "white")
    plt.text(0, 0.86*np.max(n), text3, fontsize=12, backgroundcolor = "white")

    if filename is not None:
        plt.savefig(filename, dpi=600)
    
    plt.show()


def similarity_histogram(M_sim, M_sim_ref, 
                         score_threshold,
                         num_bins = 50, 
                         exclude_IDs = None,
                         filename = None,
                         exclude_diagonal = True):
    """ Plot histogram of Reference scores (from matrix M_sim_ref) for all pairs 
    with similarity score >= score_threshold. 
    
    M_sim: numpy array
        Matrix with similarities between pairs.
    M_sim_ref: numpy array
        Matrix with reference scores/similarity values between pairs.
    
    filename: str
        If not none: save figure to file with given name.
    """
    fig, ax = plt.subplots(figsize=(10,10))
    
    if exclude_IDs is not None:
        # Remove elements in exclude_IDs array
        IDs = np.arange(0,M_sim.shape[0])
        M_sim = np.delete(M_sim, IDs[exclude_IDs], axis=0)
        M_sim = np.delete(M_sim, IDs[exclude_IDs], axis=1)
        M_sim_ref = np.delete(M_sim_ref, IDs[exclude_IDs], axis=0)
        M_sim_ref = np.delete(M_sim_ref, IDs[exclude_IDs], axis=1)
        
        IDs = np.delete(IDs, IDs[exclude_IDs])
        
    if exclude_diagonal == True:
        # Exclude diagonal
        M_sim[np.arange(0,M_sim.shape[0]), np.arange(0,M_sim.shape[0])] = score_threshold - 1
    
    selection = np.where(M_sim[:,:] >= score_threshold)
    X = M_sim_ref[selection].reshape(len(selection[0]))
    n, bins, patches = plt.hist(X, num_bins, weights=np.ones(len(X))/len(X), facecolor='blue', edgecolor='white', alpha=0.9)
    plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
    plt.title("Total reference scores for all candidates with similarity score > " + str(score_threshold), fontsize = 16)
#    plt.set_xticklabels(rotation=0, fontsize=12)
    ax.tick_params(labelsize=12)   
    plt.xlabel("Reference score.", fontsize = 14)
    plt.ylabel("Percentage", fontsize = 14)

    if filename is not None:
        plt.savefig(filename, dpi=600)
    
    plt.show()

    return n, bins


def compare_best_results(spectra_dict, 
                         spectra,
                         MS_measure,
                         tanimoto_sim,
                         molnet_sim,
                         num_candidates = 25,
                         similarity_method = ["centroid"]):
    """ Compare spectra-based similarity with smile-based similarity and mol.networking.

    Args:
    -------
    spectra_dict: dict
        Dictionary containing all spectra peaks, losses, metadata.
    MS_measure: object
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
    
    candidates_idx = np.zeros((num_candidates), dtype=int)
    candidates_sim = np.zeros((num_candidates))
    for k, method in enumerate(similarity_method):
        for i in range(num_spectra):
            # Select chosen similarity methods
            if method == "centroid":
                candidates_idx = MS_measure.list_similars_ctr_idx[i, :num_candidates]
            elif method == "pca":
                candidates_idx = MS_measure.list_similars_pca_idx[i, :num_candidates]
            elif method == "autoencoder":
                candidates_idx = MS_measure.list_similars_ae_idx[i, :num_candidates]
            elif method == "lda":
                candidates_idx = MS_measure.list_similars_lda_idx[i, :num_candidates]
            elif method == "lsi":
                candidates_idx = MS_measure.list_similars_lsi_idx[i, :num_candidates]
            elif method == "doc2vec":
                candidates_idx = MS_measure.list_similars_d2v_idx[i, :num_candidates]
            else:
                print("Chosen similarity measuring method not found.")

            candidates_sim = tanimoto_sim[i, candidates_idx]
            spec_best[i,:,k] = candidates_sim

    for i in range(num_spectra):        
        # Compare to molecular networking score
        molnet_candidates_idx = molnet_sim[i,:].argsort()[-num_candidates:][::-1]
        molnet_candidates_sim = tanimoto_sim[i, molnet_candidates_idx]
        
        # Compare to maximum possible Tanimoto score
        tanimoto_candidates_idx = tanimoto_sim[i,:].argsort()[-num_candidates:][::-1]
        tanimoto_candidates_sim = tanimoto_sim[i, tanimoto_candidates_idx]     
                
        mol_best[i,:] = molnet_candidates_sim
        tanimoto_best[i,:] = tanimoto_candidates_sim

    labels = []
    avg_best_scores = []
    labels.append('Tanimoto (best)')
    avg_best_scores.append(np.mean(tanimoto_best, axis=0))
    labels.append('Mol.networking score')
    avg_best_scores.append(np.mean(mol_best, axis=0))
    
    for k, method in enumerate(similarity_method):
        labels.append('Spectrum similarity (' + method + ')')
        avg_best_scores.append(np.mean(spec_best[:,:,k], axis=0))

    return avg_best_scores, labels


def plot_best_results(avg_best_scores,  
                      labels,
                      tanimoto_sim,
                      filename = None):
    """ Plot best candidate average results.
    """
    
    num_candidates = len(avg_best_scores[0])

    # These are the colors that will be used in the plot
    color_sequence = ['#003f5c','#882556', '#D65113', '#ffa600', '#58508d', '#bc5090', 
                      '#2651d1', '#2f4b7c', '#ff6361', '#a05195', '#d45087'] 
    markers = ['^', 'v', 'o']#, 'v']
                      
    fig, ax = plt.subplots(figsize=(10,16))
    plt.subplot(211)
    for i, label in enumerate(labels):
        plt.plot(np.arange(0,num_candidates), avg_best_scores[i], 
                 label=label, linewidth=1, markersize=12,
                 marker=markers[min(i,len(markers)-1)], linestyle=':', color=color_sequence[i])
    
    # Add mean Tanimoto baseline
    plt.plot(np.arange(0,num_candidates), np.mean(tanimoto_sim)*np.ones((num_candidates)),
             label='Average Tanimoto similarity', linewidth=2, color='black')    

    plt.legend(fontsize = 12)
    plt.xticks(range(0, num_candidates), fontsize=12)
    plt.xlabel("Top 'x' candidates")
    plt.ylabel("Average Tanimoto score.")
    
#    fig, ax = plt.subplots(figsize=(10,8))
    plt.subplot(212)
    for i, label in enumerate(labels[1:], start=1):
        plt.plot(np.arange(1,num_candidates), avg_best_scores[i][1:]/avg_best_scores[0][1:], 
                 label=label+'/Tanimoto max', linewidth=1, markersize=12,
                 marker=markers[min(i,len(markers)-1)], linestyle=':', color=color_sequence[i])

    # Add mean Tanimoto baseline
    plt.plot(np.arange(1,num_candidates), np.mean(tanimoto_sim)*np.ones((num_candidates-1))/avg_best_scores[0][1:],
             label='Baseline: random candidate selection', linewidth=2, color='black')  
    
    plt.legend(fontsize = 12)
    plt.xticks(range(1, num_candidates), fontsize=12)
    plt.xlabel("Top 'x' candidates")
    plt.ylabel("Fraction of max. possible average Tanimoto score")
    
    if filename is not None:
        plt.savefig(filename, dpi=600)
        
        
def MS_similarity_network(MS_measure, 
                          similarity_method="centroid", 
                          link_method = "single", 
                          filename="MS_word2vec_test.graphml", 
                          cutoff = 0.7,
                          max_links = 10,
                          extern_matrix = None):
    """ Built network from closest connections found
        Using networkx
        
    Args:
    -------
    MS_measure: SimilarityMeasures object   
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
        list_similars_idx = MS_measure.list_similars_ctr_idx
        list_similars = MS_measure.list_similars_ctr
    elif similarity_method == "pca":
        list_similars_idx = MS_measure.list_similars_pca_idx
        list_similars = MS_measure.list_similars_pca
    elif similarity_method == "autoencoder":
        list_similars_idx = MS_measure.list_similars_ae_idx
        list_similars = MS_measure.list_similars_ae
    elif similarity_method == "lda":
        list_similars_idx = MS_measure.list_similars_lda_idx
        list_similars = MS_measure.list_similars_lda
    elif similarity_method == "lsi":
        list_similars_idx = MS_measure.list_similars_lsi_idx
        list_similars = MS_measure.list_similars_lsi
    elif similarity_method == "doc2vec":
        list_similars_idx = MS_measure.list_similars_d2v_idx
        list_similars = MS_measure.list_similars_d2v
    elif similarity_method == "extern":
        num_candidates = MS_measure.list_similars_ctr_idx.shape[1]
        list_similars = np.zeros((MS_measure.list_similars_ctr_idx.shape))
        list_similars_idx = np.zeros((MS_measure.list_similars_ctr_idx.shape)).astype(int)
        
        if extern_matrix is None:
            print("Need externally derived similarity matrix to proceed.")
        else:
            if extern_matrix.shape[0] == extern_matrix.shape[1] == list_similars.shape[0]: 
                for i in range(0, list_similars.shape[0]):
                    list_similars_idx[i,:] = (-extern_matrix[i]).argsort()[:num_candidates].astype(int)
                    list_similars[i,:] = extern_matrix[i, list_similars_idx[i,:]]
            else:
                print("External matrix with similarity scores does not have the right dimensions.")
    else:
        print("Wrong method given. Or method not yet implemented in function.")

        
    if max_links > (list_similars_idx.shape[1] - 1):
        print("Maximum number of candidate links exceeds dimension of 'list_similars'-array.")

    
    dimension = list_similars_idx.shape[0]
    
    # Initialize network graph
    import networkx as nx
    MSnet = nx.Graph()               
    MSnet.add_nodes_from(np.arange(0, dimension))   
       
    for i in range(0, dimension):      
        idx = np.where(list_similars[i,:] > cutoff)[0][:max_links]
        if link_method == "single":
            new_edges = [(i, int(list_similars_idx[i,x]), float(list_similars[i,x])) for x in idx if list_similars_idx[i,x] != i]
        elif link_method == "mutual":
            new_edges = [(i, int(list_similars_idx[i,x]), float(list_similars[i,x])) for x in idx if list_similars_idx[i,x] != i if i in list_similars_idx[x,:]]
        else:
            print("Link method not kown")
        MSnet.add_weighted_edges_from(new_edges)
        
    # Export graph for drawing (e.g. using Cytoscape)
    nx.write_graphml(MSnet, filename)
    print("Network stored as graphml file under: ", filename)
