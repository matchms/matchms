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
import helper_functions as functions
import fnmatch
import copy
import numpy as np
from scipy.optimize import curve_fit
from scipy.optimize import linear_sum_assignment
import random
import pandas as pd

from pyteomics import mgf

from openbabel import openbabel as ob
from openbabel import pybel

from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.Chem import AllChem

# Add multi core parallelization
from concurrent.futures import ThreadPoolExecutor, as_completed


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
    def __init__(self, min_frag = 0.0, max_frag = 1000.0,
                 min_loss = 10.0, max_loss = 200.0,
                 min_intensity_perc = 0.0,
                 exp_intensity_filter = 0.01,
                 peaks_per_mz = 20/200,
                 min_peaks = 10,
                 max_peaks = None,
                 merge_energies = True,
                 merge_ppm = 10,
                 replace = 'max'):

        self.id = []
        self.filename = []
        self.peaks = []
        self.precursor_mz = []
        self.parent_mz = []
        self.metadata = {}
        self.family = None
        self.annotations = []
        self.smiles = []
        self.inchi = []
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
        self.peaks_per_mz = peaks_per_mz
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
        peaks = process_peaks(peaks, self.min_frag, self.max_frag,
                              self.min_intensity_perc, self.exp_intensity_filter,
                              self.min_peaks, self.max_peaks)
        
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

        peaks = list(zip(spectrum_mgf['m/z array'], spectrum_mgf['intensity array']))
        if len(peaks) >= self.min_peaks:
            peaks = process_peaks(peaks, self.min_frag, self.max_frag,
                                  self.min_intensity_perc, self.exp_intensity_filter,
                                  self.min_peaks, self.max_peaks)
        
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


## --------------------------------------------------------------------------------------------------
## ---------------------------- Spectrum processing functions ---------------------------------------
## --------------------------------------------------------------------------------------------------

        
def dict_to_spectrum(spectra_dict): 
    """ Create spectrum object from spectra_dict.
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


def process_peaks(peaks, min_frag, max_frag, 
                  min_intensity_perc,
                  exp_intensity_filter,
                  min_peaks,
                  max_peaks = None):
    """ Process peaks
    
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
   
    """
    def exponential_func(x, a, b):
        return a*np.exp(-b*x)
   
    if isinstance(peaks, list):
        peaks = np.array(peaks)
        if peaks.shape[1] != 2:
            print("Peaks were given in unexpected format...")
    
    if min_intensity_perc > 0:
        intensity_thres = np.max(peaks[:,1]) * min_intensity_perc/100
        keep_idx = np.where((peaks[:,0] > min_frag) & (peaks[:,0] < max_frag) & (peaks[:,1] > intensity_thres))[0]
        if (len(keep_idx) < min_peaks): 
            # If not enough peaks selected, try again without intensity threshold
            keep_idx2 = np.where((peaks[:,0] > min_frag) & (peaks[:,0] < max_frag))[0]
            peaks = peaks[keep_idx2,:]
        else:
            peaks = peaks[keep_idx,:]
    else: 
        keep_idx = np.where((peaks[:,0] > min_frag) & (peaks[:,0] < max_frag))[0]
        peaks = peaks[keep_idx,:]

    if (exp_intensity_filter is not None) and len(peaks) > 2*min_peaks:
        # Fit exponential to peak intensity distribution 
        num_bins = 100  # number of bins for histogram

        # Ignore highest peak for further analysis 
        peaks2 = peaks.copy()
        peaks2[np.where(peaks2[:,1] == np.max(peaks2[:,1])),:] = 0
        hist, bins = np.histogram(peaks2[:,1], bins=num_bins)
        start = np.where(hist == np.max(hist))[0][0]  # Take maximum intensity bin as starting point
        last = int(num_bins/2)
        x = bins[start:last]
        y = hist[start:last]
        try:
            popt, pcov = curve_fit(exponential_func, x , y, p0=(peaks.shape[0], 1e-4)) 
            threshold = -np.log(exp_intensity_filter)/popt[1]
        except RuntimeError:
            print("RuntimeError for ", len(peaks), " peaks. Use mean intensity as threshold.")
            threshold = np.mean(peaks2[:,1])
        except TypeError:
            print("Unclear TypeError for ", len(peaks), " peaks. Use mean intensity as threshold.")
            print(x, "and y: ", y)
            threshold = np.mean(peaks2[:,1])

        keep_idx = np.where(peaks[:,1] > threshold)[0]
        if len(keep_idx) < min_peaks:
            peaks = peaks[np.lexsort((peaks[:,0], peaks[:,1])),:][-min_peaks:]
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
                 peaks_per_mz = 20/200,
                 min_peaks = 10,
                 max_peaks = None,
                 merge_energies = True,
                 merge_ppm = 10,
                 replace = 'max',
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
            
            if peaks_per_mz != 0:
                # TODO: remove following BAD BAD hack:
                # Import first (acutally only needed is PRECURSOR MASS)
                spec = Spectrum(min_frag = min_frag, 
                        max_frag = max_frag,
                        min_loss = min_loss, 
                        max_loss = max_loss,
                        min_intensity_perc = min_intensity_perc,
                        exp_intensity_filter = None,
                        peaks_per_mz = peaks_per_mz,
                        min_peaks = min_peaks,
                        max_peaks = max_peaks,
                        merge_energies = merge_energies,
                        merge_ppm = merge_ppm,
                        replace = replace)
                
                # Load spectrum data from file:
                spec.read_spectrum(path_data, filename, i)
                
                # Scale the min_peak filter
                def min_peak_scaling(x, A, B):
                    return int(A + B * x)
                
                min_peaks_scaled = min_peak_scaling(spec.precursor_mz, min_peaks, peaks_per_mz)        
            else:
                min_peaks_scaled = min_peaks
            
            spectrum = Spectrum(min_frag = min_frag, 
                                max_frag = max_frag,
                                min_loss = min_loss, 
                                max_loss = max_loss,
                                min_intensity_perc = min_intensity_perc,
                                exp_intensity_filter = exp_intensity_filter,
                                peaks_per_mz = peaks_per_mz,
                                min_peaks = min_peaks_scaled,
                                max_peaks = max_peaks,
                                merge_energies = merge_energies,
                                merge_ppm = merge_ppm,
                                replace = replace)
            
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

        MS_documents, MS_documents_intensity, spectra_metadata = create_MS_documents(spectra, num_decimals, 
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
                 num_decimals = 3,
                 min_frag = 0.0, max_frag = 1000.0,
                 min_loss = 10.0, max_loss = 200.0,
                 min_intensity_perc = 0.0,
                 exp_intensity_filter = 0.01,
                 peaks_per_mz = 20/200,
                 min_peaks = 10,
                 max_peaks = None,
                 peak_loss_words = ['peak_', 'loss_']):        
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
        of the highest peak intensity (Default = 0.0, essentially meaning: OFF).
    exp_intensity_filter: float
        Filter out peaks by applying an exponential fit to the intensity histogram.
        Intensity threshold will be set at where the exponential function will have dropped 
        to exp_intensity_filter (Default = 0.01).
    peaks_per_mz: float
        Factor to describe linear increase of mininum peaks per spectrum with increasing
        parentmass. Formula is: int(min_peaks + peaks_per_mz * parentmass).
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
    
    spectra = []
    spectra_dict = {}
    MS_documents = []
    MS_documents_intensity = []
    collect_new_data = True
        
    if file_json is not None:
        try: 
            spectra_dict = functions.json_to_dict(file_json)
            spectra_metadata = pd.read_csv(file_json[:-5] + "_metadata.csv")
            print("Spectra json file found and loaded.")
            spectra = dict_to_spectrum(spectra_dict)
            collect_new_data = False
            
            with open(file_json[:-4] + "txt", "r") as f:
                for line in f:
                    line = line.replace('"', '').replace("'", "").replace("[", "").replace("]", "").replace("\n", "")
                    MS_documents.append(line.split(", "))
                    
            with open(file_json[:-5] + "_intensity.txt", "r") as f:
                for line in f:
                    line = line.replace("[", "").replace("]", "")
                    MS_documents_intensity.append([int(x) for x in line.split(", ")])
                
        except FileNotFoundError: 
            print("Could not find file ", file_json) 
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

                # Scale the min_peak filter
                if spec is not None:
                    min_peaks_scaled = min_peak_scaling(spec['params']['pepmass'][0], min_peaks, peaks_per_mz)
                
                    spectrum = Spectrum(min_frag = min_frag, 
                                        max_frag = max_frag,
                                        min_loss = min_loss, 
                                        max_loss = max_loss,
                                        min_intensity_perc = min_intensity_perc,
                                        exp_intensity_filter = exp_intensity_filter,
                                        peaks_per_mz = peaks_per_mz,
                                        min_peaks = min_peaks_scaled,
                                        max_peaks = max_peaks)
                    
                    id = i #spec.spectrum_id
                    spectrum.read_spectrum_mgf(spec, id)
                    spectrum.get_losses
        
                    # Calculate losses:
                    if len(spectrum.peaks) >= min_peaks: 
                        spectrum.get_losses()
                    
                    # Collect in form of list of spectrum objects
                    spectra.append(spectrum)
                    
                else:
                    print("Found empty spectra for ID: ", i)
            
        # Filter out spectra with few peaks
        min_peaks_absolute = min_peaks
        num_spectra_initial = len(spectra)
        spectra = [copy.deepcopy(x) for x in spectra if len(x.peaks) >= min_peaks_absolute]
        print("Take ", len(spectra), "spectra out of ", num_spectra_initial, ".")

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

        # Create documents from peaks (and losses)
        MS_documents, MS_documents_intensity, spectra_metadata = create_MS_documents(spectra, num_decimals, 
                                                                                             peak_loss_words, 
                                                                                             min_loss, max_loss)

        # Save collected data
        if collect_new_data == True:
            spectra_metadata.to_csv(file_json[:-5] + "_metadata.csv", index=False)
            
            functions.dict_to_json(spectra_dict, file_json)     
            # Store documents
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
            else:
                print("No losses detected for: ", spec_id, spectrum.id)

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


def get_mol_fingerprints(spectra_dict, method = "daylight"):
    """ Calculate molecule fingerprints based on given inchi or smiles (using RDkit).
    
    Output: exclude_IDs list with spectra that had no inchi or smiles or problems when deriving fingerprint
    
    Args:
    --------
    spectra_dict: dict
        Dictionary containing all spectrum objects information (peaks, losses, metadata...).
    method: str
        Determine method for deriving molecular fingerprints. Supported choices are 'daylight', 
        'morgan1', 'morgan2', 'morgan3'.
    """
    
    # If spectra is given as a dictionary
    keys = []
    exclude_IDs = []
    molecules = []
    for key, value in spectra_dict.items():
        if "inchi" in value["metadata"]:
            mol = 1
            keys.append(key) 
            try:
                mol = Chem.MolFromInchi(value["metadata"]["inchi"], 
                                                   sanitize=True, 
                                                   removeHs=True, 
                                                   logLevel=None, 
                                                   treatWarningAsError=True)
            except:
                print('error handling inchi:', value["metadata"]["inchi"])
                mol = 0
        
        if "smiles" in value or mol == 0:  # Smiles but no InChikey or inchi handling failed
            keys.append(key) 
            try:
                mol = Chem.MolFromSmiles(value["smiles"])
            except:
                print('error handling smiles:', value["smiles"])
                mol = 0
        if mol == 0 or mol == 1:
            print("No smiles found for spectra ", key, ".")
            mol = Chem.MolFromSmiles("H20") # Just have some water when you get stuck
            exclude_IDs.append(int(value["id"]))
        molecules.append(mol)   
        
    fingerprints = []
    for i in range(len(molecules)):
        if molecules[i] is None:
            print("Problem with molecule " + str(spectra_dict[keys[i]]["id"]))
            fp = 0
            exclude_IDs.append(int(spectra_dict[keys[i]]["id"]))
        else:
            if method == "daylight":
                fp = FingerprintMols.FingerprintMol(molecules[i])
            elif method == "morgan1":
                fp = AllChem.GetMorganFingerprint(molecules[i],1)
            elif method == "morgan2":
                fp = AllChem.GetMorganFingerprint(molecules[i],2)
            elif method == "morgan3":
                fp = AllChem.GetMorganFingerprint(molecules[i],3)
            else:
                print("Unkown fingerprint method given...")

        fingerprints.append(fp)
    
    return molecules, fingerprints, exclude_IDs


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

    zero_pairs = find_pairs(peaks1, peaks2, tol=tol, shift=0.0)
    
    if method == 'cosine':
        matching_pairs = zero_pairs
    elif method == 'molnet':
        shift = spectra[ID1].parent_mz - spectra[ID2].parent_mz
        nonzero_pairs = find_pairs(peaks1, peaks2, tol=tol, shift=shift)
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
