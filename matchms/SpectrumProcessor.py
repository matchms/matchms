from collections import defaultdict
from functools import partial
import pandas as pd
from tqdm import tqdm
import matchms.filtering as msfilters


class SpectrumProcessor:
    """
    A class to process spectra using a series of filters. 

    The class enables a user to define a custom spectrum processing workflow by setting multiple
    flags and parameters.

    Parameters
    ----------
    predefined_pipeline : str
        Name of a predefined processing pipeline. Options: 'minimal', 'basic', 'default',
        'fully_annotated', or None. Default is 'default'.
    """

    def __init__(self, predefined_pipeline='default'):
        self.filters = []
        if predefined_pipeline is not None :
            if predefined_pipeline not in PREDEFINED_PIPELINES:
                raise ValueError(f"Unknown processing pipeline '{predefined_pipeline}'. Available pipelines: {list(PREDEFINED_PIPELINES.keys())}")
            for fname in PREDEFINED_PIPELINES[predefined_pipeline]:
                self.add_filter(fname)

    def add_filter(self, filter_spec):
        """
        Add a filter to the processing pipeline.

        Parameters
        ----------
        filter_spec : str or tuple
            Name of the filter function to add, or a tuple where the first element is the name of the 
            filter function and the second element is a dictionary containing additional arguments for the function.
        """
        if isinstance(filter_spec, str):
            filter_func = FILTER_FUNCTIONS[filter_spec]
        elif isinstance(filter_spec, tuple):
            filter_name, filter_args = filter_spec
            filter_func = partial(FILTER_FUNCTIONS[filter_name], **filter_args)
            filter_func.__name__ = FILTER_FUNCTIONS[filter_name].__name__
        else:
            raise TypeError("filter_spec should be a string or a tuple")

        self.filters.append(filter_func)
        # Sort filters according to their order in ALL_FILTERS
        self.filters.sort(key=lambda f: [x.__name__ for x in ALL_FILTERS].index(f.__name__))
        return self

    def process_spectrum(self, spectrum):
        """
        Process the given spectrum with all filters in the processing pipeline.

        Parameters
        ----------
        spectrum : Spectrum
            The spectrum to process.
        
        Returns
        -------
        Spectrum
            The processed spectrum.
        """
        if self.filters == []:
            raise TypeError("No filters to process")
        for filter_func in self.filters:
            spectrum = filter_func(spectrum)
            if spectrum is None:
                break
        return spectrum

    def process_spectrums(self, spectrums: list,
                          create_report=False,
                          progress_bar=True,
                          ):
        """
        Process a list of spectrums with all filters in the processing pipeline.

        Parameters
        ----------
        spectrums : list[Spectrum]
            The spectrums to process.
        create_report: book, optional
            Creates and outputs a report of the main changes during processing.
            The report will be returned as pandas DataFrame. Default is set to False.
        progress_bar : bool, optional
            Displays progress bar if set to True. Default is True.

        Returns
        -------
        Spectrums
            List containing the processed spectrums.
        """
        if create_report:
            filtering_report = FilteringReport()

        processed_spectrums = []
        for s in tqdm(spectrums, disable=(not progress_bar), desc="Processing spectrums"):
            if s is None:
                continue  # empty spectra will be discarded
            processed_spectrum = self.process_spectrum(s)
            if create_report:
                filtering_report.add_to_report(s, processed_spectrum)
            processed_spectrums.append(processed_spectrum)

        if create_report:
            processing_report = filtering_report.to_dataframe()
            return processed_spectrums, processing_report
        return processed_spectrums


# List all filters in a functionally working order
ALL_FILTERS = [msfilters.make_charge_int,
               msfilters.add_compound_name,
               msfilters.derive_adduct_from_name,
               msfilters.derive_formula_from_name,
               msfilters.clean_compound_name,
               msfilters.interpret_pepmass,
               msfilters.add_precursor_mz,
               msfilters.add_retention_index,
               msfilters.add_retention_time,
               msfilters.derive_ionmode,
               msfilters.correct_charge,
               # msfilters.derive_adduct_from_name,  # run again? Or improve those filters?
               # msfilters.derive_formula_from_name,  # run again? Or improve those filters?
               msfilters.require_precursor_mz,
               msfilters.add_parent_mass,
               msfilters.harmonize_undefined_inchikey,
               msfilters.harmonize_undefined_inchi,
               msfilters.harmonize_undefined_smiles,
               msfilters.repair_inchi_inchikey_smiles,
               msfilters.repair_parent_mass_match_smiles_wrapper,
               msfilters.require_correct_ionmode,
               msfilters.require_precursor_below_mz,
               msfilters.require_parent_mass_match_smiles,
               msfilters.require_valid_annotation,
               msfilters.normalize_intensities,
               msfilters.select_by_intensity,
               msfilters.select_by_mz,
               msfilters.select_by_relative_intensity,
               msfilters.remove_peaks_around_precursor_mz,
               msfilters.remove_peaks_outside_top_k,
               msfilters.require_minimum_number_of_peaks,
               msfilters.require_minimum_of_high_peaks,
               msfilters.add_fingerprint,
               msfilters.add_losses,
              ]

FILTER_FUNCTIONS = {x.__name__: x for x in ALL_FILTERS}

MINIMAL_FILTERS = ["make_charge_int",
                   "interpret_pepmass",
                   "derive_ionmode",
                   "correct_charge",
                   ]
BASIC_FILTERS = MINIMAL_FILTERS \
    + ["add_compound_name",
       "derive_adduct_from_name",
       "derive_formula_from_name",
       "clean_compound_name",
       "add_precursor_mz",
    ]
DEFAULT_FILTERS = BASIC_FILTERS \
    + ["require_precursor_mz",
       "add_parent_mass",
       "harmonize_undefined_inchikey",
       "harmonize_undefined_inchi",
       "harmonize_undefined_smiles",
       "repair_inchi_inchikey_smiles",
       "repair_parent_mass_match_smiles_wrapper",
       "require_correct_ionmode",
       "normalize_intensities",
    ]
FULLY_ANNOTATED_PROCESSING = DEFAULT_FILTERS \
    + ["require_parent_mass_match_smiles",
       "require_valid_annotation",
    ]

PREDEFINED_PIPELINES = {
    "minimal": MINIMAL_FILTERS,
    "basic": BASIC_FILTERS,
    "default": DEFAULT_FILTERS,
    "fully_annotated": FULLY_ANNOTATED_PROCESSING,
}


class FilteringReport:
    """Class to keep track of spectrum changes during filtering.
    """
    def __init__(self):
        self.fields = ["precursor_mz", "parent_mass", "ionmode", "charge", "smiles", "inchikey", "inchi"]
        self.counter_changed = defaultdict(int)
        self.counter_added = defaultdict(int)
        self.counter_removed_spectrums = 0

    def add_to_report(self, spectrum_old, spectrum_new):
        """Add changes between spectrum_old and spectrum_new to the report.
        """
        if spectrum_new is None:
            self.counter_removed_spectrums += 1
        for field in self.fields:
            if spectrum_old.get(field) != spectrum_new.get(field):
                if spectrum_old.get(field) is None:
                    self.counter_added[field] += 1
                else:
                    self.counter_changed[field] += 1

    def to_dataframe(self):
        """Create Pandas DataFrame Report of counted spectrum changes."""
        changes = pd.DataFrame(self.counter_changed.items(),
                               columns=["field", "number of changes"])
        additions = pd.DataFrame(self.counter_added.items(),
                                 columns=["field", "number of additions"])
        processing_report = pd.merge(changes, additions, how="outer", on="field")
        processing_report = processing_report.set_index("field").fillna(0)
        return processing_report.astype(int)
