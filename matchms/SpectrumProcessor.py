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
        self.filter_order = [x.__name__ for x in ALL_FILTERS]
        if predefined_pipeline is not None:
            if not isinstance(predefined_pipeline, str):
                raise ValueError("Predefined pipeline parameter should be a string")
            if predefined_pipeline not in PREDEFINED_PIPELINES:
                raise ValueError(f"Unknown processing pipeline '{predefined_pipeline}'. Available pipelines: {list(PREDEFINED_PIPELINES.keys())}")
            for fname in PREDEFINED_PIPELINES[predefined_pipeline]:
                self.add_matchms_filter(fname)

    def add_filter(self, filter_function):
        """Add a filter to the processing pipeline. Takes both matchms filter names (and parameters)
        as wel as custom-made functions.
        """
        if isinstance(filter_function, str):
            self.add_matchms_filter(filter_function)
        elif isinstance(filter_function, (tuple, list)) and isinstance(filter_function[0], str):
            self.add_matchms_filter(filter_function)
        else:
            self.add_custom_filter(filter_function[0], filter_function[1])

    def add_matchms_filter(self, filter_spec):
        """
        Add a filter to the processing pipeline.

        Parameters
        ----------
        filter_spec : str or tuple
            Name of the filter function to add, or a tuple where the first element is the name of the 
            filter function and the second element is a dictionary containing additional arguments for the function.
        """
        if isinstance(filter_spec, str):
            if filter_spec not in FILTER_FUNCTIONS:
                raise ValueError("Unknown filter type. Should be known filter name or function.")
            filter_func = FILTER_FUNCTIONS[filter_spec]
        elif isinstance(filter_spec, (tuple, list)):
            filter_name, filter_args = filter_spec
            if filter_name not in FILTER_FUNCTIONS:
                raise ValueError("Unknown filter type. Should be known filter name or function.")
            filter_func = partial(FILTER_FUNCTIONS[filter_name], **filter_args)
            filter_func.__name__ = FILTER_FUNCTIONS[filter_name].__name__
        else:
            raise TypeError("filter_spec should be a string or a tuple/list")

        self.filters.append(filter_func)
        # Sort filters according to their order in self.filter_order
        self.filters.sort(key=lambda f: self.filter_order.index(f.__name__))

    def add_custom_filter(self, filter_function, filter_params=None):
        """
        Add a custom filter function to the processing pipeline.

        Parameters
        ----------
        filter_function: callable
            Custom function to add to the processing pipeline.
            Expects a function that takes a matchms Spectrum object as input and returns a Spectrum object
            (or None).
            Regarding the order of execution: the added filter will be executed where it is introduced to the
            processing pipeline.
        filter_params: dict
            If needed, add dictionary with all filter parameters. Default is set to None.
        """
        if not callable(filter_function):
            raise TypeError("Expected callable filter function.")
        filter_position = 0
        for filter_func in self.filters[::-1]:
            if filter_func.__name__ in self.filter_order:
                filter_position = self.filter_order.index(filter_func.__name__)
        self.filter_order.insert(filter_position + 1, filter_function.__name__)
        if filter_params is None:
            self.filters.append(filter_function)
        else:
            filter_func = partial(filter_function, **filter_params)
            filter_func.__name__ = filter_function.__name__
            self.filters.append(filter_func)

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
            processing_report = ProcessingReport()

        processed_spectrums = []
        for s in tqdm(spectrums, disable=(not progress_bar), desc="Processing spectrums"):
            if s is None:
                continue  # empty spectra will be discarded
            processed_spectrum = self.process_spectrum(s)
            if create_report:
                processing_report.add_to_report(s, processed_spectrum)
            if processed_spectrum is not None:
                processed_spectrums.append(processed_spectrum)

        if create_report:
            return processed_spectrums, processing_report
        return processed_spectrums

    @property
    def processing_steps(self):
        return [x.__name__ for x in self.filters]

    def __str__(self):
        summary_string = "SpectrumProcessor\nProcessing steps:\n - "
        return summary_string + "\n - ".join(self.processing_steps)


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
       "normalize_intensities",
    ]
FULLY_ANNOTATED_PROCESSING = DEFAULT_FILTERS \
    + [("require_correct_ionmode", {"ion_mode_to_keep": "both"}),
       "require_parent_mass_match_smiles",
       "require_valid_annotation",
    ]

PREDEFINED_PIPELINES = {
    "minimal": MINIMAL_FILTERS,
    "basic": BASIC_FILTERS,
    "default": DEFAULT_FILTERS,
    "fully_annotated": FULLY_ANNOTATED_PROCESSING,
}


class ProcessingReport:
    """Class to keep track of spectrum changes during filtering.
    """
    def __init__(self):
        self.fields = ["precursor_mz", "parent_mass", "ionmode", "charge", "smiles", "inchikey", "inchi"]
        self.counter_changed = defaultdict(int)
        self.counter_added = defaultdict(int)
        self.counter_removed_spectrums = 0
        self.counter_number_processed = 0

    def add_to_report(self, spectrum_old, spectrum_new):
        """Add changes between spectrum_old and spectrum_new to the report.
        """
        self.counter_number_processed += 1
        if spectrum_new is None:
            self.counter_removed_spectrums += 1
        else:
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

    def __str__(self):
        report_str = f"""\
----- Spectrum Processing Report -----
Number of spectrums processed: {self.counter_number_processed}
Number of spectrums removed: {self.counter_removed_spectrums}
Changes during processing:
{str(self.to_dataframe())}
"""
        return report_str

    def __repr__(self):
        return f"Report({self.counter_number_processed},\
        {self.counter_removed_spectrums},\
        {dict(self.counter_changed)},\
        {dict(self.counter_added)})"
