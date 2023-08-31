from collections import defaultdict
from functools import partial
from typing import Dict, Optional, Tuple, Union, Callable
import numpy as np
import pandas as pd
from tqdm import tqdm
from matchms import Spectrum
from matchms.filtering.filter_order_and_default_pipelines import (
    ALL_FILTERS, FILTER_FUNCTION_NAMES, PREDEFINED_PIPELINES)


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

    def __init__(self, predefined_pipeline: Optional[str] = 'default'):
        self.filters = []
        self.filter_order = [x.__name__ for x in ALL_FILTERS]
        if predefined_pipeline is not None:
            if not isinstance(predefined_pipeline, str):
                raise ValueError("Predefined pipeline parameter should be a string")
            if predefined_pipeline not in PREDEFINED_PIPELINES:
                raise ValueError(f"Unknown processing pipeline '{predefined_pipeline}'. Available pipelines: {list(PREDEFINED_PIPELINES.keys())}")
            for filter_name in PREDEFINED_PIPELINES[predefined_pipeline]:
                self.add_matchms_filter(filter_name)

    def add_filter(self,
                   filter_function: Union[Tuple[str, Dict[str, any]], str, Tuple[Callable, Dict[str, any]], Callable]):
        """Add a filter to the processing pipeline. Takes both matchms filter names (and parameters)
        as well as custom-made functions.
        """
        if isinstance(filter_function, str):
            self.add_matchms_filter(filter_function)
        elif isinstance(filter_function, (tuple, list)) and isinstance(filter_function[0], str):
            self.add_matchms_filter(filter_function)
        else:
            self.add_custom_filter(filter_function[0], filter_function[1])

    def add_matchms_filter(self, filter_spec: Union[Tuple[str, Dict[str, any]], str]):
        """
        Add a filter to the processing pipeline.

        Parameters
        ----------
        filter_spec : str or tuple
            Name of the filter function to add, or a tuple where the first element is the name of the
            filter function and the second element is a dictionary containing additional arguments for the function.
        """
        if isinstance(filter_spec, str):
            if filter_spec not in FILTER_FUNCTION_NAMES:
                raise ValueError(f"Unknown filter type: {filter_spec} Should be known filter name or function.")
            filter_func = FILTER_FUNCTION_NAMES[filter_spec]
        elif isinstance(filter_spec, (tuple, list)):
            filter_name, filter_args = filter_spec
            if filter_name not in FILTER_FUNCTION_NAMES:
                raise ValueError(f"Unknown filter type: {filter_name} Should be known filter name or function.")
            filter_func = partial(FILTER_FUNCTION_NAMES[filter_name], **filter_args)
            filter_func.__name__ = FILTER_FUNCTION_NAMES[filter_name].__name__
        else:
            raise TypeError("filter_spec should be a string or a tuple or list")

        self.filters.append(filter_func)
        # Sort filters according to their order in self.filter_order
        self.filters.sort(key=lambda f: self.filter_order.index(f.__name__))

    def add_custom_filter(self, filter_function: Union[Tuple[Callable, Dict[str, any]], Callable], filter_params=None):
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

    def process_spectrum(self, spectrum,
                         processing_report: Optional["ProcessingReport"] = None):
        """
        Process the given spectrum with all filters in the processing pipeline.

        Parameters
        ----------
        spectrum : Spectrum
            The spectrum to process.
        processing_report:
            A ProcessingReport object When passed the progress will be added to the object.

        Returns
        -------
        Spectrum
            The processed spectrum.
        """
        if not self.filters:
            raise TypeError("No filters to process")
        if processing_report is not None:
            processing_report.counter_number_processed += 1
        for filter_func in self.filters:
            spectrum_out = filter_func(spectrum)
            if processing_report is not None:
                processing_report.add_to_report(spectrum, spectrum_out, filter_func.__name__)
            if spectrum_out is None:
                break
            spectrum = spectrum_out
        return spectrum_out

    def process_spectrums(self, spectrums: list,
                          create_report: bool = False,
                          progress_bar: bool = True,
                          ):
        """
        Process a list of spectrums with all filters in the processing pipeline.

        Parameters
        ----------
        spectrums : list[Spectrum]
            The spectrums to process.
        create_report: bool, optional
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
        else:
            processing_report = None

        processed_spectrums = []
        for s in tqdm(spectrums, disable=(not progress_bar), desc="Processing spectrums"):
            if s is None:
                continue  # empty spectra will be discarded
            processed_spectrum = self.process_spectrum(s, processing_report)
            if processed_spectrum is not None:
                processed_spectrums.append(processed_spectrum)

        if create_report:
            return processed_spectrums, processing_report
        return processed_spectrums

    @property
    def processing_steps(self):
        filter_list = []
        for filter_step in self.filters:
            if isinstance(filter_step, partial):
                filter_params = filter_step.keywords
                filter_list.append((filter_step.__name__, filter_params))
            else:
                filter_list.append(filter_step.__name__)
        return filter_list

    def __str__(self):
        summary_string = "SpectrumProcessor\nProcessing steps:"
        for processing_step in self.processing_steps:
            if isinstance(processing_step, str):
                summary_string += "\n- " + processing_step
            elif isinstance(processing_step, tuple):
                filter_name = processing_step[0]
                summary_string += "\n- - " + filter_name
                filter_params = processing_step[1]
                for filter_param in filter_params:
                    summary_string += "\n  - " + str(filter_param)
        return summary_string


class ProcessingReport:
    """Class to keep track of spectrum changes during filtering.
    """
    def __init__(self):
        self.counter_changed_metadata = defaultdict(int)
        self.counter_removed_spectrums = defaultdict(int)
        self.counter_number_processed = 0

    def add_to_report(self, spectrum_old, spectrum_new: Spectrum,
                      filter_function_name: str):
        """Add changes between spectrum_old and spectrum_new to the report.
        """
        if spectrum_new is None:
            self.counter_removed_spectrums[filter_function_name] += 1
        else:
            if spectrum_new.metadata != spectrum_old.metadata:
                self.counter_changed_metadata[filter_function_name] += 1

    def to_dataframe(self):
        """Create Pandas DataFrame Report of counted spectrum changes."""
        metadata_changed = pd.DataFrame(self.counter_changed_metadata.items(),
                                        columns=["filter", "changed metadata"])
        removed = pd.DataFrame(self.counter_removed_spectrums.items(),
                               columns=["filter", "removed spectra"])
        processing_report = pd.merge(removed, metadata_changed, how="outer", on="filter")

        processing_report = processing_report.set_index("filter").fillna(0)
        return processing_report.astype(int)

    def __str__(self):
        report_str = f"""\
----- Spectrum Processing Report -----
Number of spectrums processed: {self.counter_number_processed}
Number of spectrums removed: {sum(self.counter_removed_spectrums.values())}
Changes during processing:
{str(self.to_dataframe())}
"""
        return report_str

    def __repr__(self):
        return f"Report({self.counter_number_processed},\
        {self.counter_removed_spectrums},\
        {dict(self.counter_changed_spectrum)})"


def objects_differ(obj1, obj2):
    """Test if two objects are different. Supposed to work for standard
    Python data types as well as numpy arrays.
    """
    if isinstance(obj1, np.ndarray) or isinstance(obj2, np.ndarray):
        return not np.array_equal(obj1, obj2)
    return obj1 != obj2
