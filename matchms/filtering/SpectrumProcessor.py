import inspect
import logging
import os
from collections import OrderedDict, defaultdict
from functools import partial
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from deprecated import deprecated
from tqdm import tqdm
from matchms import Spectrum
from matchms.exporting import save_spectra
from matchms.filtering.filter_order import ALL_FILTERS, FILTER_FUNCTION_NAMES
from matchms.yaml_file_functions import ordered_dump


logger = logging.getLogger("matchms")
FunctionWithParametersType = Tuple[Union[Callable, str], Dict[str, any]]


class SpectrumProcessor:
    """
    A class to process spectra using a series of filters.

    The class enables a user to define a custom spectrum processing workflow by setting multiple
    flags and parameters.

    Parameters
    ----------
    filters:
        A list of filter functions, see add_filter for all the allowed formats.
    """

    def __init__(self,
                 filters: Iterable[Union[str,
                                         Callable,
                                         FunctionWithParametersType]]):
        self.filters = []
        self.filter_order = [x.__name__ for x in ALL_FILTERS]
        for filter_name in filters:
            self.parse_and_add_filter(filter_name)

    def parse_and_add_filter(self, filter_description: Union[str,
                                                             Callable,
                                                             FunctionWithParametersType,
                                                             ],
                             filter_position: Optional[int] = None):
        """Adds a filter, by parsing the different allowed inputs.

        filter:
            Allowed formats:
            str (has to be a matchms function name)
            (str, {str, any} (has to be a matchms function name, followed by parameters)
            Callable (can be matchms filter or custom made filter)
            Callable, {str, any} (the dict should be parameters.
        filter_position:
            If None: Matchms filters are automatically ordered.
            Custom filters will be added at the end of the filter list.
            If not None, the filter will be added to the given position in the filter order list.
        """
        filter_args = None
        if isinstance(filter_description, (tuple, list)):
            if len(filter_description) == 1:
                filter_function = filter_description[0]
            elif len(filter_description) == 2:
                filter_function = filter_description[0]
                filter_args = filter_description[1]
            else:
                raise ValueError("The filter_function should contain only two values, "
                                 "the first should be string or callable and the second a dictionary with settings")
        else:
            filter_function = filter_description
        if isinstance(filter_function, str):
            filter_function = load_matchms_filter_from_string(filter_function)
        self._add_filter_to_filter_order(filter_function.__name__,
                                         filter_position=filter_position)
        self._store_filter(filter_function, filter_args)

    def _store_filter(self,
                      new_filter_function: Callable,
                      filter_params: Optional[Dict[str, any]]):
        """Stores filter, removes duplicates and sorts filters"""
        if not callable(new_filter_function):
            raise TypeError("Expected callable filter function.")
        new_filter_function = create_partial_function(new_filter_function, filter_params)
        check_all_parameters_given(new_filter_function)
        self._replace_already_stored_filters(new_filter_function)
        # Sort filters according to their order in self.filter_order
        self.filters.sort(key=lambda f: self.filter_order.index(f.__name__))

    def _replace_already_stored_filters(self,
                                       new_filter_function: Callable):
        """Replaces filters that are already stored

        This will also overwrite the parameter settings, with the settings that are added last"""
        filter_already_added = False
        for i, filter_function in enumerate(self.filters):
            if new_filter_function.__name__ == filter_function.__name__:
                logger.warning("The filter %s was already in the filter list, "
                               "the last added filter parameters are used, "
                               "check yaml file for details", new_filter_function.__name__)
                self.filters[i] = new_filter_function
                filter_already_added = True
        if not filter_already_added:
            self.filters.append(new_filter_function)

    def _add_filter_to_filter_order(self,
                                    filter_function_name,
                                    filter_position: Optional[int] = None):
        """Adds the filter name to the filter order list if it is not yet there.

        filter_function_name:
            The name of the filter function.
        filter_position:
            The position where the filter should be added.
            This overrides the position if it is already stored.
            If None. A filter already in the order list stays at the same position.
            If None. A new filter name is appended to the end.
        """
        # Check if already stored. If a filter position is given the order will be adjusted
        if filter_function_name in self.filter_order:
            if filter_position is None:
                return None
            # Remove the filter (so it is not duplicated and can be added again)
            self.filter_order.remove(filter_function_name)

        # Add filter position at the end of the list
        if filter_position is None or filter_position >= len(self.filters):
            self.filter_order.append(filter_function_name)
        else:
            current_filter_at_position = self.filters[filter_position].__name__
            order_index = self.filter_order.index(current_filter_at_position)
            self.filter_order.insert(order_index, filter_function_name)
        return None

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
        if processing_report is not None:
            processing_report.counter_number_processed += 1

        clone = processing_report is not None

        for filter_func in self.filters:
            method_params = inspect.signature(filter_func).parameters

            if not filter_func.__name__.startswith("require_"):
                if clone and "clone" not in method_params:
                    logger.error(
                        "Processing report is set to True, but the filter function %s does not support cloning.",
                        filter_func.__name__)

            # Ensures that filter function can be used, even if it does not support cloning
            spectrum_out = filter_func(spectrum, **({"clone": clone} if "clone" in method_params else {}))

            if processing_report is not None:
                processing_report.add_to_report(spectrum, spectrum_out, filter_func.__name__)
            if spectrum_out is None:
                return None
            spectrum = spectrum_out
        return spectrum

    @deprecated(version="0.26.5",
                reason="This method is deprecated and will be removed in the future. Use 'process_spectra()' instead.")
    def process_spectrums(self, spectra: list,
                          progress_bar: bool = True,
                          cleaned_spectra_file=None,
                          create_report: Optional[bool] = False
                          ):
        """
        Wrapper method for process_spectra()

        Parameters
        ----------
        spectra : list[Spectrum]
            The spectra to process.
        create_report: bool, optional
            Creates and outputs a report of the main changes during processing.
            The report will be returned as pandas DataFrame. Default is set to False.
        progress_bar : bool, optional
            Displays progress bar if set to True. Default is True.
        cleaned_spectra_file:
            Path to where the cleaned spectra should be saved.
        Returns
        -------
        spectra
            List containing the processed spectra.
        processing_report
            A ProcessingReport containing the effect of the filters.
        """
        return self.process_spectra(spectra, progress_bar, cleaned_spectra_file, create_report)

    def process_spectra(self, spectra: list,
                          progress_bar: bool = True,
                          cleaned_spectra_file=None,
                          create_report: Optional[bool] = False
                          ):
        """
        Process a list of spectra with all filters in the processing pipeline.

        Parameters
        ----------
        spectra : list[Spectrum]
            The spectra to process.
        create_report: bool, optional
            Creates and outputs a report of the main changes during processing.
            The report will be returned as pandas DataFrame. Default is set to False.
        progress_bar : bool, optional
            Displays progress bar if set to True. Default is True.
        cleaned_spectra_file:
            Path to where the cleaned spectra should be saved.
        Returns
        -------
        spectra
            List containing the processed spectra.
        processing_report
            A ProcessingReport containing the effect of the filters.
        """
        if cleaned_spectra_file is not None:
            if os.path.exists(cleaned_spectra_file):
                raise FileExistsError("The specified save references file already exists")
            ftype = os.path.splitext(cleaned_spectra_file)[1].lower()[1:]
            incremental_save = ftype in ('mgf', 'msp')
        else:
            incremental_save = False

        if not self.filters:
            logger.warning("No filters have been specified, so spectra were not filtered")

        processing_report = ProcessingReport(self.filters) if create_report else None

        processed_spectra = []
        for s in tqdm(spectra, disable=(not progress_bar), desc="Processing spectra"):
            if s is None:
                continue  # empty spectra will be discarded

            # Clone spectrum once, if no ProcessingReport is created. ProcessingReport needs cloning in every filter.
            spectrum = s.clone() if not create_report else s
            processed_spectrum = self.process_spectrum(spectrum, processing_report)

            if processed_spectrum is not None:
                processed_spectra.append(processed_spectrum)

                if cleaned_spectra_file is not None and incremental_save:
                    save_spectra(processed_spectrum, cleaned_spectra_file, append=True)

        if cleaned_spectra_file is not None and not incremental_save:
            save_spectra(processed_spectra, cleaned_spectra_file)

        return processed_spectra, processing_report

    @property
    def processing_steps(self):
        filter_list = []
        for filter_step in self.filters:
            parameter_settings = get_parameter_settings(filter_step)
            if parameter_settings is not None:
                filter_list.append((filter_step.__name__, parameter_settings))
            else:
                filter_list.append(filter_step.__name__)
        return filter_list

    def __str__(self):
        workflow = OrderedDict()
        workflow["Processing steps"] = self.processing_steps
        return ordered_dump(workflow)


def load_matchms_filter_from_string(filter_name):
    if not isinstance(filter_name, str):
        raise ValueError("Expected a string")
    if filter_name not in FILTER_FUNCTION_NAMES:
        raise ValueError(f"Unknown filter type: {filter_name} Should be known filter name or function.")
    return FILTER_FUNCTION_NAMES[filter_name]


def create_partial_function(filter_function: Callable,
                            filter_params: Optional[Dict[str, any]]):
    """Adds the filter params to the filter function"""
    if filter_params is not None:
        if not isinstance(filter_params, dict):
            raise ValueError(f"Expected a dictionary for filter_args got {filter_params}")
        partial_filter_func = partial(filter_function, **filter_params)
        partial_filter_func.__name__ = filter_function.__name__
        return partial_filter_func
    return filter_function


def check_all_parameters_given(func: Callable):
    """Asserts that all added parameters for a function are given (except spectrum_in)"""
    signature = inspect.signature(func)
    parameters_without_value = []
    for parameter, value in signature.parameters.items():
        if value.default is inspect.Parameter.empty:
            parameters_without_value.append(parameter)
    assert len(parameters_without_value) == 1, \
        f"More than one parameter of the function {func.__name__} is not specified, " \
        f"the parameters not specified are {parameters_without_value}"


def get_parameter_settings(func):
    """Returns all parameters and parameter values for a function

    This includes default parameter settings and, but also the settings stored in partial"""
    signature = inspect.signature(func)
    parameter_settings = {
            parameter: value.default
            for parameter, value in signature.parameters.items()
            if value.default is not inspect.Parameter.empty
        }
    if parameter_settings == {}:
        return None
    return parameter_settings


class ProcessingReport:
    """Class to keep track of spectrum changes during filtering.
    """
    def __init__(self, filter_functions: Optional[List[Callable]] = None):
        if filter_functions:
            self.filter_names = [filter_function.__name__ for filter_function in filter_functions]
        else:
            self.filter_names = []
        self.counter_changed_metadata = defaultdict(int)
        self.counter_removed_spectra = defaultdict(int)
        self.counter_changed_peaks = defaultdict(int)
        self.counter_number_processed = 0

    def add_to_report(self, spectrum_old, spectrum_new: Spectrum,
                      filter_function_name: str):
        """Add changes between spectrum_old and spectrum_new to the report.
        """
        if spectrum_new is None:
            self.counter_removed_spectra[filter_function_name] += 1
        else:
            # Add metadata changes
            if spectrum_new.metadata != spectrum_old.metadata:
                self.counter_changed_metadata[filter_function_name] += 1
            # Add peak changes
            if spectrum_new.peaks != spectrum_old.peaks:
                self.counter_changed_peaks[filter_function_name] += 1

    def to_dataframe(self):
        """Create Pandas DataFrame Report of counted spectrum changes."""
        metadata_changed = pd.DataFrame(self.counter_changed_metadata.items(),
                                        columns=["filter", "changed metadata"])
        removed = pd.DataFrame(self.counter_removed_spectra.items(),
                               columns=["filter", "removed spectra"])
        peaks_changed = pd.DataFrame(self.counter_changed_peaks.items(),
                                     columns=["filter", "changed mass spectrum"])
        processing_report = pd.merge(removed, metadata_changed, how="outer", on="filter")
        processing_report = pd.merge(processing_report, peaks_changed, how="outer", on="filter")

        # Add filters that did not do any changes:
        for filter_name in self.filter_names:
            if filter_name not in processing_report["filter"].values:
                processing_report.loc[len(processing_report)] = {"filter": filter_name}

        try:
            with pd.option_context("future.no_silent_downcasting", True):
                processing_report = processing_report.set_index("filter").infer_objects().fillna(0)
        except pd.errors.OptionError:
            processing_report = processing_report.set_index("filter").fillna(0)

        return processing_report.astype(int)

    def __str__(self):
        pd.set_option('display.max_columns', 4)
        pd.set_option('display.width', 1000)
        report_str = ("----- Spectrum Processing Report -----\n"
                      f"Number of spectra processed: {self.counter_number_processed}\n"
                      f"Number of spectra removed: {sum(self.counter_removed_spectra.values())}\n"
                      "Changes during processing:\n"
                      f"{str(self.to_dataframe())}")
        return report_str

    def __repr__(self):
        return f"Report({self.counter_number_processed},\
        {self.counter_removed_spectra},\
        {dict(self.counter_removed_spectra)},\
        {dict(self.counter_changed_metadata)},\
        {dict(self.counter_changed_peaks)})"


def objects_differ(obj1, obj2):
    """Test if two objects are different. Supposed to work for standard
    Python data types as well as numpy arrays.
    """
    if isinstance(obj1, np.ndarray) or isinstance(obj2, np.ndarray):
        return not np.array_equal(obj1, obj2)
    return obj1 != obj2
