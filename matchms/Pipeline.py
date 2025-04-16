import logging
import os
from collections import OrderedDict
from datetime import datetime
from typing import Callable, Iterable, List, Optional, Union
from deprecated import deprecated
import matchms.similarity as mssimilarity
from matchms import calculate_scores
from matchms.filtering.filter_order import ALL_FILTERS
from matchms.filtering.SpectrumProcessor import FunctionWithParametersType, SpectrumProcessor
from matchms.importing.load_spectra import load_list_of_spectrum_files
from matchms.logging_functions import add_logging_to_file, reset_matchms_logger, set_matchms_logger_level
from matchms.typing import SpectrumType
from matchms.yaml_file_functions import load_workflow_from_yaml_file, ordered_dump


_masking_functions = ["filter_by_range"]
_score_functions = {key.lower(): f for key, f in mssimilarity.__dict__.items() if callable(f)}
logger = logging.getLogger("matchms")
# ruff: noqa: E501


def create_workflow(
    yaml_file_name: Optional[str] = None,
    query_filters: Iterable[Union[str, Callable, FunctionWithParametersType]] = (),
    reference_filters: Iterable[Union[str, Callable, FunctionWithParametersType]] = (),
    score_computations: Iterable[Union[str, List[dict]]] = (),
) -> OrderedDict:
    """Creates a workflow that specifies the filters and scores needed to be run by Pipeline

    Example code can be found in the docstring of Pipeline.

    :param yaml_file_name:
        A yaml file containing the workflow settings will be saved if a file name is specified.
        If None no yaml file will be saved.
    :param query_filters:
        Additional filters that should be applied to the query spectra.
    :param reference_filters:
        Additional filters that should be applied to the reference spectra
    :param score_computations:
        Score computations that should be performed.
    """
    workflow = OrderedDict()
    queries_processor = SpectrumProcessor(query_filters)
    workflow["query_filters"] = queries_processor.processing_steps
    reference_processor = SpectrumProcessor(reference_filters)
    workflow["reference_filters"] = reference_processor.processing_steps
    workflow["score_computations"] = score_computations
    if yaml_file_name is not None:
        assert not os.path.exists(yaml_file_name), (
            "This yaml file name already exists. "
            "To use the settings in the yaml file, please use the load_workflow_from_yaml_file function "
            "in yaml_file_functions.py or check the tutorial."
        )
        with open(yaml_file_name, "w", encoding="utf-8") as file:
            file.write("# Matchms pipeline config file \n")
            file.write("# Change and adapt fields where necessary \n")
            file.write("# " + 20 * "=" + " \n")
            ordered_dump(workflow, file)
    workflow["query_filters"] = queries_processor.filters
    workflow["reference_filters"] = reference_processor.filters
    return workflow


class Pipeline:
    """Central pipeline class.

    The matchms Pipeline class is meant to make running extensive analysis pipelines
    fast and easy. It can be used in two different ways. First, a pipeline can be defined
    using a config file (a yaml file, best to start from the template provided to define
    your own pipline).

    Once a config file is defined, the pipeline can be executed with the following code:

    .. code-block:: python
        from matchms.Pipeline import Pipeline, load_workflow_from_yaml_file

        workflow = load_workflow_from_yaml_file("my_config_file.yaml")
        pipeline = Pipeline(workflow)

        # Optional steps
        pipeline.logging_file = "my_pipeline.log"
        pipeline.logging_level = "ERROR"

        pipeline.run("my_spectra.mgf")

    The second way to define a pipeline is via a Python script. The following code is an
    example of how this works:

    .. code-block:: python
        from matchms.Pipeline import Pipeline, create_workflow

        workflow = create_workflow(
            yaml_file_name="my_config_file.yaml",  # The workflow will be stored in a yaml file.
            query_filters=[
                ["add_parent_mass"],
                ["normalize_intensities"],
                ["select_by_relative_intensity", {"intensity_from": 0.0, "intensity_to": 1.0}],
                ["select_by_mz", {"mz_from": 0, "mz_to": 1000}],
                ["require_minimum_number_of_peaks", {"n_required": 5}],
            ],
            reference_filters=["add_fingerprint"],
            score_computations=[
                ["precursormzmatch", {"tolerance": 120.0}],
                ["cosinegreedy", {"tolerance": 1.0}]["filter_by_range", {"name": "CosineGreedy_score", "low": 0.3}],
                ["modifiedcosine", {"tolerance": 1.0}],
                ["filter_by_range", {"name": "ModifiedCosine_score", "low": 0.3}],
            ],
        )

        pipeline = Pipeline(workflow)
        pipeline.logging_file = "my_pipeline.log"
        pipeline.logging_level = "WARNING"
        pipeline.run("my_query_spectra.mgf", "my_reference_spectra.mgf")


    To combine this with custom made scores or available matchms-compatible scores
    such as `Spec2Vec` or `MS2DeepScore`, it is also possible to pass objects instead of
    names to create_workflow

    .. code-block:: python

        from spec2vec import Spec2Vec

        workflow = create_workflow(
            score_computations=[
                ["precursormzmatch", {"tolerance": 120.0}],
                [Spec2Vec, {"model": "my_spec2vec_model.model"}],
                ["filter_by_range", {"name": "Spec2Vec", "low": 0.3}],
            ]
        )
    """

    def __init__(self,
                 workflow: OrderedDict,
                 progress_bar=True,
                 logging_level: str = "WARNING",
                 logging_file: Optional[str] = None):
        """
        Parameters
        ----------
        workflow:
            Contains an orderedDict containing the workflow settings. Can be created using create_workflow.
        progress_bar:
            Default is True. Set to False if no progress bar should be displayed.
        """
        self._spectra_queries = None
        self._spectra_references = None
        self.is_symmetric = False
        self.scores = None

        self.logging_level = logging_level
        self.logging_file = logging_file
        self.progress_bar = progress_bar
        self.__workflow = workflow
        self.check_workflow()

        self._initialize_spectrum_processor_queries()
        if self.is_symmetric is False:
            self._initialize_spectrum_processor_references()

    def _initialize_spectrum_processor_queries(self):
        """Initialize spectrum processing workflow for the query spectra."""
        self.write_to_logfile("--- Processing pipeline query spectra: ---")
        self.processing_queries = SpectrumProcessor(self.__workflow["query_filters"])
        self.write_to_logfile(str(self.processing_queries))
        if self.processing_queries.processing_steps != self.__workflow["query_filters"]:
            logger.warning("The order of the filters has been changed compared to the Yaml file.")

    def _initialize_spectrum_processor_references(self):
        """Initialize spectrum processing workflow for the reference spectra."""
        self.write_to_logfile("--- Processing pipeline reference spectra: ---")

        self.processing_references = SpectrumProcessor(self.__workflow["reference_filters"])
        self.write_to_logfile(str(self.processing_references))
        if self.processing_queries.processing_steps != self.__workflow["query_filters"]:
            logger.warning("The order of the filters has been changed compared to the Yaml file.")

    def check_workflow(self):
        """Define Pipeline workflow based on a yaml file (config_file)."""
        assert isinstance(self.__workflow, OrderedDict), f"Workflow is expectd to be a OrderedDict, instead it was of type {type(self.__workflow)}"
        expected_keys = {"query_filters", "reference_filters", "score_computations"}
        assert set(self.__workflow.keys()) == expected_keys
        check_score_computation(score_computations=self.score_computations)

    def run(self, query_files, reference_files=None, cleaned_query_file=None, cleaned_reference_file=None):
        """Execute the defined Pipeline workflow.

        This method will execute all steps of the workflow.
        1) Initializing the log file and importing the spectra
        2) Spectrum processing (using matchms filters)
        3) Score Computations
        """
        if cleaned_reference_file is not None:
            if os.path.exists(cleaned_reference_file):
                raise FileExistsError("The specified save references file already exists")
        if cleaned_query_file is not None:
            if os.path.exists(cleaned_query_file):
                raise FileExistsError("The specified save queries file already exists")

        self.set_logging()
        self.write_to_logfile("--- Start running matchms pipeline. ---")
        self.write_to_logfile(f"Start time: {str(datetime.now())}")
        self.import_spectra(query_files, reference_files)

        # Processing
        self.write_to_logfile("--- Processing spectra ---")
        self.write_to_logfile(f"Time: {str(datetime.now())}")
        # Process query spectra
        spectra, report = self.processing_queries.process_spectra(
            self._spectra_queries, progress_bar=self.progress_bar, cleaned_spectra_file=cleaned_query_file
        )
        self._spectra_queries = spectra
        self.write_to_logfile(str(report))
        if cleaned_query_file is not None:
            self.write_to_logfile(f"--- Query spectra written to {cleaned_query_file} ---")

        # Process reference spectra (if necessary)
        if self.is_symmetric is False:
            self._spectra_references, report = self.processing_references.process_spectra(
                self._spectra_references, progress_bar=self.progress_bar, cleaned_spectra_file=cleaned_reference_file
            )
            self.write_to_logfile(str(report))
            if cleaned_reference_file is not None:
                self.write_to_logfile(f"--- Reference spectra written to {cleaned_reference_file} ---")
        else:
            self._spectra_references = self._spectra_queries

        # Score computation and masking
        self.write_to_logfile("--- Computing scores ---")
        for i, computation in enumerate(self.score_computations):
            self.write_to_logfile(f"Time: {str(datetime.now())}")
            if not isinstance(computation, list):
                computation = [computation]
            if isinstance(computation[0], str) and computation[0] in _masking_functions:
                self.write_to_logfile(f"-- Score masking: {computation} --")
                self._apply_score_masking(computation)
            else:
                self.write_to_logfile(f"-- Score computation: {computation} --")
                self._apply_similarity_measure(computation, i)
        self.write_to_logfile(f"--- Pipeline run finished ({str(datetime.now())}) ---")
        return report

    def _apply_score_masking(self, computation):
        """Apply filter to remove scores which are out of the set range."""
        if len(computation) == 1:
            name = self.scores.score_names[-1]
            self.scores.filter_by_range(name=name)
        elif "name" not in computation[1]:
            name = self.scores.scores.score_names[-1]
            self.scores.filter_by_range(name=name, **computation[1])
        else:
            self.scores.filter_by_range(**computation[1])

    def _apply_similarity_measure(self, computation, i):
        """Run score computations for all listed methods and on all loaded and processed spectra."""

        def get_similarity_measure(computation):
            if isinstance(computation[0], str):
                if len(computation) > 1:
                    return _score_functions[computation[0]](**computation[1])
                return _score_functions[computation[0]]()
            if callable(computation[0]):
                if len(computation) > 1:
                    return computation[0](**computation[1])
                return computation[0]()
            raise TypeError("Unknown similarity measure.")

        similarity_measure = get_similarity_measure(computation)
        # If this is the first score computation:
        if i == 0:
            self.scores = calculate_scores(
                self._spectra_references, self._spectra_queries, similarity_measure, array_type="sparse", is_symmetric=self.is_symmetric
            )
        else:
            new_scores = similarity_measure.sparse_array(
                references=self._spectra_references,
                queries=self._spectra_queries,
                idx_row=self.scores.scores.row,
                idx_col=self.scores.scores.col,
                is_symmetric=self.is_symmetric,
            )
            self.scores.scores.add_sparse_data(self.scores.scores.row, self.scores.scores.col, new_scores, similarity_measure.__class__.__name__)

    def set_logging(self):
        """Set the matchms logger to write messages to file (if defined)."""
        reset_matchms_logger()
        set_matchms_logger_level(self.logging_level)
        if self.logging_file is not None:
            add_logging_to_file(self.logging_file, loglevel=self.logging_level, remove_stream_handlers=True)
        else:
            logger.warning("No logging file was defined.Logging messages will not be written to file.")

    def write_to_logfile(self, line):
        """Write message to log file."""
        if self.logging_file is not None:
            with open(self.logging_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    @deprecated(version="0.26.5", reason="This method is deprecated and will be removed in the future. Use import_spectra() instead.")
    def import_spectrums(self, query_files: Union[List[str], str], reference_files: Optional[Union[List[str], str]] = None):
        """Wrapper method for import_spectra()

        Parameters
        ----------
        query_files
            List of files, or single filename, containing the query spectra.
        reference_files
            List of files, or single filename, containing the reference spectra.
            If set to None (default) then all query spectra will be compared to each other.
        """
        return self.import_spectra(query_files, reference_files)

    def import_spectra(self, query_files: Union[List[str], str], reference_files: Optional[Union[List[str], str]] = None):
        """Import spectra from file(s).

        Parameters
        ----------
        query_files
            List of files, or single filename, containing the query spectra.
        reference_files
            List of files, or single filename, containing the reference spectra.
            If set to None (default) then all query spectra will be compared to each other.
        """
        # import query spectra
        self.write_to_logfile("--- Importing data ---")
        self._spectra_queries = load_list_of_spectrum_files(query_files)

        self.write_to_logfile(f"Loaded query spectra from {query_files}")

        # import reference spectra
        if reference_files is None:
            self.is_symmetric = True
            self._spectra_references = self._spectra_queries
            self.write_to_logfile("Reference spectra are equal to the query spectra (is_symmetric = True)")
        else:
            self._spectra_references = load_list_of_spectrum_files(reference_files)
            self.write_to_logfile(f"Loaded reference spectra from {reference_files}")

    # Getter & Setters
    @property
    def score_computations(self) -> Iterable[Union[str, List[dict]]]:
        return self.__workflow.get("score_computations")

    @score_computations.setter
    def score_computations(self, computations):
        self.__workflow["score_computations"] = computations
        check_score_computation(score_computations=self.score_computations)

    @property
    def query_filters(self) -> Iterable[Union[str, List[dict]]]:
        return self.__workflow.get("query_filters")

    @query_filters.setter
    def query_filters(self, filters: Iterable[Union[str, List[dict]]]):
        self.__workflow["query_filters"] = filters
        self._initialize_spectrum_processor_queries()

    @property
    def reference_filters(self) -> Iterable[Union[str, List[dict]]]:
        return self.__workflow.get("reference_filters")

    @reference_filters.setter
    def reference_filters(self, filters: Iterable[Union[str, List[dict]]]):
        self.__workflow["reference_filters"] = filters
        self._initialize_spectrum_processor_references()

    @property
    @deprecated(version="0.26.5", reason="This property is deprecated and will be removed in the future. Use spectra_queries instead.")
    def spectrums_queries(self) -> List[SpectrumType]:
        return self._spectra_queries

    @property
    def spectra_queries(self) -> List[SpectrumType]:
        return self._spectra_queries

    @property
    @deprecated(version="0.26.5", reason="This property is deprecated and will be removed in the future. Use spectra_references instead.")
    def spectrums_references(self) -> List[SpectrumType]:
        return self._spectra_references

    @property
    def spectra_references(self) -> List[SpectrumType]:
        return self._spectra_references


def get_unused_filters(yaml_file):
    """Prints all filter names that are in ALL_FILTERS, but not in the yaml file"""
    workflow = load_workflow_from_yaml_file(yaml_file)
    processor = SpectrumProcessor(workflow["query_filters"])

    filters_used = [filter_function.__name__ for filter_function in processor.filters]
    for filter_function in ALL_FILTERS:
        if filter_function.__name__ not in filters_used:
            print(filter_function.__name__)


def check_score_computation(score_computations: Iterable[Union[str, List[dict]]]):
    """Check if the score computations seem OK before running.
    Aim is to avoid pipeline crashing after long computation.
    """
    # Check if all score compuation steps exist
    for computation in score_computations:
        if not isinstance(computation, list):
            computation = [computation]
        if isinstance(computation[0], str) and computation[0] in _masking_functions:
            continue
        if isinstance(computation[0], str) and computation[0] in _score_functions:
            continue
        if callable(computation[0]):
            continue
        raise ValueError(f"Unknown score computation: {computation[0]}.")
