import logging
import os
from collections import OrderedDict
from datetime import datetime
from typing import List, Optional, Union
from deprecated import deprecated
from matchms.filtering.filter_order import ALL_FILTERS
from matchms.filtering.SpectrumProcessor import SpectrumProcessor
from matchms.importing.load_spectra import load_list_of_spectrum_files
from matchms.logging_functions import add_logging_to_file, reset_matchms_logger, set_matchms_logger_level
from matchms.similarity.ComputeScores import ComputeScores, parse_similarity_methods_and_masks
from matchms.Spectrum import Spectrum
from matchms.yaml_file_functions import load_workflow_from_yaml_file, ordered_dump


logger = logging.getLogger("matchms")
# ruff: noqa: E501


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
                ["mask", {"operation": "==", "value": True}],
                ["cosinegreedy", {"tolerance": 1.0}],
                ["mask", {"operation": ">=", "value": 0.3}],
                ["modifiedcosinegreedy", {"tolerance": 1.0}],
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
                ["mask", {"operation": ">=", "value": 0.3}],
                [Spec2Vec, {"model": "my_spec2vec_model.model"}],
            ]
        )
    """

    def __init__(
        self,
        similarity_methods_and_masks: Optional[ComputeScores] = None,
        query_filters: Optional[SpectrumProcessor] = None,
        reference_filters: Optional[SpectrumProcessor] = None,
        progress_bar=True,
        logging_level: str = "WARNING",
        logging_file: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        workflow:
            Contains an orderedDict containing the workflow settings. Can be created using create_workflow.
        progress_bar:
            Default is True. Set to False if no progress bar should be displayed.
        """
        self._spectra_queries = []
        self._spectra_references = []
        self.is_symmetric = False
        if self.is_symmetric and reference_filters is not None:
            raise ValueError("Reference filters cannot be defined if is_symmetric is True.")
        self.logging_level = logging_level
        self.logging_file = logging_file
        self.progress_bar = progress_bar

        self.similarity_score_compute_pipeline = similarity_methods_and_masks
        self.scores = None
        self.query_filters = query_filters
        self.reference_filters = reference_filters
        if self.query_filters is not None:
            self.write_to_logfile("--- Processing pipeline query spectra: ---")
            self.write_to_logfile(str(self.query_filters))
        if self.reference_filters is not None:
            self.write_to_logfile("--- Processing pipeline reference spectra: ---")
            self.write_to_logfile(str(self.reference_filters))

    @classmethod
    def from_yaml(
        cls,
        yaml_file_name,
        progress_bar=True,
        logging_level: str = "WARNING",
        logging_file: Optional[str] = None,
    ):
        """Initialize Pipeline from yaml file."""
        workflow = load_workflow_from_yaml_file(yaml_file_name)
        similarity_score_compute_pipeline = None
        processing_references = None
        processing_queries = None
        if "score_computations" in workflow:
            # Set up score computation pipeline
            similarity_methods_and_masks = parse_similarity_methods_and_masks(
                score_computations=workflow["score_computations"]
            )
            similarity_score_compute_pipeline = ComputeScores(
                similarity_methods_and_masks=similarity_methods_and_masks,
                progress_bar=progress_bar,
                logging_level=logging_level,
                logging_file=logging_file,
            )
        if "query_filters" in workflow:
            processing_queries = SpectrumProcessor(workflow["query_filters"])
            if processing_queries.processing_steps != workflow["query_filters"]:
                logger.warning("The order of the query filters has been changed compared to the Yaml file.")
        if "reference_filters" in workflow:
            processing_references = SpectrumProcessor(workflow["reference_filters"])
            if processing_references.processing_steps != workflow["reference_filters"]:
                logger.warning("The order of the reference filters has been changed compared to the Yaml file.")
        return cls(
            similarity_methods_and_masks=similarity_score_compute_pipeline,
            query_filters=processing_queries,
            reference_filters=processing_references,
            progress_bar=progress_bar,
            logging_level=logging_level,
            logging_file=logging_file,
        )

    def run(
        self,
        query_files,
        reference_files=None,
        cleaned_query_file=None,
        cleaned_reference_file=None,
        create_report=True,
    ):
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

        if self.query_filters is not None:
            # Processing
            self.write_to_logfile("--- Processing spectra ---")
            self.write_to_logfile(f"Time: {str(datetime.now())}")
            # Process query spectra
            spectra, report = self.query_filters.process_spectra(
                self._spectra_queries,
                progress_bar=self.progress_bar,
                cleaned_spectra_file=cleaned_query_file,
                create_report=create_report,
            )
            self._spectra_queries = spectra
            self.write_to_logfile(str(report))
            if cleaned_query_file is not None:
                self.write_to_logfile(f"--- Query spectra written to {cleaned_query_file} ---")

        if self.is_symmetric:
            self._spectra_references = self._spectra_queries

        # Process reference spectra (if necessary)
        if self.reference_filters is not None:
            self._spectra_references, report = self.reference_filters.process_spectra(
                self._spectra_references,
                progress_bar=self.progress_bar,
                cleaned_spectra_file=cleaned_reference_file,
                create_report=create_report,
            )
            self.write_to_logfile(str(report))
            if cleaned_reference_file is not None:
                self.write_to_logfile(f"--- Reference spectra written to {cleaned_reference_file} ---")

        if self.similarity_score_compute_pipeline is not None:
            self.write_to_logfile("--- Computing scores ---")
            self.scores = self.similarity_score_compute_pipeline.run(
                self.spectra_queries, self.spectra_references, self.is_symmetric
            )
        self.write_to_logfile(f"--- Pipeline run finished ({str(datetime.now())}) ---")

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

    @deprecated(
        version="0.26.5",
        reason="This method is deprecated and will be removed in the future. Use import_spectra() instead.",
    )
    def import_spectrums(
        self, query_files: Union[List[str], str], reference_files: Optional[Union[List[str], str]] = None
    ):
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

    def import_spectra(
        self, query_files: Union[List[str], str], reference_files: Optional[Union[List[str], str]] = None
    ):
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

    @property
    @deprecated(
        version="0.26.5",
        reason="This property is deprecated and will be removed in the future. Use spectra_queries instead.",
    )
    def spectrums_queries(self) -> List[Spectrum]:
        return self._spectra_queries

    @property
    def spectra_queries(self) -> List[Spectrum]:
        return self._spectra_queries

    @property
    @deprecated(
        version="0.26.5",
        reason="This property is deprecated and will be removed in the future. Use spectra_references instead.",
    )
    def spectrums_references(self) -> List[Spectrum]:
        return self._spectra_references

    @property
    def spectra_references(self) -> List[Spectrum]:
        return self._spectra_references

    def save_as_yaml(self, yaml_file_name):
        """Save the current workflow settings as a yaml file."""
        workflow = OrderedDict()
        if self.query_filters is not None:
            workflow["query_filters"] = self.query_filters.processing_steps
        if self.reference_filters is not None:
            workflow["reference_filters"] = self.reference_filters.processing_steps
        if self.similarity_score_compute_pipeline is not None:
            workflow["score_computations"] = self.similarity_score_compute_pipeline.to_yaml()
        if os.path.exists(yaml_file_name):
            raise FileExistsError("The specified yaml file already exists")
        with open(yaml_file_name, "w", encoding="utf-8") as file:
            file.write("# Matchms pipeline config file \n")
            file.write("# Change and adapt fields where necessary \n")
            file.write("# " + 20 * "=" + " \n")
            ordered_dump(workflow, file)


def get_unused_filters(yaml_file):
    """Prints all filter names that are in ALL_FILTERS, but not in the yaml file"""
    workflow = load_workflow_from_yaml_file(yaml_file)
    processor = SpectrumProcessor(workflow["query_filters"])

    filters_used = [filter_function.__name__ for filter_function in processor.filters]
    for filter_function in ALL_FILTERS:
        if filter_function.__name__ not in filters_used:
            print(filter_function.__name__)
