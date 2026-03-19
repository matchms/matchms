import logging
import operator
import os
from collections import OrderedDict
from datetime import datetime
from typing import Callable, Iterable, List, Optional, Sequence, Union
import numpy as np
import matchms.similarity as mssimilarity
from matchms.filtering.filter_order import ALL_FILTERS
from matchms.filtering.SpectrumProcessor import (
    FunctionWithParametersType,
    SpectrumProcessor,
)
from matchms.importing.load_spectra import load_list_of_spectrum_files
from matchms.logging_functions import (
    add_logging_to_file,
    reset_matchms_logger,
    set_matchms_logger_level,
)
from matchms.Scores import Scores, ScoresMask
from matchms.similarity.BaseSimilarity import BaseSimilarity
from matchms.typing import SpectrumType
from matchms.yaml_file_functions import load_workflow_from_yaml_file, ordered_dump


logger = logging.getLogger("matchms")

_MASKING_FUNCTIONS = {"mask"}

_SCORE_FUNCTIONS = {}
for key, value in mssimilarity.__dict__.items():
    if isinstance(value, type) and issubclass(value, BaseSimilarity) and value is not BaseSimilarity:
        _SCORE_FUNCTIONS[key.lower()] = value


def create_workflow(
    yaml_file_name: Optional[str] = None,
    spectra_1_filters: Iterable[Union[str, Callable, FunctionWithParametersType]] = (),
    spectra_2_filters: Iterable[Union[str, Callable, FunctionWithParametersType]] = (),
    score_computations: Iterable[Union[str, List[Union[str, dict]]]] = (),
) -> OrderedDict:
    """Create a workflow specification for Pipeline."""
    workflow = OrderedDict()

    processor_1 = SpectrumProcessor(spectra_1_filters)
    processor_2 = SpectrumProcessor(spectra_2_filters)

    workflow["spectra_1_filters"] = processor_1.processing_steps
    workflow["spectra_2_filters"] = processor_2.processing_steps
    workflow["score_computations"] = list(score_computations)

    if yaml_file_name is not None:
        if os.path.exists(yaml_file_name):
            raise FileExistsError(
                "This yaml file already exists. "
                "Use load_workflow_from_yaml_file(...) to load an existing workflow."
            )
        with open(yaml_file_name, "w", encoding="utf-8") as file:
            file.write("# Matchms pipeline config file\n")
            file.write("# Change and adapt fields where necessary\n")
            file.write("# " + 20 * "=" + "\n")
            ordered_dump(workflow, file)

    return workflow


class Pipeline:
    """Central pipeline class.

    The pipeline applies filters to one or two collections of spectra and then
    executes a sequence of similarity computations and mask steps.

    Notes
    -----
    - If only ``spectra_1`` is provided during :meth:`run`, the pipeline assumes
      a symmetric all-vs-all computation and sets ``is_symmetric=True``.
    - If ``spectra_2`` is also provided, the pipeline computes ``spectra_1`` vs
      ``spectra_2`` and sets ``is_symmetric=False``.
    """

    def __init__(
        self,
        workflow: OrderedDict,
        progress_bar: bool = True,
        logging_level: str = "WARNING",
        logging_file: Optional[str] = None,
    ):
        self._spectra_1: List[SpectrumType] = []
        self._spectra_2: Optional[List[SpectrumType]] = None

        self.scores: Optional[Scores] = None
        self.mask: Optional[ScoresMask] = None

        self.progress_bar = progress_bar
        self.logging_level = logging_level
        self.logging_file = logging_file

        self.__workflow = workflow
        self.check_workflow()

        self.processing_spectra_1: Optional[SpectrumProcessor] = None
        self.processing_spectra_2: Optional[SpectrumProcessor] = None

        self._initialize_spectrum_processor_1()
        self._initialize_spectrum_processor_2()

    @classmethod
    def from_yaml(
        cls,
        yaml_file_name: str,
        progress_bar: bool = True,
        logging_level: str = "WARNING",
        logging_file: Optional[str] = None,
    ) -> "Pipeline":
        workflow = load_workflow_from_yaml_file(yaml_file_name)
        return cls(
            workflow=workflow,
            progress_bar=progress_bar,
            logging_level=logging_level,
            logging_file=logging_file,
        )

    def _initialize_spectrum_processor_1(self) -> None:
        self.write_to_logfile("--- Processing pipeline spectra_1: ---")
        self.processing_spectra_1 = SpectrumProcessor(self.__workflow["spectra_1_filters"])
        self.write_to_logfile(str(self.processing_spectra_1))
        if self.processing_spectra_1.processing_steps != self.__workflow["spectra_1_filters"]:
            logger.warning("The order of spectra_1 filters has been changed compared to the yaml file.")

    def _initialize_spectrum_processor_2(self) -> None:
        self.write_to_logfile("--- Processing pipeline spectra_2: ---")
        self.processing_spectra_2 = SpectrumProcessor(self.__workflow["spectra_2_filters"])
        self.write_to_logfile(str(self.processing_spectra_2))
        if self.processing_spectra_2.processing_steps != self.__workflow["spectra_2_filters"]:
            logger.warning("The order of spectra_2 filters has been changed compared to the yaml file.")

    def check_workflow(self) -> None:
        if not isinstance(self.__workflow, OrderedDict):
            raise TypeError(
                "Workflow is expected to be an OrderedDict, "
                f"but got type {type(self.__workflow)}."
            )
        expected_keys = {"spectra_1_filters", "spectra_2_filters", "score_computations"}
        if set(self.__workflow.keys()) != expected_keys:
            raise ValueError(
                f"Workflow must contain exactly keys {expected_keys}, "
                f"but got {set(self.__workflow.keys())}."
            )

        check_score_computation(self.__workflow["score_computations"])

    def run(
        self,
        spectra_1,
        spectra_2=None,
        cleaned_spectra_1_file=None,
        cleaned_spectra_2_file=None,
        create_report: bool = True,
    ):
        """Execute the pipeline workflow."""
        if cleaned_spectra_1_file is not None and os.path.exists(cleaned_spectra_1_file):
            raise FileExistsError("The specified cleaned spectra_1 file already exists.")
        if cleaned_spectra_2_file is not None and os.path.exists(cleaned_spectra_2_file):
            raise FileExistsError("The specified cleaned spectra_2 file already exists.")

        self.set_logging()
        self.write_to_logfile("--- Start running matchms pipeline. ---")
        self.write_to_logfile(f"Start time: {datetime.now()}")

        self.import_spectra(spectra_1, spectra_2)

        self.write_to_logfile("--- Processing spectra ---")
        self.write_to_logfile(f"Time: {datetime.now()}")

        report = None

        if self.processing_spectra_1 is not None:
            self._spectra_1, report = self.processing_spectra_1.process_spectra(
                self._spectra_1,
                progress_bar=self.progress_bar,
                cleaned_spectra_file=cleaned_spectra_1_file,
                create_report=create_report,
            )
            self.write_to_logfile(str(report))
            if cleaned_spectra_1_file is not None:
                self.write_to_logfile(f"--- Spectra_1 written to {cleaned_spectra_1_file} ---")

        if self.processing_spectra_2 is not None and self._spectra_2 is not None:
            self._spectra_2, report = self.processing_spectra_2.process_spectra(
                self._spectra_2,
                progress_bar=self.progress_bar,
                cleaned_spectra_file=cleaned_spectra_2_file,
                create_report=create_report,
            )
            self.write_to_logfile(str(report))
            if cleaned_spectra_2_file is not None:
                self.write_to_logfile(f"--- Spectra_2 written to {cleaned_spectra_2_file} ---")

        self.scores = None
        self.mask = None

        self.write_to_logfile("--- Computing scores ---")
        for computation in self.score_computations:
            self.write_to_logfile(f"Time: {datetime.now()}")

            if not isinstance(computation, list):
                computation = [computation]

            step_name = computation[0]
            if isinstance(step_name, str) and step_name.lower() in _MASKING_FUNCTIONS:
                self.write_to_logfile(f"-- Score masking: {computation} --")
                self._apply_score_masking(computation)
            else:
                self.write_to_logfile(f"-- Score computation: {computation} --")
                self._apply_similarity_measure(computation)

        self.write_to_logfile(f"--- Pipeline run finished ({datetime.now()}) ---")
        return report

    def _apply_score_masking(self, computation) -> None:
        if self.scores is None:
            raise ValueError("No scores have been computed yet, so masking cannot be applied.")

        params = computation[1] if len(computation) > 1 else {}
        operation_name = params.get("operation")
        value = params.get("value")
        field = params.get("field")

        if operation_name is None or "value" not in params:
            raise ValueError(
                "Mask computation requires parameters {'operation': ..., 'value': ...}."
            )

        self.mask = _build_mask_from_scores(
            scores=self.scores,
            operation_name=operation_name,
            value=value,
            field=field,
        )

    def _apply_similarity_measure(self, computation) -> None:
        similarity_measure = _instantiate_similarity(computation)

        if self.mask is None:
            self.scores = similarity_measure.matrix(
                spectra_1=self._spectra_1,
                spectra_2=self._spectra_2,
                progress_bar=self.progress_bar,
            )
            return

        idx_row, idx_col = _mask_to_index_arrays(self.mask)

        self.scores = similarity_measure.sparse_matrix(
            spectra_1=self._spectra_1,
            spectra_2=self._spectra_2,
            idx_row=idx_row,
            idx_col=idx_col,
            progress_bar=self.progress_bar,
        )

    def set_logging(self) -> None:
        reset_matchms_logger()
        set_matchms_logger_level(self.logging_level)
        if self.logging_file is not None:
            add_logging_to_file(
                self.logging_file,
                loglevel=self.logging_level,
                remove_stream_handlers=True,
            )
        else:
            logger.warning("No logging file was defined. Logging messages will not be written to file.")

    def write_to_logfile(self, line: str) -> None:
        if self.logging_file is not None:
            with open(self.logging_file, "a", encoding="utf-8") as file:
                file.write(line + "\n")

    def import_spectra(
        self,
        spectra_1: Union[List[str], str],
        spectra_2: Optional[Union[List[str], str]] = None,
    ) -> None:
        """Import one or two spectra collections from file(s)."""
        self.write_to_logfile("--- Importing data ---")
        self._spectra_1 = load_list_of_spectrum_files(spectra_1)
        self.write_to_logfile(f"Loaded spectra_1 from {spectra_1}")

        if spectra_2 is None:
            self.is_symmetric = True
            self._spectra_2 = None
            self.write_to_logfile("No spectra_2 given, using symmetric computation (is_symmetric = True)")
        else:
            self.is_symmetric = False
            self._spectra_2 = load_list_of_spectrum_files(spectra_2)
            self.write_to_logfile(f"Loaded spectra_2 from {spectra_2}")

    def save_as_yaml(self, yaml_file_name: str) -> None:
        if os.path.exists(yaml_file_name):
            raise FileExistsError("The specified yaml file already exists.")

        workflow = OrderedDict()
        workflow["spectra_1_filters"] = self.processing_spectra_1.processing_steps
        workflow["spectra_2_filters"] = self.processing_spectra_2.processing_steps
        workflow["score_computations"] = list(self.score_computations)

        with open(yaml_file_name, "w", encoding="utf-8") as file:
            file.write("# Matchms pipeline config file\n")
            file.write("# Change and adapt fields where necessary\n")
            file.write("# " + 20 * "=" + "\n")
            ordered_dump(workflow, file)

    @property
    def score_computations(self) -> Sequence[Union[str, List[dict]]]:
        return self.__workflow["score_computations"]

    @score_computations.setter
    def score_computations(self, computations):
        check_score_computation(computations)
        self.__workflow["score_computations"] = computations

    @property
    def spectra_1_filters(self):
        return self.__workflow["spectra_1_filters"]

    @spectra_1_filters.setter
    def spectra_1_filters(self, filters):
        self.__workflow["spectra_1_filters"] = filters
        self._initialize_spectrum_processor_1()

    @property
    def spectra_2_filters(self):
        return self.__workflow["spectra_2_filters"]

    @spectra_2_filters.setter
    def spectra_2_filters(self, filters):
        self.__workflow["spectra_2_filters"] = filters
        self._initialize_spectrum_processor_2()

    @property
    def spectra_1(self) -> List[SpectrumType]:
        return self._spectra_1

    @property
    def spectra_2(self) -> Optional[List[SpectrumType]]:
        return self._spectra_2


def _instantiate_similarity(computation) -> BaseSimilarity:
    name_or_callable = computation[0]
    params = computation[1] if len(computation) > 1 else {}

    if isinstance(name_or_callable, str):
        similarity_cls = _SCORE_FUNCTIONS.get(name_or_callable.lower())
        if similarity_cls is None:
            raise ValueError(f"Unknown score computation: {name_or_callable!r}.")
        return similarity_cls(**params)

    if callable(name_or_callable):
        return name_or_callable(**params)

    raise TypeError(f"Unknown similarity specification: {name_or_callable!r}.")


def _mask_to_index_arrays(mask: ScoresMask) -> tuple[np.ndarray, np.ndarray]:
    if mask.is_sparse:
        return mask.row, mask.col
    row, col = np.nonzero(mask.dense_mask)
    return row.astype(np.int_), col.astype(np.int_)


def _build_mask_from_scores(
    scores: Scores,
    operation_name: str,
    value,
    field: Optional[str] = None,
) -> ScoresMask:
    operations = {
        ">": operator.gt,
        ">=": operator.ge,
        "<": operator.lt,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
    }

    if operation_name not in operations:
        raise ValueError(
            f"Unknown mask operation {operation_name!r}. Supported operations are: {tuple(operations)}."
        )

    target = scores if field is None else scores[field]

    try:
        mask = operations[operation_name](target, value)
    except TypeError as exc:
        if field is None and not scores.is_scalar:
            raise TypeError(
                "Masking a multi-field Scores object requires specifying a field, "
                "for example ['mask', {'field': 'score', 'operation': '>=', 'value': 0.3}]."
            ) from exc
        raise

    if not isinstance(mask, ScoresMask):
        raise TypeError("Score comparison did not produce a ScoresMask as expected.")

    return mask


def get_unused_filters(yaml_file):
    workflow = load_workflow_from_yaml_file(yaml_file)
    processor = SpectrumProcessor(workflow["spectra_1_filters"])

    filters_used = [filter_function.__name__ for filter_function in processor.filters]
    for filter_function in ALL_FILTERS:
        if filter_function.__name__ not in filters_used:
            print(filter_function.__name__)


def check_score_computation(score_computations: Sequence[Union[str, List[dict]]]) -> None:
    if score_computations is None:
        return

    n_steps = len(score_computations)

    for i, computation in enumerate(score_computations):
        if not isinstance(computation, list):
            computation = [computation]

        if len(computation) == 0:
            raise ValueError("Empty score computation step is not allowed.")

        step = computation[0]
        params = computation[1] if len(computation) > 1 else {}

        if isinstance(step, str) and step.lower() in _MASKING_FUNCTIONS:
            if "operation" not in params or "value" not in params:
                raise ValueError(
                    "Mask computation requires parameters {'operation': ..., 'value': ...}."
                )
            if i == 0 or i == n_steps - 1:
                raise ValueError("Masking at the start or end of score computations is not allowed.")
            continue

        if isinstance(step, str):
            if step.lower() not in _SCORE_FUNCTIONS:
                raise ValueError(f"Unknown score computation: {step!r}.")
            continue

        if callable(step):
            continue

        raise ValueError(f"Unknown score computation: {step!r}.")
