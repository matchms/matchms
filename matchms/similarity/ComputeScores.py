import logging
from datetime import datetime
from typing import List, Optional, Sequence, Tuple, Union
import matchms.similarity as mssimilarity
from matchms.logging_functions import add_logging_to_file, reset_matchms_logger, set_matchms_logger_level
from matchms.similarity.BaseSimilarity import BaseSimilarity
from matchms.similarity.ScoresMask import ScoresMask, _get_operator
from matchms.Spectrum import Spectrum


logger = logging.getLogger("matchms")


class MaskSetting:
    """Class to define settings for masking of scores."""

    def __init__(self, operation: str, value: float):
        try:
            _get_operator(operation)
        except ValueError:
            raise ValueError(f"Unknown masking operation: {operation}. Available operations are: >, >= <=, ==, !=")
        self.operation = operation
        self.value = value


class ComputeScores:
    def __init__(
        self,
        similarity_methods_and_masks: List[BaseSimilarity | MaskSetting],
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
        self.logging_level = logging_level
        self.logging_file = logging_file
        self.progress_bar = progress_bar
        self.similarity_methods_and_masks = similarity_methods_and_masks
        for i, similarity_method_or_mask in enumerate(self.similarity_methods_and_masks):
            if not isinstance(similarity_method_or_mask, (BaseSimilarity, MaskSetting)):
                raise ValueError(
                    "Elements of similarity_methods_and_masks should be instances of BaseSimilarity or MaskSetting."
                )
            if i == 0 or i == len(self.similarity_methods_and_masks) - 1:
                if isinstance(similarity_method_or_mask, MaskSetting):
                    raise ValueError("Masking at the start or end of the score computations is not allowed.")
        self.set_logging()

    def run(
        self,
        query_spectra: List[Spectrum],
        reference_spectra=None,
        is_symmetric=False,
    ):
        """Computes the scores"""
        mask = None
        scores = None
        if is_symmetric:
            reference_spectra = query_spectra
        if reference_spectra is None:
            raise ValueError("Reference spectra should be provided if is_symmetric is False.")
        # Score computation and masking
        self.write_to_logfile("--- Computing scores ---")
        for similarity_method_or_mask in self.similarity_methods_and_masks:
            self.write_to_logfile(f"Time: {str(datetime.now())}")
            if isinstance(similarity_method_or_mask, BaseSimilarity):
                self.write_to_logfile(f"-- Score computation: {similarity_method_or_mask.__class__.__name__} --")
                scores = self._apply_similarity_measure(
                    similarity_method_or_mask, reference_spectra, query_spectra, is_symmetric, mask
                )
            else:
                # Perform masking of scores
                self.write_to_logfile(f"-- Score masking: {similarity_method_or_mask} --")
                if scores is None:
                    raise ValueError("No scores have been computed yet, so masking cannot be applied.")
                mask = ScoresMask.from_matrix(
                    scores, similarity_method_or_mask.operation, similarity_method_or_mask.value
                )
        return scores

    def _apply_similarity_measure(
        self,
        similarity_measure: BaseSimilarity,
        reference_spectra: Sequence[Spectrum],
        query_spectra: Sequence[Spectrum],
        is_symmetric=False,
        mask=None,
    ):
        """Run score computations for all listed methods and on all loaded and processed spectra."""
        # todo choose between matrix and sparse array based on coverage of mask
        scores = similarity_measure.matrix(
            reference_spectra, query_spectra, is_symmetric=is_symmetric, mask_indices=mask
        )
        return scores

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


# A single-item list: [str | Callable]
ComputationWithoutSettings = Tuple[Union[str, BaseSimilarity]]

# A two-item list: [str | Callable, dict]
ComputationWithSettings = Tuple[Union[str, BaseSimilarity], dict]


def parse_similarity_methods_and_masks(
    score_computations: Sequence[Union[str, BaseSimilarity, ComputationWithoutSettings, ComputationWithSettings]],
) -> List[Union[BaseSimilarity, MaskSetting]]:
    """Parses the score computations and masks defined in the workflow to return SimilarityMeasures and MaskSettings

    Acceptable input is a list with:
    - str, Which is a known matchms similarity method.
    - Callable, A custom similarity method provided as a function.
    - A list of two elements, where the first element is a str or Callable as defined
    """
    cleaned_score_computations = []
    # Check if all score computation steps exist
    for i, computation in enumerate(score_computations):
        if isinstance(computation, (str, BaseSimilarity)):
            cleaned_score_computations.append(parse_similarity_measure_without_settings(computation))
        elif computation[0] == "mask":
            if i == 0 or len(score_computations) == i:
                raise ValueError("Masking at the start or end of the score computations is not allowed.")
            if len(computation) != 2:
                raise ValueError("Masking settings should be provided as a list of two elements: ['mask', dict].")
            cleaned_score_computations.append(parse_masking_operation(computation[1]))
        elif len(computation) == 1:
            cleaned_score_computations.append(parse_similarity_measure_without_settings(computation[0]))
        elif len(computation) == 2:
            cleaned_score_computations.append(parse_similarity_measure_with_settings(computation))
        else:
            raise ValueError("Invalid computation format.")
    return cleaned_score_computations


def parse_masking_operation(mask_settings: dict) -> MaskSetting:
    if not isinstance(mask_settings, dict):
        raise ValueError("Masking settings should be provided as a dictionary.")
    if "operation" in mask_settings and "value" in mask_settings:
        return MaskSetting(mask_settings["operation"], mask_settings["value"])
    else:
        raise ValueError("Masking settings should contain 'operation' and 'value' keys.")


def get_existing_similarity_measure_by_name(name: str):
    _score_functions = {}
    for key, f in mssimilarity.__dict__.items():
        if isinstance(f, type) and issubclass(f, BaseSimilarity) and f is not BaseSimilarity:
            _score_functions[key.lower()] = f
    if name not in _score_functions:
        raise ValueError(
            f"Unknown similarity measure name: {name}. "
            f"Available measures are: {list(_score_functions.keys())}. "
            f" If you want to use a custom similarity measure, please provide it as a callable function."
        )
    return _score_functions[name]


def parse_similarity_measure_with_settings(computation: ComputationWithSettings) -> BaseSimilarity:
    if not isinstance(computation, (list, tuple)) or len(computation) != 2:
        raise ValueError(
            "Similarity measure with settings should be provided as a list of two elements: [str | Callable, dict]."
        )
    if not isinstance(computation[1], dict):
        raise ValueError("Settings for similarity measure should be provided as a dictionary.")

    if isinstance(computation[0], str):
        return get_existing_similarity_measure_by_name(computation[0])(**computation[1])
    if (
        isinstance(computation[0], type)
        and issubclass(computation[0], BaseSimilarity)
        and computation[0] is not BaseSimilarity
    ):
        return computation[0](**computation[1])
    raise TypeError("Unknown similarity measure.")


def parse_similarity_measure_without_settings(computation: Union[str, BaseSimilarity]) -> BaseSimilarity:
    if isinstance(computation, str):
        return get_existing_similarity_measure_by_name(computation)()
    if isinstance(computation, type) and issubclass(computation, BaseSimilarity) and computation is not BaseSimilarity:
        return computation()
    raise TypeError("Unknown similarity measure.")
