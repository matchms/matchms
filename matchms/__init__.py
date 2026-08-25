from matchms.filtering.spectra_collection_processor import SpectraCollectionProcessor
from matchms.filtering.spectrum_processor import SpectrumProcessor
from . import exporting, filtering, importing, networking, plotting, similarity
from .__version__ import __version__
from .calculate_scores import calculate_scores
from .fingerprints import Fingerprints
from .fragments import Fragments
from .logging_functions import _init_logger, set_matchms_logger_level
from .metadata import Metadata
from .metadata_collection import MetadataCollection
from .pipeline import Pipeline
from .scores import Scores
from .spectra_collection import SpectraCollection
from .spectrum import Spectrum


_init_logger()


__author__ = "Matchms developers community"
__email__ = "florian.huber@hs-duesseldorf.de"
__all__ = [
    "Fingerprints",
    "Fragments",
    "Metadata",
    "MetadataCollection",
    "Pipeline",
    "Scores",
    "SpectraCollection",
    "SpectraCollectionProcessor",
    "Spectrum",
    "SpectrumProcessor",
    "__version__",
    "calculate_scores",
    "exporting",
    "filtering",
    "importing",
    "networking",
    "plotting",
    "set_matchms_logger_level",
    "similarity",
]
