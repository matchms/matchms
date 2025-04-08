from matchms.filtering.SpectrumProcessor import SpectrumProcessor
from . import exporting, filtering, importing, networking, plotting, similarity
from .__version__ import __version__
from .calculate_scores import calculate_scores
from .Fingerprints import Fingerprints
from .Fragments import Fragments
from .logging_functions import _init_logger, set_matchms_logger_level, set_rdkit_logger_level
from .Metadata import Metadata
from .Pipeline import Pipeline
from .Scores import Scores
from .Spectrum import Spectrum


_init_logger()

try:  # rdkit is not included in pip package
    from rdkit import Chem

    set_rdkit_logger_level("rdApp.error")
except ImportError:
    _has_rdkit = False
    from collections import UserString

    class ChemMock(UserString):
        def __call__(self, *args, **kwargs):
            return self

        def __getattr__(self, key):
            return self

    Chem = AllChem = ChemMock("")


__author__ = "Matchms developers community"
__email__ = "florian.huber@hs-duesseldorf.de"
__all__ = [
    "__version__",
    "calculate_scores",
    "exporting",
    "filtering",
    "Fingerprints",
    "Fragments",
    "importing",
    "Metadata",
    "networking",
    "Pipeline",
    "plotting",
    "Scores",
    "set_matchms_logger_level",
    "similarity",
    "Spectrum",
    "SpectrumProcessor",
]
