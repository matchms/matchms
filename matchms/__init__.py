import logging
from . import exporting
from . import filtering
from . import importing
from . import similarity
from .__version__ import __version__
from .calculate_scores import calculate_scores
from .Scores import Scores
from .Spectrum import Spectrum
from .Spikes import Spikes


logging.getLogger(__name__).addHandler(logging.NullHandler())

__author__ = "Netherlands eScience Center"
__email__ = 'generalization@esciencecenter.nl'
__all__ = [
    "__version__",
    "calculate_scores",
    "exporting",
    "filtering",
    "importing",
    "Scores",
    "similarity",
    "Spectrum",
    "Spikes"
]
