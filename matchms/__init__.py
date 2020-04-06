# -*- coding: utf-8 -*-
"""Documentation about matchms"""

from .calculate_scores import calculate_scores
from .Scores import Scores
from .Spectrum import Spectrum

import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

__author__ = "Netherlands eScience Center"
__email__ = 'generalization@esciencecenter.nl'
__all__ = [
    "calculate_scores",
    "exporting",
    "filtering",
    "importing",
    "Scores",
    "similarity",
    "Spectrum"
]
