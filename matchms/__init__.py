# -*- coding: utf-8 -*-
"""Documentation about matchms"""

from .exporting import *
from .harmonization import *
from .importing import *
from .similarity import *

from .Spectrum import Spectrum
from .Scores import Scores
from .calculate_scores import calculate_scores

from .__version__ import __version__

import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

__author__ = "Netherlands eScience Center"
__email__ = 'generalization@esciencecenter.nl'
