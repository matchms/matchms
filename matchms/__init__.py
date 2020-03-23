# -*- coding: utf-8 -*-
"""Documentation about matchms"""
import logging

from .helper_functions import *
from .MS_functions import *
from .MS_library_search import *
from .MS_similarity_classical import *
from .networking import *
from .plotting_functions import *
from .similarity_measure import *

from .__version__ import __version__

logging.getLogger(__name__).addHandler(logging.NullHandler())

__author__ = "Netherlands eScience Center"
__email__ = 'generalization@esciencecenter.nl'
