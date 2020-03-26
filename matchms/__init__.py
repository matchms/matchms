# -*- coding: utf-8 -*-
"""Documentation about matchms"""

from .helper_functions import *
from .ms_functions import *
from .ms_library_search import *
from .ms_similarity_classical import *
from .networking import *
from .plotting_functions import *
from .similarity_measure import *
from .similarity_measure import *

from .__version__ import __version__

import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

__author__ = "Netherlands eScience Center"
__email__ = 'generalization@esciencecenter.nl'
