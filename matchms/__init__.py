# -*- coding: utf-8 -*-
"""Documentation about matchms"""

import matchms.importing
import matchms.exporting
import matchms.harmonization
import matchms.similarity
from matchms.Spectrum import Spectrum
import matchms.Scores

from .__version__ import __version__

import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

__author__ = "Netherlands eScience Center"
__email__ = 'generalization@esciencecenter.nl'
