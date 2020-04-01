# -*- coding: utf-8 -*-
"""Documentation about matchms"""


import matchms.exporting
import matchms.harmonization
import matchms.similarity

from matchms.importing.load_from_mgf import load_from_mgf
from matchms.Spectrum import Spectrum
from matchms.Scores import Scores

from .__version__ import __version__

import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())

__author__ = "Netherlands eScience Center"
__email__ = 'generalization@esciencecenter.nl'
