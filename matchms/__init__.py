# -*- coding: utf-8 -*-
"""Documentation about matchms"""
import logging

import matchms.helper_functions as helper_functions
import matchms.MS_functions as MS_functions
import matchms.MS_library_search as MS_library_search
import matchms.MS_similarity_classical as MS_similarity_classical 
import matchms.networking as networking
import matchms.plotting_functions as plotting_functions
import matchms.similarity_measure as similarity_measure

from .__version__ import __version__

logging.getLogger(__name__).addHandler(logging.NullHandler())

__author__ = "Netherlands eScience Center"
__email__ = 'generalization@esciencecenter.nl'
