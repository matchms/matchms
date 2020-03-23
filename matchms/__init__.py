# -*- coding: utf-8 -*-
"""Documentation about matchms"""
import logging

from matchms import helper_functions
from matchms import MS_functions
from matchms import MS_library_search
from matchms import MS_similarity_classical
from matchms import networking
from matchms import plotting_functions
from matchms import similarity_measure

from .__version__ import __version__

logging.getLogger(__name__).addHandler(logging.NullHandler())

__author__ = "Netherlands eScience Center"
__email__ = 'generalization@esciencecenter.nl'
