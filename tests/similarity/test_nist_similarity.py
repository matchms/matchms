from matchms.similarity.BaseSimilarity import BaseSimilarity
from matchms.similarity import NISTSimilarity
from matchms import Fragments

import numpy as np

from .test_spectrum_similarity_functions import spectra

def test_is_base_similarity():
    sut = NISTSimilarity()
    assert issubclass(type(sut), BaseSimilarity)

def get_peak_pairs(peaks_a: Fragments, peaks_b: Fragments, tolerance: float):
    peaks_b


def test_vectorize_spectrum(spectra):
    
