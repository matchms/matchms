from typing import List
from typing import Tuple
from typing import Union
from typing import Callable
import numpy
from .Spectrum import Spectrum


SpectrumType = Union[Spectrum, None]
ReferencesType = QueriesType = Union[List[object], Tuple[object], numpy.ndarray]

"""Input for a similarity function"""
Sample = Union[Spectrum, None, object]

"""Result of a similarity function"""
Score = Union[float, Tuple[float, int]]

"""Signature of a similarity function"""
SimilarityFunction = Callable[[Sample, Sample], Score]
