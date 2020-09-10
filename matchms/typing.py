from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
import numpy
from .Spectrum import Spectrum


SpectrumType = Optional[Spectrum]
ReferencesType = QueriesType = Union[List[object], Tuple[object], numpy.ndarray]

"""Input for a similarity function"""
Sample = Union[Spectrum, None, object]

"""Result of a similarity function"""
Score = Union[float, Tuple[float, int]]
