from typing import Union, List, Tuple
import numpy
from .Spectrum import Spectrum


SpectrumType = Union[Spectrum, None]
ReferencesType = QueriesType = Union[List[object], Tuple[object], numpy.ndarray]
