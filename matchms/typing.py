from typing import Union, List, Tuple
from numpy import ndarray
from .Spectrum import Spectrum


SpectrumType = Union[Spectrum, None]
ReferencesType = QueriesType = Union[List[object], Tuple[object], ndarray]
