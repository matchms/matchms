from typing import List
from typing import Tuple
from typing import Union
from numpy import ndarray
from .Spectrum import Spectrum


SpectrumType = Union[Spectrum, None]
ReferencesType = QueriesType = Union[List[object], Tuple[object], ndarray]
