from typing import List
from typing import Tuple
from typing import Union
import numpy
from .Spectrum import Spectrum


SpectrumType = Union[Spectrum, None]
ReferencesType = QueriesType = Union[List[object], Tuple[object], numpy.ndarray]
