from typing import List
from typing import NewType
from typing import Tuple
from typing import Union
import numpy


SpectrumType = NewType("Spectum", object)
ReferencesType = QueriesType = Union[List[object], Tuple[object], numpy.ndarray]


"""Result of a similarity function"""
Score = Union[float, Tuple[float, int]]
