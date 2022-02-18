from typing import List, NewType, Tuple, Union
import numpy


SpectrumType = NewType("Spectrum", object)
ReferencesType = QueriesType = Union[List[object], Tuple[object], numpy.ndarray]


"""Result of a similarity function"""
Score = Union[float, Tuple[float, int]]
