from collections.abc import Callable
from typing import TYPE_CHECKING, Any, TypeAlias
import numpy as np


if TYPE_CHECKING:
    from matchms.spectrum import Spectrum

    SpectrumType: TypeAlias = Spectrum
else:
    SpectrumType: TypeAlias = Any
ReferencesType = QueriesType = list[object] | tuple[object] | np.ndarray
ScoreFilter = Callable[[np.ndarray], bool]

"""Result of a similarity function"""
Score = float | tuple[float, int]
