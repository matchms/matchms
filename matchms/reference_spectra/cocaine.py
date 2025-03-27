"""Submodule providing a reference spectrum for cocaine."""

import numpy as np
from ..Spectrum import Spectrum


COCAINE_PRECURSOR_MZ: float = 304.153137

COCAINE_MZ: np.ndarray = np.array(
    [
        82.064789,
        105.033249,
        109.213745,
        119.04921,
        150.0914,
        182.117676,
        185.804688,
        226.579071,
        304.153137,
    ]
)

COCAINE_INTENSITIES: np.ndarray = np.array(
    [
        13342.493164,
        3264.133545,
        1584.27478,
        2382.930908,
        3257.366211,
        133504.296875,
        1849.140137,
        1391.734497,
        86052.375,
    ]
)


def cocaine() -> Spectrum:
    """Return a reference spectrum for cocaine."""
    return Spectrum(
        mz=COCAINE_MZ,
        intensities=COCAINE_INTENSITIES,
        metadata={"precursor_mz": COCAINE_PRECURSOR_MZ},
    )
