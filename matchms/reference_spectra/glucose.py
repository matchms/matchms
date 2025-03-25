"""Submodule providing a reference spectrum for glucose."""

import numpy as np
from ..Spectrum import Spectrum


GLUCOSE_PRECURSOR_MZ: float = 203.05

GLUCOSE_MZ: np.ndarray = np.array(
    [
        82.952148,
        105.270447,
        112.789398,
        121.208,
        129.116699,
        131.104095,
        131.990692,
        135.007935,
        142.50119,
        143.102905,
        158.092255,
        160.235291,
        173.100464,
        185.152679,
    ]
)

GLUCOSE_INTENSITIES: np.ndarray = np.array(
    [
        798.858887,
        1257.253418,
        3923.249023,
        1952.965454,
        169.587341,
        412.055176,
        309.939514,
        520.569153,
        555.742432,
        13786.814453,
        1758.816406,
        408.889771,
        892.346924,
        9220.535156,
    ]
)


def glucose() -> Spectrum:
    """Return a reference spectrum for glucose."""
    return Spectrum(
        mz=GLUCOSE_MZ,
        intensities=GLUCOSE_INTENSITIES,
        metadata={"precursor_mz": GLUCOSE_PRECURSOR_MZ},
    )
