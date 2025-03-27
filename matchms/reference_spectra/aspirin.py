"""Submodule providing a reference spectrum for aspirin."""

import numpy as np
from ..Spectrum import Spectrum


ASPIRIN_PRECURSOR_MZ: float = 181.0490

ASPIRIN_MZ: np.ndarray = np.array(
    [
        50.0149,
        51.0228,
        53.0385,
        55.0177,
        65.0384,
        77.0383,
        79.054,
        80.0254,
        81.0334,
        91.0541,
        92.0254,
        93.0333,
        94.0411,
        95.049,
        98.0361,
        105.0333,
        105.0445,
        107.0489,
        111.0439,
        120.0203,
        121.0282,
        121.0394,
        133.0282,
        135.0438,
        138.0308,
        149.0231,
        163.0386,
        167.0337,
        181.0491,
    ]
)

ASPIRIN_INTENSITIES: np.ndarray = np.array(
    [
        49377.4,
        53422.1,
        454244.5,
        57881.0,
        1997532.0,
        825848.2,
        1153465.5,
        96202.8,
        58626.8,
        44573.8,
        394779.8,
        1129287.8,
        56357.2,
        1654496.5,
        72487.3,
        707899.4,
        1119356.4,
        207437.4,
        587441.8,
        166384.0,
        9695889.0,
        2506571.2,
        5824675.0,
        6124332.0,
        78621.8,
        34285450.0,
        18191732.0,
        69049.3,
        120675.9,
    ]
)


def aspirin() -> Spectrum:
    """Return a reference spectrum for aspirin."""
    return Spectrum(
        mz=ASPIRIN_MZ,
        intensities=ASPIRIN_INTENSITIES,
        metadata={"precursor_mz": ASPIRIN_PRECURSOR_MZ},
    )
