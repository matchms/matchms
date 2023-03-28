"""Helper functions related to hashing.
"""
import hashlib
import json
from .Fragments import Fragments


def spectrum_hash(peaks: Fragments, hash_length: int = 20,
                  mz_precision: int = 5, intensity_precision: int = 2):
    """Compute hash from mz-intensity pairs of all peaks in spectrum.
    Method is inspired by SPLASH (doi:10.1038/nbt.3689).
    """
    mz_precision_factor = 10 ** mz_precision
    intensity_precision_factor = 10 ** intensity_precision

    def format_mz(mz):
        return int(mz * mz_precision_factor)

    def format_intensity(intensity):
        return int(intensity * intensity_precision_factor)

    peak_list = [(format_mz(peak[0]), format_intensity(peak[1])) for peak in peaks.to_numpy]
    # Sort by increasing m/z and then by decreasing intensity
    peak_list.sort(key=lambda x: (x[0], - x[1]))

    encoded = " ".join(":".join(map(str, x)) for x in peak_list).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:hash_length]


def metadata_hash(metadata: dict, hash_length: int = 20):
    """Compute hash from metadata dictionary.
    """
    encoded = json.dumps(metadata, sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()[:hash_length]
