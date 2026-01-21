import numpy as np
import pytest
from matchms import Spectrum
from matchms.hashing import metadata_hash, spectrum_hash
from .builder_Spectrum import SpectrumBuilder


@pytest.fixture
def spectrum() -> Spectrum:
    mz = np.array([100, 200, 290, 490, 490.5], dtype="float")
    intensities = np.array([0.1, 0.11, 1.0, 0.3, 0.4], dtype="float")
    metadata = {"precursor_mz": 505.0}
    builder = SpectrumBuilder().with_mz(mz).with_intensities(intensities).with_metadata(metadata)
    return builder.build()


def test_spectrum_hash(spectrum):
    generated_hash = spectrum_hash(spectrum.peaks)
    assert len(generated_hash) == 20, "Expected hash of length 20."
    assert generated_hash == "9b78750e231ff7ea020b", "Expected different hash"


def test_spectrum_hash_changed_length(spectrum):
    generated_hash = spectrum_hash(spectrum.peaks, hash_length=15)
    assert len(generated_hash) == 15, "Expected hash of length 15."
    assert generated_hash == "9b78750e231ff7e", "Expected different hash"


def test_spectrum_hash_changed_mz_precision(spectrum: Spectrum):
    mz2 = spectrum.peaks.mz
    mz2[0] += 0.01
    spectrum2 = SpectrumBuilder().from_spectrum(spectrum).with_mz(mz2).build()

    generated_hash_1a = spectrum_hash(spectrum.peaks, mz_precision=1)

    generated_hash_1b = spectrum_hash(spectrum2.peaks, mz_precision=1)
    assert generated_hash_1a == generated_hash_1b, "Expected hashes to be the same"
    # mz_precision = 2
    generated_hash_1a = spectrum_hash(spectrum.peaks, mz_precision=2)

    generated_hash_1b = spectrum_hash(spectrum2.peaks, mz_precision=2)
    assert generated_hash_1a != generated_hash_1b, "Expected hashes to be different"


def test_spectrum_hash_changed_precision(spectrum):
    generated_hash = spectrum_hash(spectrum.peaks, intensity_precision=1)
    assert generated_hash == "aef640708220769ce327", "Expected different hash"


def test_metadata_hash(spectrum):
    generated_hash = metadata_hash(spectrum.metadata)
    assert generated_hash == "23da883766f6cdb37a35", "Expected different hash"


def test_metadata_hash_changed_length(spectrum):
    generated_hash = metadata_hash(spectrum.metadata, hash_length=15)
    assert generated_hash == "23da883766f6cdb", "Expected different hash"
