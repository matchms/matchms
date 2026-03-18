import numpy as np
import pytest
from scipy.sparse import csr_array
from matchms import Spectrum
from matchms.hashing import metadata_hash, spectra_hashes, spectrum_hash, spectrum_hash_arrays
from .builder_Spectrum import SpectrumBuilder


@pytest.fixture
def spectrum() -> Spectrum:
    mz = np.array([100, 200, 290, 490, 490.5], dtype="float")
    intensities = np.array([0.1, 0.11, 1.0, 0.3, 0.4], dtype="float")
    metadata = {"precursor_mz": 505.0}
    builder = SpectrumBuilder().with_mz(mz).with_intensities(intensities).with_metadata(metadata)
    return builder.build()


@pytest.fixture
def sample_csr_data():
    data = np.array([0.5, 1.0, 0.8], dtype=float)
    indices = np.array([10, 20, 15], dtype=int)
    indptr = np.array([0, 2, 3, 3], dtype=int)

    return csr_array((data, indices, indptr), shape=(3, 30))


def test_spectrum_hash(spectrum):
    generated_hash = spectrum_hash(spectrum.peaks)
    assert len(generated_hash) == 20, "Expected hash of length 20."
    assert generated_hash == "9b78750e231ff7ea020b", "Expected different hash"


def test_spectrum_hash_arrays(spectrum):
    mz = spectrum.peaks.mz
    intensities = spectrum.peaks.intensities

    hash_standard = spectrum_hash(spectrum.peaks)
    hash_vectorized = spectrum_hash_arrays(mz, intensities)

    assert hash_vectorized == hash_standard, "hash_arrays and spectrum_hash are different"
    assert len(hash_vectorized) == 20


def test_spectrum_hash_changed_length(spectrum):
    generated_hash = spectrum_hash(spectrum.peaks, hash_length=15)
    assert len(generated_hash) == 15, "Expected hash of length 15."
    assert generated_hash == "9b78750e231ff7e", "Expected different hash"


def test_spectrum_hash_arrays_custom_params(spectrum):
    mz = spectrum.peaks.mz
    intensities = spectrum.peaks.intensities

    hash_v = spectrum_hash_arrays(mz, intensities, hash_length=15, mz_precision=2)
    hash_s = spectrum_hash(spectrum.peaks, hash_length=15, mz_precision=2)

    assert hash_v == hash_s
    assert len(hash_v) == 15


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


def mock_bin_to_mz(bin_idx):
    return (bin_idx * 0.1) + (0.1 / 2)


def mock_mz_to_bin(mz, bin_size):
    return np.floor(mz / bin_size).astype(np.int64)


def test_spectra_hashes_consistency(sample_csr_data):
    hashes = spectra_hashes(sample_csr_data, mock_bin_to_mz)

    mz_0 = mock_bin_to_mz(sample_csr_data.indices[0:2])
    int_0 = sample_csr_data.data[0:2]
    expected_hash_0 = spectrum_hash_arrays(mz_0, int_0, hash_length=20)

    assert len(hashes) == 3
    assert hashes[0] == expected_hash_0


def test_spectra_hashes_custom_params(sample_csr_data):
    length = 10
    hashes = spectra_hashes(sample_csr_data, mock_bin_to_mz, hash_length=length)

    for h in hashes:
        assert len(h) == length


def test_spectra_hashes_with_collection_mock(spectrum):
    bin_size = 0.1
    mz = np.array([100.0, 200.0])
    intensities = np.array([1.0, 0.5])

    bins = mock_mz_to_bin(mz, bin_size)
    csr = csr_array((intensities, bins, [0, 2]), shape=(1, 5000))

    hashes = spectra_hashes(csr, mock_bin_to_mz)

    assert isinstance(hashes[0], str)
    assert len(hashes[0]) == 20
