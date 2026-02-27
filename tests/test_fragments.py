import numpy as np
import pytest
from matchms.Fragments import Fragments


@pytest.mark.parametrize(
    "dtype",
    [
        np.float16,
        np.float32,
        np.float64,
        float,
        "float",
    ],
)
def test_fragments_init(dtype):
    mz = np.array([10, 20, 30], dtype=dtype)
    intensities = np.array([100, 20, 300], dtype=dtype)

    peaks = Fragments(mz=mz, intensities=intensities)

    assert peaks is not None
    assert np.allclose(mz, peaks.mz)
    assert np.allclose(intensities, peaks.intensities)


def test_fragments_mz_wrong_numpy_dtype():
    mz = np.array([10, 20, 30], dtype="int")
    intensities = np.array([100, 20, 300], dtype="float")

    with pytest.raises(AssertionError) as msg:
        _ = Fragments(mz=mz, intensities=intensities)

    assert str(msg.value) == "Input argument 'mz' should be an array of type float."


def test_fragments_intensities_wrong_numpy_dtype():
    mz = np.array([10, 20, 30], dtype="float")
    intensities = np.array([100, 20, 300], dtype="int")

    with pytest.raises(AssertionError) as msg:
        _ = Fragments(mz=mz, intensities=intensities)

    assert str(msg.value) == "Input argument 'intensities' should be an array of type float."


def test_fragments_same_shape():
    mz = np.array([10, 20, 30, 40], dtype="float")
    intensities = np.array([100, 20, 300], dtype="float")

    with pytest.raises(AssertionError) as msg:
        _ = Fragments(mz=mz, intensities=intensities)

    assert str(msg.value) == "Input arguments 'mz' and 'intensities' should be the same shape."


def test_fragments_mz_wrong_data_type():
    mz = [10, 20, 30]
    intensities = np.array([100, 20, 300], dtype="float")

    with pytest.raises(AssertionError) as msg:
        _ = Fragments(mz=mz, intensities=intensities)

    assert str(msg.value) == "Input argument 'mz' should be a np.array."


def test_fragments_intensities_wrong_data_type():
    mz = np.array([10, 20, 30], dtype="float")
    intensities = [100, 20, 300]

    with pytest.raises(AssertionError) as msg:
        _ = Fragments(mz=mz, intensities=intensities)

    assert str(msg.value) == "Input argument 'intensities' should be a np.array."


def test_fragments_dot_clone():
    mz = np.array([10, 20, 30], dtype="float")
    intensities = np.array([100, 20, 300], dtype="float")

    peaks = Fragments(mz=mz, intensities=intensities)

    peaks_cloned = peaks.clone()

    assert peaks == peaks_cloned
    assert peaks is not peaks_cloned


def test_fragments_getitem():
    mz = np.array([10, 20, 30], dtype="float")
    intensities = np.array([100, 20, 300], dtype="float")

    peaks = Fragments(mz=mz, intensities=intensities)

    assert np.allclose(peaks[1], np.array(mz[1], intensities[1]))
    assert np.allclose(peaks[:], np.stack((peaks.mz, peaks.intensities)))


def test_fragments_to_numpy():
    """Test conversion to stacked numpy array"""
    mz = np.array([10, 20, 30], dtype="float")
    intensities = np.array([100, 99.9, 300], dtype="float")

    peaks = Fragments(mz=mz, intensities=intensities)

    assert np.allclose(peaks.to_numpy, np.array([[10.0, 100.0], [20.0, 99.9], [30.0, 300.0]]))
