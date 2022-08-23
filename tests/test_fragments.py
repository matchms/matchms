import numpy as np
import pytest
from matchms import Fragments


def test_fragments_init():

    mz = numpy.array([10, 20, 30], dtype="float")
    intensities = numpy.array([100, 20, 300], dtype="float")

    peaks = Fragments(mz=mz, intensities=intensities)

    assert peaks is not None
    assert numpy.allclose(mz, peaks.mz)
    assert numpy.allclose(intensities, peaks.intensities)


def test_fragments_mz_wrong_numpy_dtype():

    mz = numpy.array([10, 20, 30], dtype="int")
    intensities = numpy.array([100, 20, 300], dtype="float")

    with pytest.raises(AssertionError) as msg:
        _ = Fragments(mz=mz, intensities=intensities)

    assert str(msg.value) == "Input argument 'mz' should be an array of type float."


def test_fragments_intensities_wrong_numpy_dtype():

    mz = numpy.array([10, 20, 30], dtype="float")
    intensities = numpy.array([100, 20, 300], dtype="int")

    with pytest.raises(AssertionError) as msg:
        _ = Fragments(mz=mz, intensities=intensities)

    assert str(msg.value) == "Input argument 'intensities' should be an array of type float."


def test_fragments_same_shape():

    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([100, 20, 300], dtype="float")

    with pytest.raises(AssertionError) as msg:
        _ = Fragments(mz=mz, intensities=intensities)

    assert str(msg.value) == "Input arguments 'mz' and 'intensities' should be the same shape."


def test_fragments_mz_wrong_data_type():

    mz = [10, 20, 30]
    intensities = numpy.array([100, 20, 300], dtype="float")

    with pytest.raises(AssertionError) as msg:
        _ = Fragments(mz=mz, intensities=intensities)

    assert str(msg.value) == "Input argument 'mz' should be a numpy.array."


def test_fragments_intensities_wrong_data_type():

    mz = numpy.array([10, 20, 30], dtype="float")
    intensities = [100, 20, 300]

    with pytest.raises(AssertionError) as msg:
        _ = Fragments(mz=mz, intensities=intensities)

    assert str(msg.value) == "Input argument 'intensities' should be a numpy.array."


def test_fragments_dot_clone():

    mz = numpy.array([10, 20, 30], dtype="float")
    intensities = numpy.array([100, 20, 300], dtype="float")

    peaks = Fragments(mz=mz, intensities=intensities)

    peaks_cloned = peaks.clone()

    assert peaks == peaks_cloned
    assert peaks is not peaks_cloned


def test_fragments_getitem():

    mz = numpy.array([10, 20, 30], dtype="float")
    intensities = numpy.array([100, 20, 300], dtype="float")

    peaks = Fragments(mz=mz, intensities=intensities)

    assert numpy.allclose(peaks[1], numpy.array(mz[1], intensities[1]))
    assert numpy.allclose(peaks[:],
                          numpy.stack((peaks.mz, peaks.intensities)))


def test_fragments_to_numpy():
    """Test conversion to stacked numpy array"""
    mz = numpy.array([10, 20, 30], dtype="float")
    intensities = numpy.array([100, 99.9, 300], dtype="float")

    peaks = Fragments(mz=mz, intensities=intensities)

    assert numpy.allclose(peaks.to_numpy, numpy.array([[10., 100.],
                                                       [20., 99.9],
                                                       [30., 300.]]))
