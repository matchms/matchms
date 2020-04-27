from matchms import Spikes
import numpy
import pytest


def test_spikes_init():

    mz = numpy.array([10, 20, 30], dtype="float")
    intensities = numpy.array([100, 20, 300], dtype="float")

    peaks = Spikes(mz=mz, intensities=intensities)

    assert peaks is not None
    assert numpy.allclose(mz, peaks.mz)
    assert numpy.allclose(intensities, peaks.intensities)


def test_spikes_mz_wrong_numpy_dtype():

    mz = numpy.array([10, 20, 30], dtype="int")
    intensities = numpy.array([100, 20, 300], dtype="float")

    with pytest.raises(AssertionError) as msg:
        _ = Spikes(mz=mz, intensities=intensities)

    assert str(msg.value) == "Input argument 'mz' should be an array of type float."


def test_spikes_intensities_wrong_numpy_dtype():

    mz = numpy.array([10, 20, 30], dtype="float")
    intensities = numpy.array([100, 20, 300], dtype="int")

    with pytest.raises(AssertionError) as msg:
        _ = Spikes(mz=mz, intensities=intensities)

    assert str(msg.value) == "Input argument 'intensities' should be an array of type float."


def test_spikes_same_shape():

    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([100, 20, 300], dtype="float")

    with pytest.raises(AssertionError) as msg:
        _ = Spikes(mz=mz, intensities=intensities)

    assert str(msg.value) == "Input arguments 'mz' and 'intensities' should be the same shape."


def test_spikes_mz_wrong_data_type():

    mz = [10, 20, 30]
    intensities = numpy.array([100, 20, 300], dtype="float")

    with pytest.raises(AssertionError) as msg:
        _ = Spikes(mz=mz, intensities=intensities)

    assert str(msg.value) == "Input argument 'mz' should be a numpy.array."


def test_spikes_intensities_wrong_data_type():

    mz = numpy.array([10, 20, 30], dtype="float")
    intensities = [100, 20, 300]

    with pytest.raises(AssertionError) as msg:
        _ = Spikes(mz=mz, intensities=intensities)

    assert str(msg.value) == "Input argument 'intensities' should be a numpy.array."


def test_spikes_dot_clone():

    mz = numpy.array([10, 20, 30], dtype="float")
    intensities = numpy.array([100, 20, 300], dtype="float")

    peaks = Spikes(mz=mz, intensities=intensities)

    peaks_cloned = peaks.clone()

    assert peaks == peaks_cloned
    assert peaks is not peaks_cloned


def test_spikes_unpack():

    mz = numpy.array([10, 20, 30], dtype="float")
    intensities = numpy.array([100, 20, 300], dtype="float")

    peaks = Spikes(mz=mz, intensities=intensities)

    mz_unpacked, intensities_unpacked = peaks

    assert numpy.allclose(mz, mz_unpacked)
    assert numpy.allclose(intensities, intensities_unpacked)
