import numpy
from matchms import Spectrum
from matchms.similarity import ParentmassMatchParallel


def test_parentmass_match():
    "Test with default tolerance."
    spectrum_1 = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"parentmass": 100.0})

    spectrum_2 = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"parentmass": 101.0})

    spectrum_a = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"parentmass": 99.0})

    spectrum_b = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"parentmass": 98.0})

    similarity_score = ParentmassMatchParallel()
    scores = similarity_score([spectrum_1, spectrum_2], [spectrum_a, spectrum_b])
    assert numpy.all(scores == numpy.array([[0., 0.], [0., 0.]])), "Expected different score."


def test_parentmass_match_tolerance2():
    """Test with tolerance=2."""
    spectrum_1 = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"parentmass": 100.0})

    spectrum_2 = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"parentmass": 101.0})

    spectrum_a = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"parentmass": 99.0})

    spectrum_b = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"parentmass": 98.0})

    similarity_score = ParentmassMatchParallel(tolerance=2.0)
    scores = similarity_score([spectrum_1, spectrum_2], [spectrum_a, spectrum_b])
    assert numpy.all(scores == numpy.array([[1., 1.], [1., 0.]])), "Expected different score."
