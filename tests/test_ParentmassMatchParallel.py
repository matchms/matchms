import numpy
from matchms import Spectrum
from matchms.similarity import ParentmassMatchParallel
from matchms.similarity.ParentmassMatchParallel import \
    calculate_parentmass_scores


def test_parentmass_match():
    "Test with default tolerance."
    spectrum_1 = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"parent_mass": 100.0})

    spectrum_2 = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"parent_mass": 101.0})

    spectrum_a = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"parent_mass": 99.0})

    spectrum_b = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"parent_mass": 98.0})

    similarity_score = ParentmassMatchParallel()
    scores = similarity_score([spectrum_1, spectrum_2], [spectrum_a, spectrum_b])
    assert numpy.all(scores == numpy.array([[False, False],
                                            [False, False]])), "Expected different scores."


def test_parentmass_match_tolerance2():
    """Test with tolerance=2."""
    spectrum_1 = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"parent_mass": 100.0})

    spectrum_2 = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"parent_mass": 101.0})

    spectrum_a = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"parent_mass": 99.0})

    spectrum_b = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"parent_mass": 98.0})

    similarity_score = ParentmassMatchParallel(tolerance=2.0)
    scores = similarity_score([spectrum_1, spectrum_2], [spectrum_a, spectrum_b])
    assert numpy.all(scores == numpy.array([[True, True],
                                            [True, False]])), "Expected different scores."


def test_calculate_parentmass_scores_compiled():
    """Test the underlying score function (numba compiled)."""
    parentmasses_ref = numpy.asarray([101, 200, 300])
    parentmasses_query = numpy.asarray([100, 301])
    scores = calculate_parentmass_scores(parentmasses_ref, parentmasses_query, tolerance=2.0)
    assert numpy.all(scores == numpy.array([[1., 0.],
                                            [0., 0.],
                                            [0., 1.]])), "Expected different scores."


def test_calculate_parentmass_scores():
    """Test the underlying score function (non-compiled)."""
    parentmasses_ref = numpy.asarray([101, 200, 300])
    parentmasses_query = numpy.asarray([100, 301])
    scores = calculate_parentmass_scores.py_func(parentmasses_ref, parentmasses_query, tolerance=2.0)
    assert numpy.all(scores == numpy.array([[True, False],
                                            [False, False],
                                            [False, True]])), "Expected different scores."
