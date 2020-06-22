import numpy
import pytest
from matchms import Spectrum
from matchms.similarity import ParentmassMatch


def test_parentmass_match():
    "Test with default tolerance."
    spectrum_1 = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"parent_mass": 100.0})

    spectrum_2 = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"parent_mass": 101.0})

    similarity_score = ParentmassMatch()
    score = similarity_score(spectrum_1, spectrum_2)
    assert not score, "Expected different score."


def test_parentmass_match_tolerance2():
    "Test with tolerance > difference."
    spectrum_1 = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"parent_mass": 100.0})

    spectrum_2 = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"parent_mass": 101.0})

    similarity_score = ParentmassMatch(tolerance=2.0)
    score = similarity_score(spectrum_1, spectrum_2)
    assert score, "Expected different score."


def test_parentmass_match_missing_parentmass():
    "Test with missing parentmass."
    spectrum_1 = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={"parent_mass": 100.0})

    spectrum_2 = Spectrum(mz=numpy.array([], dtype="float"),
                          intensities=numpy.array([], dtype="float"),
                          metadata={})

    similarity_score = ParentmassMatch(tolerance=2.0)

    with pytest.raises(AssertionError) as msg:
        _ = similarity_score(spectrum_1, spectrum_2)

    expected_message_part = "Missing parent mass."
    assert expected_message_part in str(msg.value), "Expected particular error message."
