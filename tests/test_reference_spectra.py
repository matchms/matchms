"""Simple test submodule to verify that all reference spectra can be instantiated."""

from matchms.reference_spectra import (aspirin, cocaine, glucose,
                                       hydroxy_cholesterol, phenylalanine,
                                       salicin)


def test_aspirin():
    """Test if aspirin reference spectrum can be instantiated."""
    aspirin_spectrum = aspirin()
    assert aspirin_spectrum is not None


def test_cocaine():
    """Test if cocaine reference spectrum can be instantiated."""
    cocaine_spectrum = cocaine()
    assert cocaine_spectrum is not None


def test_glucose():
    """Test if glucose reference spectrum can be instantiated."""
    glucose_spectrum = glucose()
    assert glucose_spectrum is not None


def test_hydroxy_cholesterol():
    """Test if hydroxy-cholesterol reference spectrum can be instantiated."""
    hydroxy_cholesterol_spectrum = hydroxy_cholesterol()
    assert hydroxy_cholesterol_spectrum is not None


def test_phenylalanine():
    """Test if phenylalanine reference spectrum can be instantiated."""
    phenylalanine_spectrum = phenylalanine()
    assert phenylalanine_spectrum is not None


def test_salicin():
    """Test if salicin reference spectrum can be instantiated."""
    salicin_spectrum = salicin()
    assert salicin_spectrum is not None
