import os
from types import GeneratorType

import numpy as np
import pytest
from matchms.Spectrum import Spectrum
from matchms.importing import load_from_mzspeclib


module_root = os.path.join(os.path.dirname(__file__), "..")

@pytest.fixture
def testdata():
    return os.path.join(module_root, 'testdata','rcx_gc-ei_ms_20201028_perylene.mzspeclib')

@pytest.fixture
def spectrum(testdata) -> Spectrum:
    return list(load_from_mzspeclib(testdata))[0]

def test_is_generator(testdata):
    actual = load_from_mzspeclib(testdata)
    assert isinstance(actual, GeneratorType)


def test_has_spectrum(testdata):
    actual = list(load_from_mzspeclib(testdata))
    assert len(actual) == 1


def test_spectrum_has_name(spectrum):
    assert spectrum.get('compound_name') == 'Perylene'


def test_spectrum_has_correct_num_peaks(spectrum):
    assert len(spectrum.peaks) == 19
    assert spectrum.get('num_peaks') == '19'

def test_spectrum_has_correct_mzs(spectrum):
    actual = spectrum.peaks.mz
    expected = np.array([
        112.03071,113.03854,124.03076,124.53242,125.03855,
        125.54019,126.04636,126.54804,222.04645,224.06192,
        226.04175,246.04646,248.06204,249.07072,250.07765,
        251.07967,252.09323,253.09656,254.09985
    ], dtype=np.float32)

    assert np.array_equal(actual, expected)


def test_has_multiple_spectra():
    multi_spectra_file = os.path.join(module_root, 'testdata','rcx_ei_mzspeclib')
    assert len(list(load_from_mzspeclib(multi_spectra_file))) == 10