from unittest.mock import Mock, patch
import numpy as np
from matchms import Spectrum
from matchms.importing.load_from_usi import load_from_usi


@patch('matchms.importing.load_from_usi.requests.get')
def normal_test(mock_get):
    mock_get.return_value = Mock(ok=True)
    mock_get.return_value.json.return_value = {'peaks': [[1.,2.],[3.,4.]]}
    spec = load_from_usi('something')
    expected_metadata = {'usi': 'something', 'url': 'https://metabolomics-usi.ucsd.edu/json/?usi=something', 'precursor_mz': None}
    expected = Spectrum(np.array([1.,3. ]),np.array([2.,4.]),expected_metadata)
    assert spec == expected

@patch('matchms.importing.load_from_usi.requests.get')
def test_404(mock_get):
    mock_get.return_value = Mock(ok=True)
    mock_get.return_value.status_code = 404
    mock_get.return_value.json.return_value = None
    spec = load_from_usi('something')
    expected = None
    assert spec == expected

@patch('matchms.importing.load_from_usi.requests.get')
def test_no_peaks(mock_get):
    mock_get.return_value = Mock(ok=True)
    mock_get.return_value.json.return_value = {'peaks': []}
    spec = load_from_usi('something')
    expected = None
    assert spec == expected