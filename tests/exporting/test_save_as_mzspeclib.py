from ..builder_Spectrum import SpectrumBuilder
import numpy as np
import os

from matchms.exporting import save_as_mzspeclib

def test_creates_file(tmp_path):
    outpath = tmp_path / "test.mzspeclib"
    save_as_mzspeclib([], outpath)
    assert os.path.isfile(outpath)

def test_has_header(tmp_path):
    name = 'test'
    outpath = tmp_path / f"{name}.mzspeclib"
    save_as_mzspeclib([], outpath)

    with open(outpath,'r') as file:
        lines = file.readlines()

        assert lines[0] == '<mzSpecLib>\n'
        assert lines[1] == 'MS:1003186|library format version=1.0\n'
        assert lines[2] == 'MS:1003188|library name=test\n'
        assert lines[3] == '<AttributeSet Spectrum=all>\n'
        assert lines[4] == '<AttributeSet Analyte=all>\n'
        assert lines[5] == '<AttributeSet Interpretation=all>\n'

def test_has_spectrum(tmp_path):
    spectrum = SpectrumBuilder().with_mz(np.array([10, 20, 30], dtype=np.float32)).with_intensities(np.array([100, 150, 80], dtype=np.float32)).build()
    outpath = tmp_path / "test.mzspeclib"
    save_as_mzspeclib([spectrum], outpath)

    with open(outpath, 'r') as file:
        lines = file.readlines()
        assert '<Spectrum=1>\n' in lines
        assert '<Peaks>\n' in lines
        assert len(lines) == 12
 

