import filecmp
import os
from pathlib import Path
import numpy as np
from matchms.exporting import save_as_mzspeclib
from ..builder_Spectrum import SpectrumBuilder


def test_creates_file(tmp_path):
    outpath = tmp_path / "test.mzspeclib"
    save_as_mzspeclib([], outpath)
    assert os.path.isfile(outpath)

def test_has_header(tmp_path):
    name = 'test'
    outpath = tmp_path / f"{name}.mzspeclib"
    save_as_mzspeclib([], outpath)

    with open(outpath,'r', encoding='UTF-8') as file:
        lines = file.readlines()

        assert lines[0] == '<mzSpecLib>\n'
        assert lines[1] == 'MS:1003186|library format version=1.0\n'
        assert lines[2] == 'MS:1003188|library name=test\n'
        assert lines[3] == '<AttributeSet Spectrum=all>\n'
        assert lines[4] == '<AttributeSet Analyte=all>\n'
        assert lines[5] == '<AttributeSet Interpretation=all>\n'

def test_has_spectrum(tmp_path):
    spectrum = SpectrumBuilder() \
      .with_mz(np.array([10, 20, 30], dtype=np.float32)) \
      .with_intensities(np.array([100, 150, 80], dtype=np.float32)) \
      .build()
    outpath = tmp_path / "test.mzspeclib"
    save_as_mzspeclib([spectrum], outpath)

    with open(outpath, 'r', encoding='UTF-8') as file:
        lines = file.readlines()
        assert '<Spectrum=1>\n' in lines
        assert '<Peaks>\n' in lines
        assert len(lines) == 13


def test_has_analyte(tmp_path):
    mz = np.array([35, 36, 37, 38], dtype=np.float64)
    intensities = np.array([169.85, 999, 53.95, 323.71], dtype=np.float64)
    metadata = {'formula': 'ClH', 'compound_name': 'Hydrogen chloride'}
    spectrum = SpectrumBuilder().with_mz(mz).with_intensities(intensities).with_metadata(metadata).build()

    outpath = tmp_path / "test.mzspeclib"
    save_as_mzspeclib([spectrum], outpath)
    current_dir = Path(__file__).parent
    expected = os.path.join(current_dir, "../testdata/Hydrogen_chloride.mzspeclib")
    assert filecmp.cmp(outpath, expected)


def test_write_custom_keys(tmp_path):
    spectrum = SpectrumBuilder().with_metadata({'instrument': 'GC Orbitrap'}).build()
    outpath = tmp_path / "test.mzspeclib"
    save_as_mzspeclib([spectrum], outpath)

    with open(outpath, 'r', encoding='UTF-8') as file:
        lines = file.readlines()
        assert '[1]MS:1003275|other attribute name=instrument\n' in lines
        assert '[1]MS:1003276|other attribute value=GC Orbitrap\n' in lines


def test_mapped_attributes(tmp_path):
    spectrum = SpectrumBuilder().with_metadata({'ionmode': 'positive'}).build()
    outpath = tmp_path / "test.mzspeclib"
    save_as_mzspeclib([spectrum], outpath)
    with open(outpath, 'r', encoding='UTF-8') as file:
        lines = file.readlines()
        assert 'MS:1000465|scan polarity=MS:1000130|positive scan\n' in lines

def test_peak_comments(tmp_path):
    mz = np.array([35.7], dtype=np.float64)
    intensities = np.array([1337], dtype=np.float64)
    metadata = {'peak_comments': {35.7: 'test'}}
    spectrum = SpectrumBuilder().with_mz(mz).with_intensities(intensities).with_metadata(metadata).build()
    outpath = tmp_path / "test.mzspeclib"

    save_as_mzspeclib([spectrum], outpath)
    with open(outpath, 'r', encoding='UTF-8') as file:
        lines = file.readlines()
        assert '35.7\t1337\ttest\n' in lines

def test_has_attributes_with_units(tmp_path):
    spectrum = SpectrumBuilder().with_metadata({'collision_energy': '70eV'}).build()
    outpath = tmp_path / "test.mzspeclib"
    save_as_mzspeclib([spectrum], outpath)

    with open(outpath, 'r', encoding='UTF-8') as file:
        lines = file.readlines()
        assert '[1]MS:1000045|collision energy=70\n' in lines
        assert '[1]UO:0000000|unit=UO:0000266|electronvolt\n' in lines
