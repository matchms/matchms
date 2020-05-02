import numpy
from matchms import Spectrum
from matchms.filtering import harmonize_undefined_smiles


def test_harmonize_undefined_smiles_empty_string():

    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"smiles": ""})

    spectrum = harmonize_undefined_smiles(spectrum_in)
    assert spectrum.get("smiles") == ""


def test_harmonize_undefined_smiles_na_1():

    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"smiles": "n/a"})

    spectrum = harmonize_undefined_smiles(spectrum_in)
    assert spectrum.get("smiles") == ""


def test_harmonize_undefined_smiles_na_2():

    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"smiles": "N/A"})

    spectrum = harmonize_undefined_smiles(spectrum_in)
    assert spectrum.get("smiles") == ""


def test_harmonize_undefined_smiles_na_3():

    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"smiles": "NA"})

    spectrum = harmonize_undefined_smiles(spectrum_in)
    assert spectrum.get("smiles") == ""


def test_harmonize_undefined_smiles_no_data():

    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"smiles": "no data"})

    spectrum = harmonize_undefined_smiles(spectrum_in)
    assert spectrum.get("smiles") == ""


def test_harmonize_undefined_smiles_alias_nan():

    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"smiles": "nan"})

    spectrum = harmonize_undefined_smiles(spectrum_in, aliases=["nodata", "NaN", "Nan", "nan"])
    assert spectrum.get("smiles") == ""


def test_harmonize_undefined_smiles_alias_nan_undefined_is_na():

    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"smiles": "nan"})

    spectrum = harmonize_undefined_smiles(spectrum_in, aliases=["nodata", "NaN", "Nan", "nan"], undefined="n/a")
    assert spectrum.get("smiles") == "n/a"
