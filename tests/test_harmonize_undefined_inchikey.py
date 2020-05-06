import numpy
from matchms import Spectrum
from matchms.filtering import harmonize_undefined_inchikey


def test_harmonize_undefined_inchikey_empty_string():

    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"inchikey": ""})

    spectrum = harmonize_undefined_inchikey(spectrum_in)
    assert spectrum.get("inchikey") == ""


def test_harmonize_undefined_inchikey_na_1():

    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"inchikey": "n/a"})

    spectrum = harmonize_undefined_inchikey(spectrum_in)
    assert spectrum.get("inchikey") == ""


def test_harmonize_undefined_inchikey_na_2():

    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"inchikey": "N/A"})

    spectrum = harmonize_undefined_inchikey(spectrum_in)
    assert spectrum.get("inchikey") == ""


def test_harmonize_undefined_inchikey_na_3():

    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"inchikey": "NA"})

    spectrum = harmonize_undefined_inchikey(spectrum_in)
    assert spectrum.get("inchikey") == ""


def test_harmonize_undefined_inchikey_no_data():

    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"inchikey": "no data"})

    spectrum = harmonize_undefined_inchikey(spectrum_in)
    assert spectrum.get("inchikey") == ""


def test_harmonize_undefined_inchikey_alias_nan():

    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"inchikey": "nan"})

    spectrum = harmonize_undefined_inchikey(spectrum_in, aliases=["nodata", "NaN", "Nan", "nan"])
    assert spectrum.get("inchikey") == ""


def test_harmonize_undefined_inchikey_alias_nan_undefined_is_na():

    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"inchikey": "nan"})

    spectrum = harmonize_undefined_inchikey(spectrum_in, aliases=["nodata", "NaN", "Nan", "nan"], undefined="n/a")
    assert spectrum.get("inchikey") == "n/a"
