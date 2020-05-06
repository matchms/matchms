import numpy
from matchms import Spectrum
from matchms.filtering import harmonize_undefined_inchi


def test_harmonize_undefined_inchi_empty_string():

    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"inchi": ""})

    spectrum = harmonize_undefined_inchi(spectrum_in)
    assert spectrum.get("inchi") == ""


def test_harmonize_undefined_inchi_na_1():

    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"inchi": "n/a"})

    spectrum = harmonize_undefined_inchi(spectrum_in)
    assert spectrum.get("inchi") == ""


def test_harmonize_undefined_inchi_na_2():

    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"inchi": "N/A"})

    spectrum = harmonize_undefined_inchi(spectrum_in)
    assert spectrum.get("inchi") == ""


def test_harmonize_undefined_inchi_na_3():

    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"inchi": "NA"})

    spectrum = harmonize_undefined_inchi(spectrum_in)
    assert spectrum.get("inchi") == ""


def test_harmonize_undefined_inchi_alias_nan():

    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"inchi": "nan"})

    spectrum = harmonize_undefined_inchi(spectrum_in, aliases=["nodata", "NaN", "Nan", "nan"])
    assert spectrum.get("inchi") == ""


def test_harmonize_undefined_inchi_alias_nan_undefined_is_na():

    spectrum_in = Spectrum(mz=numpy.array([], dtype="float"),
                           intensities=numpy.array([], dtype="float"),
                           metadata={"inchi": "nan"})

    spectrum = harmonize_undefined_inchi(spectrum_in, aliases=["nodata", "NaN", "Nan", "nan"], undefined="n/a")
    assert spectrum.get("inchi") == "n/a"
