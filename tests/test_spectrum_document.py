import numpy
from matchms.similarity.spec2vec import SpectrumDocument
from matchms.filtering import add_losses
from matchms import Spectrum


def test_spectrum_document_init_n_decimals_default_value_no_losses():

    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([0, 0.01, 0.1, 1], dtype="float")
    metadata = dict(precursor_mz=100.0)
    spectrum = Spectrum(mz=mz, intensities=intensities, metadata=metadata)
    spectrum_document = SpectrumDocument(spectrum)

    assert spectrum_document.n_decimals == 1
    assert len(spectrum_document) == 4
    assert spectrum_document.words == [
        "peak@10.0", "peak@20.0", "peak@30.0", "peak@40.0"
    ]
    assert next(spectrum_document) == "peak@10.0"


def test_spectrum_document_init_n_decimals_2_no_losses():
    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([0, 0.01, 0.1, 1], dtype="float")
    metadata = dict(precursor_mz=100.0)
    spectrum = Spectrum(mz=mz, intensities=intensities, metadata=metadata)
    spectrum_document = SpectrumDocument(spectrum, n_decimals=2)

    assert spectrum_document.n_decimals == 2
    assert len(spectrum_document) == 4
    assert spectrum_document.words == [
        "peak@10.00", "peak@20.00", "peak@30.00", "peak@40.00"
    ]
    assert next(spectrum_document) == "peak@10.00"


def test_spectrum_document_init_n_decimals_default_value():

    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([0, 0.01, 0.1, 1], dtype="float")
    metadata = dict(precursor_mz=100.0)
    spectrum_in = Spectrum(mz=mz, intensities=intensities, metadata=metadata)
    spectrum = add_losses(spectrum_in)
    spectrum_document = SpectrumDocument(spectrum)

    assert spectrum_document.n_decimals == 1
    assert len(spectrum_document) == 8
    assert spectrum_document.words == [
        "peak@10.0", "peak@20.0", "peak@30.0", "peak@40.0",
        "loss@60.0", "loss@70.0", "loss@80.0", "loss@90.0"
    ]
    assert next(spectrum_document) == "peak@10.0"


def test_spectrum_document_init_n_decimals_2():
    mz = numpy.array([10, 20, 30, 40], dtype="float")
    intensities = numpy.array([0, 0.01, 0.1, 1], dtype="float")
    metadata = dict(precursor_mz=100.0)
    spectrum_in = Spectrum(mz=mz, intensities=intensities, metadata=metadata)
    spectrum = add_losses(spectrum_in)
    spectrum_document = SpectrumDocument(spectrum, n_decimals=2)

    assert spectrum_document.n_decimals == 2
    assert len(spectrum_document) == 8
    assert spectrum_document.words == [
        "peak@10.00", "peak@20.00", "peak@30.00", "peak@40.00",
        "loss@60.00", "loss@70.00", "loss@80.00", "loss@90.00"
    ]
    assert next(spectrum_document) == "peak@10.00"
