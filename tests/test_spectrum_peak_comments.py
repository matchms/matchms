import numpy
from matchms import Spectrum
from matchms import Spikes
from matchms.filtering import normalize_intensities


def _create_test_spectrum():
    intensities = numpy.array([1, 1, 5, 5, 5, 5, 7, 7, 7, 9, 9], dtype="float")
    return _create_test_spectrum_with_intensities(intensities)


def _create_test_spectrum_with_intensities(intensities):
    mz = numpy.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110], dtype="float")
    return Spectrum(mz=mz, intensities=intensities)


def test_peak_comments_after_filter():
    spectrum_in: Spectrum = _create_test_spectrum()
    spectrum_in.set("peak_comments", {10: "blub"})

    spectrum = normalize_intensities(spectrum_in)
    assert spectrum.get("peak_comments")[10] == "blub"


def test_reiterating_peak_comments():
    mz = numpy.array([100.0003, 100.0004, 100.0005, 110., 200., 300., 400.0176], dtype='float')
    intensities = numpy.array([1, 2, 3, 4, 5, 6, 7], dtype='float')
    peak_comments = ["m/z 100.0003", None, "m/z 100.0005", "m/z 110.", "m/z 200.", "m/z 300.", "m/z 400.0176"]
    peak_comments = {mz[i]: peak_comments[i] for i in range(len(mz))}
    spectrum = Spectrum(mz=mz, intensities=intensities,
                        metadata={"peak_comments": peak_comments})

    spectrum.peaks = Spikes(mz=numpy.array([100.0004, 110., 400.018], dtype='float'),
                            intensities=numpy.array([5, 4, 7], dtype='float'))

    assert spectrum.peak_comments == {100.0004: "m/z 100.0003; m/z 100.0005", 110.: "m/z 110.", 400.018: "m/z 400.0176"}
