import numpy as np
import pytest
from scipy.sparse import csr_array
from matchms import Spectrum
from matchms.FragmentCollection import CSRFragmentCollection
from tests.builder_Spectrum import SpectrumBuilder


@pytest.fixture
def sample_spectra():
    mz1 = np.array([100.00003, 110.2, 200.581], dtype="float")
    intensities1 = np.array([0.51, 1.0, 0.011], dtype="float")

    mz2 = np.array([111.00213, 180.2, 332.342], dtype="float")
    intensities2 = np.array([0.52, 1.0, 0.7], dtype="float")

    mz3 = np.array([111.00213, 200.1, 200.213], dtype="float")
    intensities3 = np.array([0.52, 0.5, 1.0], dtype="float")

    s1 = SpectrumBuilder().with_mz(mz1).with_intensities(intensities1).build()
    s2 = SpectrumBuilder().with_mz(mz2).with_intensities(intensities2).build()
    s3 = SpectrumBuilder().with_mz(mz3).with_intensities(intensities3).build()

    return [s1, s2, s3]


@pytest.fixture
def fragments(sample_spectra):
    return CSRFragmentCollection(sample_spectra, bin_size=0.01)


def test_construct_from_spectra(sample_spectra):
    fragments = CSRFragmentCollection(sample_spectra, bin_size=0.01)

    assert len(fragments) == 3
    assert fragments.n_spectra == 3
    assert fragments.shape[0] == 3
    assert fragments.n_bins > 0
    assert isinstance(fragments.array, csr_array)


def test_construct_from_array(fragments):
    cloned = CSRFragmentCollection.from_array(fragments.array, bin_size=fragments.bin_size)

    assert len(cloned) == len(fragments)
    assert cloned.bin_size == fragments.bin_size
    np.testing.assert_array_equal(cloned.array.toarray(), fragments.array.toarray())


def test_construct_invalid_bin_size_raises(sample_spectra):
    with pytest.raises(ValueError, match="bin_size must be > 0"):
        CSRFragmentCollection(sample_spectra, bin_size=0.0)


def test_construct_empty_spectra_raises():
    with pytest.raises(ValueError, match="Spectra must contain at least one Spectrum"):
        CSRFragmentCollection([], bin_size=0.01)


def test_construct_missing_input_raises():
    with pytest.raises(ValueError, match="Either spectra or array must be provided"):
        CSRFragmentCollection()


def test_construct_array_and_spectra_raises(sample_spectra):
    dummy = csr_array((2, 3))
    with pytest.raises(ValueError, match="Pass either spectra or array, not both"):
        CSRFragmentCollection(sample_spectra, array=dummy, bin_size=0.01)


def test_repr(fragments):
    rep = repr(fragments)
    assert "CSRFragmentCollection" in rep
    assert "n_spectra=3" in rep
    assert "bin_size=0.01" in rep


def test_copy(fragments):
    cloned = fragments.copy()

    assert cloned is not fragments
    assert cloned.array is not fragments.array
    np.testing.assert_array_equal(cloned.array.toarray(), fragments.array.toarray())


def test_mz_bin_conversion(fragments):
    mz = 123.456
    bin_idx = fragments.mz_to_bin(mz)
    back_mz = fragments.bin_to_mz(bin_idx)

    assert bin_idx == 12345
    assert back_mz == pytest.approx(123.455)


def test_get_row(fragments):
    mz, intensities = fragments.get_row(0)

    assert len(mz) == 3
    assert len(intensities) == 3
    assert np.sum(intensities) == pytest.approx(1.521, abs=1e-6)


def test_get_row_negative_index(fragments):
    mz, intensities = fragments.get_row(-1)

    assert len(mz) == 3
    assert np.sum(intensities) == pytest.approx(2.02, abs=1e-6)


def test_get_row_out_of_range_raises(fragments):
    with pytest.raises(IndexError, match="row index out of range"):
        fragments.get_row(3)


def test_take(fragments):
    subset = fragments.take([0, 2])

    assert len(subset) == 2
    np.testing.assert_allclose(subset.sum(axis=1), [1.521, 2.02], atol=1e-6)


def test_reorder_alias(fragments):
    subset = fragments.reorder([2, 0])

    assert len(subset) == 2
    np.testing.assert_allclose(subset.sum(axis=1), [2.02, 1.521], atol=1e-6)


def test_filter(fragments):
    subset = fragments.filter([True, False, True])

    assert len(subset) == 2
    np.testing.assert_allclose(subset.sum(axis=1), [1.521, 2.02], atol=1e-6)


def test_filter_invalid_length_raises(fragments):
    with pytest.raises(ValueError, match="Mask length \\(2\\) does not match number of spectra \\(3\\)"):
        fragments.filter([True, False])


def test_drop(fragments):
    subset = fragments.drop([1])

    assert len(subset) == 2
    np.testing.assert_allclose(subset.sum(axis=1), [1.521, 2.02], atol=1e-6)


def test_drop_empty(sample_spectra):
    empty_spec = Spectrum(mz=np.array([]), intensities=np.array([]), metadata={})
    fragments = CSRFragmentCollection(sample_spectra + [empty_spec], bin_size=0.01)

    assert len(fragments) == 4
    cleaned = fragments.drop_empty()
    assert len(cleaned) == 3


def test_slice_rows_with_slice(fragments):
    subset = fragments.slice_rows(slice(0, 2))

    assert len(subset) == 2
    np.testing.assert_allclose(subset.sum(axis=1), [1.521, 2.22], atol=1e-6)


def test_slice_rows_with_int(fragments):
    subset = fragments.slice_rows(1)

    assert len(subset) == 1
    np.testing.assert_allclose(subset.sum(axis=1), [2.22], atol=1e-6)


def test_slice_rows_with_bool_mask(fragments):
    subset = fragments.slice_rows(np.array([False, True, True]))

    assert len(subset) == 2
    np.testing.assert_allclose(subset.sum(axis=1), [2.22, 2.02], atol=1e-6)


def test_slice_rows_invalid_selector_raises(fragments):
    with pytest.raises(TypeError, match="Unsupported row selector"):
        fragments.slice_rows("invalid")


def test_slice_mz(fragments):
    subset = fragments.slice_mz(100.0, 150.0)

    assert len(subset) == 3
    assert subset.shape[1] <= fragments.shape[1]

    # First spectrum should keep 100.00003 and 110.2 peaks only
    mz, intensities = subset.get_row(0)
    assert len(mz) == 2
    assert np.sum(intensities) == pytest.approx(1.51, abs=1e-6)


def test_slice_mz_invalid_range_raises(fragments):
    with pytest.raises(ValueError, match="mz_max must be >?= mz_min|mz_max must be >= mz_min"):
        fragments.slice_mz(200.0, 100.0)


def test_getitem_row_slice(fragments):
    subset = fragments[:2]

    assert isinstance(subset, CSRFragmentCollection)
    assert len(subset) == 2
    np.testing.assert_allclose(subset.sum(axis=1), [1.521, 2.22], atol=1e-6)


def test_getitem_row_list(fragments):
    subset = fragments[[0, 2]]

    assert len(subset) == 2
    np.testing.assert_allclose(subset.sum(axis=1), [1.521, 2.02], atol=1e-6)


def test_getitem_tuple_row_and_mz_slice(fragments):
    subset = fragments[:2, 100.0:200.0]

    assert isinstance(subset, CSRFragmentCollection)
    assert len(subset) == 2

    mz0, int0 = subset.get_row(0)
    assert np.all(mz0 < 200.01)
    assert np.sum(int0) == pytest.approx(1.51, abs=1e-6)


def test_getitem_invalid_tuple_length_raises(fragments):
    with pytest.raises(IndexError, match="Expected at most two indexers"):
        _ = fragments[0, 1, 2]


def test_getitem_invalid_column_selector_raises(fragments):
    with pytest.raises(TypeError, match="Unsupported column selector"):
        _ = fragments[:, [1, 2]]


def test_sum_axis_1(fragments):
    sums = fragments.sum(axis=1)
    np.testing.assert_allclose(sums, [1.521, 2.22, 2.02], atol=1e-6)


def test_count_axis_1(fragments):
    counts = fragments.count(axis=1)
    np.testing.assert_array_equal(counts, [3, 3, 3])


def test_count_axis_0(fragments):
    counts = fragments.count(axis=0)
    assert counts.shape[0] == fragments.shape[1]
    assert counts.sum() == 9


def test_count_invalid_axis_raises(fragments):
    with pytest.raises(ValueError, match="axis must be 0 or 1"):
        fragments.count(axis=2)


def test_row_intensity_sums(fragments):
    np.testing.assert_allclose(fragments.row_intensity_sums(), [1.521, 2.22, 2.02], atol=1e-6)


def test_row_peak_counts(fragments):
    np.testing.assert_array_equal(fragments.row_peak_counts(), [3, 3, 3])


def test_fragment_hashes_cached_property(fragments):
    hashes_1 = fragments.fragment_hashes
    hashes_2 = fragments.fragment_hashes

    assert len(hashes_1) == len(fragments)
    assert hashes_1 is hashes_2


def test_fragment_hashes_equal_for_identical_input(sample_spectra):
    fragments_1 = CSRFragmentCollection(sample_spectra, bin_size=0.01)
    fragments_2 = CSRFragmentCollection(sample_spectra, bin_size=0.01)

    assert np.all(fragments_1.fragment_hashes == fragments_2.fragment_hashes)
