import math
import numpy as np
import pytest
from matchms import SpectraCollection
from matchms.similarity.flash_utils import (
    _build_library_index_from_prepared,
    _clean_and_weight,
    _compute_l2_by_row,
    _entropy_weight,
    _entropy_weight_packed,
    _LibraryIndex,
    _prepare_collection,
    _rebuild_offsets,
    _row_ids_from_offsets,
)
from matchms.similarity.flash_utils_spectrum_list import _clean_and_weight as _clean_and_weight_old
from ..builder_spectrum import SpectrumBuilder


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def build_spectrum(mz, intens, precursor_mz=None):
    builder = (
        SpectrumBuilder()
        .with_mz(np.asarray(mz, dtype=float))
        .with_intensities(np.asarray(intens, dtype=float))
    )

    if precursor_mz is not None:
        if hasattr(builder, "with_precursor_mz"):
            builder = builder.with_precursor_mz(float(precursor_mz))
        elif hasattr(builder, "with_metadata"):
            builder = builder.with_metadata(
                {"precursor_mz": float(precursor_mz)}
            )

    return builder.build()


def build_collection(*spectra):
    return SpectraCollection(
        list(spectra),
        mz_precision=1e-6,
    )


def prepared_row(prepared, row):
    start = prepared.spec_offsets[row]
    end = prepared.spec_offsets[row + 1]
    return np.column_stack(
        (
            prepared.spec_mz[start:end],
            prepared.spec_int[start:end],
        )
    )


# -------------------------------------------------------------------------
# Existing single-spectrum helper behaviour
# -------------------------------------------------------------------------

def test_entropy_weight_behaviour():
    intensities = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)

    p = intensities / intensities.sum()
    entropy = float(-np.sum(p * np.log(p)))
    weight = 0.25 + 0.25 * entropy

    expected = np.power(intensities, weight).astype(np.float32)
    result = _entropy_weight(intensities, np.float32)

    assert np.allclose(
        result,
        expected,
        rtol=0,
        atol=1e-7,
    )


def test_entropy_weight_zero_total_returns_zero():
    intensities = np.zeros(4, dtype=float)

    result = _entropy_weight(
        intensities,
        np.float32,
    )

    assert result.dtype == np.float32
    assert np.allclose(result, 0.0)


def test_entropy_weight_saturation_at_three_nats():
    intensities = np.ones(32, dtype=float)

    result = _entropy_weight(
        intensities,
        np.float32,
    )

    assert np.allclose(
        result,
        intensities.astype(np.float32),
        atol=0.0,
    )


def test_clean_and_weight_matches_old_implementation():
    peaks = np.array(
        [
            [100.00, 0.05],
            [150.00, 1.00],
            [150.03, 0.50],
            [198.00, 0.20],
            [199.00, 0.90],
        ],
        dtype=float,
    )

    kwargs = {
        "precursor_mz": 200.0,
        "remove_precursor": True,
        "offset_to_precursor": -1.6,
        "noise_cutoff": 0.05,
        "normalize_to_half": True,
        "merge_within_da": 0.05,
        "weighing_type": "entropy",
        "dtype": np.float32,
    }

    old = _clean_and_weight_old(
        peaks,
        **kwargs,
    )
    new = _clean_and_weight(
        peaks,
        **kwargs,
    )

    assert new.dtype == np.float32
    assert np.allclose(
        new,
        old,
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "weighing_type",
    ["cosine", "entropy"],
)
def test_clean_and_weight_empty_input(weighing_type):
    peaks = np.empty((0, 2), dtype=float)

    result = _clean_and_weight(
        peaks,
        precursor_mz=None,
        remove_precursor=False,
        offset_to_precursor=-1.6,
        noise_cutoff=0.0,
        normalize_to_half=False,
        merge_within_da=0.0,
        weighing_type=weighing_type,
        dtype=np.float32,
    )

    assert result.shape == (0, 2)
    assert result.dtype == np.float32


def test_clean_and_weight_rejects_unknown_weighing_type():
    peaks = np.array([[100.0, 1.0]])

    with pytest.raises(
        ValueError,
        match="Score type '.*' not recognized",
    ):
        _clean_and_weight(
            peaks,
            precursor_mz=None,
            remove_precursor=False,
            offset_to_precursor=-1.6,
            noise_cutoff=0.0,
            normalize_to_half=False,
            merge_within_da=0.0,
            weighing_type="invalid",
        )


# -------------------------------------------------------------------------
# Packed-row helpers
# -------------------------------------------------------------------------

def test_row_ids_from_offsets():
    offsets = np.array(
        [0, 2, 2, 5],
        dtype=np.int64,
    )

    result = _row_ids_from_offsets(offsets)

    assert np.array_equal(
        result,
        np.array(
            [0, 0, 2, 2, 2],
            dtype=np.int32,
        ),
    )


def test_rebuild_offsets_with_empty_rows():
    spec_idx = np.array(
        [0, 0, 2, 2, 2],
        dtype=np.int32,
    )

    offsets = _rebuild_offsets(
        spec_idx,
        n_specs=4,
    )

    assert np.array_equal(
        offsets,
        np.array(
            [0, 2, 2, 5, 5],
            dtype=np.int64,
        ),
    )


def test_compute_l2_by_row():
    intensities = np.array(
        [3.0, 4.0, 2.0],
        dtype=np.float64,
    )
    offsets = np.array(
        [0, 2, 2, 3],
        dtype=np.int64,
    )

    result = _compute_l2_by_row(
        intensities,
        offsets,
        np.float64,
    )

    assert np.allclose(
        result,
        [5.0, 0.0, 2.0],
        atol=1e-12,
    )


def test_entropy_weight_packed_matches_per_spectrum_weighting():
    intensities = np.array(
        [1.0, 2.0, 3.0, 4.0, 2.0],
        dtype=np.float64,
    )
    spec_idx = np.array(
        [0, 0, 1, 1, 1],
        dtype=np.int32,
    )

    result = _entropy_weight_packed(
        intensities,
        spec_idx,
        n_specs=2,
        dtype=np.float64,
    )

    expected = np.concatenate(
        [
            _entropy_weight(
                intensities[:2],
                np.float64,
            ),
            _entropy_weight(
                intensities[2:],
                np.float64,
            ),
        ]
    )

    assert np.allclose(
        result,
        expected,
        atol=1e-12,
    )


# -------------------------------------------------------------------------
# SpectraCollection preprocessing
# -------------------------------------------------------------------------

@pytest.mark.parametrize(
    "weighing_type,normalize_to_half,compute_l2",
    [
        ("cosine", False, True),
        ("cosine", True, True),
        ("entropy", False, False),
        ("entropy", True, False),
    ],
)
def test_prepare_collection_matches_single_spectrum_preprocessing(
    weighing_type,
    normalize_to_half,
    compute_l2,
):
    spectra = [
        build_spectrum(
            [100.0, 150.0, 150.03, 199.0],
            [0.05, 1.0, 0.5, 0.2],
            precursor_mz=200.0,
        ),
        build_spectrum(
            [80.0, 120.0, 250.0],
            [0.1, 0.7, 1.0],
            precursor_mz=300.0,
        ),
        build_spectrum(
            [50.0, 70.0],
            [0.0, 0.0],
            precursor_mz=None,
        ),
    ]

    collection = build_collection(*spectra)

    kwargs = {
        "intensity_power": 0.5,
        "remove_precursor": True,
        "offset_to_precursor": -1.6,
        "noise_cutoff": 0.05,
        "normalize_to_half": normalize_to_half,
        "merge_within_da": 0.05,
        "weighing_type": weighing_type,
        "compute_l2_norm": compute_l2,
        "dtype": np.float64,
    }

    prepared = _prepare_collection(
        collection,
        **kwargs,
    )

    assert prepared.n_specs == len(spectra)
    assert prepared.spec_offsets.shape == (
        len(spectra) + 1,
    )

    for row, spectrum in enumerate(spectra):
        expected = _clean_and_weight(
            spectrum.peaks.to_numpy,
            precursor_mz=spectrum.get(
                "precursor_mz",
                None,
            ),
            remove_precursor=kwargs["remove_precursor"],
            offset_to_precursor=kwargs[
                "offset_to_precursor"
            ],
            noise_cutoff=kwargs["noise_cutoff"],
            normalize_to_half=kwargs[
                "normalize_to_half"
            ],
            merge_within_da=kwargs[
                "merge_within_da"
            ],
            weighing_type=kwargs[
                "weighing_type"
            ],
            intensity_power=kwargs[
                "intensity_power"
            ],
            dtype=kwargs["dtype"],
        )

        actual = prepared_row(
            prepared,
            row,
        )

        assert actual.shape == expected.shape
        assert np.allclose(
            actual,
            expected,
            atol=2e-6,
        )


def test_prepare_collection_precursor_filter_is_row_specific():
    collection = build_collection(
        build_spectrum(
            [100.0, 199.0],
            [1.0, 1.0],
            precursor_mz=200.0,
        ),
        build_spectrum(
            [100.0, 199.0],
            [1.0, 1.0],
            precursor_mz=250.0,
        ),
    )

    prepared = _prepare_collection(
        collection,
        remove_precursor=True,
        offset_to_precursor=-1.6,
        noise_cutoff=0.0,
        weighing_type="cosine",
        dtype=np.float64,
    )

    first = prepared_row(
        prepared,
        0,
    )
    second = prepared_row(
        prepared,
        1,
    )

    assert np.allclose(
        first[:, 0],
        [100.0],
    )
    assert np.allclose(
        second[:, 0],
        [100.0, 199.0],
    )


def test_prepare_collection_missing_precursor_keeps_peaks():
    collection = build_collection(
        build_spectrum(
            [100.0, 300.0],
            [1.0, 1.0],
        ),
    )

    prepared = _prepare_collection(
        collection,
        remove_precursor=True,
        offset_to_precursor=-1.6,
        noise_cutoff=0.0,
        weighing_type="cosine",
        dtype=np.float64,
    )

    row = prepared_row(
        prepared,
        0,
    )

    assert np.allclose(
        row[:, 0],
        [100.0, 300.0],
    )
    assert np.isnan(
        prepared.precursor_mz[0]
    )


def test_prepare_collection_relative_noise_filter_is_per_row():
    collection = build_collection(
        build_spectrum(
            [100.0, 200.0],
            [100.0, 5.0],
        ),
        build_spectrum(
            [100.0, 200.0],
            [1.0, 0.2],
        ),
    )

    prepared = _prepare_collection(
        collection,
        remove_precursor=False,
        noise_cutoff=0.1,
        weighing_type="cosine",
        dtype=np.float64,
    )

    row_0 = prepared_row(
        prepared,
        0,
    )
    row_1 = prepared_row(
        prepared,
        1,
    )

    assert np.allclose(
        row_0[:, 0],
        [100.0],
    )
    assert np.allclose(
        row_1[:, 0],
        [100.0, 200.0],
    )


def test_prepare_collection_normalizes_each_row_to_half():
    collection = build_collection(
        build_spectrum(
            [100.0, 200.0],
            [1.0, 3.0],
        ),
        build_spectrum(
            [100.0, 200.0],
            [2.0, 2.0],
        ),
    )

    prepared = _prepare_collection(
        collection,
        remove_precursor=False,
        noise_cutoff=0.0,
        normalize_to_half=True,
        weighing_type="cosine",
        dtype=np.float64,
    )

    for row in range(2):
        actual = prepared_row(
            prepared,
            row,
        )
        assert actual[:, 1].sum() == pytest.approx(
            0.5,
            abs=1e-12,
        )


def test_prepare_collection_computes_l2_after_preprocessing():
    collection = build_collection(
        build_spectrum(
            [100.0, 200.0],
            [3.0, 4.0],
        ),
    )

    prepared = _prepare_collection(
        collection,
        remove_precursor=False,
        noise_cutoff=0.0,
        normalize_to_half=False,
        weighing_type="cosine",
        compute_l2_norm=True,
        dtype=np.float64,
    )

    assert prepared.spec_l2 is not None
    assert prepared.spec_l2[0] == pytest.approx(
        5.0,
        abs=1e-12,
    )


def test_prepare_collection_rejects_unknown_weighing_type():
    collection = build_collection(
        build_spectrum(
            [100.0],
            [1.0],
        )
    )

    with pytest.raises(
        ValueError,
        match="Score type '.*' not recognized",
    ):
        _prepare_collection(
            collection,
            weighing_type="invalid",
        )


# -------------------------------------------------------------------------
# Library index
# -------------------------------------------------------------------------

def test_library_index_from_prepared_fragment_arrays():
    collection = build_collection(
        build_spectrum(
            [100.0, 300.0],
            [0.5, 0.2],
            precursor_mz=500.0,
        ),
        build_spectrum(
            [150.0],
            [0.8],
            precursor_mz=510.0,
        ),
    )

    prepared = _prepare_collection(
        collection,
        remove_precursor=False,
        noise_cutoff=0.0,
        weighing_type="cosine",
        compute_l2_norm=True,
        dtype=np.float32,
    )

    idx = _build_library_index_from_prepared(
        prepared,
        compute_neutral_loss=True,
        compute_l2_norm=True,
    )

    assert isinstance(
        idx,
        _LibraryIndex,
    )
    assert idx.n_specs == 2

    # Spectrum-major representation.
    assert np.array_equal(
        idx.spec_offsets,
        np.array(
            [0, 2, 3],
            dtype=np.int64,
        ),
    )
    assert np.allclose(
        idx.spec_mz,
        [100.0, 300.0, 150.0],
        atol=1e-6,
    )
    assert np.allclose(
        idx.spec_int,
        [0.5, 0.2, 0.8],
        atol=1e-6,
    )

    # Globally sorted fragment index.
    assert np.allclose(
        idx.peaks_mz,
        [100.0, 150.0, 300.0],
        atol=1e-6,
    )
    assert np.array_equal(
        idx.peaks_spec_idx,
        np.array(
            [0, 1, 0],
            dtype=np.int32,
        ),
    )


def test_library_index_neutral_loss_mapping_points_to_product_peak():
    collection = build_collection(
        build_spectrum(
            [100.0, 300.0],
            [0.5, 0.2],
            precursor_mz=500.0,
        ),
        build_spectrum(
            [150.0],
            [0.8],
            precursor_mz=510.0,
        ),
    )

    prepared = _prepare_collection(
        collection,
        remove_precursor=False,
        noise_cutoff=0.0,
        weighing_type="cosine",
        dtype=np.float64,
    )

    idx = _build_library_index_from_prepared(
        prepared,
        compute_neutral_loss=True,
    )

    assert np.all(
        np.diff(idx.nl_mz) >= 0
    )

    for k in range(idx.nl_mz.size):
        spectrum_idx = int(
            idx.nl_spec_idx[k]
        )
        precursor_mz = float(
            idx.precursor_mz[spectrum_idx]
        )
        neutral_loss = float(
            idx.nl_mz[k]
        )

        product_position = int(
            idx.nl_product_idx[k]
        )
        product_mz = float(
            idx.peaks_mz[
                product_position
            ]
        )

        assert product_mz == pytest.approx(
            precursor_mz - neutral_loss,
            abs=1e-8,
        )


def test_library_index_omits_neutral_losses_for_missing_precursor():
    collection = build_collection(
        build_spectrum(
            [100.0, 200.0],
            [1.0, 2.0],
        ),
    )

    prepared = _prepare_collection(
        collection,
        remove_precursor=False,
        noise_cutoff=0.0,
        weighing_type="cosine",
        dtype=np.float64,
    )

    idx = _build_library_index_from_prepared(
        prepared,
        compute_neutral_loss=True,
    )

    assert idx.nl_mz.size == 0
    assert idx.nl_int.size == 0
    assert idx.nl_spec_idx.size == 0
    assert idx.nl_product_idx.size == 0


def test_library_index_reuses_precomputed_l2():
    collection = build_collection(
        build_spectrum(
            [100.0, 200.0],
            [3.0, 4.0],
        ),
    )

    prepared = _prepare_collection(
        collection,
        remove_precursor=False,
        noise_cutoff=0.0,
        weighing_type="cosine",
        compute_l2_norm=True,
        dtype=np.float64,
    )

    idx = _build_library_index_from_prepared(
        prepared,
        compute_l2_norm=True,
    )

    assert idx.spec_l2 is prepared.spec_l2
    assert idx.spec_l2[0] == pytest.approx(
        5.0,
        abs=1e-12,
    )