# tests/test_flash_spectral_entropy.py
import math
import numpy as np
import pytest
from matchms.similarity.flash_utils import (
    _clean_and_weight,
    _entropy_weight,
    _LibraryIndex,
    _merge_within,
)
from matchms.similarity.FlashSpectralEntropy import (
    FlashSpectralEntropy,
    _accumulate_fragment_row_numba,
    _accumulate_nl_row_numba,
    _xlog2_scalar,
    _xlog2_vec,
    _search_window_halfwidth_nb,
)
from ..builder_Spectrum import SpectrumBuilder


# ----------------------------
# Helpers
# ----------------------------

def build_spectrum(mz, intens, precursor_mz=None):
    """Use SpectrumBuilder, robustly setting precursor via builder if available."""
    b = SpectrumBuilder().with_mz(np.asarray(mz, dtype="float")).with_intensities(
        np.asarray(intens, dtype="float")
    )
    if hasattr(b, "with_precursor_mz") and precursor_mz is not None:
        b = b.with_precursor_mz(float(precursor_mz))
    elif precursor_mz is not None:
        # Fallback: many builders support setting metadata
        if hasattr(b, "with_metadata"):
            b = b.with_metadata({"precursor_mz": float(precursor_mz)})
    return b.build()

def xlog2_array(x):
    """Reference x*log2(x) with safe 0 handling."""
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x, dtype=float)
    mask = x > 0
    out[mask] = x[mask] * np.log2(x[mask])
    return out

def expected_fragment_entropy_contrib(Iq, Ilib):
    # v = xlog2(Ilib + Iq) - xlog2(Iq) - xlog2(Ilib)
    return xlog2_array(Ilib + Iq) - xlog2_array(Iq) - xlog2_array(Ilib)

def getattr_py_and_compiled(func):
    """Return a list of (callable, label) for uncompiled and compiled flavors if available."""
    py = getattr(func, "py_func", func)
    out = [(py, "py")]
    if py is not func:
        out.append((func, "compiled"))
    return out

# ----------------------------
# Basic properties / helpers
# ----------------------------

@pytest.mark.parametrize("vals", [
    [0.0, 0.5, 1.0, 2.0],
    np.array([0.0, 0.5, 1.0, 2.0], dtype=np.float32),
])
def test_xlog2_helpers(vals):
    vals = np.asarray(vals)
    expected = xlog2_array(vals)
    got_vec = _xlog2_vec(vals.astype(np.float32), np.float32)
    assert np.allclose(got_vec, expected, rtol=0, atol=1e-7)

    for v in vals:
        assert math.isclose(
            _xlog2_scalar(float(v), np.float32),
            float(v * math.log2(v)) if v > 0 else 0.0, rel_tol=0, abs_tol=1e-12
        )

def test_search_window_halfwidth_ppm():
    m = 500.0
    tol_ppm = 400.0
    hw = _search_window_halfwidth_nb(m, tol_ppm, True)
    # Symmetric ppm -> approximately (ppm*1e-6)*m /(1-0.5*ppm*1e-6)
    c = tol_ppm * 1e-6
    expected = (c * m) / (1.0 - 0.5 * c)
    assert math.isclose(hw, expected, rel_tol=0, abs_tol=1e-12)

# ----------------------------
# Preprocessing
# ----------------------------

def test_entropy_weight_behaviour():
    intens = np.array([1.0, 1.0, 1.0, 1.0])
    w = 0.25 + 0.25 * 2.0  # entropy of uniform over 4 bins = 2 bits
    expected = np.power(intens, w).astype(np.float32)
    got = _entropy_weight(intens, np.float32)
    assert np.allclose(got, expected, rtol=0, atol=1e-7)

def test_merge_within_weighted_average():
    # two close peaks at 100.00 and 100.03 within 0.05 should merge
    peaks = np.column_stack([
        np.array([100.00, 100.03, 120.0], dtype=np.float32),
        np.array([1.0,    3.0,    2.0], dtype=np.float32),
    ])
    out = _merge_within(peaks, max_delta_da=0.05)
    # first two merged into weighted center: (100*1 + 100.03*3) / (1+3) = 100.0225
    assert out.shape == (2, 2)
    assert math.isclose(out[0, 0], np.array(100.0225, dtype=np.float32), rel_tol=0, abs_tol=1e-6)
    assert math.isclose(out[0, 1], 4.0, rel_tol=0, abs_tol=1e-6)
    assert math.isclose(out[1, 0], 120.0, rel_tol=0, abs_tol=1e-6)
    assert math.isclose(out[1, 1], 2.0, rel_tol=0, abs_tol=1e-6)

def test_clean_and_weight_pipeline_precursor_noise_norm_merge():
    mz = np.array([100, 199.0, 199.5, 199.9, 210], dtype=float)
    intens = np.array([0.05, 0.2, 0.01, 0.2,  0.01], dtype=float)
    pmz = 200.0
    out = _clean_and_weight(
        np.column_stack([mz, intens]).T,
        precursor_mz=pmz,
        remove_precursor=True,
        precursor_window=1.6,   # keep only <= 198.4
        noise_cutoff=0.05,      # remove peaks below 5% of max (max after precursor filtering)
        normalize_to_half=True,
        merge_within_da=0.5,    # merge anything within 0.5 Da
        score_type="entropy",
        dtype=np.float32
    )
    # Only m/z=100 survives precursor filter (<=198.4); others removed
    assert out.shape[0] == 1
    assert math.isclose(out[0, 0], 100.0, rel_tol=0, abs_tol=1e-12)
    # Intensities are entropy-weighted then normalized to sum 0.5
    assert math.isclose(float(out[:, 1].sum()), 0.5, rel_tol=0, abs_tol=1e-7)

# ----------------------------
# Pair scoring (fragment / NL / hybrid / ppm / identity / dtype / symmetry)
# ----------------------------

@pytest.mark.parametrize("use_ppm,tol", [(False, 0.1), (True, 400.0)])
def test_pair_fragment_basic(use_ppm, tol):
    # two clear fragment matches
    s1 = build_spectrum([100, 200, 300], [0.2, 1.0, 0.3], precursor_mz=500.0)
    s2 = build_spectrum([100, 200, 305], [0.2, 0.5, 0.4], precursor_mz=500.0)

    fse = FlashSpectralEntropy(tolerance=tol, use_ppm=use_ppm, mode="fragment",
                               remove_precursor=False, noise_cutoff=0.0,
                               normalize_to_half=False, merge_within=0.0, dtype=np.float32)
    score12 = float(fse.pair(s1, s2))
    score21 = float(fse.pair(s2, s1))
    # Commutative (up to tiny FP jitter)
    assert score12 == pytest.approx(score21, 1e-6)
    assert score12 > 0.0

def test_pair_returns_zero_when_empty_after_cleanup():
    # After cleanup everything is removed (precursor filter kills all)
    s1 = build_spectrum([199.0, 199.5], [1.0, 0.5], precursor_mz=200.0)
    s2 = build_spectrum([199.2, 199.7], [1.0, 0.5], precursor_mz=200.0)
    fse = FlashSpectralEntropy(tolerance=0.02, mode="fragment",
                               remove_precursor=True, precursor_window=1.6,
                               noise_cutoff=0.0, normalize_to_half=True, merge_within=0.0)
    assert float(fse.pair(s1, s2)) == 0.0

def test_identity_gate_da_and_ppm():
    s1 = build_spectrum([100, 200], [1.0, 1.0], precursor_mz=500.0)
    s2 = build_spectrum([100, 200], [1.0, 1.0], precursor_mz=500.3)

    # Base score (no identity gating) is positive
    base = FlashSpectralEntropy(tolerance=0.02, mode="fragment", remove_precursor=False, noise_cutoff=0.0)
    base_score = float(base.pair(s1, s2))
    assert base_score > 0.0

    # Strict Da gating: 0.2 Da allowed -> zero, 0.4 Da allowed -> non-zero
    gate_da_tight = FlashSpectralEntropy(tolerance=0.02, mode="fragment",
                                         identity_precursor_tolerance=0.2, identity_use_ppm=False,
                                         remove_precursor=False, noise_cutoff=0.0)
    assert float(gate_da_tight.pair(s1, s2)) == 0.0

    gate_da_loose = FlashSpectralEntropy(tolerance=0.02, mode="fragment",
                                         identity_precursor_tolerance=0.5, identity_use_ppm=False,
                                         remove_precursor=False, noise_cutoff=0.0)
    assert float(gate_da_loose.pair(s1, s2)) == pytest.approx(base_score, rel=0, abs=1e-7)

    # PPM gating around ~500 m/z: 400 ppm ~0.2 Da window
    gate_ppm_tight = FlashSpectralEntropy(tolerance=0.02, mode="fragment",
                                          identity_precursor_tolerance=300.0, identity_use_ppm=True,
                                          remove_precursor=False, noise_cutoff=0.0)
    assert float(gate_ppm_tight.pair(s1, s2)) == 0.0

    gate_ppm_loose = FlashSpectralEntropy(tolerance=0.02, mode="fragment",
                                          identity_precursor_tolerance=800.0, identity_use_ppm=True,
                                          remove_precursor=False, noise_cutoff=0.0)
    assert float(gate_ppm_loose.pair(s1, s2)) == pytest.approx(base_score, rel=0, abs=1e-7)

def test_neutral_loss_vs_hybrid_prefer_fragments():
    # Construct spectra where NL hits would duplicate fragment hits,
    # so hybrid (fragment-priority) should drop NL contributions.
    q = build_spectrum([100, 200], [1.0, 1.0], precursor_mz=500.0)
    r = build_spectrum([100, 200], [1.0, 1.0], precursor_mz=500.0)

    # Use lenient windows so both fragment and NL will match.
    kwargs = dict(tolerance=0.1, use_ppm=False, remove_precursor=False,
                  noise_cutoff=0.0, normalize_to_half=False, merge_within=0.0)

    frag = FlashSpectralEntropy(mode="fragment", **kwargs)
    nl   = FlashSpectralEntropy(mode="neutral_loss", **kwargs)
    hyb  = FlashSpectralEntropy(mode="hybrid", **kwargs)

    s_frag = float(frag.pair(r, q))
    s_nl   = float(nl.pair(r, q))       # fragment + NL (implementation accumulates fragments first)
    s_hyb  = float(hyb.pair(r, q))      # should equal fragment-only score

    assert s_frag > 0.0
    assert s_hyb == pytest.approx(s_frag, rel=0, abs=1e-7)
    assert s_nl >= s_hyb  # NL adds (or equals if none)

def test_dtype_output_and_commutativity():
    a = build_spectrum([100, 150, 300], [0.5, 1.0, 0.4], precursor_mz=600.0)
    b = build_spectrum([100, 155, 295], [0.5, 0.8, 0.6], precursor_mz=600.0)
    f32 = FlashSpectralEntropy(dtype=np.float32, remove_precursor=False, noise_cutoff=0.0)
    f64 = FlashSpectralEntropy(dtype=np.float64, remove_precursor=False, noise_cutoff=0.0)

    s_ab_32 = f32.pair(a, b)
    s_ba_32 = f32.pair(b, a)
    s_ab_64 = f64.pair(a, b)
    s_ba_64 = f64.pair(b, a)

    assert s_ab_32.dtype == np.float32 and s_ba_32.dtype == np.float32
    assert s_ab_64.dtype == np.float64 and s_ba_64.dtype == np.float64
    assert float(s_ab_32) == pytest.approx(float(s_ba_32), 1e-6)
    assert float(s_ab_64) == pytest.approx(float(s_ba_64), 1e-12)

# ----------------------------
# Compiled vs. uncompiled (if Numba present) for helpers
# ----------------------------

@pytest.mark.parametrize("impl,label", getattr_py_and_compiled(_accumulate_fragment_row_numba))
def test_accumulate_fragment_row_py_vs_compiled(impl, label):
    # Tiny synthetic library of one spectrum with two peaks
    lib_mz  = np.array([100.0, 200.0], dtype=np.float32)
    lib_int = np.array([0.5, 0.5], dtype=np.float32)
    lib_sid = np.array([0, 0], dtype=np.int32)

    # Query with two matching peaks
    q_mz  = np.array([100.0, 200.0], dtype=np.float32)
    q_int = np.array([0.25, 0.75], dtype=np.float32)

    scores = np.zeros(1, dtype=np.float32)
    # UPDATED SIGNATURE: no dtype arg
    impl(scores, q_mz, q_int, lib_mz, lib_int, lib_sid, 0.02, False)
    # Expected: both peaks contribute to spectrum 0
    assert scores.shape == (1,)
    assert scores[0] > 0.0

@pytest.mark.parametrize("impl,label", getattr_py_and_compiled(_accumulate_nl_row_numba))
def test_accumulate_nl_row_py_vs_compiled(impl, label):
    # Library with pmz=500 and a product at 100 (loss 400)
    nl_mz  = np.array([400.0], dtype=np.float32)
    nl_int = np.array([0.5], dtype=np.float32)
    nl_sid = np.array([0], dtype=np.int32)
    nl_pid = np.array([0], dtype=np.int64)
    # Product table (for hybrid exclusions)
    peaks_mz  = np.array([100.0], dtype=np.float32)
    peaks_sid = np.array([0], dtype=np.int32)

    q_mz  = np.array([100.0], dtype=np.float32)
    q_int = np.array([1.0], dtype=np.float32)
    q_pmz = 500.0

    scores = np.zeros(1, dtype=np.float32)
    # UPDATED SIGNATURE: no dtype arg; add empty prod_min/prod_max for prefer_fragments=False
    prod_min = np.empty(0, dtype=np.int64)
    prod_max = np.empty(0, dtype=np.int64)
    impl(scores, q_mz, q_int, q_pmz, nl_mz, nl_int, nl_sid, nl_pid,
         peaks_mz, peaks_sid, 0.1, False, False, prod_min, prod_max)
    assert scores[0] > 0.0

# ----------------------------
# Matrix path (dense + sparse)
# ----------------------------

def test_matrix_dense_matches_pair():
    refs = [
        build_spectrum([100, 200], [1.0, 0.5], precursor_mz=500.0),
        build_spectrum([110, 300], [0.3, 1.0], precursor_mz=600.0),
    ]
    qs = [
        build_spectrum([100, 205], [1.0, 0.5], precursor_mz=500.0),
        build_spectrum([110, 300], [1.0, 0.3], precursor_mz=600.0),
    ]
    fse = FlashSpectralEntropy(tolerance=0.1, use_ppm=False, mode="fragment",
                               remove_precursor=False, noise_cutoff=0.0,
                               normalize_to_half=False, merge_within=0.0)
    M = fse.matrix(refs, qs, array_type="numpy", n_jobs=0)
    assert M.shape == (2, 2)
    # Check each cell equals pair()
    for i, r in enumerate(refs):
        for j, q in enumerate(qs):
            expected = float(fse.pair(r, q))
            assert float(M[i, j]) == pytest.approx(expected, rel=0, abs=1e-6)


def test_matrix_sparse_basic():
    refs = [
        build_spectrum([100, 200], [1.0, 0.5], precursor_mz=500.0),
        build_spectrum([110, 300], [0.3, 1.0], precursor_mz=600.0),
    ]
    qs = [
        build_spectrum([100, 205], [1.0, 0.5], precursor_mz=500.0),
        build_spectrum([110, 300], [1.0, 0.3], precursor_mz=600.0),
    ]
    fse = FlashSpectralEntropy(tolerance=0.1, use_ppm=False, mode="fragment",
                               remove_precursor=False, noise_cutoff=0.0,
                               normalize_to_half=False, merge_within=0.0)
    S = fse.matrix(refs, qs, array_type="sparse", n_jobs=0)
    assert S.shape == (2, 2, 1)
    dense = S.to_array()
    # Verify against pair() wherever nonzero
    for i in range(2):
        for j in range(2):
            expect = float(fse.pair(refs[i], qs[j]))
            assert float(dense[i, j]) == pytest.approx(expect, rel=0, abs=1e-6)

# ----------------------------
# End-to-end pair expected check using internals (sanity)
# ----------------------------

def test_pair_matches_manual_accumulation_fragment_only():
    # Small pair, fragment-only, no cleanup to simplify
    r = build_spectrum([100, 200, 300], [0.2, 1.0, 0.3], precursor_mz=500.0)
    q = build_spectrum([100, 205, 300], [0.2, 0.5, 0.6], precursor_mz=500.0)

    fse = FlashSpectralEntropy(tolerance=0.1, use_ppm=False, mode="fragment",
                               remove_precursor=False, noise_cutoff=0.0,
                               normalize_to_half=False, merge_within=0.0, dtype=np.float32)

    # Run through the same internal preprocessing to build a tiny library from query
    A, pmzA = fse._prepare(r)
    B, pmzB = fse._prepare(q)

    lib = _LibraryIndex(np.float32)
    lib.n_specs = 1
    lib.peaks_mz = B[:, 0]
    lib.peaks_int = B[:, 1]
    lib.peaks_spec_idx = np.zeros(B.shape[0], dtype=np.int32)
    lib.nl_mz = np.zeros(0, dtype=np.float32)
    lib.nl_int = np.zeros(0, dtype=np.float32)
    lib.nl_spec_idx = np.zeros(0, dtype=np.int32)
    lib.nl_product_idx = np.zeros(0, dtype=np.int64)
    lib.precursor_mz = np.array([pmzB if pmzB is not None else np.nan], dtype=np.float32)

    scores = np.zeros(1, dtype=np.float32)
    # UPDATED SIGNATURE: no dtype arg
    _accumulate_fragment_row_numba(
        scores, A[:, 0], A[:, 1],
        lib.peaks_mz, lib.peaks_int, lib.peaks_spec_idx,
        fse.tolerance, fse.use_ppm
    )

    expected = float(scores[0])
    got = float(fse.pair(r, q))
    assert got == pytest.approx(expected, rel=0, abs=1e-7)
