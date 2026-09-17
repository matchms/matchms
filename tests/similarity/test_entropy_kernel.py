"""Independent mathematical reference tests"""
from types import SimpleNamespace
import numpy as np
import pytest
from matchms.similarity._flash_entropy import entropy_rows as search_postings_into
from matchms.similarity.flash_index import build_entropy_index


def pack(peaks, precursors, dtype=np.float64):
    lengths = [len(x) for x in peaks]
    flat = np.concatenate(peaks).astype(dtype) if sum(lengths) else np.empty((0, 2), dtype)
    return SimpleNamespace(n_specs=len(peaks), spec_offsets=np.r_[0, np.cumsum(lengths)].astype(np.int64),
                           spec_mz=np.ascontiguousarray(flat[:, 0]), spec_int=np.ascontiguousarray(flat[:, 1]),
                           precursor_mz=np.asarray(precursors, dtype=dtype))


def oracle(a, b, pa, pb, tolerance, ppm, mode):
    """A two-pointer reference independent of the posting/index code."""
    consumed_a = set()
    consumed_b = set()
    def sweep(loss):
        nonlocal consumed_a, consumed_b
        aa = [(i, float(pa) - float(x[0]) if loss else float(x[0]), float(x[1]))
              for i, x in enumerate(a) if x[1] > 0 and i not in consumed_a]
        bb = [(j, float(pb) - float(x[0]) if loss else float(x[0]), float(x[1]))
              for j, x in enumerate(b) if x[1] > 0 and j not in consumed_b]
        if loss:
            aa.reverse()
            bb.reverse()
        i = j = 0
        result = 0.0
        while i < len(aa) and j < len(bb):
            ai, x, u = aa[i]
            bj, y, v = bb[j]
            allowed = tolerance * 1e-6 * 0.5 * (x+y) if ppm else tolerance
            if abs(x-y) <= allowed:
                result += (u+v)*np.log2(u+v) - u*np.log2(u) - v*np.log2(v)
                consumed_a.add(ai)
                consumed_b.add(bj)
                i += 1
                j += 1
            elif x < y:
                i += 1
            else:
                j += 1
        return result
    score = sweep(False) if mode != "neutral_loss" else 0.0
    if mode != "fragment" and np.isfinite(pa) and np.isfinite(pb):
        score += sweep(True)
    return score


def run(a, b, pa, pb, mode="fragment", dtype=np.float64, tol=0.02, ppm=False, gate=-1., gate_ppm=False):
    qp, rp = pack(a, pa, dtype), pack(b, pb, dtype)
    index = build_entropy_index(rp, mode).entropy_data()
    out = np.empty((len(a), len(b)), dtype=dtype)
    diag = np.empty((len(a), 3), dtype=np.int64)
    search_postings_into(out, qp.spec_offsets, qp.spec_mz, qp.spec_int, qp.precursor_mz,
                         index.fragment, index.neutral_loss, index.precursor_mz, index.n_peaks,
                         tol, ppm, {"fragment":0,"neutral_loss":1,"hybrid":2}[mode],
                         gate, gate_ppm, 0, len(a))
    return out, diag


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("mode, expected", [("fragment", .5), ("neutral_loss", .5), ("hybrid", 1.)])
def test_modes(dtype, mode, expected):
    a = [np.array([[100., .25], [200., .25]])]
    b = [np.array([[100., .25], [210., .25]])]
    out, _ = run(a, b, [500.], [510.], mode, dtype)
    assert out.dtype == dtype
    assert out[0, 0] == pytest.approx(expected, abs=2e-7)


@pytest.mark.parametrize("seed", range(12))
@pytest.mark.parametrize("mode", ["fragment", "neutral_loss", "hybrid"])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("ppm,tol", [(False, .035), (True, 500.)])
def test_random_overlaps_duplicates_zeros_and_missing_precursors(seed, mode, dtype, ppm, tol):
    rng = np.random.default_rng(seed)
    def spectra(n):
        result = []
        for _ in range(n):
            k = int(rng.integers(0, 75))  # Includes multi-word masks (>64 peaks).
            mz = np.sort(rng.choice([100., 100.01, 100.02, 110., 110.03, 120., 200., 210., 300., 520.], k))
            values = rng.random(k)
            values[values < .2] = 0
            if values.sum() > 0:
                values *= .5 / values.sum()
            result.append(np.column_stack((mz, values)).astype(dtype))
        return result
    a, b = spectra(7), spectra(9)
    pa = rng.choice([500., 510., 520., np.nan], len(a)).astype(dtype)
    pb = rng.choice([500., 510., 520., np.nan], len(b)).astype(dtype)
    out, _ = run(a, b, pa, pb, mode, dtype, tol, ppm)
    expected = np.array([[oracle(x,y,p,q,tol,ppm,mode) for y,q in zip(b,pb)] for x,p in zip(a,pa)])
    np.testing.assert_allclose(out, expected, atol=2e-7 if dtype == np.float32 else 2e-14, rtol=2e-7)
    assert np.all(out >= -1e-7) and np.all(out <= 1.0+1e-6)
    reverse, _ = run(b, a, pb, pa, mode, dtype, tol, ppm)
    np.testing.assert_allclose(out, reverse.T, atol=2e-7, rtol=2e-7)


@pytest.mark.parametrize("mode", ["fragment", "neutral_loss", "hybrid"])
def test_exact_da_boundary_and_outside(mode):
    # Binary-exact values avoid making the fixture boundary ambiguous.
    a = [np.array([[100., .5]])]
    b = [np.array([[100.125, .5]]), np.array([[np.nextafter(100.125, np.inf), .5]])]
    out, _ = run(a,b,[500.],[500.,500.],mode,tol=.125)
    # For NL the subtraction 500-mz can erase a single ulp; compare the actual
    # prepared coordinates rather than impose an incorrect mathematical limit.
    expected = [[oracle(a[0],x,500.,500.,.125,False,mode) for x in b]]
    np.testing.assert_allclose(out,expected,atol=1e-14)


@pytest.mark.parametrize("mode", ["neutral_loss", "hybrid"])
def test_fragment_priority_blocks_both_sides(mode):
    a = [np.array([[100., .25], [110., .25]])]
    b = [np.array([[110., .25], [120., .25]])]
    out, _ = run(a,b,[500.],[510.],mode,tol=.01)
    assert out[0,0] == pytest.approx(1. if mode=="neutral_loss" else .5)


def test_empty_inputs_and_zero_scores():
    out,_ = run([], [], [], [], "hybrid")
    assert out.shape == (0,0)
    out,_ = run([np.empty((0,2))], [np.empty((0,2))], [500.], [500.])
    assert out[0,0] == 0


def test_identity_gate():
    a = [np.array([[100.,.5]])]
    b = a * 3
    out,_ = run(a,b,[500.],[500.,510.,np.nan],gate=.1)
    np.testing.assert_allclose(out, [[1.,0.,0.]])


def test_inputs_unchanged():
    a=np.array([[100., .2],[110.,.3]])
    before=a.copy()
    run([a],[a],[500.],[500.],"hybrid")
    np.testing.assert_array_equal(a,before)
