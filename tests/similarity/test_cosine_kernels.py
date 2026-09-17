"""Compiled-vs-posting parity and independent cosine expectations."""
from types import SimpleNamespace
import numpy as np
import pytest
from reference_kernels._flash_cosine import cosine_rows as reference_cosine_rows
from upgrade_kernels._flash_cosine import cosine_rows


def packed(peaks, precursor, dtype):
    peaks = [np.asarray(a, dtype=dtype).reshape(-1, 2) for a in peaks]
    flat = np.concatenate(peaks) if peaks else np.empty((0, 2), dtype=dtype)
    return SimpleNamespace(
        spec_offsets=np.r_[0, np.cumsum([len(a) for a in peaks])].astype(np.int64),
        spec_mz=np.ascontiguousarray(flat[:, 0]), spec_int=np.ascontiguousarray(flat[:, 1]),
        precursor_mz=np.asarray(precursor, dtype=dtype).astype(np.float64),
        spec_l2=np.asarray([np.linalg.norm(a[:, 1].astype(np.float64)) for a in peaks]),
        n_specs=len(peaks), dtype=np.dtype(dtype),
    )


def index(p, mode):
    owners=np.repeat(np.arange(p.n_specs, dtype=np.int64), np.diff(p.spec_offsets))
    order=np.argsort(p.spec_mz)
    inverse=np.empty(len(order), np.int64)
    inverse[order]=np.arange(len(order))
    valid=np.where(np.isfinite(p.precursor_mz[owners]))[0] if mode != 0 else np.empty(0, np.int64)
    losses=(p.precursor_mz[owners[valid]]-p.spec_mz[valid]).astype(p.dtype)
    nlorder=np.argsort(losses)
    return (p.spec_mz[order], p.spec_int[order], owners[order], losses[nlorder],
            owners[valid][nlorder], inverse[valid][nlorder], p.precursor_mz, p.spec_l2)


def run(ref, query, pmz_r, pmz_q, mode=0, dtype=np.float64, tol=.03, ppm=False, gate=-1):
    r,q=packed(ref,pmz_r,dtype),packed(query,pmz_q,dtype)
    lib=index(r,mode)
    args=(q.spec_offsets,q.spec_mz,q.spec_int,q.precursor_mz,q.spec_l2,*lib,tol,ppm,mode,gate,False)
    normal=np.empty((q.n_specs,r.n_specs),dtype)
    fast=np.empty_like(normal)
    counts=np.empty(normal.shape,np.int32)
    reference_cosine_rows(normal,counts,*args,0,q.n_specs)
    posting_counts=np.empty_like(counts)
    cosine_rows(fast,posting_counts,*args,0,q.n_specs)
    np.testing.assert_array_equal(posting_counts,counts)
    score_only=np.empty_like(fast)
    cosine_rows(score_only,np.empty((0,0),np.int32),*args,0,q.n_specs)
    np.testing.assert_array_equal(score_only,fast)
    return normal,fast,counts


def independent_greedy(a,b,pa,pb,tol,mode):
    matches=[]
    for i,(x,u) in enumerate(a):
        if u <= 0:
            continue
        for j,(y,v) in enumerate(b):
            if mode!=1 and abs(x-y)<=tol:
                matches.append((i,j,u*v,1))
    if mode!=0 and np.isfinite(pa) and np.isfinite(pb) and (mode==1 or abs(pa-pb)>tol):
        for i,(x,u) in enumerate(a):
            if u<=0:
                continue
            for j,(y,v) in enumerate(b):
                if abs((pa-x)-(pb-y))<=tol:
                    matches.append((i,j,u*v,0))
    order=np.argsort([-x[2]-1e-12*x[3] for x in matches])
    useda=set();usedb=set();dot=0
    for k in order:
        i,j,w,_=matches[k]
        if i not in useda and j not in usedb:
            useda.add(i);usedb.add(j);dot+=w
    denom=np.linalg.norm(a[:,1])*np.linalg.norm(b[:,1])
    return dot/denom if denom>0 else 0, len(useda)


@pytest.mark.parametrize('dtype',[np.float32,np.float64])
@pytest.mark.parametrize('mode',[0,1,2])
@pytest.mark.parametrize('ppm,tol',[(False,.031),(True,310.)])
@pytest.mark.parametrize('seed',range(16))
def test_posting_reproduces_compiled_with_conflicts(dtype,mode,ppm,tol,seed):
    rng=np.random.default_rng(seed)
    def spectra(n):
        result=[]
        for _ in range(n):
            k=int(rng.integers(0,140))
            mz=np.sort(rng.choice([100.,100.01,100.02,110.,120.,200.,210.,510.],k))
            intensity=rng.random(k)
            intensity[intensity<.1]=0
            result.append(np.column_stack((mz,intensity)))
        return result
    refs,queries=spectra(7),spectra(5)
    pa=rng.choice([500.,510.,520.,np.nan],len(refs))
    pb=rng.choice([500.,510.,520.,np.nan],len(queries))
    normal,fast,counts=run(refs,queries,pa,pb,mode,dtype,tol,ppm)
    np.testing.assert_allclose(fast,normal,atol=2e-7 if dtype==np.float32 else 5e-15,rtol=1e-7)
    assert np.all(normal>=-1e-12) and np.all(normal<=1+1e-6)
    assert np.all(counts>=0)


@pytest.mark.parametrize('mode',[0,1,2])
@pytest.mark.parametrize('seed',range(12))
def test_compiled_against_independent_greedy(mode,seed):
    rng=np.random.default_rng(seed)
    a=np.column_stack((np.sort(rng.choice([100.,100.01,110.,200.,300.],12)),rng.random(12)))
    b=np.column_stack((np.sort(rng.choice([100.,110.,110.01,200.,310.],13)),rng.random(13)))
    normal,fast,counts=run([b],[a],[510.],[500.],mode=mode)
    expected,count=independent_greedy(a,b,500.,510.,.03,mode)
    np.testing.assert_allclose(normal[0,0],expected,atol=1e-14)
    assert counts[0,0]==count


@pytest.mark.parametrize('mode',[0,1,2])
def test_unambiguous_and_small_shift(mode):
    a=np.array([[100.,1.],[150.,.5],[199.,10.]])
    b=np.array([[100.,1.],[170.,.5],[209.,10.]])
    normal,fast,counts=run([a],[b],[200.],[210.],mode=mode,tol=.01)
    expected=1/101.25 if mode==0 else 100/101.25 if mode==1 else 101/101.25
    np.testing.assert_allclose(normal,expected)
    np.testing.assert_allclose(fast,normal)
    a=np.array([[100.,1.]])
    b=np.array([[99.985,1.],[100.,.1]])
    normal,fast,counts=run([b],[a],[500.],[500.009],mode=mode,tol=.01)
    np.testing.assert_allclose(fast,normal,atol=1e-15)


def test_mz_order_is_not_greedy_cosine():
    r=np.array([[100.,.1],[100.01,1.]])
    q=np.array([[100.005,1.]])
    normal,fast,counts=run([r],[q],[500.],[500.],tol=.02)
    assert normal[0,0]==pytest.approx(1/np.sqrt(1.01))
    assert fast[0,0]==pytest.approx(normal[0,0])
    assert counts[0,0]==1


def test_hybrid_does_not_impose_fragment_priority():
    r=np.array([[100.,.1],[110.,1.]])
    q=np.array([[100.,1.]])
    normal,fast,counts=run([r],[q],[510.],[500.],mode=2,tol=.01)
    assert fast[0,0]==pytest.approx(1/np.sqrt(1.01))
    np.testing.assert_allclose(normal,fast)


@pytest.mark.parametrize('mode',[0,1,2])
def test_empty_library_empty_query_and_zero_norm(mode):
    for refs,qs,pr,pq in [([],[],[],[]),([],[[[100.,1.]]],[],[500.]),
                         ([[[100.,1.]]],[],[500.],[]),([[]],[[]],[500.],[500.]),
                         ([[[100.,0.]]],[[[100.,1.]]],[500.],[500.])]:
        normal,fast,counts=run(refs,qs,pr,pq,mode=mode)
        assert normal.shape==(len(qs),len(refs))
        assert np.all(normal==0) and np.all(fast==0)


@pytest.mark.parametrize('mode',[0,1,2])
def test_enabled_identity_gate_rejects_missing_on_either_side(mode):
    refs=[[[100.,1.]]]*3
    queries=[[[100.,1.]]]*2
    normal,fast,counts=run(refs,queries,[500.,510.,np.nan],[500.,np.nan],mode=mode,gate=.1)
    expected=np.array([[1.,0.,0.],[0.,0.,0.]])
    np.testing.assert_allclose(normal,expected)
    np.testing.assert_allclose(fast,expected)


@pytest.mark.parametrize("mode", [0, 1, 2])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_match_counts_on_independent_postings(mode, dtype):
    """Fast-path counts equal accepted edges, including shifted-only matches."""
    reference = np.array([[100.0, 0.8], [200.0, 0.5], [300.0, 0.2]])
    query = np.array([[100.0, 0.7], [210.0, 0.3], [310.0, 0.6]])
    normal, posting, counts = run(
        [reference], [query], [500.0], [510.0], mode=mode, dtype=dtype, tol=.01,
    )
    assert counts[0, 0] == [1, 2, 3][mode]
    np.testing.assert_allclose(posting, normal, atol=1e-7)


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_zero_norm_and_precursor_gate_clear_counts(mode):
    """A rejected pair must not retain the candidate count from the first pass."""
    zero = [[100.0, 0.0], [200.0, 0.0]]
    nonzero = [[100.0, 1.0], [200.0, 0.5]]
    _, _, counts = run([nonzero], [zero], [500.0], [500.0], mode=mode)
    assert counts[0, 0] == 0
    _, _, counts = run([nonzero], [nonzero], [501.0], [500.0], mode=mode, gate=.1)
    assert counts[0, 0] == 0


def test_counts_are_not_the_number_of_candidate_edges():
    reference = [[100.0, 0.2], [100.01, 1.0], [200.0, 0.8]]
    query = [[100.005, 0.6], [200.0, 1.0]]
    _, _, counts = run([reference], [query], [500.0], [500.0], tol=.02)
    # Three candidate edges, but only two accepted assignments.
    assert counts[0, 0] == 2


@pytest.mark.parametrize("mode", [0, 1, 2])
def test_thread_blocks_do_not_share_count_state(mode):
    from concurrent.futures import ThreadPoolExecutor

    library = packed([[[100., 1.], [100.01, .2]], [[110., .5], [210., 1.]]], [500., 510.], np.float64)
    queries = packed(
        [[[100.005, .8], [200., 1.]], [[110., 1.]], [], [[210., 1.]]],
        [500., 510., 500., 510.], np.float64,
    )
    args = (
        queries.spec_offsets, queries.spec_mz, queries.spec_int,
        queries.precursor_mz, queries.spec_l2, *index(library, mode),
        .02, False, mode, -1.0, False,
    )
    expected = np.empty((4, 2))
    expected_counts = np.empty((4, 2), np.int32)
    cosine_rows(expected, expected_counts, *args, 0, 4)
    actual = np.empty_like(expected)
    actual_counts = np.empty_like(expected_counts)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(cosine_rows, actual, actual_counts, *args, start, stop)
            for start, stop in [(0, 2), (2, 4)]
        ]
        for future in futures:
            future.result()
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_counts, expected_counts)
