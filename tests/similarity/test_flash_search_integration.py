"""Public API, persistence, and threading regressions for indexed similarities."""
from concurrent.futures import ThreadPoolExecutor
import inspect
import json
import numpy as np
import pytest

pytest.importorskip('matchms.spectra_collection')
from matchms import Spectrum, SpectraCollection
from matchms.similarity.flash_index import FlashIndex
from matchms.similarity.flash_similarity import CosineFlash, FlashEntropy
from matchms.similarity.cosine import Cosine
from matchms.similarity.entropy import Entropy
from matchms.similarity.modified_cosine import ModifiedCosine


def spectrum(mz, intensity, precursor=500.):
    return Spectrum(np.asarray(mz,dtype=float),np.asarray(intensity,dtype=float),
                    metadata={'precursor_mz':precursor},metadata_harmonization=False)


def inputs():
    queries = SpectraCollection([
        spectrum([100.,100.01,200.],[.2,.3,.5]),
        spectrum([110.,210.],[.6,.4],510.),
        spectrum([],[],520.),
    ])
    library = SpectraCollection([
        spectrum([100.005,200.],[.4,.6]),
        spectrum([110.,210.],[.5,.5],510.),
    ])
    return queries,library


@pytest.mark.parametrize('cls',[Cosine,Entropy,ModifiedCosine,CosineFlash,FlashEntropy])
def test_public_search_signature(cls):
    params=inspect.signature(cls.search).parameters
    assert list(params)==['self','query_spectra','library_index','score_fields','progress_bar','n_jobs']
    assert params['score_fields'].kind is inspect.Parameter.KEYWORD_ONLY
    assert params['score_fields'].default is None
    assert params['progress_bar'].default is True
    assert params['n_jobs'].default == -1


@pytest.mark.parametrize('cls',[CosineFlash,FlashEntropy])
@pytest.mark.parametrize('mode',['fragment','neutral_loss','hybrid'])
@pytest.mark.parametrize('dtype',[np.float32,np.float64])
def test_search_matrix_prepared_persistence_and_threads(cls,mode,dtype,tmp_path):
    """Test that all public search methods agree, and that persistence and threads work."""
    queries,library=inputs()
    s=cls(matching_mode=mode,tolerance=.02,dtype=dtype,remove_precursor=False,noise_cutoff=0.)
    before=[x.peaks.to_numpy.copy() for x in library]
    index=s.build_index(library)
    assert isinstance(index,FlashIndex)
    assert index.metadata['mz_precision']==library.mz_precision
    assert index.metadata['fragment_backend']==type(library.fragments).__name__
    expected=s.matrix(queries,library,progress_bar=False,n_jobs=1)
    a=s.search(queries,index,progress_bar=False,n_jobs=1)
    b=s.search(query_spectra=queries,library_index=index,progress_bar=False,n_jobs=2)
    ready=s.prepare_queries(queries)
    c=s.search_prepared(ready,index,n_jobs=2)
    file=tmp_path/'library.npz'
    s.save_index(index,filename=file)
    d=s.search(queries,s.load_index(filename=file),progress_bar=False,n_jobs=1)
    assert expected.shape==a.shape==b.shape==c.shape==d.shape==(3,2)
    atol=3e-7 if dtype==np.float32 else 1e-12
    for field in s.score_fields:
        for result in (a,b,c,d):
            np.testing.assert_allclose(result.to_array(field),expected.to_array(field),atol=atol,rtol=atol)
    with ThreadPoolExecutor(max_workers=2) as pool:
        values=list(pool.map(lambda _:s.search_prepared(ready,index,n_jobs=1),range(4)))
    for result in values:
        np.testing.assert_array_equal(result.to_array('score'),a.to_array('score'))
    for x,y in zip(before,library,strict=True):
        np.testing.assert_array_equal(x,y.peaks.to_numpy)


@pytest.mark.parametrize('cls',[Cosine,Entropy,ModifiedCosine])
def test_public_empty_shapes_and_no_implicit_rebuild(cls,monkeypatch):
    queries,library=inputs()
    s=cls(tolerance=.02,remove_precursor=False)
    index=s.build_index(library)
    empty=s.build_index([])
    assert s.search([],index,progress_bar=False,n_jobs=1).shape==(0,2)
    assert s.search(queries,empty,progress_bar=False,n_jobs=1).shape==(3,0)
    def fail(*args,**kwargs):
        raise AssertionError('search must not rebuild or revalidate all index arrays')
    monkeypatch.setattr(s,'build_index_prepared',fail)
    monkeypatch.setattr(index,'_validate_arrays',fail)
    s.search(queries,index,progress_bar=False,n_jobs=1)
    with pytest.raises(TypeError,match='library_index'):
        s.search(index,queries,progress_bar=False,n_jobs=1)


@pytest.mark.parametrize('cls',[CosineFlash,FlashEntropy])
def test_matching_mode_is_capability_not_preprocessing(cls):
    queries,library=inputs()
    kwargs=dict(tolerance=.02,noise_cutoff=0.,remove_precursor=False)
    hybrid=cls(matching_mode='hybrid',**kwargs)
    index=hybrid.build_index(library)
    assert 'matching_mode' not in index.config
    assert 'tolerance' not in index.config
    for mode in ('fragment','neutral_loss','hybrid'):
        scorer=cls(matching_mode=mode,**kwargs)
        a=scorer.search(queries,index,progress_bar=False,n_jobs=1)
        b=scorer.matrix(queries,library,progress_bar=False,n_jobs=1)
        np.testing.assert_allclose(a.to_array('score'),b.to_array('score'),atol=1e-12)
    fragment=cls(matching_mode='fragment',**kwargs)
    with pytest.raises(ValueError,match='neutral-loss index'):
        hybrid.search(queries,fragment.build_index(library),progress_bar=False,n_jobs=1)


@pytest.mark.parametrize('mode', ['fragment', 'neutral_loss', 'hybrid'])
def test_score_and_match_field_selection_share_public_index(mode):
    queries, library = inputs()
    scorer = CosineFlash(
        matching_mode=mode, tolerance=.02, noise_cutoff=0., remove_precursor=False,
    )
    index = scorer.build_index(library)
    full = scorer.search(queries, index, progress_bar=False, n_jobs=1)
    score_only = scorer.search(
        queries, index, score_fields=('score',), progress_bar=False, n_jobs=1,
    )
    count_only = scorer.search(
        queries, index, score_fields=('matches',), progress_bar=False, n_jobs=1,
    )
    assert full.score_fields == ('score', 'matches')
    assert score_only.score_fields == ('score',)
    assert count_only.score_fields == ('matches',)
    np.testing.assert_array_equal(score_only.to_array(), full.to_array('score'))
    np.testing.assert_array_equal(count_only.to_array(), full.to_array('matches'))


@pytest.mark.parametrize('cls',[CosineFlash,FlashEntropy])
def test_search_after_loading_original_version_two(tmp_path,cls):
    queries,library=inputs()
    s=cls(matching_mode='hybrid',tolerance=.02,noise_cutoff=0.,remove_precursor=False)
    index=s.build_index(library)
    file=tmp_path/'new.npz';index.save(file)
    with np.load(file,allow_pickle=False) as f:
        payload={key:f[key] for key in f.files}
    meta=json.loads(str(payload['__metadata__'].item()));meta['version']=2
    for name in ('peaks_pid','peaks_xlog2','nl_xlog2'):
        payload.pop(name,None);meta['optional_arrays'].pop(name,None)
    payload['__metadata__']=np.asarray(json.dumps(meta))
    legacy=tmp_path/'legacy.npz';np.savez(legacy,**payload)
    loaded=s.load_index(legacy)
    assert isinstance(loaded,FlashIndex)
    a=s.search(queries,index,progress_bar=False,n_jobs=1)
    b=s.search(queries,loaded,progress_bar=False,n_jobs=1)
    for field in s.score_fields:
        np.testing.assert_allclose(a.to_array(field),b.to_array(field),atol=1e-12)


def test_cosine_matrix_search_agree_even_for_direction_sensitive_ties():
    # This loss-matching graph gives different greedy choices if the inputs
    # are swapped. matrix must keep query/library direction for cosine.
    a=spectrum([100.875,101.,102.,102.125,102.625,103.125,103.375,103.625],
               [.5,.5,.5,.25,1.,1.,.25,.25],500.)
    b=spectrum([100.,100.375,100.5,100.625,100.875,101.375,102.5,102.75,103.125,103.75,104.],
               [.25,.25,.25,1.,.5,.5,1.,.5,1.,.5,.25],501.)
    s=CosineFlash(matching_mode='neutral_loss',tolerance=.5,noise_cutoff=0.,remove_precursor=False)
    queries=SpectraCollection([a,a]);library=SpectraCollection([b])
    expected=s.search(queries,s.build_index(library),progress_bar=False,n_jobs=1)
    actual=s.matrix(queries,library,progress_bar=False,n_jobs=1)
    np.testing.assert_array_equal(actual.to_array('score'),expected.to_array('score'))
    np.testing.assert_array_equal(actual.to_array('matches'),expected.to_array('matches'))


@pytest.mark.parametrize('cls',[CosineFlash,FlashEntropy])
def test_settings_rejection_and_search_time_tolerance_reuse(cls):
    queries,library=inputs()
    a=cls(tolerance=.01)
    index=a.build_index(library)
    cls(tolerance=.2,use_ppm=True).search(queries,index,progress_bar=False,n_jobs=1)
    with pytest.raises(ValueError,match='noise_cutoff'):
        cls(noise_cutoff=.05).search(queries,index,progress_bar=False,n_jobs=1)
    if cls is FlashEntropy:
        with pytest.raises(ValueError,match='weighing_type'):
            CosineFlash().search(queries,index,progress_bar=False,n_jobs=1)


def test_hungarian_index_rejected_with_original_error_kind():
    queries,library=inputs()
    with pytest.raises(ValueError,match='use_hungarian=True'):
        Cosine(use_hungarian=True).build_index(library)


@pytest.mark.parametrize('cls',[Cosine,Entropy,ModifiedCosine,CosineFlash,FlashEntropy])
def test_to_dict_stays_constructor_only(cls):
    source=cls(tolerance=.02,noise_cutoff=None,dtype=np.float32)
    config=json.loads(json.dumps(source.to_dict()))
    assert config.pop('__Similarity__')==cls.__name__
    assert cls(**config).to_dict()==source.to_dict()
