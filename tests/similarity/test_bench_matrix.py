import numpy as np
import pytest
from matchms import Spectrum
from matchms.similarity import CosineGreedy, CosineHungarian, FlashSimilarity, LinearCosine


def _make_synthetic_spectra(n_spectra, n_peaks=30, n_common=50, tolerance=0.02, seed=42):
    rng = np.random.default_rng(seed)
    # Build a shared pool of well-separated m/z values (gaps > 2*tolerance)
    # so that merge/preprocessing is a no-op — we benchmark pure scoring.
    min_gap = 2.5 * tolerance
    pool_mz = np.empty(n_common)
    pool_mz[0] = 50.0
    for i in range(1, n_common):
        pool_mz[i] = pool_mz[i - 1] + rng.uniform(min_gap, 10.0)
    spectra = []
    for _ in range(n_spectra):
        chosen = rng.choice(pool_mz, size=n_peaks, replace=False)
        # Tiny jitter keeps peaks within tolerance of the pool value
        # but still well-separated from neighboring pool values.
        mz = np.sort(chosen + rng.uniform(-tolerance * 0.25, tolerance * 0.25, size=n_peaks))
        intensities = rng.exponential(scale=1.0, size=n_peaks)
        intensities /= intensities.max()
        precursor_mz = float(rng.uniform(200, 600))
        spectra.append(Spectrum(mz=mz, intensities=intensities, metadata={"precursor_mz": precursor_mz}))
    return spectra


SIZES = [50, 100, 200]


@pytest.mark.parametrize("n_spectra", SIZES)
def test_bench_cosine_hungarian(benchmark, n_spectra):
    spectra = _make_synthetic_spectra(n_spectra)
    sim = CosineHungarian(tolerance=0.02)
    benchmark(sim.matrix, spectra, spectra, is_symmetric=True, progress_bar=False)


@pytest.mark.parametrize("n_spectra", SIZES)
def test_bench_cosine_greedy(benchmark, n_spectra):
    spectra = _make_synthetic_spectra(n_spectra)
    sim = CosineGreedy(tolerance=0.02)
    benchmark(sim.matrix, spectra, spectra, is_symmetric=True, progress_bar=False)


@pytest.mark.parametrize("n_spectra", SIZES)
def test_bench_linear_cosine(benchmark, n_spectra):
    spectra = _make_synthetic_spectra(n_spectra)
    sim = LinearCosine(tolerance=0.02)
    benchmark(sim.matrix, spectra, spectra, is_symmetric=True, progress_bar=False)


@pytest.mark.parametrize("n_spectra", SIZES)
def test_bench_flash_similarity(benchmark, n_spectra):
    spectra = _make_synthetic_spectra(n_spectra)
    sim = FlashSimilarity(tolerance=0.02)
    benchmark(sim.matrix, spectra, spectra, is_symmetric=True, n_jobs=1)
