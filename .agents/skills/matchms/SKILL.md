---
name: matchms
description: Import, filter, clean and compare (similarities) between MS/MS or MS2 mass spectra and identify unknown compounds from spectral libraries. Use when working with mass spectrometry data files (e.g., .mgf, .msp, .mzml, .parquet). Use this skill when the user asks to "import or load MS/MS spectra", "clean or filter MS/MS spectra", "convert a spectra file", "compare spectra", "compute similarity scores or a similarity matrix", "build a spectral library index", "search queries against a reference library", "build a similarity network" or "spectral network", "export spectra to MGF/MSP/JSON", or mentions matchms, MGF/MSP/MzML files, cosine/entropy spectral similarity. Provides the collection-first matchms workflows via the matchms CLI and the Python API.
---

# Installation
Install the library using `uv`:
```bash
uv pip install matchms
```

# matchms workflows

Matchms is a Python package for importing, processing, cleaning, exporting, and
comparing tandem mass spectrometry (MS/MS) data. The modern, collection-first
workflow is: import a dataset as a `SpectraCollection`, apply filters, compute
similarity scores (one-shot matrix, or index + search for repeated queries),
optionally build a similarity network, and export the processed spectra.
Individual spectra remain available as `Spectrum` objects (legacy / spectrum
wise work), but for complete datasets prefer collections throughout.

The same workflows exist in two forms — pick one per task:

- **CLI** (`matchms ...`): use for file-to-file operations on complete datasets
  (describe, convert, clean, score matrix, build index, search). It validates
  inputs, prints structured JSON errors, and never silently overwrites.
  The CLI covers importing, filtering (and filter composition), score computation and exporting.
- **Python API**: use for anything in-process — custom filters, custom
  similarity parameters, network generation (not in the CLI), or embedding
  matchms in a larger analysis script.

## Stage 1: importing spectra

CLI — inspect a file before anything else, or convert it:

```bash
matchms spectra describe my_spectra.mgf     # n_spectra, peak/intensity/entropy stats
matchms spectra convert in.msp out.json     # pure format conversion, no filtering
```

Python — load a complete dataset as a `SpectraCollection`:

```python
from matchms.importing import load_ms2_dataset

collection = load_ms2_dataset("my_spectra.mgf")   # format detected from extension
# explicit type if the extension is ambiguous:
collection = load_ms2_dataset("my_file.txt", ftype="mgf")

print(collection.n_spectra, collection.metadata.head(), collection.describe())
spectrum_0 = collection[0]                         # rows are regular Spectrum objects
```

- Input formats: `mgf`, `msp`, `mzml`, `mzxml`, `json`, `pickle` and `parquet`
  (CLI input detection is extension-based; the same holds for the Python API
  unless `ftype` is given).
- Collection-level operations that keep metadata and peaks synchronized:
  `collection.filter(mask)` (pandas-style selection on
  `collection.metadata`), `collection.sort("precursor_mz")`,
  `collection[:100, 50.0:500.0]` (rows, then m/z window),
  `collection.drop_empty_spectra()`, `collection.drop_duplicates()`.
- Legacy / low-level importers still work for lists of `Spectrum` objects:
  `load_spectra`, `load_from_mgf`, `load_from_msp`, `load_from_mzml`,
  `load_from_mzxml`, `load_from_json`, `load_from_pickle` in
  `matchms.importing`.

## Stage 2: filtering / cleaning

**Filter categories:**
- **Metadata processing**: Harmonize compound names, derive chemical structures, standardize adducts, correct charges etc.
- **Peak filtering**: Normalize intensities, select by m/z or intensity, remove precursor peaks
- **Quality control**: Require minimum peaks, validate precursor m/z, ensure metadata completeness
- **Chemical annotation**: Add fingerprints, derive InChI/SMILES, repair structural mismatches

CLI — apply a named pipeline (see `matchms filter list` and
`matchms filter pipelines` for names; a full flag reference lives in
`references/cli-reference.md`):

```bash
matchms filter run in.mgf out.mgf --pipeline LIBRARY_CLEANING --report
# extra filters on top: --filter <NAME> (repeatable),
# parameters: --param filter_name.param=value
```

The `--report` flag writes `<out>_processing_report.json` with per-filter
input/output/removed/changed counts. Default pipeline when `--pipeline` is
omitted: `DEFAULT_FILTERS`.

Python — individual filters are plain functions that accept a collection
(failing spectra are dropped from both metadata and peaks) or a single
spectrum (returns `None` on failure):

```python
from matchms.filtering import (
    harmonize_missing_entries,
    select_by_relative_intensity,
    require_minimum_number_of_peaks,
)

collection = harmonize_missing_entries(collection)
collection = select_by_relative_intensity(collection, intensity_from=0.01)
collection = require_minimum_number_of_peaks(collection, n_required=5)
```

For a whole ordered pipeline use `SpectraProcessor` (same API as the CLI,
including processing reports via `create_processing_report()`):

```python
from matchms import SpectraProcessor

processor = SpectraProcessor(filters=[
    "harmonize_missing_entries",
    ("select_by_relative_intensity", {"intensity_from": 0.01}),
    ("require_minimum_number_of_peaks", {"n_required": 5}),
])
processed = processor.process_collection(collection)      # or process_spectrum(s)
```

Known filters are placed in canonical filter order; custom callables are
appended (or placed explicitly). Consult `filter_order.py` to get the correct order of custom filter sets.
Consult `filter_effects.py` to investigate, which filter changes what (Metadata or Fragments/Peaks)
and if its removing or changing Spectra.

When the goal is to compare query and reference spectra, use the exact same Fragment/Peak Processing filters.

### Filtering and Sorting SpectraCollection
A loaded SpectraCollection and be filtered (by Metadata or Fragments/Peaks) during runtime with the Python API.
When one Filter/Sorting is applied the SpectraCollection is synced:
```python
# Select spectra by metadata
positive = collection.filter(collection.metadata["ionmode"] == "positive")

# Sort spectra by metadata
sorted_collection = collection.sort("precursor_mz")

# Select spectra and restrict m/z range
selected = collection[:100, 50.0:500.0]

# Drop spectra without peaks
collection = collection.drop_empty_spectra()

# Drop duplicate spectra
collection = collection.drop_duplicates()
```

## Stage 3: computing similarity scores

Which method (high-level choices; `matchms similarity list` is the source of
truth):

| Similarity | Recommended class | Typical use | Specialized implementations |
|---|---|---|---|
| Cosine | `Cosine` | Standard peak-based spectral similarity with one-to-one peak matching. Supports reusable library indices for repeated searches. | `CosineGreedy`, `CosineHungarian`, `CosineLinear`, `CosineFlash`, `CosineBlink` |
| Modified cosine | `ModifiedCosine` | Cosine similarity allowing both direct fragment matches and matches shifted by the difference in precursor m/z. Supports reusable library indices for repeated searches. | `ModifiedCosineGreedy`, `ModifiedCosineHungarian`, `ModifiedCosineLinear`; `CosineFlash` with `matching_mode="hybrid"` |
| Spectral entropy | `Entropy` | General-purpose spectral entropy similarity with explicit one-to-one matching. Supports fragment, neutral-loss, and hybrid matching as well as reusable library indices. | `EntropyGreedy`, `EntropyFlash` |
| Search-optimized spectral entropy | `EntropySearch` | High-throughput fragment-only entropy searches against large reference libraries. Requires sufficiently separated peaks and can optionally merge close peaks during preparation. | — |
| Neutral-loss cosine | `NeutralLossesCosine` | Compare spectra based on neutral-loss rather than fragment m/z patterns. | — |
| Binned spectra | `BinnedEmbeddingSimilarity` | Compare fixed-width binned spectrum representations using cosine or Euclidean similarity. | — |
| Molecular structure | `FingerprintSimilarity` | Compare molecular fingerprints derived from structure metadata. | — |
| Metadata | `MetadataMatch` | Compare arbitrary metadata fields using exact or tolerance-based matching. | — |
| Precursor or parent mass | `PrecursorMzMatch`, `ParentMassMatch` | Simple matching based on precursor m/z or parent mass. | — |

One-shot matrix — CLI:

```bash
# all-vs-all (symmetric) over one file:
matchms similarity matrix library.mgf --method ModifiedCosine --tolerance 0.01 -o scores.npz
# queries vs references:
matchms similarity matrix queries.mgf library.mgf --method Cosine -o scores.npz
# large data: sparse mode keeps only pairs >= threshold (.tsv/.csv + --id-field for readable ids)
matchms similarity matrix big.mgf --method Cosine --mode sparse --score-min 0.1 -o pairs.tsv
```

Output: `.npz` (a `Scores` artifact, loadable via `Scores.load`), or
`.tsv`/`.csv` long format (one row per pair).

Python — collections go straight into the similarity object:

```python
from matchms.similarity import ModifiedCosine

similarity = ModifiedCosine(tolerance=0.01)
scores = similarity.matrix(collection)            # or (queries, references)
scores.shape, scores.score_fields                 # e.g. ('score', 'matches')
scores.to_array("score")                          # dense numpy matrix
mask = scores > 0.5                               # comparison operators work
high = scores.to_array("score")[mask]
```

For classic list-of-`Spectrum` code, the top-level helper is equivalent:

```python
from matchms import calculate_scores
from matchms.similarity import Cosine

scores = calculate_scores(spectra_1, spectra_2, Cosine(tolerance=0.01))
```

Repeated searches against a fixed library — build the index once, search many
times. Only index-capable classes qualify (`matchms similarity list` shows
`build_index` in each method's `methods` field): `Cosine`, `ModifiedCosine`,
`Entropy`, `EntropySearch`.

```bash
matchms similarity build-index library.mgf --method Cosine --tolerance 0.01 -o lib.index.npz
matchms similarity search queries.mgf lib.index.npz --method Cosine --tolerance 0.01 \
    --top-k 5 --min-score 0.3 -o hits.tsv
```

The search command also accepts a plain spectra file as the library (index
built on the fly) for one-off searches; with an index, pass
`--library-spectra <file>` to enable `--library-id-field`.

Python:

```python
library_index = similarity.build_index(library_collection)
scores = similarity.search(query_collection, library_index)   # rows = queries
similarity.save_index(library_index, "lib_index.npz")
library_index = similarity.load_index("lib_index.npz")        # same configuration!
```

An index is bound to the exact method + parameters it was built with. A
mismatch is an `index_incompatible` error (CLI) — rebuild the index, never
hand-edit it. Note: indexed search for `Cosine`/`ModifiedCosine` is
unavailable with `use_hungarian=True`.

## Stage 4: generating a similarity network (Python only)

The CLI has no network command; use `matchms.networking.SimilarityNetwork`
(wraps `networkx`, a core dependency). It needs a **square all-vs-all**
`Scores` object, so compute a symmetric matrix first:

```python
from matchms.networking import SimilarityNetwork

net = SimilarityNetwork(
    top_n=20,          # candidate edges = top_n hits per node (must be >= max_links)
    max_links=10,      # max outgoing edges kept per node
    score_cutoff=0.7,  # edge only if score >= cutoff
    link_method="single",   # or "mutual" (edge only if both nodes rank each other)
    keep_unconnected_nodes=True,
)
net.create_network(scores, identifiers=[s.get("compound_name") for s in spectra])
```

`identifiers` must be unique, one per row/column of the square matrix.
The resulting `networkx.Graph` is in `net.graph` (edge weight = score);
analyze it directly or export it:

```python
net.export_to_file("network.graphml")          # default
net.export_to_file("network.json", graph_format="cyjs")  # also: json, gexf, gml
```

Tip: for very large datasets, pre-filter the matrix (e.g. via a sparse-mode
matrix with a score threshold) before building the network instead of relying
on a dense all-vs-all result.

## Stage 5: exporting spectra

CLI:

```bash
matchms spectra convert processed.mgf processed.json
matchms filter run in.mgf out.msp --export-style gnps   # styles: matchms, massbank, nist, riken, gnps
```

Python — `SpectraCollection` exports directly:

```python
collection.to_mgf("processed_spectra.mgf")
collection.to_msp("processed_spectra.msp")
collection.to_json("processed_spectra.json")
collection.to_mgf("combined.mgf", append=True)          # mgf/msp only
collection.to_msp("nist_style.msp", export_style="nist")
```

For list-of-`Spectrum` code the classic wrapper works:
`from matchms.exporting import save_spectra; save_spectra(spectra, "out.mgf")`.
Score artifacts round-trip too: `scores.save("scores.npz")` /
`Scores.load("scores.npz")`.

# Plotting Spectra
Only a single spectrum can be plotted or a spectrum can be plotted against another (reference) spectrum.
Plotting cannot be applied to a collection.
To plot a single spectrum and plot it against a reference:
```python
import numpy as np
from matchms import Spectrum

spectrum = Spectrum(
    mz=np.array([100.0, 150.0, 200.0]),
    intensities=np.array([0.2, 1.0, 0.4]),
    metadata={"spectrum_id": "query-1", "precursor_mz": 250.5},
)

print(spectrum.peaks.mz)
print(spectrum.get("precursor_mz"))
losses = spectrum.compute_losses(loss_mz_from=5.0, loss_mz_to=200.0)
spectrum.plot()
spectrum.plot_against(reference_spectrum)
```

# Guardrails

- Never overwrite an input file. CLI: existing outputs need `--append`
  (`.mgf`/`.msp` only) or fail with `file_exists`; similarity outputs are
  replaced. Python: write to a new filename.
- Check the CLI exit code (`0` ok, `1` handled error with a JSON error
  payload, `2` usage error) and the `ok` field of the JSON before using an
  output file. Key error codes: `input_not_found`, `unsupported_format`,
  `empty_spectra`, `unknown_value` (name not found — read `valid_values` and
  the "Did you mean" hint), `unsupported_method` (method lacks the required
  capability), `index_incompatible` (rebuild the index), `matrix_too_large`
  (switch to `--mode sparse --score-min`), `missing_dependency` (e.g. RDKit —
  do not retry, install it or change method).
- Formats are detected from the file extension only (CLI); rename files with
  non-standard extensions or use `ftype` in Python / `--ftype` in the CLI.
- Peak data in a `SpectraCollection` is binned: reconstructed spectra may
  carry bin-center m/z values. Compare m/z numerically with a tolerance,
  never by exact equality.
- `matchms similarity list` / `matchms filter list` output is the source of
  truth for available methods/filters and their capabilities — do not assume
  a hardcoded list.
- For large matrices, prefer sparse mode with a `--score-min` threshold;
  dense mode refuses matrices above 50,000,000 entries.
- Network generation requires a square all-vs-all score matrix and unique
  node identifiers; it is Python-only.
- Never compare raw query spectra against differently processed reference spectra.
- Never deserialize untrusted pickle data.


# Additional resources

- **`references/cli-reference.md`** — full CLI reference: every subcommand's
  flags, JSON payload shapes, supported formats, and the complete error
  vocabulary.
- User documentation: https://matchms.readthedocs.io/en/latest/
