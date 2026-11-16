# matchms CLI reference

All commands accept the global output flags (before or after the subcommand):

- `--json` — force machine-readable JSON on stdout
- `--table` — force human-readable tabular text on stdout
- without either — JSON when stdout is not a terminal, text otherwise
- `--quiet` — suppress log output (errors only)
- `--verbose` — debug logging (stderr)

Logs and progress bars always go to stderr; stdout carries only the payload.
Exit codes: `0` ok, `1` handled CLI error (JSON error payload), `2` usage error
(unknown command or bad arguments).

Success payloads start with `"ok": true` and an `"operation"` string.
Error payloads have `"ok": false`, `"error"`, `"message"`, `"operation"`,
optional `input_file` / `parameter` / `valid_values` / `hint`, and
`"output_produced": {"any": bool, "files": [...]}`.

## Formats

- Input: `json`, `mgf`, `msp`, `mzml`, `mzxml`, `pickle` (detected from the
  extension; `filter run` also takes `--ftype` to force one).
- Output: `json`, `mgf`, `msp`, `pickle` (`filter run`, `spectra convert`).
  The output *directory* must already exist.
- `--append` (where supported) is only valid for `.mgf` / `.msp` outputs.

## matchms info

`matchms info [--json]` — versions, supported formats, and the command tree.
Payload: `ok`, `operation`, `cli` (`{name, schema_version, python, platform,
commands}` — one entry per subcommand), `matchms` (version), `filters`
(count), `input_formats`, `output_formats`.

## matchms filter

### filter list

`matchms filter list` — all filters in default execution order.
Payload: `ok`, `operation`, `n_filters`, `filters: [{order, name, description}]`.

### filter pipelines

`matchms filter pipelines [NAME]` — overview of all default pipelines
(`{name, description, n_filters, is_default}`), or, with `NAME`, the filters of
that pipeline in execution order (with pipeline-specific parameters).
The names are the valid values for `filter run --pipeline`.

### filter info

`matchms filter info <name>` — description, parameters (name, type, default,
description) and input type of one filter. Payload: `ok`, `operation`, `name`,
`order`, `description`, `collection_supported`, `docstring`, `required`
(list), `parameters` (list of `{name, type, required, default, description}`).

### filter run

```
matchms filter run <input> <output>
    [--ftype FTYPE]
    [--pipeline {DEFAULT_FILTERS, BASIC_FILTERS, LIBRARY_CLEANING, ...}]
    [--filter NAME]...            # added at the canonical filter-order position
    [--param FILTER_NAME.PARAM=VALUE]...
    [--report]
    [--export-style {matchms, massbank, nist, riken, gnps}]
    [--append]
```

Base pipeline defaults to `DEFAULT_FILTERS`. Every `--param` must target a
filter that is part of the resulting pipeline, and that filter must accept the
parameter. `--report` writes `<output>_processing_report.json` next to the
output. Payload: `ok`, `operation`, `input_file`, `input_format`,
`output_file`, `output_format`, `pipeline` (name), `n_spectra_in`,
`n_spectra_out`, `n_removed`, `filters` (ordered list of
`{name, parameters?}`), `report_file` (path) and `report` (list, one entry per
filter: `{filter, input_spectra, output_spectra, removed_spectra,
changed_metadata, changed_fragments}` — only with `--report`).

Pipelines (from `matchms.filtering.default_pipelines`): `DEFAULT_FILTERS`,
`BASIC_FILTERS`, `LIBRARY_CLEANING`, `CLEAN_PEAKS`, `HARMONIZE_METADATA_ENTRIES`,
`HARMONIZE_METADATA_FIELD_NAMES`, `DERIVE_MISSING_METADATA`,
`DERIVE_METADATA_IN_WRONG_FIELD`, `REPAIR_ANNOTATION`,
`REQUIRE_COMPLETE_ANNOTATION`, `REQUIRE_COMPLETE_METADATA`,
`MS2DEEPSCORE_TRAINING`, `OTHER_FILTERS`.

## matchms similarity

### similarity list

`matchms similarity list` — all similarity classes grouped as in the README
("Cosine", "Modified cosine", "Spectral entropy", ...; ungrouped classes go to
"Other"). Each entry: `{group, name, description, score_fields, methods}` where
`methods` ⊆ `pair, matrix, sparse_matrix, build_index, search` — the
`methods` field is the capability source of truth (e.g. `build_index` present
means the class supports `build-index`/`search`).

### similarity info

`matchms similarity info <name>` — group, full description, score fields,
supported methods, constructor parameters (name, type, default, description),
and whether the class is commutative.

### similarity matrix

```
matchms similarity matrix <spectra_1> [spectra_2]
    --method NAME
    [--param NAME=VALUE]... | [--tolerance FLOAT]
    [--mode {dense, sparse}] [--score-min FLOAT]
    -o <out.{npz,tsv,csv}>
    [--id-field FIELD] [--max-dense-entries N] [--top N] [--no-progress]
```

- Without `spectra_2`: symmetric all-vs-all matrix over `spectra_1`.
- With `spectra_2`: rows = `spectra_1`, columns = `spectra_2`.
- `--mode sparse` requires a sparse-capable method and keeps only pairs with
  main score `>= --score-min`.
- Output: `.npz` stores the Scores artifact; `.tsv`/`.csv` are long format
  (one row per pair; `--id-field` adds `row_id`/`col_id` metadata columns).
  An existing output file is replaced.
- Dense mode fails with `matrix_too_large` when rows × cols >
  `--max-dense-entries` (default 50,000,000).
- Payload: `ok`, `operation`, `method` (`{name, class, params}` — the effective
  constructor parameters), `mode`, `inputs` (`spectra_1`/`spectra_2` with
  `file`, `n_spectra`; `symmetric`), `scores` (`shape`, `kind`
  dense/sparse, `score_fields`, `stats` — per score field
  `{count, min, max, mean, std}`; sparse adds `density`, `n_stored`),
  `output`, `top_pairs` (with `--top`), `elapsed_seconds`.

### similarity build-index

```
matchms similarity build-index <library>
    --method NAME
    [--param NAME=VALUE]... | [--tolerance FLOAT]
    -o <out.index.npz>
```

- The method must implement `build_index()`; the valid values in the error
  payload list the index-capable classes.
- Output must end in `.index.npz`; an existing file is replaced.
- The index is bound to the exact method + parameters.
- `Entropy` methods: `matching_mode` (`fragment`, `neutral_loss`, `hybrid`)
  is validated up front (`invalid_parameter` for other values).
- `EntropySearch`: `peak_separation` may only be `merge` or `raise`.
- Payload: `ok`, `operation`, `method`, `library` (`{file, n_spectra}`),
  `index` (`{file, format, size_bytes}`), `elapsed_seconds`, `note`.

### similarity search

```
matchms similarity search <queries> <library>
    --method NAME
    [--param NAME=VALUE]... | [--tolerance FLOAT]
    [--top-k N] [--min-score FLOAT] [--score-field NAME]
    [--query-id-field FIELD] [--library-id-field FIELD]
    [--library-spectra FILE]
    [--batch-size N]
    -o <out.{tsv,csv}>
    [--top N]
```

- `library` is either a spectra file (index built on the fly) or a saved
  `.index.npz` index. Queries are rows; the library the columns. A query may
  be its own best hit.
- For an index library, the method and parameters must match those used to
  build the index, otherwise the error is `index_incompatible`.
- `--library-spectra` is only for index libraries: the spectra file the index
  was built from (count must match); it enables `--library-id-field`.
- `--score-field` selects the ranking/score field (default `score`; e.g.
  `matches` for cosine-family). `--min-score` filters on that field.
- `--batch-size` (default 1000) limits memory for large query files.
- Output is a hit list (one row per kept pair): `query_index`,
  `reference_index`, `rank`, then all score fields of the method (e.g.
  `score`, `matches`), plus id columns when id fields are set.
- Payload: `ok`, `operation`, `method`, `queries` (`{file, n_spectra}`),
  `library` (`{file, kind: "spectra"|"index", n_spectra, spectra_file}`),
  `search` (`{top_k, min_score, score_field}`), `results`
  (`{n_hits, n_queries_with_hits, n_queries_without_hits}`), `top_hits`,
  `output`, `elapsed_seconds`.

## matchms spectra

### spectra describe

`matchms spectra describe <file>` — per-collection statistics. Payload: `ok`,
`operation`, `input_file`, `n_spectra`, `statistics` (`peak_counts`,
`intensity_sums`, `intensity_entropy` — each with `{count, mean, std, min,
"25%", "50%", "75%", max}` over spectra).

### spectra convert

```
matchms spectra convert <input> <output>
    [--ftype FTYPE]
    [--export-style {matchms, massbank, nist, riken, gnps}]
    [--append]
```

Pure format conversion, no filtering. Payload: `ok`, `operation`, `input_file`,
`input_format`, `output_file`, `output_format`, `n_spectra`.

## Error codes

`input_not_found`, `unsupported_format`, `file_exists`, `empty_spectra`,
`invalid_parameter`, `missing_parameter`, `invalid_input`, `compute_error`,
`save_failed`, `missing_dependency`, `unsupported_method`, `unknown_value`,
`matrix_too_large`, `index_incompatible`.

- `unknown_value` carries `parameter` (the option, e.g. `method`,
  `pipeline_name`), the offending value in `message`, `valid_values`, and a
  `hint` with the closest suggestion ("Did you mean ...?").
- `unsupported_method` carries `valid_values` = the names that do have the
  required capability (e.g. the index-capable classes).
- `file_exists` occurs only when the caller did not pass `--append`.
- Progress bars and log lines are stderr-only; JSON payloads on stdout are
  always parseable in machine mode.
