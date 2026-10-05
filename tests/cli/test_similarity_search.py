import json
import os
import pandas as pd
import pytest
from matchms.cli.main import main
from matchms.importing.load_spectra import SUPPORTED_FILE_FORMATS as INPUT_FORMATS


TEST_DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "testdata"))
MGF_FILE = os.path.join(TEST_DATA, "testdata.mgf")

INDEX_CAPABLE = ("Cosine", "CosineFlash", "ModifiedCosine", "Entropy", "EntropyFlash", "EntropySearch")

# Expected JSON payload key structure (schema snapshot).
TOP_KEYS = {
    "ok", "operation", "method", "queries", "library",
    "search", "results", "top_hits", "elapsed_seconds", "output",
}
METHOD_KEYS = {"name", "class", "params"}
QUERIES_KEYS = {"file", "n_spectra"}
LIBRARY_KEYS = {"file", "kind", "n_spectra", "spectra_file"}
SEARCH_KEYS = {"top_k", "min_score", "score_field"}
RESULTS_KEYS = {"n_hits", "n_queries_with_hits", "n_queries_without_hits"}
OUTPUT_KEYS = {"file", "format", "size_bytes"}
TOP_HIT_KEYS = {"query_index", "reference_index", "value"}

# A small library of three clearly distinct spectra (used as queries as well).
LIBRARY_SPECS = [
    (100.0, [(10.0, 1.0), (20.0, 2.0), (30.0, 3.0)]),
    (200.0, [(100.0, 1.0), (200.0, 2.0), (300.0, 3.0)]),
    (300.0, [(50.0, 1.0), (70.0, 2.0), (90.0, 3.0)]),
]


def make_mgf(specs):
    """Render a list of (precursor_mz, [(mz, intensity), ...]) as MGF text."""
    chunks = []
    for precursor, peaks in specs:
        lines = [f"PEPMASS={precursor}"]
        lines += [f"{mz} {intensity}" for mz, intensity in peaks]
        chunks.append("BEGIN IONS\n" + "\n".join(lines) + "\nEND IONS\n")
    return "".join(chunks)


SMALL_LIBRARY_MGF = make_mgf(LIBRARY_SPECS)


def run_cli(*argv, capsys):
    """Run the CLI and return (exit_code, parsed_json_stdout)."""
    exit_code = main(list(argv))
    out = capsys.readouterr().out
    return exit_code, json.loads(out)


def read_tsv(path):
    # "matches" is only present for cosine-family methods; keep it optional.
    return pd.read_csv(str(path), sep="\t", dtype={"matches": "Int64"})


def write_mgf(tmp_path, text, name="lib.mgf"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


def write_mgf_with_ids(tmp_path, specs, prefix, name="library.mgf"):
    """Write an MGF file whose spectra carry spectrum_id and compound_name."""
    lines = []
    for index, (precursor, peaks) in enumerate(specs, start=1):
        lines.append("BEGIN IONS")
        lines.append(f"spectrum_id={prefix}{index}")
        lines.append(f"compound_name={prefix}_compound_{index}")
        lines.append(f"PEPMASS={precursor}")
        lines += [f"{mz} {intensity}" for mz, intensity in peaks]
        lines.append("END IONS")
        lines.append("")
    path = tmp_path / name
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# Success cases
# ---------------------------------------------------------------------------


def test_search_spectra_file(tmp_path, capsys):
    out = tmp_path / "hits.tsv"
    exit_code, payload = run_cli(
        "similarity", "search", MGF_FILE, MGF_FILE,
        "--method", "Cosine", "--top-k", "5",
        "-o", str(out), "--json", capsys=capsys,
    )
    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["operation"] == "similarity search"
    assert payload["queries"]["n_spectra"] == 30
    assert payload["library"]["kind"] == "spectra"
    assert payload["library"]["n_spectra"] == 30
    assert payload["library"]["spectra_file"] == MGF_FILE
    assert payload["search"] == {"top_k": 5, "min_score": None, "score_field": "score"}
    assert payload["results"]["n_queries_with_hits"] + payload["results"]["n_queries_without_hits"] == 30
    assert payload["results"]["n_hits"] > 0
    assert payload["output"]["format"] == "tsv"
    assert out.exists()
    assert os.path.getsize(str(out)) == payload["output"]["size_bytes"]


def test_search_spectra_and_index_give_identical_hits(tmp_path, capsys):
    """Searching a spectra file and a saved index of the same file agree exactly."""
    lib = write_mgf(tmp_path, SMALL_LIBRARY_MGF)
    index = tmp_path / "lib.index.npz"
    exit_code, _ = run_cli(
        "similarity", "build-index", lib, "--method", "Cosine",
        "-o", str(index), "--json", capsys=capsys,
    )
    assert exit_code == 0

    out_spectra = tmp_path / "hits_spectra.tsv"
    exit_code, _ = run_cli(
        "similarity", "search", lib, lib, "--method", "Cosine", "--top-k", "3",
        "-o", str(out_spectra), "--json", capsys=capsys,
    )
    assert exit_code == 0

    out_index = tmp_path / "hits_index.tsv"
    exit_code, _ = run_cli(
        "similarity", "search", lib, str(index), "--method", "Cosine", "--top-k", "3",
        "-o", str(out_index), "--json", capsys=capsys,
    )
    assert exit_code == 0

    assert read_tsv(out_spectra).equals(read_tsv(out_index))


@pytest.mark.parametrize("method", INDEX_CAPABLE)
def test_search_each_method_identity_best_hit(tmp_path, capsys, method):
    """For a query taken from the library, the best hit is itself."""
    lib = write_mgf(tmp_path, SMALL_LIBRARY_MGF)
    out = tmp_path / "hits.tsv"
    args = ["similarity", "search", lib, lib, "--method", method, "--top-k", "1",
            "-o", str(out), "--json"]
    if method == "EntropySearch":
        args += ["--param", "max_tolerance=0.01"]
    exit_code, _ = run_cli(*args, capsys=capsys)
    assert exit_code == 0

    frame = read_tsv(out)
    assert len(frame) == 3
    for query_index in range(3):
        row = frame[frame["query_index"] == query_index].iloc[0]
        assert row["rank"] == 1
        assert row["reference_index"] == query_index
        assert row["score"] > 0.5


def test_search_orientation_queries_are_rows(tmp_path, capsys):
    """Rows are the queries; reference_index points into the library order."""
    library = write_mgf(tmp_path, make_mgf(LIBRARY_SPECS), name="library.mgf")
    queries = write_mgf(tmp_path, make_mgf(LIBRARY_SPECS[:2]), name="queries.mgf")
    out = tmp_path / "hits.tsv"
    exit_code, payload = run_cli(
        "similarity", "search", queries, library, "--method", "Cosine",
        "--top-k", "1", "-o", str(out), "--json", capsys=capsys,
    )
    assert exit_code == 0
    assert payload["queries"]["n_spectra"] == 2
    assert payload["library"]["n_spectra"] == 3

    frame = read_tsv(out)
    assert set(frame["query_index"].tolist()) == {0, 1}
    # query 0 == library row 0, query 1 == library row 1
    assert frame[frame["query_index"] == 0].iloc[0]["reference_index"] == 0
    assert frame[frame["query_index"] == 1].iloc[0]["reference_index"] == 1


def test_search_top_k_limits_hits_per_query(tmp_path, capsys):
    out = tmp_path / "hits.tsv"
    for top_k, expected_rows in ((1, 30), (2, 60)):
        exit_code, payload = run_cli(
            "similarity", "search", MGF_FILE, MGF_FILE, "--method", "Cosine",
            "--top-k", str(top_k), "-o", str(out), "--json", capsys=capsys,
        )
        assert exit_code == 0
        frame = read_tsv(out)
        assert payload["results"]["n_hits"] == expected_rows
        assert (frame.groupby("query_index").size() <= top_k).all()
        assert set(frame["rank"].unique().tolist()) <= set(range(1, top_k + 1))


def test_search_min_score_filters_and_counts(tmp_path, capsys):
    out = tmp_path / "hits.tsv"
    exit_code, payload = run_cli(
        "similarity", "search", MGF_FILE, MGF_FILE, "--method", "Cosine",
        "--top-k", "5", "--min-score", "0.999",
        "-o", str(out), "--json", capsys=capsys,
    )
    assert exit_code == 0
    frame = read_tsv(out)
    if len(frame):
        assert (frame["score"] >= 0.999).all()
    # A strict threshold is reported, not an error.
    assert payload["results"]["n_queries_without_hits"] >= 0
    assert payload["results"]["n_queries_with_hits"] + \
        payload["results"]["n_queries_without_hits"] == 30


def test_search_min_score_can_empty_a_query(tmp_path, capsys):
    """A low top-k / high min-score can leave queries with no rows; that is not an error."""
    lib = write_mgf(tmp_path, SMALL_LIBRARY_MGF)
    out = tmp_path / "hits.tsv"
    exit_code, payload = run_cli(
        "similarity", "search", lib, lib, "--method", "Cosine",
        "--top-k", "1", "--min-score", "0.9999",
        "-o", str(out), "--json", capsys=capsys,
    )
    assert exit_code == 0
    frame = read_tsv(out)
    if len(frame):
        assert (frame["score"] >= 0.9999).all()
    assert payload["results"]["n_queries_with_hits"] + \
        payload["results"]["n_queries_without_hits"] == 3


def test_search_batching_is_deterministic(tmp_path, capsys):
    """--batch-size 1 and a large batch size give identical output."""
    out_a = tmp_path / "hits_1.tsv"
    out_b = tmp_path / "hits_big.tsv"
    exit_code, _ = run_cli(
        "similarity", "search", MGF_FILE, MGF_FILE, "--method", "Cosine",
        "--top-k", "3", "--batch-size", "1", "-o", str(out_a), "--json", capsys=capsys,
    )
    assert exit_code == 0
    exit_code, _ = run_cli(
        "similarity", "search", MGF_FILE, MGF_FILE, "--method", "Cosine",
        "--top-k", "3", "--batch-size", "1000", "-o", str(out_b), "--json", capsys=capsys,
    )
    assert exit_code == 0
    assert read_tsv(out_a).equals(read_tsv(out_b))


def test_search_score_field_matches_ranks_by_matches(tmp_path, capsys):
    """--score-field ranks (and applies --min-score) on that field."""
    out = tmp_path / "hits.tsv"
    exit_code, payload = run_cli(
        "similarity", "search", MGF_FILE, MGF_FILE, "--method", "Cosine",
        "--top-k", "3", "--score-field", "matches",
        "-o", str(out), "--json", capsys=capsys,
    )
    assert exit_code == 0
    assert payload["search"]["score_field"] == "matches"
    frame = read_tsv(out)
    for _, group in frame.groupby("query_index", sort=True):
        matches = group["matches"].tolist()
        assert matches == sorted(matches, reverse=True)


def test_search_tied_scores_ranked_in_library_order(tmp_path, capsys):
    out = tmp_path / "hits.tsv"
    exit_code, _ = run_cli(
        "similarity", "search", MGF_FILE, MGF_FILE, "--method", "Cosine",
        "--top-k", "5", "-o", str(out), "--json", capsys=capsys,
    )
    assert exit_code == 0
    frame = read_tsv(out)
    # For a tied best score the first reference in library order comes first.
    row0 = frame[(frame["query_index"] == 0) & (frame["rank"] == 1)].iloc[0]
    assert row0["reference_index"] == 0


def test_search_with_library_ids(tmp_path, capsys):
    library = write_mgf_with_ids(tmp_path, LIBRARY_SPECS, "Q", name="library.mgf")
    out = tmp_path / "hits.tsv"
    exit_code, _ = run_cli(
        "similarity", "search", library, library, "--method", "Cosine",
        "--query-id-field", "spectrum_id", "--library-id-field", "compound_name",
        "--top-k", "2", "-o", str(out), "--json", capsys=capsys,
    )
    assert exit_code == 0
    frame = read_tsv(out)
    assert "query_id" in frame.columns and "reference_id" in frame.columns
    # The best hit of query 0 is itself, with its own identifier.
    row0 = frame[(frame["query_index"] == 0) & (frame["rank"] == 1)].iloc[0]
    assert row0["query_id"] == "Q1"
    assert row0["reference_id"] == "Q_compound_1"
    # The id columns are omitted when the flag is not set.
    out2 = tmp_path / "hits_noid.tsv"
    exit_code, _ = run_cli(
        "similarity", "search", library, library, "--method", "Cosine",
        "--top-k", "2", "-o", str(out2), "--json", capsys=capsys,
    )
    assert exit_code == 0
    assert "query_id" not in read_tsv(out2).columns
    assert "reference_id" not in read_tsv(out2).columns


def test_search_index_with_library_spectra_ids(tmp_path, capsys):
    """An index library can resolve ids when --library-spectra is provided."""
    lib = write_mgf_with_ids(tmp_path, LIBRARY_SPECS, "L", name="library.mgf")
    index = tmp_path / "lib.index.npz"
    exit_code, _ = run_cli(
        "similarity", "build-index", lib, "--method", "Cosine",
        "-o", str(index), "--json", capsys=capsys,
    )
    assert exit_code == 0

    out = tmp_path / "hits.tsv"
    exit_code, payload = run_cli(
        "similarity", "search", lib, str(index), "--method", "Cosine",
        "--library-id-field", "compound_name",
        "--library-spectra", lib, "--top-k", "2",
        "-o", str(out), "--json", capsys=capsys,
    )
    assert exit_code == 0
    assert payload["library"]["kind"] == "index"
    assert payload["library"]["spectra_file"] == lib
    frame = read_tsv(out)
    assert "reference_id" in frame.columns
    # The best hit of query 0 resolves to library row 0's identifier.
    row0 = frame[(frame["query_index"] == 0) & (frame["rank"] == 1)].iloc[0]
    assert row0["reference_index"] == 0
    assert row0["reference_id"] == "L_compound_1"


def test_search_output_csv(tmp_path, capsys):
    out = tmp_path / "hits.csv"
    exit_code, payload = run_cli(
        "similarity", "search", MGF_FILE, MGF_FILE, "--method", "Cosine",
        "--top-k", "1", "-o", str(out), "--json", capsys=capsys,
    )
    assert exit_code == 0
    assert payload["output"]["format"] == "csv"
    frame = pd.read_csv(str(out), sep=",")
    assert list(frame.columns[:3]) == ["query_index", "reference_index", "rank"]


def test_search_output_replaced(tmp_path, capsys):
    out = tmp_path / "hits.tsv"
    out.write_text("stale-header\nstale\nrow\n", encoding="utf-8")
    exit_code, payload = run_cli(
        "similarity", "search", MGF_FILE, MGF_FILE, "--method", "Cosine",
        "--top-k", "1", "-o", str(out), "--json", capsys=capsys,
    )
    assert exit_code == 0
    frame = read_tsv(out)
    assert "stale" not in " ".join(frame.columns)
    assert payload["results"]["n_hits"] == 30


def test_search_json_payload_schema(tmp_path, capsys):
    out = tmp_path / "hits.tsv"
    exit_code, payload = run_cli(
        "similarity", "search", MGF_FILE, MGF_FILE, "--method", "Cosine",
        "--top-k", "3", "--top", "3",
        "--query-id-field", "spectrum_id", "--library-id-field", "compound_name",
        "-o", str(out), "--json", capsys=capsys,
    )
    assert exit_code == 0
    assert set(payload) == TOP_KEYS
    assert set(payload["method"]) == METHOD_KEYS
    assert payload["method"]["name"] == "Cosine"
    assert payload["method"]["class"] == "Cosine"
    assert set(payload["queries"]) == QUERIES_KEYS
    assert set(payload["library"]) == LIBRARY_KEYS
    assert set(payload["search"]) == SEARCH_KEYS
    assert set(payload["results"]) == RESULTS_KEYS
    assert set(payload["output"]) == OUTPUT_KEYS
    assert isinstance(payload["elapsed_seconds"], float)
    assert len(payload["top_hits"]) <= 3
    for hit in payload["top_hits"]:
        assert set(hit) == TOP_HIT_KEYS | {"query_id", "reference_id"}


def test_search_tsv_column_order(tmp_path, capsys):
    lib = write_mgf_with_ids(tmp_path, LIBRARY_SPECS, "Q", name="library.mgf")
    out = tmp_path / "hits.tsv"
    exit_code, _ = run_cli(
        "similarity", "search", lib, lib, "--method", "Cosine",
        "--query-id-field", "spectrum_id", "--library-id-field", "compound_name",
        "--top-k", "2", "-o", str(out), "--json", capsys=capsys,
    )
    assert exit_code == 0
    with open(out) as handle:
        header = handle.readline().strip()
    # Cosine has two score fields; ids are present in this run.
    assert header == "query_index\tquery_id\treference_index\treference_id\trank\tscore\tmatches"


def test_search_human_table_output(tmp_path, capsys):
    out = tmp_path / "hits.tsv"
    exit_code = main([
        "similarity", "search", MGF_FILE, MGF_FILE, "--method", "Cosine",
        "--top-k", "3", "-o", str(out), "--table",
    ])
    text = capsys.readouterr().out
    assert exit_code == 0
    assert "Similarity search:" in text
    assert "hit list:" in text
    with pytest.raises(json.JSONDecodeError):
        json.loads(text)


def test_search_method_is_case_insensitive(tmp_path, capsys):
    out = tmp_path / "hits.tsv"
    exit_code, payload = run_cli(
        "similarity", "search", MGF_FILE, MGF_FILE,
        "--method", "entropysearch", "--param", "max_tolerance=0.01",
        "--top-k", "1", "-o", str(out), "--json", capsys=capsys,
    )
    assert exit_code == 0
    assert payload["method"]["name"] == "EntropySearch"


def test_search_effective_params_include_defaults(tmp_path, capsys):
    out = tmp_path / "hits.tsv"
    exit_code, payload = run_cli(
        "similarity", "search", MGF_FILE, MGF_FILE, "--method", "Cosine",
        "--tolerance", "0.1", "--top-k", "1", "-o", str(out), "--json", capsys=capsys,
    )
    assert exit_code == 0
    params = payload["method"]["params"]
    assert params["tolerance"] == pytest.approx(0.1)
    assert params["remove_precursor"] is True


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad", ["hits.npz", "hits.json", "hits", "hits.txt"])
def test_search_wrong_output_extension(tmp_path, capsys, bad):
    exit_code, payload = run_cli(
        "similarity", "search", MGF_FILE, MGF_FILE, "--method", "Cosine",
        "-o", str(tmp_path / bad), "--json", capsys=capsys,
    )
    assert exit_code == 1
    assert payload["error"] == "invalid_parameter"
    assert payload["parameter"] == "output"
    assert set(payload["valid_values"]) == {".csv", ".tsv"}


def test_search_missing_queries(tmp_path, capsys):
    missing = tmp_path / "nope.mgf"
    exit_code, payload = run_cli(
        "similarity", "search", str(missing), MGF_FILE, "--method", "Cosine",
        "-o", str(tmp_path / "hits.tsv"), "--json", capsys=capsys,
    )
    assert exit_code == 1
    assert payload["error"] == "input_not_found"
    assert payload["input_file"] == str(missing)


def test_search_missing_library(tmp_path, capsys):
    missing = tmp_path / "nope.mgf"
    exit_code, payload = run_cli(
        "similarity", "search", MGF_FILE, str(missing), "--method", "Cosine",
        "-o", str(tmp_path / "hits.tsv"), "--json", capsys=capsys,
    )
    assert exit_code == 1
    assert payload["error"] == "input_not_found"
    assert set(payload["valid_values"]) == set(INPUT_FORMATS) | {".index.npz"}


def test_search_unsupported_query_extension(tmp_path, capsys):
    fake = tmp_path / "queries.txt"
    fake.write_text("not a spectra file\n", encoding="utf-8")
    exit_code, payload = run_cli(
        "similarity", "search", str(fake), MGF_FILE, "--method", "Cosine",
        "-o", str(tmp_path / "hits.tsv"), "--json", capsys=capsys,
    )
    assert exit_code == 1
    assert payload["error"] == "unsupported_format"
    assert payload["input_file"] == str(fake)


def test_search_unsupported_library_extension(tmp_path, capsys):
    fake = tmp_path / "library.txt"
    fake.write_text("not a spectra file\n", encoding="utf-8")
    exit_code, payload = run_cli(
        "similarity", "search", MGF_FILE, str(fake), "--method", "Cosine",
        "-o", str(tmp_path / "hits.tsv"), "--json", capsys=capsys,
    )
    assert exit_code == 1
    assert payload["error"] == "unsupported_format"
    assert payload["input_file"] == str(fake)


@pytest.mark.parametrize("name", ["CosineGreedy", "MetadataMatch", "ParentMassMatch", "EntropyGreedy"])
def test_search_non_indexed_method(tmp_path, capsys, name):
    exit_code, payload = run_cli(
        "similarity", "search", MGF_FILE, MGF_FILE, "--method", name,
        "-o", str(tmp_path / "hits.tsv"), "--json", capsys=capsys,
    )
    assert exit_code == 1
    assert payload["error"] == "unsupported_method"
    assert set(payload["valid_values"]) == set(INDEX_CAPABLE)


def test_search_unknown_method_has_suggestion(tmp_path, capsys):
    exit_code, payload = run_cli(
        "similarity", "search", MGF_FILE, MGF_FILE, "--method", "Cosin",
        "-o", str(tmp_path / "hits.tsv"), "--json", capsys=capsys,
    )
    assert exit_code == 1
    assert payload["error"] == "unknown_value"
    assert payload["parameter"] == "method"
    assert "Cosine" in payload["valid_values"]


def test_search_rejects_use_hungarian(tmp_path, capsys):
    exit_code, payload = run_cli(
        "similarity", "search", MGF_FILE, MGF_FILE, "--method", "ModifiedCosine",
        "--param", "use_hungarian=true",
        "-o", str(tmp_path / "hits.tsv"), "--json", capsys=capsys,
    )
    assert exit_code == 1
    assert payload["error"] == "unsupported_method"
    assert payload["parameter"] == "use_hungarian"


def test_search_unknown_param_name(tmp_path, capsys):
    exit_code, payload = run_cli(
        "similarity", "search", MGF_FILE, MGF_FILE, "--method", "Cosine",
        "--param", "not_a_param=1",
        "-o", str(tmp_path / "hits.tsv"), "--json", capsys=capsys,
    )
    assert exit_code == 1
    assert payload["error"] == "invalid_parameter"
    assert payload["parameter"] == "not_a_param"


def test_search_unknown_score_field(tmp_path, capsys):
    exit_code, payload = run_cli(
        "similarity", "search", MGF_FILE, MGF_FILE, "--method", "Cosine",
        "--score-field", "scoree",
        "-o", str(tmp_path / "hits.tsv"), "--json", capsys=capsys,
    )
    assert exit_code == 1
    assert payload["error"] == "unknown_value"
    assert payload["parameter"] == "score_field"
    assert set(payload["valid_values"]) == {"score", "matches"}


def test_search_bad_top_k(tmp_path, capsys):
    exit_code, payload = run_cli(
        "similarity", "search", MGF_FILE, MGF_FILE, "--method", "Cosine",
        "--top-k", "0", "-o", str(tmp_path / "hits.tsv"), "--json", capsys=capsys,
    )
    assert exit_code == 1
    assert payload["error"] == "invalid_parameter"
    assert payload["parameter"] == "top_k"


def test_search_bad_batch_size(tmp_path, capsys):
    exit_code, payload = run_cli(
        "similarity", "search", MGF_FILE, MGF_FILE, "--method", "Cosine",
        "--batch-size", "0", "-o", str(tmp_path / "hits.tsv"), "--json", capsys=capsys,
    )
    assert exit_code == 1
    assert payload["error"] == "invalid_parameter"
    assert payload["parameter"] == "batch_size"


def test_search_tolerance_shorthand_conflict(tmp_path, capsys):
    exit_code, payload = run_cli(
        "similarity", "search", MGF_FILE, MGF_FILE, "--method", "Cosine",
        "--tolerance", "0.1", "--param", "tolerance=0.2",
        "-o", str(tmp_path / "hits.tsv"), "--json", capsys=capsys,
    )
    assert exit_code == 1
    assert payload["error"] == "invalid_parameter"
    assert payload["parameter"] == "tolerance"


def test_search_library_spectra_with_spectra_library(tmp_path, capsys):
    lib = write_mgf(tmp_path, SMALL_LIBRARY_MGF)
    exit_code, payload = run_cli(
        "similarity", "search", lib, lib, "--method", "Cosine",
        "--library-spectra", lib,
        "-o", str(tmp_path / "hits.tsv"), "--json", capsys=capsys,
    )
    assert exit_code == 1
    assert payload["error"] == "invalid_parameter"
    assert payload["parameter"] == "library_spectra"


def test_search_library_id_field_without_library_spectra(tmp_path, capsys):
    """An index library needs --library-spectra to resolve library ids."""
    lib = write_mgf(tmp_path, SMALL_LIBRARY_MGF)
    index = tmp_path / "lib.index.npz"
    exit_code, _ = run_cli(
        "similarity", "build-index", lib, "--method", "Cosine",
        "-o", str(index), "--json", capsys=capsys,
    )
    assert exit_code == 0

    exit_code, payload = run_cli(
        "similarity", "search", lib, str(index), "--method", "Cosine",
        "--library-id-field", "compound_name",
        "-o", str(tmp_path / "hits.tsv"), "--json", capsys=capsys,
    )
    assert exit_code == 1
    assert payload["error"] == "invalid_parameter"
    assert payload["parameter"] == "library_id_field"


def test_search_library_spectra_wrong_count(tmp_path, capsys):
    lib = write_mgf_with_ids(tmp_path, LIBRARY_SPECS, "L", name="library.mgf")
    index = tmp_path / "lib.index.npz"
    exit_code, _ = run_cli(
        "similarity", "build-index", lib, "--method", "Cosine",
        "-o", str(index), "--json", capsys=capsys,
    )
    assert exit_code == 0

    two = write_mgf_with_ids(tmp_path, LIBRARY_SPECS[:2], "L", name="two.mgf")
    exit_code, payload = run_cli(
        "similarity", "search", lib, str(index), "--method", "Cosine",
        "--library-id-field", "compound_name", "--library-spectra", two,
        "-o", str(tmp_path / "hits.tsv"), "--json", capsys=capsys,
    )
    assert exit_code == 1
    assert payload["error"] == "invalid_input"
    assert payload["input_file"] == two


def test_search_index_incompatible_preprocessing(tmp_path, capsys):
    """An index built with different preprocessing settings is rejected."""
    lib = write_mgf(tmp_path, SMALL_LIBRARY_MGF, name="library.mgf")
    index = tmp_path / "lib.index.npz"
    exit_code, _ = run_cli(
        "similarity", "build-index", lib, "--method", "Cosine",
        "--param", "remove_precursor=false",
        "-o", str(index), "--json", capsys=capsys,
    )
    assert exit_code == 0

    # Search with the default remove_precursor=true: the index no longer matches.
    exit_code, payload = run_cli(
        "similarity", "search", lib, str(index), "--method", "Cosine",
        "-o", str(tmp_path / "hits.tsv"), "--json", capsys=capsys,
    )
    assert exit_code == 1
    assert payload["error"] == "index_incompatible"


def test_search_index_incompatible_same_params_succeeds(tmp_path, capsys):
    lib = write_mgf(tmp_path, SMALL_LIBRARY_MGF, name="library.mgf")
    index = tmp_path / "lib.index.npz"
    exit_code, _ = run_cli(
        "similarity", "build-index", lib, "--method", "Cosine",
        "--param", "remove_precursor=false",
        "-o", str(index), "--json", capsys=capsys,
    )
    assert exit_code == 0

    exit_code, payload = run_cli(
        "similarity", "search", lib, str(index), "--method", "Cosine",
        "--param", "remove_precursor=false",
        "-o", str(tmp_path / "hits.tsv"), "--json", capsys=capsys,
    )
    assert exit_code == 0
    assert payload["results"]["n_hits"] > 0


def test_search_missing_index_file(tmp_path, capsys):
    exit_code, payload = run_cli(
        "similarity", "search", MGF_FILE, str(tmp_path / "absent.index.npz"),
        "--method", "Cosine", "-o", str(tmp_path / "hits.tsv"), "--json", capsys=capsys,
    )
    assert exit_code == 1
    assert payload["error"] == "input_not_found"


def test_search_corrupt_index_file(tmp_path, capsys):
    lib = write_mgf(tmp_path, SMALL_LIBRARY_MGF)
    bad_index = tmp_path / "bad.index.npz"
    bad_index.write_bytes(b"this is not an npz archive")
    exit_code, payload = run_cli(
        "similarity", "search", lib, str(bad_index), "--method", "Cosine",
        "-o", str(tmp_path / "hits.tsv"), "--json", capsys=capsys,
    )
    assert exit_code == 1
    assert payload["error"] in {"compute_error", "index_incompatible"}


def test_search_empty_queries(tmp_path, capsys):
    """A query file with no spectra is rejected, symmetric with the library."""
    empty = write_mgf(tmp_path, "# a query file with no spectra\n", name="empty.mgf")
    exit_code, payload = run_cli(
        "similarity", "search", empty, MGF_FILE, "--method", "Cosine",
        "-o", str(tmp_path / "hits.tsv"), "--json", capsys=capsys,
    )
    assert exit_code == 1
    assert payload["error"] == "empty_spectra"
    assert payload["input_file"] == empty


def test_search_empty_library(tmp_path, capsys):
    """A library file with no spectra cannot be indexed and is rejected."""
    empty = write_mgf(tmp_path, "# a library with no spectra\n", name="empty.mgf")
    exit_code, payload = run_cli(
        "similarity", "search", MGF_FILE, empty, "--method", "Cosine",
        "-o", str(tmp_path / "hits.tsv"), "--json", capsys=capsys,
    )
    assert exit_code == 1
    assert payload["error"] == "empty_spectra"


def test_search_unknown_query_id_field(tmp_path, capsys):
    exit_code, payload = run_cli(
        "similarity", "search", MGF_FILE, MGF_FILE, "--method", "Cosine",
        "--query-id-field", "not_a_column",
        "-o", str(tmp_path / "hits.tsv"), "--json", capsys=capsys,
    )
    assert exit_code == 1
    assert payload["error"] == "invalid_parameter"
    assert payload["parameter"] == "query_id_field"


def test_search_output_directory_must_exist(tmp_path, capsys):
    missing_dir = tmp_path / "no_such_dir" / "hits.tsv"
    exit_code, payload = run_cli(
        "similarity", "search", MGF_FILE, MGF_FILE, "--method", "Cosine",
        "-o", str(missing_dir), "--json", capsys=capsys,
    )
    assert exit_code == 1
    assert payload["error"] == "save_failed"
    assert payload["input_file"] == str(missing_dir)
