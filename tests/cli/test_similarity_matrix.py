import json
import os
import pandas as pd
import pytest
from matchms.cli.commands import similarity_matrix
from matchms.cli.main import build_parser, main
from matchms.exporting import save_as_mgf
from matchms.importing.load_spectra import SUPPORTED_FILE_FORMATS as INPUT_FORMATS
from matchms.scores import Scores
from matchms.similarity import __all__ as SIMILARITY_NAMES


TEST_DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "testdata"))
MGF_FILE = os.path.join(TEST_DATA, "testdata.mgf")
MSP_FILE = os.path.join(TEST_DATA, "Hydrogen_chloride.msp")
FINGERPRINT_JSON = os.path.join(TEST_DATA, "test_remove_fingerprint.json")

OUTPUT_FORMATS = ("csv", "npz", "tsv")

# Sparse-capable methods implement sparse_matrix(); the rest are dense-only.
SPARSE_METHODS = (
    "CosineGreedy",
    "CosineHungarian",
    "EntropyGreedy",
    "MetadataMatch",
    "ModifiedCosineGreedy",
    "ModifiedCosineHungarian",
    "NeutralLossesCosine",
    "ParentMassMatch",
    "PrecursorMzMatch",
)


def run_cli(*argv, capsys):
    """Run the CLI and return (exit_code, parsed_json_stdout)."""
    exit_code = main(list(argv))
    out = capsys.readouterr().out
    return exit_code, json.loads(out)


# ---------------------------------------------------------------------------
# Success cases
# ---------------------------------------------------------------------------


def test_matrix_dense_npz_success(tmp_path, capsys):
    out = tmp_path / "m.npz"
    exit_code, payload = run_cli(
        "similarity", "matrix", MGF_FILE, "--method", "CosineGreedy",
        "-o", str(out), "--no-progress", "--json", capsys=capsys,
    )

    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["operation"] == "similarity matrix"
    assert payload["method"]["name"] == "CosineGreedy"
    assert payload["method"]["class"] == "CosineGreedy"
    assert payload["mode"] == "dense"

    inputs = payload["inputs"]
    assert inputs["symmetric"] is True
    assert inputs["spectra_1"]["file"] == MGF_FILE
    assert inputs["spectra_1"]["n_spectra"] == 30
    assert inputs["spectra_2"] is None

    scores = payload["scores"]
    assert scores["shape"] == [30, 30]
    assert scores["kind"] == "dense"
    assert "score" in scores["score_fields"]
    assert scores["n_stored"] == 30 * 30
    assert scores["density"] == pytest.approx(1.0)
    for field in scores["score_fields"]:
        stat = scores["stats"][field]
        assert stat["count"] == 30 * 30
        assert stat["min"] <= stat["mean"] <= stat["max"]

    assert isinstance(payload["elapsed_seconds"], float)
    out_info = payload["output"]
    assert out_info["file"] == str(out)
    assert out_info["format"] == "npz"
    assert out_info["size_bytes"] > 0
    assert out.exists()


def test_matrix_npz_roundtrip(tmp_path, capsys):
    out = tmp_path / "m.npz"
    exit_code, payload = run_cli(
        "similarity", "matrix", MGF_FILE, "--method", "CosineGreedy",
        "-o", str(out), "--no-progress", "--json", capsys=capsys,
    )

    assert exit_code == 0
    reloaded = Scores.load(str(out))
    assert reloaded.shape == tuple(payload["scores"]["shape"])
    assert reloaded.is_sparse is (payload["scores"]["kind"] == "sparse")
    assert list(reloaded.score_fields) == payload["scores"]["score_fields"]


def test_matrix_two_file_nonsymmetric(tmp_path, capsys):
    out = tmp_path / "m.npz"
    exit_code, payload = run_cli(
        "similarity", "matrix", MSP_FILE, MGF_FILE, "--method", "PrecursorMzMatch",
        "-o", str(out), "--no-progress", "--json", capsys=capsys,
    )

    assert exit_code == 0
    inputs = payload["inputs"]
    assert inputs["symmetric"] is False
    assert inputs["spectra_2"] is not None
    assert inputs["spectra_2"]["n_spectra"] == 30
    # rows = spectra_1 (1 spectrum), columns = spectra_2 (30 spectra)
    assert payload["scores"]["shape"] == [1, 30]


def test_matrix_method_is_case_insensitive(tmp_path, capsys):
    out = tmp_path / "m.npz"
    exit_code, payload = run_cli(
        "similarity", "matrix", MGF_FILE, "--method", "cosinegreedy",
        "-o", str(out), "--no-progress", "--json", capsys=capsys,
    )

    assert exit_code == 0
    # Resolved to the canonical class name.
    assert payload["method"]["name"] == "CosineGreedy"


@pytest.mark.parametrize("name", SIMILARITY_NAMES)
def test_every_listed_method_resolves(name, tmp_path, capsys):
    """--method must accept every name exposed by `matchms similarity list`."""
    from matchms.cli.commands.similarity_matrix import _resolve_method

    resolved, cls = _resolve_method(name, "similarity matrix")
    assert resolved == name
    assert cls.__name__ == name


def test_matrix_sparse_mode(tmp_path, capsys):
    out = tmp_path / "m.npz"
    exit_code, payload = run_cli(
        "similarity", "matrix", MGF_FILE, "--method", "CosineGreedy",
        "--mode", "sparse", "--score-min", "0.9",
        "-o", str(out), "--no-progress", "--json", capsys=capsys,
    )

    assert exit_code == 0
    scores = payload["scores"]
    assert scores["kind"] == "sparse"
    assert scores["n_stored"] < 30 * 30
    assert scores["density"] < 1.0


def test_matrix_sparse_score_min_keeps_only_high_pairs(tmp_path, capsys):
    out = tmp_path / "m.npz"
    exit_code, payload = run_cli(
        "similarity", "matrix", MGF_FILE, "--method", "CosineGreedy",
        "--mode", "sparse", "--score-min", "0.95",
        "-o", str(out), "--no-progress", "--json", capsys=capsys,
    )

    assert exit_code == 0
    # With a strict threshold every reported top pair must be >= 0.95.
    for pair in payload["top_pairs"]:
        assert pair["value"] >= 0.95


def test_matrix_top_pairs_exclude_diagonal_when_symmetric(tmp_path, capsys):
    out = tmp_path / "m.npz"
    exit_code, payload = run_cli(
        "similarity", "matrix", MGF_FILE, "--method", "CosineGreedy",
        "-o", str(out), "--no-progress", "--top", "20", "--json", capsys=capsys,
    )

    assert exit_code == 0
    assert payload["inputs"]["symmetric"] is True
    for pair in payload["top_pairs"]:
        assert pair["row"] != pair["col"]


def test_matrix_top_zero_returns_no_pairs(tmp_path, capsys):
    out = tmp_path / "m.npz"
    exit_code, payload = run_cli(
        "similarity", "matrix", MGF_FILE, "--method", "CosineGreedy",
        "-o", str(out), "--no-progress", "--top", "0", "--json", capsys=capsys,
    )

    assert exit_code == 0
    assert payload["top_pairs"] == []


def test_matrix_same_file_twice_is_nonsymmetric(tmp_path, capfd):
    out = tmp_path / "m.npz"
    exit_code = main([
        "similarity", "matrix", MGF_FILE, MGF_FILE, "--method", "PrecursorMzMatch",
        "-o", str(out), "--no-progress", "--json",
    ])
    captured = capfd.readouterr()
    payload = json.loads(captured.out)

    assert exit_code == 0
    assert payload["inputs"]["symmetric"] is False
    assert payload["scores"]["shape"] == [30, 30]
    # A warning about the repeated file is emitted on stderr.
    assert "same file" in captured.err


def test_matrix_tsv_with_id_field(tmp_path, capsys):
    out = tmp_path / "m.tsv"
    exit_code = run_cli(
        "similarity", "matrix", MGF_FILE, "--method", "CosineGreedy",
        "--mode", "sparse", "--score-min", "0.9", "--id-field", "compound_name",
        "-o", str(out), "--no-progress", "--json", capsys=capsys,
    )[0]

    assert exit_code == 0
    df = pd.read_csv(str(out), sep="\t")
    assert list(df.columns) == ["row", "col", "row_id", "col_id", "score", "matches"]
    # Only the kept (score >= 0.9) pairs are written.
    assert df["score"].min() >= 0.9
    assert (df["row_id"] != "").all()


def test_matrix_csv_format(tmp_path, capsys):
    out = tmp_path / "m.csv"
    exit_code, payload = run_cli(
        "similarity", "matrix", MGF_FILE, "--method", "CosineGreedy",
        "-o", str(out), "--no-progress", "--json", capsys=capsys,
    )

    assert exit_code == 0
    assert payload["output"]["format"] == "csv"
    df = pd.read_csv(str(out), sep=",")
    assert "row" in df.columns and "col" in df.columns and "score" in df.columns
    assert len(df) == 30 * 30


def test_matrix_output_is_replaced(tmp_path, capsys):
    out = tmp_path / "m.npz"
    out.write_bytes(b"stale-bytes")

    exit_code, payload = run_cli(
        "similarity", "matrix", MGF_FILE, "--method", "CosineGreedy",
        "-o", str(out), "--no-progress", "--json", capsys=capsys,
    )

    assert exit_code == 0
    # The stale content is gone; a fresh artifact is written.
    assert payload["output"]["size_bytes"] > 0
    assert Scores.load(str(out)).shape == tuple(payload["scores"]["shape"])


def test_matrix_human_table_output(tmp_path, capsys):
    out = tmp_path / "m.npz"
    exit_code = main([
        "similarity", "matrix", MGF_FILE, "--method", "CosineGreedy",
        "-o", str(out), "--no-progress", "--table",
    ])
    text = capsys.readouterr().out

    assert exit_code == 0
    assert "Similarity matrix" in text
    assert "CosineGreedy" in text
    # Forced --table must not emit JSON.
    with pytest.raises(json.JSONDecodeError):
        json.loads(text)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_matrix_missing_input_file(tmp_path, capsys):
    missing = tmp_path / "nope.mgf"
    out = tmp_path / "m.npz"

    exit_code, payload = run_cli(
        "similarity", "matrix", str(missing), "--method", "CosineGreedy",
        "-o", str(out), "--json", capsys=capsys,
    )

    assert exit_code == 1
    assert payload["ok"] is False
    assert payload["error"] == "input_not_found"
    assert payload["input_file"] == str(missing)
    assert set(payload["valid_values"]) == set(INPUT_FORMATS)


def test_matrix_unsupported_input_extension(tmp_path, capsys):
    fake = tmp_path / "spectra.txt"
    fake.write_text("not a spectra file\n", encoding="utf-8")
    out = tmp_path / "m.npz"

    exit_code, payload = run_cli(
        "similarity", "matrix", str(fake), "--method", "CosineGreedy",
        "-o", str(out), "--json", capsys=capsys,
    )

    assert exit_code == 1
    assert payload["error"] == "unsupported_format"
    assert payload["input_file"] == str(fake)
    assert set(payload["valid_values"]) == set(INPUT_FORMATS)


def test_matrix_unsupported_output_extension(tmp_path, capsys):
    out = tmp_path / "m.txt"

    exit_code, payload = run_cli(
        "similarity", "matrix", MGF_FILE, "--method", "CosineGreedy",
        "-o", str(out), "--json", capsys=capsys,
    )

    assert exit_code == 1
    assert payload["error"] == "unsupported_format"
    assert set(payload["valid_values"]) == set(OUTPUT_FORMATS)


def test_matrix_unknown_method(tmp_path, capsys):
    out = tmp_path / "m.npz"

    exit_code, payload = run_cli(
        "similarity", "matrix", MGF_FILE, "--method", "NotARealMethod",
        "-o", str(out), "--json", capsys=capsys,
    )

    assert exit_code == 1
    assert payload["error"] == "unknown_value"
    assert set(payload["valid_values"]) == set(SIMILARITY_NAMES)


def test_matrix_method_typo_suggests_closest(tmp_path, capsys):
    out = tmp_path / "m.npz"

    exit_code, payload = run_cli(
        "similarity", "matrix", MGF_FILE, "--method", "CosineGredy",
        "-o", str(out), "--json", capsys=capsys,
    )

    assert exit_code == 1
    assert payload["error"] == "unknown_value"
    assert "Did you mean 'CosineGreedy'?" in payload["hint"]


def test_matrix_sparse_on_nonsparse_method(tmp_path, capsys):
    out = tmp_path / "m.npz"

    exit_code, payload = run_cli(
        "similarity", "matrix", MGF_FILE, "--method", "Cosine",
        "--mode", "sparse", "-o", str(out), "--json", capsys=capsys,
    )

    assert exit_code == 1
    assert payload["error"] == "unsupported_method"
    assert "Cosine" not in payload["valid_values"]
    for name in SPARSE_METHODS:
        assert name in payload["valid_values"]


@pytest.mark.parametrize("dense_name", ["Cosine", "Entropy", "ModifiedCosine", "FingerprintSimilarity"])
def test_matrix_sparse_rejects_each_dense_only_method(tmp_path, capsys, dense_name):
    out = tmp_path / "m.npz"

    exit_code, payload = run_cli(
        "similarity", "matrix", MGF_FILE, "--method", dense_name,
        "--mode", "sparse", "-o", str(out), "--json", capsys=capsys,
    )

    assert exit_code == 1
    assert payload["error"] == "unsupported_method"


def test_matrix_score_min_in_dense_mode(tmp_path, capsys):
    out = tmp_path / "m.npz"

    exit_code, payload = run_cli(
        "similarity", "matrix", MGF_FILE, "--method", "CosineGreedy",
        "--score-min", "0.5", "-o", str(out), "--json", capsys=capsys,
    )

    assert exit_code == 1
    assert payload["error"] == "invalid_parameter"


def test_matrix_dense_too_large(tmp_path, capsys):
    out = tmp_path / "m.npz"

    exit_code, payload = run_cli(
        "similarity", "matrix", MGF_FILE, "--method", "CosineGreedy",
        "--max-dense-entries", "100", "-o", str(out), "--json", capsys=capsys,
    )

    assert exit_code == 1
    assert payload["error"] == "matrix_too_large"


def test_matrix_dense_tsv_too_large(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(similarity_matrix, "DENSE_TSV_MAX_ENTRIES", 100)
    out = tmp_path / "m.tsv"

    exit_code, payload = run_cli(
        "similarity", "matrix", MGF_FILE, "--method", "CosineGreedy",
        "-o", str(out), "--no-progress", "--json", capsys=capsys,
    )

    assert exit_code == 1
    assert payload["error"] == "matrix_too_large"


def test_matrix_tolerance_conflicts_with_param(tmp_path, capsys):
    out = tmp_path / "m.npz"

    exit_code, payload = run_cli(
        "similarity", "matrix", MGF_FILE, "--method", "CosineGreedy",
        "--tolerance", "0.1", "--param", "tolerance=0.2", "-o", str(out), "--json", capsys=capsys,
    )

    assert exit_code == 1
    assert payload["error"] == "invalid_parameter"


def test_matrix_unknown_parameter_name(tmp_path, capsys):
    out = tmp_path / "m.npz"

    exit_code, payload = run_cli(
        "similarity", "matrix", MGF_FILE, "--method", "CosineGreedy",
        "--param", "not_a_real_param=1", "-o", str(out), "--json", capsys=capsys,
    )

    assert exit_code == 1
    assert payload["error"] == "invalid_parameter"
    assert payload["parameter"] == "not_a_real_param"


def test_matrix_metadata_match_requires_field(tmp_path, capsys):
    out = tmp_path / "m.npz"

    exit_code, payload = run_cli(
        "similarity", "matrix", MGF_FILE, "--method", "MetadataMatch",
        "-o", str(out), "--json", capsys=capsys,
    )

    assert exit_code == 1
    assert payload["error"] == "invalid_parameter"
    assert payload["parameter"] == "field"


def test_matrix_entropy_search_rejects_ppm(tmp_path, capsys):
    out = tmp_path / "m.npz"

    exit_code, payload = run_cli(
        "similarity", "matrix", MGF_FILE, "--method", "EntropySearch",
        "--param", "use_ppm=true", "-o", str(out), "--json", capsys=capsys,
    )

    assert exit_code == 1
    assert payload["error"] == "invalid_parameter"
    assert payload["parameter"] == "use_ppm"


def test_matrix_entropy_search_rejects_non_fragment_mode(tmp_path, capsys):
    out = tmp_path / "m.npz"

    exit_code, payload = run_cli(
        "similarity", "matrix", MGF_FILE, "--method", "EntropySearch",
        "--param", "matching_mode=precursor", "-o", str(out), "--json", capsys=capsys,
    )

    assert exit_code == 1
    assert payload["error"] == "invalid_parameter"
    assert payload["parameter"] == "matching_mode"


def test_matrix_fingerprint_without_structure(tmp_path, capsys):
    out = tmp_path / "m.npz"

    exit_code, payload = run_cli(
        "similarity", "matrix", MGF_FILE, "--method", "FingerprintSimilarity",
        "-o", str(out), "--json", capsys=capsys,
    )

    assert exit_code == 1
    assert payload["error"] == "invalid_parameter"
    assert payload["parameter"] == "inchikey"


def test_matrix_fingerprint_requires_generator(tmp_path, capsys):
    out = tmp_path / "m.npz"

    exit_code, payload = run_cli(
        "similarity", "matrix", FINGERPRINT_JSON, "--method", "FingerprintSimilarity",
        "-o", str(out), "--json", capsys=capsys,
    )

    assert exit_code == 1
    assert payload["error"] == "invalid_parameter"
    assert payload["parameter"] == "fingerprint_generator"


def test_matrix_empty_collection(tmp_path, capsys):
    empty = tmp_path / "empty.mgf"
    save_as_mgf([], str(empty), "matchms", file_mode="w")
    out = tmp_path / "m.npz"

    exit_code, payload = run_cli(
        "similarity", "matrix", str(empty), "--method", "CosineGreedy",
        "-o", str(out), "--json", capsys=capsys,
    )

    assert exit_code == 1
    assert payload["error"] == "empty_spectra"
    assert payload["input_file"] == str(empty)


def test_matrix_requires_method(tmp_path):
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["similarity", "matrix", MGF_FILE, "-o", str(tmp_path / "m.npz")])


def test_matrix_requires_output(tmp_path):
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["similarity", "matrix", MGF_FILE, "--method", "CosineGreedy"])


def test_matrix_registered_in_info(capsys):
    exit_code = main(["info", "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    commands = payload["cli"]["commands"]
    assert "similarity matrix" in commands
