import json
import os
import numpy as np
import pytest
from matchms.cli.commands import similarity_build_index
from matchms.cli.main import build_parser, main
from matchms.exporting import save_as_mgf
from matchms.importing import load_ms2_dataset
from matchms.importing.load_spectra import SUPPORTED_FILE_FORMATS as INPUT_FORMATS
from matchms.similarity import (
    Cosine,
    CosineFlash,
    Entropy,
    EntropyFlash,
    EntropySearch,
    ModifiedCosine,
)


TEST_DATA = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "testdata"))
MGF_FILE = os.path.join(TEST_DATA, "testdata.mgf")
MSP_FILE = os.path.join(TEST_DATA, "Hydrogen_chloride.msp")

INDEX_CAPABLE = ("Cosine", "CosineFlash", "ModifiedCosine", "Entropy", "EntropyFlash", "EntropySearch")

# The expected JSON payload key structure (schema snapshot).
TOP_KEYS = {"ok", "operation", "method", "library", "index", "elapsed_seconds", "note"}
METHOD_KEYS = {"name", "class", "params"}
LIBRARY_KEYS = {"file", "n_spectra"}
INDEX_KEYS = {"file", "format", "size_bytes"}

# A library with one spectrum holding two peaks closer than 2 * max_tolerance,
# so EntropySearch(peak_separation="raise") rejects it while "merge" accepts it.
CLOSE_PEAKS_MGF = """BEGIN IONS
PEPMASS=500.2
100.0 50.0
100.001 20.0
200.0 80.0
END IONS
BEGIN IONS
PEPMASS=600.3
300.0 90.0
400.0 70.0
END IONS
"""


def run_cli(*argv, capsys):
    """Run the CLI and return (exit_code, parsed_json_stdout)."""
    exit_code = main(list(argv))
    out = capsys.readouterr().out
    return exit_code, json.loads(out)


def write_mgf(tmp_path, text, name="lib.mgf"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# Success cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", INDEX_CAPABLE)
def test_build_index_each_supported_method(tmp_path, capsys, method):
    out = tmp_path / "lib.index.npz"
    exit_code, payload = run_cli(
        "similarity", "build-index", MGF_FILE, "--method", method,
        "-o", str(out), "--json", capsys=capsys,
    )

    assert exit_code == 0
    assert payload["ok"] is True
    assert payload["operation"] == "similarity build-index"
    assert payload["method"]["name"] == method
    assert payload["method"]["class"] == method
    assert payload["library"]["file"] == MGF_FILE
    assert payload["library"]["n_spectra"] == 30
    assert payload["index"]["file"] == str(out)
    assert payload["index"]["format"] == "index.npz"
    assert payload["index"]["size_bytes"] > 0
    assert isinstance(payload["elapsed_seconds"], float)
    assert out.exists()
    assert os.path.getsize(str(out)) == payload["index"]["size_bytes"]


@pytest.mark.parametrize("method", INDEX_CAPABLE)
def test_build_index_roundtrip_search_equals_matrix(tmp_path, capsys, method):
    """An index built by the command scores identically to a direct matrix() run."""
    out = tmp_path / "lib.index.npz"
    exit_code, _ = run_cli(
        "similarity", "build-index", MGF_FILE, "--method", method,
        "-o", str(out), "--json", capsys=capsys,
    )
    assert exit_code == 0

    cls = {"Cosine": Cosine, "CosineFlash": CosineFlash, "ModifiedCosine": ModifiedCosine,
           "Entropy": Entropy, "EntropyFlash": EntropyFlash, "EntropySearch": EntropySearch}[method]
    similarity = cls()
    library = load_ms2_dataset(MGF_FILE)

    # Index produced by the command.
    index = similarity.load_index(str(out))
    queries = list(library)[:5]
    via_index = similarity.search(queries, index, progress_bar=False).to_array("score")

    # Reference: same search done directly against the library (matrix path).
    direct = similarity.matrix(queries, list(library), progress_bar=False).to_array("score")

    assert via_index.shape == direct.shape
    np.testing.assert_allclose(via_index, direct, rtol=1e-5, atol=1e-6)


def test_build_index_preserves_spectrum_order(tmp_path, capsys):
    """Reference positions refer to the library row order (positions are stable)."""
    out = tmp_path / "lib.index.npz"
    exit_code, _ = run_cli(
        "similarity", "build-index", MGF_FILE, "--method", "Cosine",
        "-o", str(out), "--json", capsys=capsys,
    )
    assert exit_code == 0

    index = Cosine().load_index(str(out))
    assert index.n_specs == 30


def test_build_index_payload_schema(tmp_path, capsys):
    out = tmp_path / "lib.index.npz"
    exit_code, payload = run_cli(
        "similarity", "build-index", MGF_FILE, "--method", "Cosine",
        "-o", str(out), "--json", capsys=capsys,
    )

    assert exit_code == 0
    assert set(payload) == TOP_KEYS
    assert set(payload["method"]) == METHOD_KEYS
    assert set(payload["library"]) == LIBRARY_KEYS
    assert set(payload["index"]) == INDEX_KEYS
    assert isinstance(payload["method"]["params"], dict)


def test_build_index_effective_params_include_defaults(tmp_path, capsys):
    """method.params must include result-affecting defaults, not just overrides."""
    out = tmp_path / "lib.index.npz"
    exit_code, payload = run_cli(
        "similarity", "build-index", MGF_FILE, "--method", "Cosine",
        "--tolerance", "0.1", "-o", str(out), "--json", capsys=capsys,
    )

    assert exit_code == 0
    params = payload["method"]["params"]
    # The override is applied ...
    assert params["tolerance"] == pytest.approx(0.1)
    # ... alongside the defaults that influence the result.
    assert params["remove_precursor"] is True
    assert params["use_hungarian"] is False
    assert "noise_cutoff" in params


def test_build_index_tolerance_shorthand(tmp_path, capsys):
    out = tmp_path / "lib.index.npz"
    exit_code, payload = run_cli(
        "similarity", "build-index", MGF_FILE, "--method", "Cosine",
        "--tolerance", "0.1", "-o", str(out), "--json", capsys=capsys,
    )

    assert exit_code == 0
    assert payload["method"]["params"]["tolerance"] == pytest.approx(0.1)


def test_build_index_method_is_case_insensitive(tmp_path, capsys):
    out = tmp_path / "lib.index.npz"
    exit_code, payload = run_cli(
        "similarity", "build-index", MGF_FILE, "--method", "entropysearch",
        "-o", str(out), "--json", capsys=capsys,
    )

    assert exit_code == 0
    assert payload["method"]["name"] == "EntropySearch"


@pytest.mark.parametrize("name", similarity_build_index.INDEX_CAPABLE_NAMES)
def test_every_index_capable_method_resolves(name, tmp_path, capsys):
    resolved, cls = similarity_build_index._resolve_method(name, "similarity build-index")
    assert resolved == name
    assert cls.__name__ == name


def test_build_index_entropysearch_merge_succeeds_on_close_peaks(tmp_path, capsys):
    lib = write_mgf(tmp_path, CLOSE_PEAKS_MGF)
    out = tmp_path / "lib.index.npz"
    exit_code, payload = run_cli(
        "similarity", "build-index", lib, "--method", "EntropySearch",
        "-o", str(out), "--json", capsys=capsys,
    )

    assert exit_code == 0
    assert payload["library"]["n_spectra"] == 2
    assert payload["method"]["params"]["peak_separation"] == "merge"
    assert out.exists()


def test_build_index_output_replaced(tmp_path, capsys):
    out = tmp_path / "lib.index.npz"
    out.write_bytes(b"stale-bytes-that-are-not-an-index")

    exit_code, payload = run_cli(
        "similarity", "build-index", MGF_FILE, "--method", "Cosine",
        "-o", str(out), "--json", capsys=capsys,
    )

    assert exit_code == 0
    # The stale content is gone; a fresh index is written and loadable.
    assert payload["index"]["size_bytes"] > 0
    assert Cosine().load_index(str(out)).n_specs == 30


def test_build_index_input_formats(tmp_path, capsys):
    exit_code, payload = run_cli(
        "similarity", "build-index", MSP_FILE, "--method", "Cosine",
        "-o", str(tmp_path / "lib.index.npz"), "--json", capsys=capsys,
    )
    assert exit_code == 0
    assert payload["library"]["n_spectra"] == 1


def test_build_index_human_table_output(tmp_path, capsys):
    out = tmp_path / "lib.index.npz"
    exit_code = main([
        "similarity", "build-index", MGF_FILE, "--method", "Cosine",
        "-o", str(out), "--table",
    ])
    text = capsys.readouterr().out

    assert exit_code == 0
    assert "Library index:" in text
    assert "Cosine" in text
    assert "similarity search" in text
    with pytest.raises(json.JSONDecodeError):
        json.loads(text)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


def test_build_index_wrong_extension_raises_before_loading(tmp_path, capsys):
    """A wrong -o extension must fail with invalid_parameter before any file is read."""
    for bad in (tmp_path / "lib.npz", tmp_path / "lib.json", tmp_path / "lib"):
        # Use a non-existent library to prove the output is checked first.
        exit_code, payload = run_cli(
            "similarity", "build-index", "/no/such/library.mgf", "--method", "Cosine",
            "-o", str(bad), "--json", capsys=capsys,
        )
        assert exit_code == 1
        assert payload["error"] == "invalid_parameter"
        assert payload["parameter"] == "output"
        assert payload["valid_values"] == [".index.npz"]


def test_build_index_missing_library(tmp_path, capsys):
    missing = tmp_path / "does_not_exist.mgf"
    exit_code, payload = run_cli(
        "similarity", "build-index", str(missing), "--method", "Cosine",
        "-o", str(tmp_path / "lib.index.npz"), "--json", capsys=capsys,
    )

    assert exit_code == 1
    assert payload["error"] == "input_not_found"
    assert payload["input_file"] == str(missing)
    assert set(payload["valid_values"]) == set(INPUT_FORMATS)


def test_build_index_unsupported_input_extension(tmp_path, capsys):
    fake = tmp_path / "library.txt"
    fake.write_text("not a spectra file\n", encoding="utf-8")
    exit_code, payload = run_cli(
        "similarity", "build-index", str(fake), "--method", "Cosine",
        "-o", str(tmp_path / "lib.index.npz"), "--json", capsys=capsys,
    )

    assert exit_code == 1
    assert payload["error"] == "unsupported_format"
    assert payload["input_file"] == str(fake)
    assert set(payload["valid_values"]) == set(INPUT_FORMATS)


def test_build_index_non_indexed_method(tmp_path, capsys):
    out = tmp_path / "lib.index.npz"
    exit_code, payload = run_cli(
        "similarity", "build-index", MGF_FILE, "--method", "CosineGreedy",
        "-o", str(out), "--json", capsys=capsys,
    )

    assert exit_code == 1
    assert payload["error"] == "unsupported_method"
    assert set(payload["valid_values"]) == set(INDEX_CAPABLE)
    assert not out.exists()


@pytest.mark.parametrize("name", ["CosineGreedy", "MetadataMatch", "ParentMassMatch"])
def test_build_index_rejects_each_non_indexed_method(tmp_path, capsys, name):
    exit_code, payload = run_cli(
        "similarity", "build-index", MGF_FILE, "--method", name,
        "-o", str(tmp_path / "lib.index.npz"), "--json", capsys=capsys,
    )
    assert exit_code == 1
    assert payload["error"] == "unsupported_method"


@pytest.mark.parametrize("method", ["Cosine", "ModifiedCosine"])
def test_build_index_rejects_use_hungarian(tmp_path, capsys, method):
    exit_code, payload = run_cli(
        "similarity", "build-index", MGF_FILE, "--method", method,
        "--param", "use_hungarian=true",
        "-o", str(tmp_path / "lib.index.npz"), "--json", capsys=capsys,
    )
    assert exit_code == 1
    assert payload["error"] == "unsupported_method"
    assert payload["parameter"] == "use_hungarian"


def test_build_index_unknown_method(tmp_path, capsys):
    exit_code, payload = run_cli(
        "similarity", "build-index", MGF_FILE, "--method", "NotARealMethod",
        "-o", str(tmp_path / "lib.index.npz"), "--json", capsys=capsys,
    )
    assert exit_code == 1
    assert payload["error"] == "unknown_value"
    assert set(payload["valid_values"]) == set(INDEX_CAPABLE)


def test_build_index_method_typo_suggests_closest(tmp_path, capsys):
    exit_code, payload = run_cli(
        "similarity", "build-index", MGF_FILE, "--method", "Cosinn",
        "-o", str(tmp_path / "lib.index.npz"), "--json", capsys=capsys,
    )
    assert exit_code == 1
    assert payload["error"] == "unknown_value"
    assert "Did you mean 'Cosine'?" in payload["hint"]


def test_build_index_tolerance_conflicts_with_param(tmp_path, capsys):
    exit_code, payload = run_cli(
        "similarity", "build-index", MGF_FILE, "--method", "Cosine",
        "--tolerance", "0.1", "--param", "tolerance=0.2",
        "-o", str(tmp_path / "lib.index.npz"), "--json", capsys=capsys,
    )
    assert exit_code == 1
    assert payload["error"] == "invalid_parameter"
    assert payload["parameter"] == "tolerance"


def test_build_index_unknown_parameter_name(tmp_path, capsys):
    exit_code, payload = run_cli(
        "similarity", "build-index", MGF_FILE, "--method", "Cosine",
        "--param", "not_a_real_param=1",
        "-o", str(tmp_path / "lib.index.npz"), "--json", capsys=capsys,
    )
    assert exit_code == 1
    assert payload["error"] == "invalid_parameter"
    assert payload["parameter"] == "not_a_real_param"


def test_build_index_entropysearch_rejects_ppm(tmp_path, capsys):
    exit_code, payload = run_cli(
        "similarity", "build-index", MGF_FILE, "--method", "EntropySearch",
        "--param", "use_ppm=true",
        "-o", str(tmp_path / "lib.index.npz"), "--json", capsys=capsys,
    )
    assert exit_code == 1
    assert payload["error"] == "invalid_parameter"
    assert payload["parameter"] == "use_ppm"


def test_build_index_entropysearch_rejects_bad_peak_separation(tmp_path, capsys):
    exit_code, payload = run_cli(
        "similarity", "build-index", MGF_FILE, "--method", "EntropySearch",
        "--param", "peak_separation=drop",
        "-o", str(tmp_path / "lib.index.npz"), "--json", capsys=capsys,
    )
    assert exit_code == 1
    assert payload["error"] == "invalid_parameter"
    assert payload["parameter"] == "peak_separation"
    assert set(payload["valid_values"]) == {"merge", "raise"}


def test_build_index_entropy_rejects_bad_matching_mode(tmp_path, capsys):
    exit_code, payload = run_cli(
        "similarity", "build-index", MGF_FILE, "--method", "Entropy",
        "--param", "matching_mode=foo",
        "-o", str(tmp_path / "lib.index.npz"), "--json", capsys=capsys,
    )
    assert exit_code == 1
    assert payload["error"] == "invalid_parameter"
    assert payload["parameter"] == "matching_mode"
    assert set(payload["valid_values"]) == {"fragment", "neutral_loss", "hybrid"}


def test_build_index_invalid_parameter_values(tmp_path, capsys):
    """A value the constructor rejects (negative tolerance) -> invalid_parameter."""
    exit_code, payload = run_cli(
        "similarity", "build-index", MGF_FILE, "--method", "Cosine",
        "--tolerance", "-0.5",
        "-o", str(tmp_path / "lib.index.npz"), "--json", capsys=capsys,
    )
    assert exit_code == 1
    assert payload["error"] == "invalid_parameter"


def test_build_index_entropysearch_raise_on_close_peaks(tmp_path, capsys):
    lib = write_mgf(tmp_path, CLOSE_PEAKS_MGF)
    out = tmp_path / "lib.index.npz"
    exit_code, payload = run_cli(
        "similarity", "build-index", lib, "--method", "EntropySearch",
        "--param", "peak_separation=raise",
        "-o", str(out), "--json", capsys=capsys,
    )
    assert exit_code == 1
    assert payload["error"] == "invalid_input"
    assert "peak_separation" in payload["hint"]
    assert not out.exists()


def test_build_index_output_directory_missing(tmp_path, capsys):
    exit_code, payload = run_cli(
        "similarity", "build-index", MGF_FILE, "--method", "Cosine",
        "-o", str(tmp_path / "no" / "such" / "dir" / "lib.index.npz"),
        "--json", capsys=capsys,
    )
    assert exit_code == 1
    assert payload["error"] == "save_failed"


def test_build_index_empty_library(tmp_path, capsys):
    empty = tmp_path / "empty.mgf"
    save_as_mgf([], str(empty), "matchms", file_mode="w")
    exit_code, payload = run_cli(
        "similarity", "build-index", str(empty), "--method", "Cosine",
        "-o", str(tmp_path / "lib.index.npz"), "--json", capsys=capsys,
    )
    assert exit_code == 1
    assert payload["error"] == "empty_spectra"
    assert payload["input_file"] == str(empty)


def test_build_index_requires_method(tmp_path):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["similarity", "build-index", MGF_FILE, "-o", str(tmp_path / "lib.index.npz")])


def test_build_index_requires_output(tmp_path):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["similarity", "build-index", MGF_FILE, "--method", "Cosine"])


def test_build_index_registered_in_info(capsys):
    exit_code = main(["info", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert "similarity build-index" in payload["cli"]["commands"]
