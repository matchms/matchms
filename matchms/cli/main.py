import argparse
import logging
from matchms import __version__ as matchms_version
from matchms.cli.commands.filter_info import run as run_filter_info
from matchms.cli.commands.filter_list import run as run_filter_list
from matchms.cli.commands.filter_pipelines import PIPELINES
from matchms.cli.commands.filter_pipelines import run as run_filter_pipelines
from matchms.cli.commands.filter_run import run as run_filter_run
from matchms.cli.commands.info import run as run_info
from matchms.cli.commands.similarity_build_index import run as run_similarity_build_index
from matchms.cli.commands.similarity_info import run as run_similarity_info
from matchms.cli.commands.similarity_list import run as run_similarity_list
from matchms.cli.commands.similarity_matrix import (
    DEFAULT_MAX_DENSE_ENTRIES,
)
from matchms.cli.commands.similarity_matrix import (
    OUTPUT_FORMATS as SIMILARITY_OUTPUT_FORMATS,
)
from matchms.cli.commands.similarity_matrix import run as run_similarity_matrix
from matchms.cli.commands.similarity_search import run as run_similarity_search
from matchms.cli.commands.spectra_convert import run as run_spectra_convert
from matchms.cli.commands.spectra_describe import run as run_spectra_describe
from matchms.cli.errors import CliError, configure_logging, emit_error, unexpected_error_payload
from matchms.cli.output import CLI_JSON_SCHEMA_VERSION, OutputContext
from matchms.exporting.save_spectra import EXPORT_STYLES


def _add_common_flags(parser: argparse.ArgumentParser, *, suppress: bool) -> None:
    """Add --json/--table/--quiet/--verbose to a parser.

    When *suppress* is True (subparsers), defaults are SUPPRESS so that flags
    given at the top level are not clobbered by subparser defaults.
    """
    default = argparse.SUPPRESS if suppress else None

    group = parser.add_argument_group(
        "output modes",
        "Machine mode (--json) prints stable JSON on stdout. Human mode (--table) "
        "prints readable text. Without either flag, JSON is used when stdout is "
        "not a terminal.",
    )
    group.add_argument(
        "--json",
        action="store_true",
        default=default,
        help="Force machine-readable JSON output on stdout.",
    )
    group.add_argument(
        "--table",
        action="store_true",
        default=default,
        help="Force human-readable tabular output on stdout.",
    )
    group.add_argument(
        "--quiet",
        action="store_true",
        default=False if not suppress else argparse.SUPPRESS,
        help="Suppress log output (only errors are reported).",
    )
    group.add_argument(
        "--verbose",
        action="store_true",
        default=False if not suppress else argparse.SUPPRESS,
        help="Enable debug-level logging (goes to stderr).",
    )


def _log_level(args) -> int:
    """Resolve the log level from --quiet/--verbose.

    ``--quiet`` silences everything except errors, ``--verbose`` drops to DEBUG,
    and the default keeps matchms warnings visible (on stderr).
    """
    if bool(getattr(args, "quiet", False)):
        return logging.ERROR
    if bool(getattr(args, "verbose", False)):
        return logging.DEBUG
    return logging.WARNING


def build_parser() -> argparse.ArgumentParser:
    """
    Build and return the main parser for the `matchms` command line interface.

    This function sets up the primary parser for handling the command-line arguments
    and subcommands used in the `matchms` CLI. The main parser includes general options
    (such as version printing) and further allows delegation to subcommands for
    specific tasks like retrieving information (`info`), working with filters (`filter`),
    or inspecting spectra files (`spectra`).

    Subcommands and their key actions include:
    - `info`: Report matchms/Python versions, supported I/O formats, filters, and the CLI schema version.
    - `filter`: List filters, inspect filtering pipelines, show filter details, or run filters on spectra files.
    - `similarity`: List the similarity measures in matchms.similarity, show one, compute a
      similarity matrix, build a reusable library index, or search a query file against a
      library (or a saved index).
    - `spectra`: Perform operations on spectra files like descriptive statistics or format conversion.

    Returns
    -------
    argparse.ArgumentParser
        The configured argparse.ArgumentParser object for use with the `matchms` CLI.
    """
    parser = argparse.ArgumentParser(
        prog="matchms",
        description=("matchms command line interface"),
        epilog=("Examples:\n  matchms info --json\n"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"matchms {matchms_version} (CLI JSON schema {CLI_JSON_SCHEMA_VERSION})",
        help="Print the matchms version and the CLI JSON schema version.",
    )
    _add_common_flags(parser, suppress=False)

    subparsers = parser.add_subparsers(dest="command", metavar="<command>")

    # common flags that also work after the subcommand
    common = argparse.ArgumentParser(add_help=False)
    _add_common_flags(common, suppress=True)

    # -- info  --------------------------------------------------------------
    p = subparsers.add_parser(
        "info",
        parents=[common],
        help="Report versions, supported formats and filters",
        description=(
            "Report the matchms/Python/CLI schema versions, supported input/output formats, "
            "all supported commands (spectra, filter, similarity) and corresponding subcommands."
        ),
    )
    p.set_defaults(func=run_info)

    # -- filter  -----------------------------------------------------------
    p = subparsers.add_parser(
        "filter",
        parents=[common],
        help="List, inspect and run matchms filters",
        description=(
            "Work with matchms filters. `filter list` shows all filters in their "
            "default execution order; `filter pipelines` shows all default filter "
            "pipelines; `filter info` shows the description and parameters of a "
            "single filter; `filter run` runs a filter pipeline on a "
            "SpectraCollection and writes the filtered spectra."
        ),
    )
    filter_subparsers = p.add_subparsers(dest="filter_command", metavar="<action>")

    # filter list ----------------------------------------------------------
    pf_list = filter_subparsers.add_parser(
        "list",
        parents=[common],
        help="List all available filters with their default order",
        description=(
            "List all available matchms filters in their default execution order "
            "as defined in matchms.filtering.filter_order.ALL_FILTERS. Shows the "
            "order, name and a short description for each filter."
        ),
    )
    pf_list.set_defaults(func=run_filter_list)

    # filter pipelines -------------------------------------------------------
    pf_pipelines = filter_subparsers.add_parser(
        "pipelines",
        parents=[common],
        help="List all available filter pipelines",
        description=(
            "List all filter pipelines defined in matchms.filtering.default_pipelines: their name, a short "
            "description and the number of filters they contain. The listed names are the valid values for "
            "--pipeline of `matchms filter run`; when --pipeline is omitted, DEFAULT_FILTERS is used. Pass a "
            "pipeline name to show the filters of that pipeline in execution order (with any pipeline-specific "
            "parameters)."
        ),
    )
    pf_pipelines.add_argument(
        "pipeline_name",
        nargs="?",
        default=None,
        metavar="NAME",
        help="Name of a pipeline to show its filters in execution order (see the overview when omitted).",
    )
    pf_pipelines.set_defaults(func=run_filter_pipelines)

    # filter info ----------------------------------------------------------
    pf_info = filter_subparsers.add_parser(
        "info",
        parents=[common],
        help="Show the description and parameters of one filter",
        description=(
            "Show detailed information about one matchms filter: where it sits in "
            "the default filter order, what it does (full docstring) and its "
            "user-settable parameters with type, default value and description."
        ),
    )
    pf_info.add_argument(
        "filter_name",
        help="Name of the filter to describe (see `matchms filter list`).",
    )
    pf_info.set_defaults(func=run_filter_info)

    # filter run -----------------------------------------------------------
    pf_run = filter_subparsers.add_parser(
        "run",
        parents=[common],
        help="Run a filter pipeline on a spectra file",
        description=(
            "Run a filter pipeline over a SpectraCollection using SpectraProcessor "
            "and write the filtered spectra to an output file. The base pipeline "
            "defaults to DEFAULT_FILTERS; select another default pipeline with "
            "--pipeline and add further filters with --filter (they are placed at "
            "the correct position of the matchms filter order). Parameters are "
            "passed with --param filter_name.param=value and must belong to a "
            "filter that is part of the pipeline. Add --report to also save a "
            "processing report next to the output file."
        ),
    )
    pf_run.add_argument(
        "input",
        help="Path to the input spectra file. Supported extensions: json, mgf, msp, mzml, mzxml, pickle.",
    )
    pf_run.add_argument(
        "output",
        help="Path to the output spectra file. Supported extensions: json, mgf, msp, pickle.",
    )
    pf_run.add_argument(
        "--ftype",
        default="auto",
        help="File type to use for import. Defaults to 'auto', which guesses the "
        "file type from the input file extension.",
    )
    pf_run.add_argument(
        "--pipeline",
        default=None,
        choices=sorted(PIPELINES),
        help="Default filter pipeline to use as the base pipeline (default: DEFAULT_FILTERS when omitted).",
    )
    pf_run.add_argument(
        "--filter",
        action="append",
        default=None,
        metavar="NAME",
        help="Filter name to add to the base pipeline; may be passed multiple "
        "times. Added filters are placed at the correct position of the "
        "canonical matchms filter order. See `matchms filter list` for names.",
    )
    pf_run.add_argument(
        "--param",
        action="append",
        default=None,
        metavar="FILTER.NAME=VALUE",
        help="Parameter for a filter in the pipeline, as filter_name.param=value; "
        "may be passed multiple times. The filter must be part of the pipeline "
        "(a --filter filter or one of the base pipeline filters). Examples: "
        "--param select_by_mz.mz_from=10.0 --param select_by_mz.mz_to=500.",
    )
    pf_run.add_argument(
        "--report",
        action="store_true",
        help="Save a processing report (per-filter input/output/removed/changed "
        "counts) to '<output>_processing_report.json' next to the output file.",
    )
    pf_run.add_argument(
        "--export-style",
        default="matchms",
        choices=EXPORT_STYLES,
        help="Metadata key style used for the export (default: %(default)s).",
    )
    pf_run.add_argument(
        "--append",
        action="store_true",
        help="Append to the output file instead of overwriting it. Only supported for .mgf and .msp output files.",
    )
    pf_run.set_defaults(func=run_filter_run)

    # -- similarity  --------------------------------------------------------
    p = subparsers.add_parser(
        "similarity",
        parents=[common],
        help="List, inspect, compute, index and search matchms similarity measures",
        description=(
            "Work with the similarity measures available in matchms.similarity. "
            "`similarity list` shows all similarity measures. "
            "`similarity info` shows the description, score fields, methods and "
            "constructor parameters of a single similarity; `similarity matrix` "
            "computes a similarity matrix between one or two spectra files; "
            "`similarity build-index` builds a reusable library index from a "
            "spectra file; `similarity search` searches a query file against a "
            "spectra library (or a saved index) and writes the best matches per "
            "query to a hit list."
        ),
    )
    similarity_subparsers = p.add_subparsers(dest="similarity_command", metavar="<action>")

    # similarity list --------------------------------------------------------
    ps_list = similarity_subparsers.add_parser(
        "list",
        parents=[common],
        help="List all available similarity measures",
        description=(
            "List all similarity classes exposed by matchms.similarity, grouped "
            "by 'Similarity measures' (Cosine, Modified cosine, Spectral entropy, "
            "...). Shows the group, name, a short description and the supported "
            "computation methods for each class. Unclassified similarity groups are "
            "listed under the 'Other' group."
        ),
    )
    ps_list.set_defaults(func=run_similarity_list)

    # similarity info --------------------------------------------------------
    ps_info = similarity_subparsers.add_parser(
        "info",
        parents=[common],
        help="Show the description and parameters of one similarity",
        description=(
            "Show detailed information about one similarity class: which similarity "
            "group it belongs to, what it does (full docstring), the score "
            "fields it produces, the computation methods it supports and its "
            "constructor parameters with type, default value and description."
        ),
    )
    ps_info.add_argument(
        "similarity_name",
        help="Name of the similarity class to describe (see `matchms similarity list`).",
    )
    ps_info.set_defaults(func=run_similarity_info)

    # similarity matrix -----------------------------------------------------
    ps_matrix = similarity_subparsers.add_parser(
        "matrix",
        parents=[common],
        help="Compute a similarity matrix between one or two spectra files",
        description=(
            "Compute a similarity matrix between the spectra of SPECTRA_1 "
            "(rows) and optionally SPECTRA_2 (columns) and write it to an output file. "
            "Without SPECTRA_2 a symmetric all-vs-all matrix over SPECTRA_1 is "
            "computed. The output extension selects the format: .npz stores the "
            "Scores artifact, .tsv/.csv store a long format (one row per pair). "
            "--method selects the similarity (see `matchms similarity list`); "
            "--mode sparse requires a sparse-capable method and, together with "
            "--score-min, keeps only pairs with a main score >= value. Stdout "
            "carries only a summary."
        ),
    )
    ps_matrix.add_argument(
        "spectra_1",
        help="Input spectra file for the rows. Supported extensions: json, mgf, msp, mzml, mzxml, pickle.",
    )
    ps_matrix.add_argument(
        "spectra_2",
        nargs="?",
        default=None,
        help="Input spectra file for the columns. If omitted, a symmetric all-vs-all "
        "computation on SPECTRA_1 is run. Supported extensions: json, mgf, msp, mzml, mzxml, pickle.",
    )
    ps_matrix.add_argument(
        "--method",
        required=True,
        metavar="NAME",
        help="Similarity class name, case-insensitive (see `matchms similarity list`).",
    )
    ps_matrix.add_argument(
        "--param",
        action="append",
        default=None,
        metavar="NAME=VALUE",
        help="Constructor parameter, auto-typed (bool, int, float, JSON); may be "
        "passed multiple times. Examples: --param tolerance=0.1 --param remove_precursor=true.",
    )
    ps_matrix.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="Shorthand for --param tolerance=... (not combined with --param tolerance).",
    )
    ps_matrix.add_argument(
        "--mode",
        default="dense",
        choices=("dense", "sparse"),
        help="dense uses matrix(); sparse uses sparse_matrix() (only sparse-capable "
        "methods support it). Default: %(default)s.",
    )
    ps_matrix.add_argument(
        "--score-min",
        type=float,
        default=None,
        help="--mode sparse only: keep pairs with a main 'score' field >= value.",
    )
    ps_matrix.add_argument(
        "-o",
        "--output",
        required=True,
        metavar="PATH",
        help=(
            "Output file; the extension selects the format (."
            + ", .".join(sorted(SIMILARITY_OUTPUT_FORMATS))
            + "). An existing file is replaced."
        ),
    )
    ps_matrix.add_argument(
        "--id-field",
        default=None,
        help="Metadata field used as the human-readable row/col identifier in TSV/CSV output "
        "(row_id/col_id columns). Without it, only indices are written.",
    )
    ps_matrix.add_argument(
        "--max-dense-entries",
        type=int,
        default=DEFAULT_MAX_DENSE_ENTRIES,
        help="Dense mode: fail when n_rows x n_cols exceeds this many entries (default: %(default)s).",
    )
    ps_matrix.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of top pairs reported in the summary (default: %(default)s).",
    )
    ps_matrix.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable the progress bar (it is printed to stderr by default).",
    )
    ps_matrix.set_defaults(func=run_similarity_matrix)

    # similarity build-index --------------------------------------------
    ps_build_index = similarity_subparsers.add_parser(
        "build-index",
        parents=[common],
        help="Build a reusable library index from a spectra file",
        description=(
            "Build a reusable library index from the reference spectra in "
            "LIBRARY and save it to -o, which must end in .index.npz. The index "
            "can later be used by `matchms similarity search` so the reference "
            "library does not have to be re-prepared for every search. --method "
            "selects an index-capable similarity (Cosine, ModifiedCosine, "
            "Entropy, EntropySearch). The index is bound to the exact method and "
            "parameters it was built with. Stdout carries only a summary."
        ),
    )
    ps_build_index.add_argument(
        "library",
        help="Reference spectra file to index. Supported extensions: json, mgf, msp, mzml, mzxml, pickle.",
    )
    ps_build_index.add_argument(
        "--method",
        required=True,
        metavar="NAME",
        help="Index-capable similarity class, case-insensitive (Cosine, ModifiedCosine, Entropy, EntropySearch).",
    )
    ps_build_index.add_argument(
        "--param",
        action="append",
        default=None,
        metavar="NAME=VALUE",
        help="Constructor parameter, auto-typed (bool, int, float, JSON); may be "
        "passed multiple times. Examples: --param tolerance=0.1 --param remove_precursor=true.",
    )
    ps_build_index.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="Shorthand for --param tolerance=... (not combined with --param tolerance).",
    )
    ps_build_index.add_argument(
        "-o",
        "--output",
        required=True,
        metavar="PATH",
        help="Output index file; must end in .index.npz. An existing file is replaced.",
    )
    ps_build_index.set_defaults(func=run_similarity_build_index)

    # similarity search -----------------------------------------------------
    ps_search = similarity_subparsers.add_parser(
        "search",
        parents=[common],
        help="Search a query file against a spectra library or a saved index",
        description=(
            "Search the spectra of QUERIES against a reference LIBRARY and write "
            "the best matches per query to an output (.tsv or .csv). LIBRARY "
            "is either a spectra file (the index is built on the fly) or an index "
            "saved by `similarity build-index` (a .index.npz file). --method "
            "selects an index-capable similarity (Cosine, ModifiedCosine, Entropy, "
            "EntropySearch). Queries are the rows of the score matrix and the "
            "library the columns; a query may be its own best hit. Stdout carries "
            "only a summary."
        ),
    )
    ps_search.add_argument(
        "queries",
        help="Query spectra file (rows of the score matrix; supported extensions: "
        "json, mgf, msp, mzml, mzxml, pickle).",
    )
    ps_search.add_argument(
        "library",
        help="Reference library: a spectra file or an index saved by `similarity build-index` "
        "(recognized by its .index.npz extension).",
    )
    ps_search.add_argument(
        "--method",
        required=True,
        metavar="NAME",
        help="Index-capable similarity class, case-insensitive (Cosine, ModifiedCosine, Entropy, EntropySearch).",
    )
    ps_search.add_argument(
        "--param",
        action="append",
        default=None,
        metavar="NAME=VALUE",
        help="Constructor parameter, auto-typed (bool, int, float, JSON); may be passed multiple "
        "times. Must match the configuration used to build the index (for an index library).",
    )
    ps_search.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="Shorthand for --param tolerance=... (not combined with --param tolerance).",
    )
    ps_search.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum number of hits kept per query (default: %(default)s).",
    )
    ps_search.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Only keep hits with a score >= this value (on the --score-field field).",
    )
    ps_search.add_argument(
        "--score-field",
        default="score",
        metavar="NAME",
        help="Score field used for ranking and for --min-score (default: %(default)s).",
    )
    ps_search.add_argument(
        "--query-id-field",
        default=None,
        metavar="FIELD",
        help="Metadata field used as a human-readable identifier for queries "
        "(e.g. spectrum_id, compound_name); the column is omitted when not set.",
    )
    ps_search.add_argument(
        "--library-id-field",
        default=None,
        metavar="FIELD",
        help="Metadata field used as a human-readable identifier for library spectra; requires "
        "library metadata (a spectra library, or --library-spectra for an index).",
    )
    ps_search.add_argument(
        "--library-spectra",
        default=None,
        metavar="FILE",
        help="Only when LIBRARY is an index: the spectra file the index was built from, used to "
        "look up library identifiers. Its spectrum count must match the index.",
    )
    ps_search.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of queries searched at once, to limit memory use (default: %(default)s).",
    )
    ps_search.add_argument(
        "-o",
        "--output",
        required=True,
        metavar="PATH",
        help="Output list (.tsv or .csv); the extension selects the format. An existing file is replaced.",
    )
    ps_search.add_argument(
        "--top",
        type=int,
        default=10,
        help="Number of best hits (over all queries) shown in the summary (default: %(default)s).",
    )
    ps_search.set_defaults(func=run_similarity_search)

    # -- spectra  ----------------------------------------------------------
    p = subparsers.add_parser(
        "spectra",
        parents=[common],
        help="Inspect and convert spectra files",
        description=(
            "Work with spectra files. `spectra describe` reports descriptive "
            "statistics of the loaded collection; `spectra convert` converts a "
            "spectra file to another supported format."
        ),
    )
    spectra_subparsers = p.add_subparsers(dest="spectra_command", metavar="<action>")

    # spectra describe -----------------------------------------------------
    pd = spectra_subparsers.add_parser(
        "describe",
        parents=[common],
        help="Describe a spectra collection (peak counts, intensity sums, entropy)",
        description=(
            "Load the spectra file with load_ms2_dataset and report the "
            "descriptive statistics of the collection (peak counts, intensity "
            "sums, intensity entropy) as computed by SpectraCollection.describe()."
        ),
    )
    pd.add_argument(
        "spectrumfile",
        help="Path to the spectra file to describe. Supported extensions: json, mgf, msp, mzml, mzxml, pickle.",
    )
    pd.add_argument(
        "--ftype",
        default="auto",
        help="File type to use for import. Defaults to 'auto', which guesses the file type from the file extension.",
    )
    pd.set_defaults(func=run_spectra_describe)

    # spectra convert ------------------------------------------------------
    pc = spectra_subparsers.add_parser(
        "convert",
        parents=[common],
        help="Convert a spectra file to another supported format",
        description=(
            "Load the input spectra file with load_ms2_dataset and save it with "
            "save_spectra in the format of the output file extension "
            "(e.g. in.mgf -> out.msp). The output file must not exist unless "
            "--append is passed (append is only supported for .mgf and .msp)."
        ),
    )
    pc.add_argument(
        "input",
        help="Path to the input spectra file. Supported extensions: json, mgf, msp, mzml, mzxml, pickle.",
    )
    pc.add_argument(
        "output",
        help="Path to the output spectra file. Supported extensions: json, mgf, msp, pickle.",
    )
    pc.add_argument(
        "--ftype",
        default="auto",
        help="File type to use for import. Defaults to 'auto', which guesses the "
        "file type from the input file extension.",
    )
    pc.add_argument(
        "--export-style",
        default="matchms",
        choices=EXPORT_STYLES,
        help="Metadata key style used for the export (default: %(default)s).",
    )
    pc.add_argument(
        "--append",
        action="store_true",
        help="Append to the output file instead of overwriting it. Only supported for .mgf and .msp output files.",
    )
    pc.set_defaults(func=run_spectra_convert)

    return parser


def main(argv: list[str] | None = None) -> int:
    """Main entrypoint for the CLI.

    Builds the actual parser with commands and subcommands.
    Configures logging, output and verbosity levels.

    Returns
    -------
    int
        Returns int of errors Codes.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "func", None):
        parser.print_help()
        return 2

    configure_logging(_log_level(args))
    ctx = OutputContext(
        json_flag=getattr(args, "json", None),
        table_flag=getattr(args, "table", None),
        quiet=bool(getattr(args, "quiet", False)),
        verbose=bool(getattr(args, "verbose", False)),
    )

    try:
        return args.func(args, ctx)
    except CliError as exc:
        emit_error(ctx, exc.to_dict())
        return exc.exit_code
    except Exception as exc:
        operation = getattr(args, "command", "") or ""
        emit_error(ctx, unexpected_error_payload(exc, operation))
        return 1
