import argparse
import logging
from matchms import __version__ as matchms_version
from matchms.cli.commands.filter_info import run as run_filter_info
from matchms.cli.commands.filter_list import run as run_filter_list
from matchms.cli.commands.filter_pipelines import PIPELINES
from matchms.cli.commands.filter_pipelines import run as run_filter_pipelines
from matchms.cli.commands.filter_run import run as run_filter_run
from matchms.cli.commands.info import run as run_info
from matchms.cli.commands.spectra_convert import EXPORT_STYLES
from matchms.cli.commands.spectra_convert import run as run_spectra_convert
from matchms.cli.commands.spectra_describe import run as run_spectra_describe
from matchms.cli.errors import CliError, configure_logging, emit_error, unexpected_error_payload
from matchms.cli.output import CLI_JSON_SCHEMA_VERSION, OutputContext


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
        version=f"matchms {matchms_version} (CLI JSON schema v{CLI_JSON_SCHEMA_VERSION})",
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
            "Report the matchms/Python versions, supported input/output formats, "
            "all filters (with default order) and the CLI JSON schema version."
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
        help="Path to the input spectra file (supported extensions: json, mgf, msp, mzml, mzxml, pickle).",
    )
    pf_run.add_argument(
        "output",
        help="Path to the output spectra file (supported extensions: json, mgf, msp, pickle).",
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
        help="Path to the spectra file to describe (supported extensions: json, mgf, msp, mzml, mzxml, pickle).",
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
        help="Path to the input spectra file (supported extensions: json, mgf, msp, mzml, mzxml, pickle).",
    )
    pc.add_argument(
        "output",
        help="Path to the output spectra file (supported extensions: json, mgf, msp, pickle).",
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
