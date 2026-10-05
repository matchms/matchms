"""Introspection helpers for filters and similarity classes.

Powers the ``filter`` and ``similarity`` CLI commands (``list`` and ``info``):
they turn a filter function or similarity class into a JSON-safe description
(signature, parameters, docstring) plus the small shared helpers the commands
use to shape and render that data.
"""

import inspect
import re
from matchms.cli.params import describe_signature
from matchms.similarity.base_similarity import BaseSimilarity


def _clean_docstring(doc: str | None) -> str:
    """Normalize a docstring: strip uniform indentation, drop blank ends."""
    if not doc:
        return ""
    return inspect.cleandoc(doc)


def short_description(docstring: str | None) -> str:
    """Return the leading sentence of a docstring.

    Cleans the docstring first, then prefers the first line; if that is
    empty, falls back to the first sentence of the docstring body.
    """
    doc = _clean_docstring(docstring)
    first_line = doc.split("\n", 1)[0].strip() if doc else ""
    if first_line:
        return first_line
    text = " ".join(doc.split()) if doc else ""
    for end in (".", "!", "?"):
        idx = text.find(end)
        if idx != -1:
            return text[: idx + 1]
    return text


def signature_parameters(info: dict, skip: tuple[str, ...] = ()) -> list[dict]:
    """Build the CLI parameter table from a ``*signature()`` result.

    ``info`` is the dict returned by :func:`filter_signature` or
    :func:`similarity_signature`. Each entry carries the parameter name, its
    type, whether it is required, its default (when any) and the description
    taken from the docstring (when documented). Parameter names listed in
    *skip* are dropped (e.g. ``clone``, which the SpectraProcessor manages).
    """
    param_docs = info["param_docs"]
    parameters = []
    for name, spec in info["signature"]["parameters"].items():
        if name in skip:
            continue
        entry = {"name": name, "type": spec["type"], "required": spec["required"]}
        if "default" in spec:
            entry["default"] = spec["default"]
        if name in param_docs:
            entry["description"] = param_docs[name]
        parameters.append(entry)
    return parameters


def filter_signature(func) -> dict:
    """Describe one filter function for list/describe commands.

    Returns a JSON-safe dict with the signature, required parameters and
    whether the filter can be applied to a whole SpectraCollection (i.e. it
    is wrapped with ``collection_filter``).
    """
    signature_info = describe_signature(func)
    # The first positional parameter is the spectrum input (matchms names it
    # 'spectrum' or 'spectrum_in'); it is not a user-settable parameter.
    first_positional = _first_positional(func)
    if first_positional:
        signature_info["parameters"].pop(first_positional, None)
        signature_info["required"] = [p for p in signature_info["required"] if p != first_positional]
    return {
        "signature": signature_info,
        "required": signature_info["required"],
        "collection_supported": hasattr(func, "__wrapped__"),
        "docstring": _clean_docstring(func.__doc__),
        "param_docs": extract_param_docs(func.__doc__),
    }


def _first_positional(func) -> str | None:
    for name, p in inspect.signature(func).parameters.items():
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            return name
    return None


def filter_description(func, parameters: dict[str, object] | None = None) -> dict:
    """JSON-safe description of one filter step (name + parameters)."""
    entry = {"name": func.__name__}
    if parameters:
        entry["parameters"] = parameters
    return entry


def filter_parameter_names(func) -> list[str]:
    """Return the user-settable parameter names of a filter.

    The first positional input parameter (``spectrum_in``) and the ``clone``
    flag, which the processor manages internally, are excluded. A trailing
    ``**kwargs`` marker is added when the filter accepts arbitrary parameters.
    """
    first_positional = _first_positional(func)
    names: list[str] = []
    has_var_keyword = False
    for name, p in inspect.signature(func).parameters.items():
        if name == first_positional or name == "clone":
            continue
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            has_var_keyword = True
        elif p.kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        ):
            names.append(name)
    if has_var_keyword:
        names.append("**kwargs")
    return names


def filter_accepts_parameter(func, param_name: str) -> bool:
    """True when *func* can be called with the given keyword parameter."""
    first_positional = _first_positional(func)
    for name, p in inspect.signature(func).parameters.items():
        if name == first_positional or name == "clone":
            continue
        if p.kind == inspect.Parameter.VAR_KEYWORD:
            return True
        if name == param_name:
            return True
    return False


_SECTION_NAMES = {
    "parameters",
    "returns",
    "yields",
    "notes",
    "examples",
    "references",
    "see also",
    "raises",
    "attributes",
}

_PARAM_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.*)$")
_BARE_PARAM_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)$")


def _section_bounds(lines: list[str]) -> list[tuple[int, str]]:
    """Return (index, name) of numpy-style section headers in a docstring."""
    bounds = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        # A header is a short name, possibly followed by an underline of dashes.
        is_dashes_next = i + 1 < len(lines) and set(lines[i + 1].strip()) == {"-"} and lines[i + 1].strip()
        name = stripped.rstrip(":").strip().lower()
        if name in _SECTION_NAMES and (is_dashes_next or re.fullmatch(r"[A-Za-z ]+", stripped)):
            bounds.append((i, name))
    return bounds


def extract_param_docs(docstring: str | None) -> dict[str, str]:
    """Extract a ``name -> description`` mapping from the Parameters section.

    Handles matchms' numpy-style docstrings where ``Parameters`` is followed
    by an underline of dashes and each ``name:`` sits at the section's base
    indent with deeper-indented description lines. Robust to the base indent
    being 0 (after ``cleandoc``) or non-zero.
    """
    doc = _clean_docstring(docstring)
    if not doc:
        return {}
    lines = doc.splitlines()

    bounds = _section_bounds(lines)
    param_start = None
    param_end = len(lines)
    for i, (idx, name) in enumerate(bounds):
        if name == "parameters":
            param_start = idx
            # skip the optional underline of dashes
            j = idx + 1
            while j < len(lines) and (not lines[j].strip() or set(lines[j].strip()) == {"-"}):
                j += 1
            param_start = j
            if i + 1 < len(bounds):
                param_end = bounds[i + 1][0]
            break
    if param_start is None:
        return {}

    section = lines[param_start:param_end]
    params: dict[str, str] = {}
    current: str | None = None
    buffer: list[str] = []
    param_indent: int | None = None

    def flush():
        nonlocal current, buffer
        if current is not None:
            params[current] = " ".join(buffer).strip()
        current = None
        buffer = []

    def _start_param(stripped: str, indent: int):
        """Start the description of the parameter named in *stripped*.

        Flushes the description of the previous parameter, if any.
        """
        nonlocal current, buffer, param_indent
        flush()
        typed = _PARAM_RE.match(stripped)
        if typed is not None:
            current = typed.group(1)
            buffer = [typed.group(2).strip()] if typed.group(2).strip() else []
        elif _BARE_PARAM_RE.match(stripped) is not None:
            current = stripped
            buffer = []
        else:
            return False
        param_indent = indent
        return True

    for line in section:
        stripped = line.strip()
        if not stripped:
            continue
        indent = len(line) - len(line.lstrip(" "))

        if param_indent is None:
            _start_param(stripped, indent)
            continue

        if indent == param_indent:
            if _start_param(stripped, indent):
                continue
            # A base-indent line that is not a parameter ends the param list.
            flush()
            break

        # Deeper indent: continuation / description of the current parameter.
        if current is not None:
            buffer.append(stripped)

    flush()
    return params


def similarity_methods(cls: type) -> list[str]:
    """The computation methods a similarity class actually implements.

    Checks the public scoring surface (``pair``, ``matrix``, ``sparse_matrix``,
    the indexed ``build_index``/``search`` workflow) against the class itself so
    inherited implementations count. This is the single source of truth for
    method/capability detection: ``similarity list``/``similarity info`` report
    these, and the index/search commands use it to decide which methods support
    a reusable library index.
    """
    methods = []
    for method in _SIMILARITY_METHODS:
        impl = inspect.getattr_static(cls, method, None)
        if impl is None:
            continue
        # ``BaseSimilarity.sparse_matrix`` only raises NotImplementedError.
        if method == "sparse_matrix" and impl is BaseSimilarity.sparse_matrix:
            continue
        methods.append(method)
    return methods


def similarity_signature(cls: type) -> dict:
    """Describe one similarity class for the ``similarity list``/``similarity info`` commands.

    Returns a JSON-safe dict with the constructor signature, the documented
    score fields and which of the common computation methods the class
    actually implements (``pair``, ``matrix``, ``sparse_matrix``, the indexed
    search workflow).
    """
    signature_info = describe_signature(cls)
    param_docs = extract_param_docs(cls.__doc__)
    param_docs.update(extract_param_docs(cls.__init__.__doc__))

    return {
        "signature": signature_info,
        "required": signature_info["required"],
        "param_docs": param_docs,
        "docstring": _clean_docstring(cls.__doc__),
        "score_fields": list(cls.score_fields),
        "is_commutative": bool(getattr(cls, "is_commutative", False)),
        "methods": similarity_methods(cls),
    }


# Methods that make up the public scoring surface of a similarity class,
# checked against the class itself so inherited implementations count.
_SIMILARITY_METHODS = ("pair", "matrix", "sparse_matrix", "build_index", "search")
