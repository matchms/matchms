"""Resolve public matchms re-export references to their documented definitions.

The generated API has one canonical object per implementation module. A public
re-export such as ``matchms.Spectrum`` remains a valid Python API but does not
have a second Sphinx object target. This hook links only when the real exported
object and its implementation documentation can be found. It never silences a
reference to a nonexistent implementation or guesses a target by suffix.
"""

from __future__ import annotations

import importlib
import inspect
from functools import lru_cache

# Only the explicitly public namespaces can be redirected. Internal module
# references such as matchms.similarity.flash_index.FlashIndex are untouched.
_PUBLIC_NAMESPACES = frozenset(
    {
        "matchms",
        "matchms.filtering",
        "matchms.importing",
        "matchms.exporting",
        "matchms.similarity",
        "matchms.plotting",
        "matchms.networking",
    }
)

# The handful of references where the public guide uses a name that is not
# re-exported at runtime. These must still exist in the Python domain before
# they can be resolved; the hook does not fabricate documentation targets.
_EXPLICIT_TARGETS = {
    "Spectrum": "matchms.spectrum.Spectrum",
    "SpectraCollection": "matchms.spectra_collection.SpectraCollection",
    "Metadata": "matchms.metadata.Metadata",
    "Scores": "matchms.scores.Scores",
    "Fragments": "matchms.fragments.Fragments",
    "Fingerprints": "matchms.fingerprints.Fingerprints",
    "matchms.filtering.SpectraProcessor": (
        "matchms.filtering.spectra_processor.SpectraProcessor"
    ),
    "matchms.importing.load_ms2_dataset": (
        "matchms.importing.load_spectra.load_ms2_dataset"
    ),
    "matchms.similarity.BaseSimilarity": (
        "matchms.similarity.base_similarity.BaseSimilarity"
    ),
    "matchms.similarity.BaseSimilarityWithSparse": (
        "matchms.similarity.base_similarity.BaseSimilarityWithSparse"
    ),
}


@lru_cache(maxsize=512)
def _implementation_for_export(target: str) -> str | None:
    """Return the actual definition of a class/function exported by matchms."""
    namespace, dot, attribute = target.rpartition(".")
    if not dot or namespace not in _PUBLIC_NAMESPACES:
        return None

    try:
        module = importlib.import_module(namespace)
    except (ImportError, AttributeError):
        return None

    exported = vars(module).get(attribute)
    if not (inspect.isclass(exported) or inspect.isfunction(exported)):
        return None

    owner = getattr(exported, "__module__", None)
    qualname = getattr(exported, "__qualname__", None)
    if not owner or not qualname or "<locals>" in qualname:
        return None

    canonical = f"{owner}.{qualname}"
    if canonical == target or not canonical.startswith("matchms."):
        return None
    return canonical


def resolve_public_matchms_reference(app, env, node, contnode):
    """Point unresolved public aliases at existing canonical Python objects.

    A failed lookup returns None to preserve Sphinx's ordinary warning. In
    particular, a bad ``:class:`` role or an undocumented implementation is not
    disguised by an ignore rule or an invented anchor.
    """
    if node.get("refdomain") != "py":
        return None

    target = node.get("reftarget")
    role = node.get("reftype")
    if not isinstance(target, str) or role not in {"class", "func", "meth", "obj"}:
        return None

    domain = env.domains["py"]
    candidates = []
    actual_definition = _implementation_for_export(target)
    if actual_definition is not None:
        candidates.append(actual_definition)
    if target in _EXPLICIT_TARGETS:
        candidates.append(_EXPLICIT_TARGETS[target])

    # Re-exported classes can also be used as prefixes of method/attribute
    # references: matchms.filtering.SpectraProcessor.process_spectrum, etc.
    # The final candidate must still be an indexed object of the right kind.
    prefix, dot, member = target.rpartition(".")
    if dot and role in {"meth", "obj"}:
        class_definition = (
            _implementation_for_export(prefix) or _EXPLICIT_TARGETS.get(prefix)
        )
        if class_definition is not None:
            candidates.append(f"{class_definition}.{member}")

    for candidate in dict.fromkeys(candidates):
        # A canonical name may be imported at runtime but not documented; do not
        # create broken links to such objects.
        if candidate not in domain.objects:
            continue
        # Never resolve a malformed role to the wrong object type merely
        # because a similarly named implementation happens to be indexed.
        objtype = domain.objects[candidate].objtype
        if role == "class" and objtype not in {"class", "exception", "type"}:
            continue
        if role == "func" and objtype not in {"function", "method"}:
            continue
        if role == "meth" and objtype not in {"method", "staticmethod", "classmethod"}:
            continue
        resolved = domain.resolve_xref(
            env,
            node["refdoc"],
            app.builder,
            role,
            candidate,
            node,
            contnode,
        )
        if resolved is not None:
            return resolved
    return None


def setup(app):
    app.connect("missing-reference", resolve_public_matchms_reference)
    return {"version": "1.0", "parallel_read_safe": True, "parallel_write_safe": True}
