from deprecated import deprecated
from matchms.filtering.metadata_processing.harmonize_missing_entries import (
    harmonize_missing_entries,
)
from matchms.typing import SpectrumType
from matchms.utils import ALIASES_FOR_NONE


@deprecated(
    version="1.0.0",
    reason="This will be dropped in a future version. Use `harmonize_missing_entries` instead."
)
def harmonize_undefined_inchi(
    spectrum_in: SpectrumType,
    undefined: str = "",
    aliases: list[str] = None,
    clone: bool | None = True,
) -> SpectrumType | None:
    """Replace all aliases for empty/undefined inchi entries by value of ``undefined`` argument."""
    if aliases is None:
        aliases = ALIASES_FOR_NONE

    return harmonize_missing_entries(
        spectrum_in,
        keys=["inchi"],
        undefined=undefined,
        aliases=aliases,
        clone=clone,
    )
