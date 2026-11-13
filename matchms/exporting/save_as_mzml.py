import logging
from collections.abc import Mapping
from pathlib import Path
import numpy as np
from psims.mzml.writer import MzMLWriter
from ..__version__ import __version__
from ..spectrum import Spectrum
from ..utils import filter_empty_spectra, fingerprint_export_warning, load_export_key_conversions


logger = logging.getLogger("matchms")

RETENTION_TIME_UNITS = ("second", "minute")
"""Units in which ``retention_time`` can be written as ``scan start time``."""

_POLARITY_TERMS = {"positive": "positive scan", "negative": "negative scan"}
_SOFTWARE_ID = "matchms"
_INSTRUMENT_ID = "unknown_instrument"
_DATA_PROCESSING_ID = "matchms_export"
_HANDLED_KEYS = (
    "precursor_mz",
    "precursor_intensity",
    "charge",
    "ionmode",
    "retention_time",
    "collision_energy",
    "ms_level",
    "title",
    "fingerprint",
    "peak_comments",
    "num_peaks",
)
"""Metadata written as PSI-MS terms, or not written at all, rather than as userParams."""


def save_as_mzml(
    spectra: list[Spectrum] | Spectrum,
    filename: str | Path,
    export_style: str = "matchms",
    ms_level: int = 2,
    retention_time_unit: str = "second",
) -> None:
    """Save spectrum(s) as an mzML file.

    Peaks are written as 64-bit m/z and intensity arrays. The metadata that mzML
    describes with controlled-vocabulary terms is written as such: ``precursor_mz``,
    ``precursor_intensity`` and ``charge`` as the selected ion, ``ionmode`` as the
    scan polarity, ``retention_time`` as the scan start time, ``collision_energy``
    as the activation and ``ms_level`` as the ms level. All other metadata entries
    are written as userParams of the spectrum, so that a reader that keeps them
    (and :func:`~matchms.importing.load_from_mzml`) can restore them.

    Example:

    .. code-block:: python

        import numpy as np
        from matchms import Spectrum
        from matchms.exporting import save_as_mzml

        spectrum = Spectrum(
            mz=np.array([100, 200, 300], dtype="float"),
            intensities=np.array([10, 10, 500], dtype="float"),
            metadata={"charge": -1, "precursor_mz": 222.2, "compound_name": "test"},
        )
        save_as_mzml(spectrum, "test.mzML")

    Parameters
    ----------
    spectra:
        Expected input are :class:`~matchms.spectrum.Spectrum` objects.
    filename:
        Provide filename to save spectrum(s). An existing file is overwritten.
    export_style:
        Converts the keys of the userParams to the required export style.
        One of ["matchms", "massbank", "nist", "riken", "gnps", "sirius"].
        Default is "matchms".
    ms_level:
        MS level written for spectra without an ``ms_level`` entry. Default is 2.
    retention_time_unit:
        Unit of the ``retention_time`` entries, written with the scan start time.
        One of ["second", "minute"]. Default is "second".
    """
    if not isinstance(spectra, list):
        # Assume that input was a single Spectrum.
        spectra = [spectra]

    spectra = filter_empty_spectra(spectra)
    fingerprint_export_warning(spectra)

    if retention_time_unit not in RETENTION_TIME_UNITS:
        raise ValueError(f"retention_time_unit must be one of {RETENTION_TIME_UNITS}, got {retention_time_unit!r}.")
    if not isinstance(ms_level, int) or isinstance(ms_level, bool) or ms_level < 1:
        raise ValueError("ms_level must be a positive integer.")

    key_conversions = _export_key_conversions(export_style)
    spectrum_arguments = [
        _spectrum_arguments(spectrum, index, export_style, key_conversions, ms_level, retention_time_unit)
        for index, spectrum in enumerate(spectra)
    ]
    ms_levels = {arguments["params"][0]["ms level"] for arguments in spectrum_arguments}

    with open(filename, "wb") as file, MzMLWriter(file, close=False) as writer:
        _write_header(writer, ms_levels)
        with (
            writer.run(id=_DATA_PROCESSING_ID, instrument_configuration=_INSTRUMENT_ID),
            writer.spectrum_list(count=len(spectrum_arguments), data_processing_method=_DATA_PROCESSING_ID),
        ):
            for arguments in spectrum_arguments:
                writer.write_spectrum(**arguments)


def _export_key_conversions(export_style: str) -> dict:
    """Return the names of the matchms keys in the export style (matchms key -> style key)."""
    if export_style == "matchms":
        return {}
    return load_export_key_conversions(export_style=export_style)


def _write_header(writer: MzMLWriter, ms_levels: set[int]) -> None:
    """Write the sections before the run: vocabularies, file description, software,
    instrument configuration and data processing. mzML requires each of them."""
    file_contents = []
    if 1 in ms_levels:
        file_contents.append("MS1 spectrum")
    if any(level > 1 for level in ms_levels):
        file_contents.append("MSn spectrum")
    file_contents.append("centroid spectrum")

    writer.controlled_vocabularies()
    writer.file_description(file_contents)
    writer.software_list(
        [{"id": _SOFTWARE_ID, "version": __version__, "params": [{"custom unreleased software tool": "matchms"}]}]
    )
    writer.instrument_configuration_list(
        [writer.InstrumentConfiguration(id=_INSTRUMENT_ID, component_list=[], params=["instrument model"])]
    )
    writer.data_processing_list(
        [
            writer.DataProcessing(
                [writer.ProcessingMethod(order=1, software_reference=_SOFTWARE_ID, params=["Conversion to mzML"])],
                id=_DATA_PROCESSING_ID,
            )
        ]
    )


def _spectrum_arguments(
    spectrum: Spectrum,
    index: int,
    export_style: str,
    key_conversions: Mapping[str, str],
    default_ms_level: int,
    retention_time_unit: str,
) -> dict:
    """Collect the arguments of ``MzMLWriter.write_spectrum`` for one spectrum."""
    metadata = spectrum.metadata_dict(export_style)
    for key in _HANDLED_KEYS:
        metadata.pop(key_conversions.get(key, key), None)

    level = _as_int(spectrum.get("ms_level"))
    params: list = [{"ms level": level if level is not None and level > 0 else default_ms_level}]
    title = spectrum.get("title")
    if title not in (None, ""):
        params.append({"spectrum title": str(title)})

    collision_energy = _as_float(spectrum.get("collision_energy"))
    if collision_energy is None and spectrum.get("collision_energy") not in (None, ""):
        # Not a plain number (e.g. "35 eV" or "ramp 20-40"): keep the text as a userParam.
        metadata[key_conversions.get("collision_energy", "collision_energy")] = spectrum.get("collision_energy")
    params.extend(_user_params(metadata))

    charge = _as_int(spectrum.get("charge"))
    arguments = {
        "mz_array": np.asarray(spectrum.peaks.mz, dtype=np.float64),
        "intensity_array": np.asarray(spectrum.peaks.intensities, dtype=np.float64),
        "id": f"index={index}",
        "centroided": True,
        "polarity": _polarity(spectrum.get("ionmode"), charge),
        "params": params,
        "encoding": {"m/z array": np.float64, "intensity array": np.float64},
    }

    retention_time = _as_float(spectrum.get("retention_time"))
    if retention_time is not None:
        arguments["scan_params"] = [{"scan start time": retention_time, "unit_name": retention_time_unit}]

    precursor_mz = _as_float(spectrum.get("precursor_mz"))
    if precursor_mz is not None:
        # mzML requires an activation element; the method itself is not known to matchms,
        # so the generic parent term is written, with the collision energy when given.
        activation: list = ["dissociation method"]
        if collision_energy is not None:
            activation.append({"collision energy": collision_energy, "unit_name": "electronvolt"})
        arguments["precursor_information"] = {
            "mz": precursor_mz,
            "intensity": _as_float(spectrum.get("precursor_intensity")),
            "charge": charge,
            "activation": activation,
        }

    return arguments


def _user_params(metadata: Mapping) -> list[dict]:
    """Return the remaining metadata entries as userParams, one per entry."""
    user_params = []
    for key, value in metadata.items():
        if value is None or (isinstance(value, str) and value == ""):
            continue
        if isinstance(value, np.generic):
            value = value.item()
        if not isinstance(value, (str, int, float)) or isinstance(value, bool):
            value = str(value)
        user_params.append({str(key): value})
    return user_params


def _polarity(ionmode, charge: int | None) -> str | int | None:
    """Return the scan polarity term, derived from the charge sign when ionmode is missing."""
    if isinstance(ionmode, str) and ionmode.lower() in _POLARITY_TERMS:
        return _POLARITY_TERMS[ionmode.lower()]
    if charge:
        return charge
    return None


def _as_float(value) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _as_int(value) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, str):
        # Accept "2", "MS2" or "ms2".
        value = value.strip().lower().removeprefix("ms")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(number) or number != int(number):
        return None
    return int(number)
