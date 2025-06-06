import re
from typing import Generator, List, Tuple
import numpy as np
from matchms.importing.parsing_utils import parse_spectrum_dict
from matchms.Spectrum import Spectrum


def load_from_msp(filename: str, metadata_harmonization: bool = True) -> Generator[Spectrum, None, None]:
    """
    MSP file to a :py:class:`~matchms.Spectrum.Spectrum` objects
    Function that reads a .msp file and converts the info
    in :py:class:`~matchms.Spectrum.Spectrum` objects.

    Parameters
    ----------
    filename:
        Path of the msp file.
    metadata_harmonization : bool, optional
        Set to False if metadata harmonization to default keys is not desired.
        The default is True.

    Yields
    ------
    Yield a spectrum object with the data of the msp file


    Example:

    .. code-block:: python

        from matchms.importing import load_from_msp

        # Download msp file from MassBank of North America repository at https://mona.fiehnlab.ucdavis.edu/
        file_msp = "MoNA-export-GC-MS-first10.msp"
        spectra = list(load_from_msp(file_msp))
    """
    for spectrum in parse_msp_file(filename):
        yield parse_spectrum_dict(spectrum=spectrum, metadata_harmonization=metadata_harmonization, spectrum_type="own")


def parse_msp_file(filename: str) -> Generator[dict, None, None]:
    """Read msp file and parse info in List of spectrum dictionaries."""

    # Lists/dicts that will contain all params, masses and intensities of each molecule
    params = {}
    masses = np.array([])
    intensities = np.array([])
    peak_comments = {}

    # Peaks counter. Used to track and count the number of peaks
    peakscount = 0

    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            rline = line.rstrip()

            if len(rline) == 0:
                continue

            if contains_metadata(rline):
                parse_metadata(rline, params)
                continue

            mz, ints, comment = _parse_line_with_peaks(rline)

            masses = np.append(masses, mz)
            intensities = np.append(intensities, ints)

            if comment is not None:
                peak_comments.update({float(masses[-1]): comment})

            peakscount += len(mz)

            # Obtaining the masses and intensities
            if int(params["num peaks"]) == peakscount:
                peakscount = 0
                # Handles edge cases with GOLM files where the nominal mass is written with a comma instead of a dot
                nominal_mass = params.get("mw")
                if nominal_mass and isinstance(nominal_mass, str):
                    params["mw"] = nominal_mass.replace(",", ".")

                yield {
                    "params": (params),
                    "m/z array": masses,
                    "intensity array": intensities,
                    "peak comments": peak_comments,
                }

                params = {}
                masses = []
                intensities = []
                peak_comments = {}


def _parse_line_with_peaks(rline: str) -> Tuple[List[float], List[float], str]:
    """Parse a line containing peaks consisting of mz and intensity values with optional comments.

    Args:
        rline (str): Line with peaks read from the MSP.

    Returns:
        Tuple[List[float], List[float], str]: mz, intensity and peak comments obtained from the line.
    """
    comment, rline = get_peak_comment(rline)
    mz, intensities = get_peak_values(rline)

    return mz, intensities, comment


def get_peak_values(peak: str) -> Tuple[List[float], List[float]]:
    """Get the m/z and intensity value from the line containing the peak information."""
    tokens = re.findall(r"(\d+(?:\.\d+)?(?:e[-+]?\d+)?)", peak)
    if len(tokens) % 2 != 0:
        raise RuntimeError("Wrong peak format detected!")

    tokens = list(map(float, tokens))
    mz = tokens[0::2]
    intensities = tokens[1::2]
    return mz, intensities


def get_peak_comment(rline: str) -> Tuple[str, str]:
    """Get the peak comment from the line containing the peak information."""
    try:
        comment = re.findall(r"[\"\'](.*)[\"\']", rline)[0]
        rline = rline[: rline.index('"')]
    except IndexError:
        comment = None
    return comment, rline


def parse_metadata(rline: str, params: dict):
    """Reads metadata contained in line into params dict.

    The complexity of this function stems from the fact that MSP allows for many different formats of metadata.

    Parameters
    ----------
    rline: str
        The line of the read MSP file that contains metadata.
    params : dict
        The params that the key value pairs of the metadata line will be added to.
    """
    splitted_line = rline.split(":", 1)
    if len(splitted_line) != 2:
        return

    key, value = splitted_line[0].strip().lower(), splitted_line[1].strip()

    if key == "comments" and "=" in value:
        _parse_comments(value, params)
    elif key == "synon" and rline.count(":") >= 2:
        _parse_synon(rline, params)
    else:
        # Fallback for generic key: value pairs
        params[key] = value


def _parse_comments(value: str, params: dict):
    """Parses key-value pairs from comments line into params."""
    value = value.replace("'", '"')  # Normalize quotes
    pattern = (
        r'(\S+)="([^"]*)"|'
        r'"(\w+)=([^"]*)"|'
        r'"([^"]*)=([^"]*)"|'
        r"(\S+)=(\d+(?:\.\d*)?)"
    )
    for match in re.findall(pattern, value):
        match = [i for i in match if i]
        if len(match) == 2:
            m_key, m_value = match
            m_key = m_key.strip().lower()
            m_value = m_value.strip()
            if m_key == "smiles" and m_key in params:
                params[f"{m_key}_2"] = m_value
            else:
                params[m_key] = m_value


def _parse_synon(rline: str, params: dict):
    """Parses synon lines with multiple colons."""
    parts = rline.split(":", 2)
    synon_key = f"{parts[0].strip().lower()}: {parts[1].strip().lower()}"
    synon_value = parts[2].strip().replace(",", ".")
    if synon_key == "synon: metb n":
        params.setdefault(synon_key, []).append(synon_value)
    else:
        params[synon_key] = synon_value


def contains_metadata(rline: str) -> bool:
    """Check if line contains Spectrum metadata."""
    has_colon = ":" in rline
    return has_colon and not _is_golm_peak_format(rline)


def _is_golm_peak_format(rline: str) -> bool:
    """This function detects whether a line is a line containing peaks in the GOLM MSP format.

    The GOLM MSP format encodes peaks as mz:intensity - this resembles a metadata line, but actually contains peaks.
    It is therefore necessary to explicitly check this corner case when determining whether a line is peaks or metadata.

    Args:
        rline (str): Line to check whether it contains peaks from GOLM

    Returns:
        bool: Whether the line is a line with peaks from GOLM or not.
    """
    return re.match(r"(\d+:{1}\d+)", rline) is not None
