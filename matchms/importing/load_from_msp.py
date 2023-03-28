import re
from typing import Generator, Iterator, Tuple
import numpy as np
from ..Spectrum import Spectrum


def load_from_msp(filename: str,
                  metadata_harmonization: bool = True) -> Generator[Spectrum, None, None]:
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
        spectrums = list(load_from_msp(file_msp))
    """

    for spectrum in parse_msp_file(filename):
        metadata = spectrum.get("params", None)
        mz = spectrum["m/z array"]
        intensities = spectrum["intensity array"]
        peak_comments = spectrum["peak comments"]
        if peak_comments != {}:
            metadata["peak_comments"] = peak_comments

        # Sort by mz (if not sorted already)
        if not np.all(mz[:-1] <= mz[1:]):
            idx_sorted = np.argsort(mz)
            mz = mz[idx_sorted]
            intensities = intensities[idx_sorted]

        yield Spectrum(mz=mz,
                       intensities=intensities,
                       metadata=metadata,
                       metadata_harmonization=metadata_harmonization)


def parse_msp_file(filename: str) -> Generator[dict, None, None]:
    """Read msp file and parse info in list of spectrum dictionaries."""

    # Lists/dicts that will contain all params, masses and intensities of each molecule
    params = {}
    masses = []
    intensities = []
    peak_comments = {}

    # Peaks counter. Used to track and count the number of peaks
    peakscount = 0

    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            rline = line.rstrip()

            if len(rline) == 0:
                continue

            if contains_metadata(rline):
                parse_metadata(rline, params)
            else:
                # Obtaining the masses and intensities
                peak_pairs = get_peak_tuples(rline)

                for peak in peak_pairs:
                    mz, intensity = get_peak_values(peak)
                    comment = get_peak_comment(peak)
                    if comment is not None:
                        peak_comments.update({mz: comment})

                    peakscount += 1
                    masses.append(mz)
                    intensities.append(intensity)

                # Obtaining the masses and intensities
                if int(params['num peaks']) == peakscount:
                    peakscount = 0
                    yield {
                        'params': (params),
                        'm/z array': np.array(masses),
                        'intensity array': np.array(intensities),
                        'peak comments': peak_comments
                    }

                    params = {}
                    masses = []
                    intensities = []
                    peak_comments = {}


def get_peak_values(peak: str) -> Tuple[float, float]:
    """ Get the m/z and intensity value from the line containing the peak information. """
    splitted_line = peak.split(maxsplit=2)
    mz = float(splitted_line[0].strip())
    intensity = float(splitted_line[1].strip())
    return mz, intensity


def get_peak_tuples(rline: str) -> Iterator[str]:
    """ Splits line at ';' and performs additional string cleaning. """
    tokens = filter(None, rline.split(";"))
    peak_pairs = map(lambda x: x.lstrip().rstrip(), tokens)
    return peak_pairs


def get_peak_comment(rline: str) -> str:
    """ Get the peak comment from the line containing the peak information. """
    try:
        comment = re.findall(r'[\"\'](.*)[\"\']', rline)[0]
    except IndexError:
        comment = None
    return comment


def parse_metadata(rline: str, params: dict):
    """ Reads metadata contained in line into params dict. """
    splitted_line = rline.split(":", 1)
    if splitted_line[0].lower() == 'comments' and "=" in splitted_line[1]:
        # Obtaining the parameters inside the comments index
        for s in splitted_line[1].split('" "'):
            splitted_line = s.replace('"', '').replace("'", "").split("=", 1)
            if splitted_line[0].lower().strip() in params.keys() and splitted_line[0].lower().strip() == 'smiles':
                params[splitted_line[0].lower()+"_2"] = splitted_line[1].strip()
            else:
                params[splitted_line[0].lower().strip()] = splitted_line[1].strip()
    else:
        params[splitted_line[0].lower()] = splitted_line[1].strip()


def contains_metadata(rline: str) -> bool:
    """ Check if line contains Spectrum metadata."""
    return ':' in rline
