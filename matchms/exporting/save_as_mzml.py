from psims.mzml.writer import MzMLWriter

from typing import List, Union
from matchms.Spectrum import Spectrum
from ..utils import (filter_empty_spectra, fingerprint_export_warning,
                     rename_deprecated_params)


@rename_deprecated_params(param_mapping={"spectrums": "spectra"}, version="0.26.5")
def save_as_mzml(spectra: Union[List[Spectrum], Spectrum],
                 filename: str,
                 export_style: str = "matchms",
                 ms_level: int = 2):
    """
    """
    if not isinstance(spectra, list):
        # Assume that input was single Spectrum
        spectra = [spectra]

    spectra = filter_empty_spectra(spectra)
    fingerprint_export_warning(spectra)

    def get_mzml_metadata(spectrum):
        """Generates dictionaries with metadata information"""
        metadata = spectrum.metadata_dict(export_style)
        if "ms level" not in metadata:
            metadata["ms level"] = ms_level

        precursor_information = {"mz": spectrum.get("precursor_mz"),
                                "intensity": spectrum.get("precursor_intensity"),
                                "charge": spectrum.get("charge")}

        for field in ["precursor_mz", "precursor_intensity", "charge", "fingerprint"]:
            if field in metadata:
                del metadata[field]
        return metadata, precursor_information

    with MzMLWriter(open(filename, 'wb'), close=True) as out:
        # Add default controlled vocabularies
        out.controlled_vocabularies()
        # Open the run and spectrum list sections
        with out.run(id="my_analysis"):
            with out.spectrum_list(count=len(spectra)):
                for id, spectrum in enumerate(spectra):
                    metadata, precursor_information = get_mzml_metadata(spectrum)
                    # Write Precursor scan
                    out.write_spectrum(
                        spectrum.mz, spectrum.intensities, id=str(id),
                        params=list(metadata.items()),
                        precursor_information=precursor_information)
