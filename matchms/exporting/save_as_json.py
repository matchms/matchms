import json
import numpy


def save_as_json(spectrums, filename):
    """Save spectrum(s) as json file.

    Args:
    ----
    spectrums: list of Spectrum() objects, Spectrum() object
        Expected input are match.Spectrum.Spectrum() objects.
    filename: str
        Provide filename to save spectrum(s).
    """
    if not isinstance(spectrums, list):
        # Assume that input was single Spectrum
        spectrums = [spectrums]

    # Write to json file
    with open(filename, 'w') as fout:
        fout.write("[")
        for i, spectrum in enumerate(spectrums):
            spec = spectrum.clone()
            peaks_list = numpy.vstack((spec.peaks.mz, spec.peaks.intensities)).T.tolist()

            # Convert matchms.Spectrum() into dictionaries
            spectrum_dict = {key: spec.metadata[key] for key in spec.metadata}
            spectrum_dict["peaks_json"] = peaks_list

            json.dump(spectrum_dict, fout)
            if i < len(spectrums) - 1:
                fout.write(",")
        fout.write("]")
