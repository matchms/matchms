from ..utils import mol_converter
from ..typing import SpectrumType
from .entry_is_empty import entry_is_empty


def clean_inchis(spectrum_in: SpectrumType, rescue_smiles=True) -> SpectrumType:
    """Make inchi style more consistent and wrongly given smiles.

    Args:
    ----
    spectrum_in: matchms.Spectrum()
        Input spectrum.
    rescue_smiles: bool
        If True, check if smiles is accidentaly given in inchi field.
        Default is True.

    Read spectrum, look for inchi. Then:
    1) Make line with inchi homogeneously looking like: '"InChI=..."'
    2) if rescue_smiles is True then try to detect inchi that are actually smiles
    and convert to proper inchi.
    """

    spectrum = spectrum_in.clone()

    if entry_is_empty(spectrum, "inchi"):
        inchi = 'n/a'
    else:
        inchi = spectrum.get("inchi").replace(" ", "")
        if inchi.split('InChI=')[-1][0] in ['C', 'c', 'O', 'N']:
            if rescue_smiles:
                # Try to 'rescue' given inchi which are actually smiles!
                assumed_smile = inchi.split('InChI=')[-1].replace('"', '')
                inchi = mol_converter(assumed_smile, "smi", "inchi")
                if not inchi:
                    inchi = 'n/a'
                if len(inchi) < 12:
                    inchi = 'n/a'
                print("New inchi:", inchi.replace('\n', ''))
                print("Derived inchi from assumed smile:", assumed_smile)

        # Make inchi string style consistent
        inchi = inchi.strip().split('InChI=')[-1]
        if inchi.endswith('"'):
            inchi = '"InChI=' + inchi
        elif inchi.endswith('\n'):
            inchi = '"InChI=' + inchi[:-2] + '"'
        elif inchi.endswith('\n"'):
            inchi = '"InChI=' + inchi[:-3] + '"'
        else:
            inchi = '"InChI=' + inchi + '"'
    spectrum.set("inchi", inchi)
    return spectrum
