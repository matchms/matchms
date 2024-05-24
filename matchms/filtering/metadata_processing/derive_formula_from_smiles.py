import logging
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from rdkit import Chem


logger = logging.getLogger("matchms")


def derive_formula_from_smiles(spectrum_in, overwrite=True):
    spectrum = spectrum_in.clone()
    if spectrum.get("formula") is not None:
        if overwrite is False:
            return spectrum

    formula = _get_formula_from_smiles(spectrum.get("smiles"))

    if formula is not None:
        if spectrum.get("formula") is not None:
            if spectrum.get("formula") != formula:
                logger.info("Overwriting formula from inchi. Original formula: %s New formula: %s",
                            spectrum.get('formula'), formula)
                spectrum.set("formula", formula)
        else:
            logger.info("Added formula from inchi. New Formula: %s", formula)
            spectrum.set("formula", formula)
    else:
        logger.warning("The smiles: %s could not be interpreted by rdkit, so no formula was set")
    return spectrum


def _get_formula_from_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return CalcMolFormula(mol)
