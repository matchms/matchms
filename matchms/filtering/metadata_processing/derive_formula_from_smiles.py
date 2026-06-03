import logging
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula
from matchms.filtering._dispatch import metadata_update_filter
from matchms.filtering.filter_utils.metadata_conversions import as_string_or_none


logger = logging.getLogger("matchms")


def _derive_formula_from_smiles(metadata, overwrite: bool = True) -> dict:
    """Add molecular formula metadata derived from SMILES.

    Parameters
    ----------
    spectrum_in
        Input spectrum or spectra collection.
    overwrite
        If ``True``, an existing ``formula`` entry will be replaced when the
        formula derived from SMILES differs from the current value.
        If ``False``, an existing ``formula`` entry will be kept unchanged.
        Default is ``True``.
    clone
        Optionally clone the input before applying the filter. If ``False``,
        the input object may be modified in place.

    Returns
    -------
    Spectrum, SpectraCollection, or None
        Input object with updated ``formula`` metadata, or ``None`` if the input
        was ``None``.
    """
    current_formula = metadata.get("formula")

    if current_formula is not None and overwrite is False:
        return {}

    smiles = as_string_or_none(metadata.get("smiles"))
    formula = _get_formula_from_smiles(smiles)

    if formula is None:
        logger.warning(
            "SMILES: %s could not be interpreted by rdkit, so no formula was set",
            smiles,
        )
        return {}

    if current_formula is not None:
        if current_formula != formula:
            logger.info(
                "Overwriting formula. Original formula: %s New formula: %s",
                current_formula,
                formula,
            )
            return {"formula": formula}
        return {}

    logger.info("Added formula from SMILES. New Formula: %s", formula)
    return {"formula": formula}


def _get_formula_from_smiles(smiles):
    smiles = as_string_or_none(smiles)

    if smiles is None:
        return None

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    return CalcMolFormula(mol)


derive_formula_from_smiles = metadata_update_filter(_derive_formula_from_smiles)