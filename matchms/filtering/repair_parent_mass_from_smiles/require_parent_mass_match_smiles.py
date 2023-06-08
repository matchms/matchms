from rdkit import Chem
from rdkit.Chem import Descriptors
from matchms import Spectrum
from matchms.constants import PROTON_MASS


def require_parent_mass_match_smiles(spectrum_in: Spectrum,
                                     mass_tolerance):
    if spectrum_in is None:
        return None

    spectrum = spectrum_in.clone()
    # Check if parent mass matches the smiles mass
    parent_mass = spectrum.get("parent_mass")
    smiles = spectrum.get("smiles")
    if _mass_diff_within_tolerance(parent_mass, smiles, mass_tolerance):
        return spectrum


def _get_expected_mass(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mass = Descriptors.ExactMolWt(mol)
    charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    neutral_mass = mass + -charge * PROTON_MASS
    return neutral_mass


def _mass_diff_within_tolerance(mass, smiles, mass_tolerance):
    smiles_mass = _get_expected_mass(smiles)
    mass_difference = mass - smiles_mass
    if abs(mass_difference) < mass_tolerance:
        return True
    else:
        return False
