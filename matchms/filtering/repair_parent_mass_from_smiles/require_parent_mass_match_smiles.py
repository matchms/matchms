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
    smiles_mass = _get_monoisotopic_neutral_mass(smiles)
    mass_difference = parent_mass - smiles_mass
    if abs(mass_difference) < mass_tolerance:
        return spectrum


def _get_monoisotopic_neutral_mass(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mass = Descriptors.ExactMolWt(mol)
    charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    neutral_mass = mass + -charge * PROTON_MASS
    return neutral_mass
