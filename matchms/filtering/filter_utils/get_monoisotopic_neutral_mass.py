from matchms.constants import PROTON_MASS


try:  # rdkit is not included in pip package
    from rdkit import Chem
    from rdkit.Chem import Descriptors
except ImportError:
    _has_rdkit = False
    from collections import UserString

    class ChemMock(UserString):
        def __call__(self, *args, **kwargs):
            return self

        def __getattr__(self, key):
            return self

    Chem = AllChem = ChemMock("")
else:
    _has_rdkit = True
rdkit_missing_message = "Conda package 'rdkit' is required for this functionality."


def get_monoisotopic_neutral_mass(smiles):
    if not _has_rdkit:
        raise ImportError(rdkit_missing_message)
    mol = Chem.MolFromSmiles(smiles)
    mass = Descriptors.ExactMolWt(mol)
    charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
    neutral_mass = mass + -charge * PROTON_MASS
    return neutral_mass
