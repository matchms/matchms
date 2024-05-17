from rdkit import Chem
import re


def get_atom_and_counts(formula):
    """Returns a dictionary of all atoms and counts of atoms"""
    parts = re.findall("[A-Z][a-z]?|[0-9]+", formula)
    atoms_and_counts = {}
    for i, atom in enumerate(parts):
        if atom.isnumeric():
            continue
        multiplier = int(parts[i + 1]) if len(parts) > i + 1 and parts[i + 1].isnumeric() else 1
        atoms_and_counts[atom] = multiplier
    return atoms_and_counts


class Formula:
    def __init__(self, formula: str):
        self.dict_representation = get_atom_and_counts(formula)

    def __add__(self, other_formula: "Formula"):
        new_formula = Formula("")
        new_formula.dict_representation = self.dict_representation.copy()
        for atom, value in other_formula.dict_representation.items():
            if atom in new_formula.dict_representation:
                new_formula.dict_representation[atom] += value
            else:
                new_formula.dict_representation[atom] = value
        return new_formula

    def __sub__(self, other_formula: "Formula"):
        new_formula = Formula("")
        new_formula.dict_representation = self.dict_representation.copy()
        for atom, value in other_formula.dict_representation.items():
            if atom in new_formula.dict_representation:
                new_formula.dict_representation[atom] -= value
                if new_formula.dict_representation[atom] < 0:
                    print(f"Removing an atom {other_formula} that does not exist in the main formula {str(self)}")
                    return None
            else:
                print(f"Removing an atom {other_formula} that does not exist in the main formula {str(self)}")
                return None
        return new_formula

    def __str__(self):
        # if "C" in self.dict_representation():
        str_representation = ""
        for atom, value in self.dict_representation.items():
            if value == 1:
                str_representation += atom
            if value > 1:
                str_representation += atom + str(value)
        return str_representation

    def get_mass(self):
        mass = 0
        periodic_table = Chem.GetPeriodicTable()
        for atom, value in self.dict_representation.items():
            try:
                atom_mass = periodic_table.GetMostCommonIsotopeMass(atom)
            except RuntimeError:
                print("The atom: %s in the formula %s is not known", atom, str(self))
                return None
            mass += atom_mass * value
        return mass
