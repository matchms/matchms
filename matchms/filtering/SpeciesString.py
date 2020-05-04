import re
from rdkit import Chem


class SpeciesString:

    def __init__(self, dirty: str):
        self.dirty = dirty
        self.target = None
        self.cleaned = None
        self.guess_target()
        self.clean()

    def __str__(self):
        if self.cleaned == "":
            return ""
        return "({0}): {1}".format(self.target, self.cleaned)

    def clean(self):

        if self.target is None:
            self.cleaned = ""

        elif self.target == "inchi":
            self.clean_as_inchi()

        elif self.target == "inchikey":
            self.clean_as_inchikey()

        elif self.target == "smiles":
            self.clean_as_smiles()

        return self

    def clean_as_inchi(self):
        """Search for valid inchi and harmonize it."""
        inchi_found = re.search("(1S\/|1\/)[0-9, A-Z, a-z,\.]{2,}\/(c|h)[0-9].*$",
                                self.dirty)
        if not inchi_found:
            self.cleaned = "unable to clean"
        else:
            inchi_cleaned = "InChI=" + inchi_found[0].replace('"', "")
            if self.is_valid_inchi(inchi_cleaned):
                self.cleaned = inchi_cleaned
            else:
                self.cleaned = "unable to clean"

    def clean_as_inchikey(self):
        """Search for valid inchikey and harmonize it."""
        inchikey_found = re.search("[A-Z]{14}-[A-Z]{10}-[A-Z]", self.dirty)
        if inchikey_found:
            self.cleaned = inchikey_found[0]
        else:
            self.cleaned = "unable to clean"

    def clean_as_smiles(self):
        """Search for valid smiles and harmonize it."""
        smiles_found = re.search("^([^J][0-9BCOHNSOPrIFla@+\-\[\]\(\)\\\/%=#$,.~&!]{6,})$",
                                self.dirty)
        if not smiles_found:
            self.cleaned = "unable to clean"
        else:
            smiles_cleaned = smiles_found[0]
            if self.is_valid_smiles(smiles_cleaned):
                self.cleaned = smiles_cleaned
            else:
                self.cleaned = "unable to clean"

    def guess_target(self):

        if self.looks_like_an_inchikey():
            self.target = "inchikey"
        elif self.looks_like_an_inchi():
            self.target = "inchi"
        elif self.looks_like_a_smiles():
            self.target = "smiles"
        else:
            self.target = None

        return self

    def looks_like_an_inchi(self):
        """Search for first piece of InChI."""
        if re.search("(InChI=1|1)(S\/|\/)[0-9, A-Z, a-z,\.]{2,}\/(c|h)[0-9]",
                     self.dirty):
            return True
        return False

    def looks_like_an_inchikey(self):
        """Return True if string has format of inchikey."""
        if re.search("[A-Z]{14}-[A-Z]{10}-[A-Z]", self.dirty):
            return True
        return False

    def looks_like_a_smiles(self):
        """Return True if string is made of allowed charcters for smiles."""
        if re.search("^([^J][0-9BCOHNSOPIFKcons@+\-\[\]\(\)\\\/%=#$,.~&!|Si|Se|Br|Mg|Na|Cl|Al]{3,})$",
                    self.dirty):
            return True
        return False

    def is_valid_inchi(self, inchi):
        """Return True if input string is valid InChI.
    
        This functions test if string can be read by rdkit as InChI.
    
        Args:
        ----
        inchi: str
            Input string to test if it has format of InChI.
        """
        mol = Chem.MolFromInchi(inchi.replace('"', ""))
        if mol:
            return True
        return False

    def is_valid_smiles(smiles):
        """Return True if input string is valid smiles.
    
        This functions test if string can be read by rdkit as smiles.
    
        Args:
        ----
        inchi: str
            Input string to test if it can be imported as smiles.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return True
        return False
