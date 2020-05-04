import re
from ..utils import is_valid_inchi, is_valid_smiles

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
        inchi_found = re.search(r"(1S\/|1\/)[0-9, A-Z, a-z,\.]{2,}\/(c|h)[0-9].*$",
                                self.dirty)
        if not inchi_found:
            self.cleaned = "issue detected:" + self.dirty
        else:
            inchi_cleaned = "InChI=" + inchi_found[0].replace('"', "")
            if is_valid_inchi(inchi_cleaned):
                self.cleaned = inchi_cleaned
            else:
                self.cleaned = "issue detected:" + self.dirty

    def clean_as_inchikey(self):
        """Search for valid inchikey and harmonize it."""
        inchikey_found = re.search(r"[A-Z]{14}-[A-Z]{10}-[A-Z]", self.dirty)
        if inchikey_found:
            self.cleaned = inchikey_found[0]
        else:
            self.cleaned = "issue detected:" + self.dirty

    def clean_as_smiles(self):
        """Search for valid smiles and harmonize it."""
        smiles_found = re.search(r"^([^J][0-9BCOHNSOPIFKcons@+\-\[\]\(\)\\\/%=#$,.~&!|Si|Se|Br|Mg|Na|Cl|Al]{3,})$",
                                 self.dirty)
        if not smiles_found:
            self.cleaned = "issue detected:" + self.dirty
        else:
            smiles_cleaned = smiles_found[0]
            if is_valid_smiles(smiles_cleaned):
                self.cleaned = smiles_cleaned
            else:
                self.cleaned = "issue detected:" + self.dirty

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
        if re.search(r"(InChI=1|1)(S\/|\/)[0-9, A-Z, a-z,\.]{2,}\/(c|h)[0-9]",
                     self.dirty):
            return True
        return False

    def looks_like_an_inchikey(self):
        """Return True if string has format of inchikey."""
        if re.search(r"[A-Z]{14}-[A-Z]{10}-[A-Z]", self.dirty):
            return True
        return False

    def looks_like_a_smiles(self):
        """Return True if string is made of allowed charcters for smiles."""
        if re.search(r"^([^J][0-9BCOHNSOPIFKcons@+\-\[\]\(\)\\\/%=#$,.~&!|Si|Se|Br|Mg|Na|Cl|Al]{3,})$",
                     self.dirty):
            return True
        return False
