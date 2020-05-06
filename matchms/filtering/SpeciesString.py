import re


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
        regexp = r"(1S\/|1\/)[0-9, A-Z, a-z,\.]{2,}\/(c|h)[0-9].*$"
        found = re.search(regexp, self.dirty)
        if found is None:
            self.cleaned = ""
        else:
            self.cleaned = "InChI=" + found[0].replace('"', "")

    def clean_as_inchikey(self):
        """Search for valid inchikey and harmonize it."""
        regexp = r"[A-Z]{14}-[A-Z]{10}-[A-Z]"
        found = re.search(regexp, self.dirty)
        if found is None:
            self.cleaned = ""
        else:
            self.cleaned = found[0]

    def clean_as_smiles(self):
        """Search for valid smiles and harmonize it."""
        regexp = r"^([^J][0-9BCOHNSOPIFKcons@+\-\[\]\(\)\\\/%=#$,.~&!|Si|Se|Br|Mg|Na|Cl|Al]{3,})$"
        found = re.search(regexp, self.dirty)
        if found is None:
            self.cleaned = ""
        else:
            self.cleaned = found[0]

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
        regexp = r"(InChI=1|1)(S\/|\/)[0-9, A-Z, a-z,\.]{2,}\/(c|h)[0-9]"
        return re.search(regexp, self.dirty) is not None

    def looks_like_an_inchikey(self):
        """Return True if string has format of inchikey."""
        regexp = r"[A-Z]{14}-[A-Z]{10}-[A-Z]"
        return re.search(regexp, self.dirty) is not None

    def looks_like_a_smiles(self):
        """Return True if string is made of allowed charcters for smiles."""
        regexp = r"^([^J][0-9BCOHNSOPIFKcons@+\-\[\]\(\)\\\/%=#$,.~&!|Si|Se|Br|Mg|Na|Cl|Al]{3,})$"
        return re.search(regexp, self.dirty) is not None
