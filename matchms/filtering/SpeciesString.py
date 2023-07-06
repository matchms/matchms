import re


class SpeciesString:
    """
    A class to process and clean different types of chemical structure strings including InChI, 
    InChIKey, and SMILES.

    The class takes a raw input string, determines the intended structure type, and then cleans 
    the string based on its type.

    Attributes
    ----------
    dirty : str
        Raw input string representing a chemical structure.
    target : str
        The intended structure type determined from the input string. Could be 'inchi', 'inchikey', 
        'smiles', or None if no valid type was identified.
    cleaned : str
        The cleaned structure string. 
    """
    def __init__(self, dirty: str):
        """
        Constructs a new instance of the SpeciesString class.

        Parameters
        ----------
        dirty : str
            The raw input string representing a chemical structure.
        """
        self.dirty = dirty
        self.target = None
        self.cleaned = None
        self.guess_target()
        self.clean()

    def __str__(self):
        if self.cleaned == "":
            return ""
        return f"({self.target}): {self.cleaned}"

    def clean(self):
        """Clean the input string based on its determined structure type."""
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
        regexp = r"(1S\/|1\/)[0-9A-Za-z\.]{2,}\/([ch])[0-9].*$"
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
        """Determine the intended structure type of the input string."""
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
        regexp = r"(InChI=1|1)(S\/|\/)[0-9, A-Z, a-z,\.]{2,}\/([ch])[0-9]"
        return re.search(regexp, self.dirty) is not None

    def looks_like_an_inchikey(self):
        """Return True if string has format of inchikey."""
        regexp = r"[A-Z]{14}-[A-Z]{10}-[A-Z]"
        return re.search(regexp, self.dirty) is not None

    def looks_like_a_smiles(self):
        """Return True if string is made of allowed charcters for smiles."""
        regexp = r"^([^J][0-9BCOHNSOPIFKcons@+\-\[\]\(\)\\\/%=#$,.~&!|Si|Se|Br|Mg|Na|Cl|Al]{3,})$"
        return re.search(regexp, self.dirty) is not None
