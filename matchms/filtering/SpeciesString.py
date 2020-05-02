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
        self.cleaned = self.dirty

    def clean_as_inchikey(self):
        self.cleaned = self.dirty

    def clean_as_smiles(self):
        self.cleaned = self.dirty

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
        return \
            self.dirty.lower().startswith("inchi=") or \
            self.dirty.lower().startswith('"inchi=')

    def looks_like_an_inchikey(self):
        return re.fullmatch("[A-Z]{14}-[A-Z]{10}-[A-Z]", self.dirty)

    def looks_like_a_smiles(self):
        # maybe something like
        # https://gist.github.com/lsauer/1312860/264ae813c2bd2c27a769d261c8c6b38da34e22fb#file-smiles_inchi_annotated-js-L31
        return "@" in self.dirty
