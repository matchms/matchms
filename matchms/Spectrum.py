# file: matchms/Spectrum.py


class Spectrum:

    def __init__(self, pyteomics_spectrum):
        self.metadata = pyteomics_spectrum.get("params", None)
        self.mz = pyteomics_spectrum["m/z array"]
        self.intensities = pyteomics_spectrum["intensity array"]
        self.PROTON_MASS = 1.00727645199076

        # Derive precursor-mass and parent-mass
        self.interpret_pepmass()
        self.precursor_mz = self.metadata['precursormass']
        self.parent_mz = self.metadata['parentmass']

    def interpret_pepmass(self):
        """Derive precursormass and parentmass from pepmass and charge.
        """
        self.metadata['precursormass'] = self.metadata.get('pepmass', None)[0]
        self.metadata['parentintensity'] = self.metadata.get('pepmass', None)[1]

        # Following corrects parentmass according to charge if charge is known.
        # This should lead to better computation of neutral losses
        precursor_mass = self.metadata['precursormass']
        int_charge = self.interpret_charge(self.metadata['charge'])

        parent_mass, single_charge_precursor_mass = self.ion_masses(
            precursor_mass, int_charge)

        self.metadata['charge_str'] = self.metadata['charge']  # Keep original
        self.metadata['charge'] = int_charge
        self.metadata['parentmass'] = parent_mass
        self.metadata[
            'singlechargeprecursormass'] = single_charge_precursor_mass

    def ion_masses(self, precursormass, int_charge):
        """Compute the parent masses.
        Single charge version is used for loss computation.
        """
        mul = abs(int_charge)
        parent_mass = precursormass * mul
        parent_mass -= int_charge * self.PROTON_MASS
        single_charge_precursor_mass = precursormass*mul
        if int_charge > 0:
            single_charge_precursor_mass -= (int_charge - 1) * self.PROTON_MASS
        elif int_charge < 0:
            single_charge_precursor_mass += (mul - 1) * self.PROTON_MASS
        else:
            parent_mass = precursormass
            single_charge_precursor_mass = precursormass
        return parent_mass, single_charge_precursor_mass

    def interpret_charge(self, charge):
        """Interpret the charge field.

        Method to interpret the ever variable charge field in the different
        formats.
        """
        if not charge:
            return 1
        try:
            if not isinstance(charge, str):
                charge = str(charge)

            # Try removing any + signs
            charge = charge.replace("+", "")

            # Remove trailing minus signs
            if charge.endswith('-'):
                charge = charge[:-1]
                if not charge.startswith('-'):
                    charge = '-' + charge
            # Turn into int
            int_charge = int(charge)
            return int_charge
        except ValueError:
            int_charge = 1
        return int_charge
