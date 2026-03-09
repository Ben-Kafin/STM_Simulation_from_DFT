import numpy as np
import os

class SpinAwareDosParser:
    """
    Decoupled DOSCAR parser optimized for spin-polarized LDOS simulations.
    """
    def __init__(self, filepath):
        self.filepath = filepath
        self.energies = None
        self.ef = 0.0
        # Data stored as [spin_channel, atom_index, energy_index, orbital_index]
        self.spin_up_dos = None 
        self.spin_down_dos = None
        self.is_polarized = False
        self._parse()

    def _parse(self):
        with open(self.filepath, 'r') as f:
            atomnum = int(f.readline().split()[0])
            [f.readline() for _ in range(4)]
            header = f.readline().split()
            nedos, self.ef = int(header[2]), float(header[3])

            # Initialize lists
            energies_list = []
            total_dos = []
            site_dos = []

            for i in range(atomnum + 1):
                if i != 0: f.readline() # Skip atom header line
                block = []
                for j in range(nedos):
                    line = [float(x) for x in f.readline().split()]
                    if i == 0:
                        energies_list.append(line[0])
                        total_dos.append(line[1:])
                    else:
                        block.append(line[1:])
                if i > 0:
                    site_dos.append(block)

        self.energies = np.array(energies_list) - self.ef
        site_dos = np.array(site_dos) # Shape: (atoms, nedos, cols)
        num_cols = site_dos.shape[2]

        # Determine Spin Polarization Logic
        # 3, 9: Non-polarized | 6, 18, 32: Polarized
        if num_cols in [6, 18, 32]:
            self.is_polarized = True
            # VASP spin format: [s_up, s_down, p_up, p_down...]
            self.spin_up_dos = site_dos[:, :, 0::2]
            self.spin_down_dos = site_dos[:, :, 1::2]
        else:
            self.is_polarized = False
            self.spin_up_dos = site_dos
            self.spin_down_dos = None

    def get_dos_for_simulator(self, spin='up'):
        """
        Returns a (num_atoms, nedos, num_orbitals) array.
        Compatible with self.dos_gpu assignment in Interactive_STM_Simulator.
        """
        if spin == 'up' or not self.is_polarized:
            data = self.spin_up_dos
        else:
            data = self.spin_down_dos
            
        # Preserve orbital dimension for partitioned LDOS mapping
        return data