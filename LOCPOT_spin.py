import os
import numpy as np
from os.path import exists, join, getsize
from pymatgen.io.vasp import Locpot

class LocpotManager:
    """
    A dedicated handler for VASP LOCPOT data extraction and caching.
    Ensures character-level fidelity with the established spin-separation logic.
    """
    def __init__(self, filepath, ispin):
        self.filepath = filepath
        self.ispin = ispin
        self.locpot_path = join(filepath, "LOCPOT")
        self.cache_path = join(filepath, "LOCPOT.npy")
        self.data = None

    def get_data(self, force_rebuild=False):
        """
        Returns the LOCPOT data array. 
        Checks for a valid cache before parsing raw VASP output.
        """
        if self._is_cache_valid() and not force_rebuild:
            print(f"[*] Valid LOCPOT cache detected at {self.cache_path}. Skipping parse.")
            self.data = np.load(self.cache_path)
            return self.data

        return self._rebuild_cache()

    def _is_cache_valid(self):
        """
        Validates existence and dimensionality of the .npy cache.
        """
        if not exists(self.cache_path) or getsize(self.cache_path) == 0:
            return False
        
        try:
            # Check dimensions against ISPIN requirement
            # ISPIN=2 requires (2, X, Y, Z); ISPIN=1 requires (X, Y, Z)
            cached_data = np.load(self.cache_path, mmap_mode='r')
            if self.ispin == 2 and cached_data.ndim != 4:
                return False
            if self.ispin == 1 and cached_data.ndim != 3:
                return False
            return True
        except Exception:
            return False

    def _rebuild_cache(self):
        """
        Parses raw LOCPOT and applies spin-separation logic:
        V_up = (V_total + V_mag) / 2
        V_dn = (V_total - V_mag) / 2
        """
        if not exists(self.locpot_path):
            raise FileNotFoundError(f"Source LOCPOT not found: {self.locpot_path}")

        print(f"[*] Parsing raw LOCPOT via pymatgen (ISPIN={self.ispin})...")
        lpt = Locpot.from_file(self.locpot_path)
        
        # Key-Blind Sequential Section Extraction
        vol_sections = list(lpt.data.values())

        if self.ispin == 2 and len(vol_sections) >= 2:
            # Reconstruct spin channels from Total Potential and Magnetization
            v_tot = vol_sections[0]
            v_mag = vol_sections[1]
            self.data = np.stack([(v_tot + v_mag) / 2.0, (v_tot - v_mag) / 2.0])
        else:
            # Default to primary section for non-magnetic systems
            self.data = vol_sections[0]

        np.save(self.cache_path, self.data)
        print(f"[*] Saved unified LOCPOT cache to {self.cache_path} with shape {self.data.shape}")
        return self.data