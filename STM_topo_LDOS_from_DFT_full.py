# -*- coding: utf-8 -*-
"""
Interactive STM Simulator: Multi-Tiered GPU Optimization
Developed by Benjamin Kafin
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp  
import cupyx.scipy.ndimage as cp_ndimage 
from os.path import exists, getsize, join
from os import chdir
from numpy.linalg import norm, inv
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import LineCollection
from matplotlib.widgets import Slider, CheckButtons, Button, RadioButtons
import matplotlib.gridspec as gridspec
import matplotlib.colors as mc, colorsys
from matplotlib.lines import Line2D
import mplcursors
from time import time

# --- SURGICAL IMPORT INTEGRATION ---
# Assuming these exist in your environment as per original source
from DOSCAR_spin_orbitals import SpinAwareDosParser
from LOCPOT_spin import LocpotManager

# --- CORE UTILITIES ---
def gpu_simpson(y, x):
    """Vectorized Simpson's Rule for GPU parity."""
    n = y.shape[1]
    if n % 2 == 0:
        return cp.trapz(y, x=x, axis=1)
    dx = (x[-1] - x[0]) / (n - 1)
    weights = cp.ones(n)
    weights[1:-1:2] = 4
    weights[2:-2:2] = 2
    return (dx / 3.0) * cp.sum(weights * y, axis=1)

def gpu_chen_tunneling_factor(V, E, phi):
    """Vectorized Julian Chen barrier model character-for-character."""
    me, h, q = 9.1093837e-31, 6.62607015e-34, 1.60217663e-19
    V_eff = cp.where(cp.abs(V) < 1e-6, 1e-6, V)
    V_j, E_j, phi_j = cp.abs(V_eff) * q, E * q, phi * q
    prefactor = (8.0 / (3.0 * V_j)) * cp.pi * cp.sqrt(2.0 * me) / h
    term1 = cp.power(cp.maximum(0.0, phi_j - E_j + V_j), 1.5)
    term2 = cp.power(cp.maximum(0.0, phi_j - E_j), 1.5)
    return prefactor * (term1 - term2)

class Unified_STM_Simulator:
    def __init__(self, filepath):
        self.filepath = filepath
        self.unit_cell_num = 4
        chdir(filepath)
        self.dev = cp.cuda.Device(0)
        print("--- INITIALIZING GPU TENSOR SIMULATOR ---")

    def _converge_tip_height(self, z_map_gpu, grid_xy_gpu, emin, emax, target_ldos, 
                             target_threshold=0.01, topo_gain=0.5, max_iter=1000, use_decay=True):
        """Exhaustive Point-Wise Convergence Engine."""
        t_start = time()
        print("--- INITIALIZING TIP CONVERGENCE ENGINE ---")
        for i in range(max_iter):
            t0 = time()
            ld_up, ld_dn, a_grid = self._calculate_ldos_at_points_gpu(cp.hstack([grid_xy_gpu, z_map_gpu[:, None]]), emin, emax, use_energy_decay=use_decay, preserve_orbitals=False)
            cur_ldos = gpu_simpson(ld_up + ld_dn if ld_dn is not None else ld_up, a_grid)
            rat = cp.maximum(cur_ldos, 1e-20) / target_ldos
            
            active_mask = cp.abs(rat - 1.0) > target_threshold
            active_count = int(cp.sum(active_mask))
            max_error = float(cp.max(cp.abs(rat - 1.0)))
            z_min, z_max = float(cp.min(z_map_gpu)), float(cp.max(z_map_gpu))
            print(f"   Iter {i+1:02d}: Active Pts ={active_count:4d} | Max Error ={max_error*100:6.2f}% | Z_min ={z_min:6.3f} Å | Z_max ={z_max:6.3f} Å | Range ={z_max-z_min:8.4f} Å | Time ={time()-t0:6.3f}s")
            if active_count == 0: break
            
            pts_active = cp.hstack([grid_xy_gpu, z_map_gpu[:, None]])[active_mask]
            idx_g = (cp.dot(pts_active, self.inv_lv_gpu) % 1.0).T * self.locpot_dims_gpu[(-3 if self.locpot_gpu.ndim==4 else 0):, None]
            pot_g = self.locpot_gpu[0] if self.locpot_gpu.ndim == 4 else self.locpot_gpu
            phi_l = cp_ndimage.map_coordinates(pot_g, idx_g, order=1, mode='wrap') - self.ef
            kappa = 0.512 * cp.sqrt(cp.maximum(0.1, phi_l))
            z_map_gpu[active_mask] += (topo_gain / (2.0 * kappa)) * cp.log(rat[active_mask])
        print(f"--- CONVERGENCE COMPLETE: Total Time ={time() - t_start:6.3f}s ---")
        return z_map_gpu

    def _parse_poscar(self, ifile):
        with open(ifile, 'r') as f:
            lines = f.readlines(); sf = float(lines[1]); lv = np.array([float(c) for c in ' '.join(lines[2:5]).split()]).reshape(3,3) * sf
            atomtypes = lines[5].split(); atomnums = [int(i) for i in lines[6].split()]
            start_line = 7 if lines[7].strip().lower()[0] in ['d', 'c'] else 8
            coord = np.array([[float(c) for c in line.split()[:3]] for line in lines[start_line+1:sum(atomnums)+start_line+1]])
            if 'direct' in lines[start_line].lower(): coord = np.dot(coord, lv)
        return lv, coord, atomtypes, atomnums

    def parse_vasp_outputs(self, locpot_path):
            poscar_path = './CONTCAR' if exists('./CONTCAR') and getsize('./CONTCAR') > 0 else './POSCAR'
            self.lv, self.coord, self.atomtypes, self.atomnums = self._parse_poscar(poscar_path)
            dos_parser = SpinAwareDosParser(join(self.filepath, 'DOSCAR'))
            self.energies, self.ef = dos_parser.energies, dos_parser.ef
            self.is_polarized = dos_parser.is_polarized
            
            num_cols = dos_parser.spin_up_dos.shape[2]
            mapping = {
                3: ['s', 'p', 'd'],
                6: ['s_up','s_down','p_up','p_down','d_up','d_down'],
                9: ['s', 'py', 'pz', 'px', 'dxy', 'dyz', 'dz2', 'dxz', 'dx2-y2'],
                18: ['s_up', 's_down', 'py_up', 'py_down', 'pz_up', 'pz_down', 'px_up', 'px_down', 'dxy_up', 'dxy_down', 'dyz_up', 'dyz_down', 'dz2_up', 'dz2_down', 'dxz_up', 'dxz_down', 'dx2-y2_up', 'dx2-y2_down'],
                32: ['s_up', 's_down', 'py_up', 'py_down', 'pz_up', 'pz_down', 'px_up', 'px_down', 'dxy_up', 'dxy_down', 'dyz_up', 'dyz_down', 'dz2_up', 'dz2_down', 'dxz_up', 'dxz_down', 'dx2-y2_up', 'dx2-y2_down', 'fy3x2_up', 'fy3x2_down', 'fxyz_up', 'fxyz_down', 'fyz2_up', 'fyz2_down', 'fz3_up', 'fz3_down', 'fxz2_up', 'fxz2_down', 'fzx2_up', 'fzx2_down', 'fx3_up', 'fx3_down']
            }
            self.orbitals = mapping.get(num_cols, [])
            
            current_global = 1
            self.vesta_label_map = {}
            for idx, t in enumerate(self.atomtypes):
                for n_rel in range(1, self.atomnums[idx] + 1):
                    self.vesta_label_map[current_global] = f"{t}{n_rel}"
                    current_global += 1

            lpt_mgr = LocpotManager(self.filepath, ispin=2 if self.is_polarized else 1)
            locpot_raw = cp.array(lpt_mgr.get_data(), dtype=cp.float32)
            
            if self.is_polarized:
                v_total, v_diff = locpot_raw[0], locpot_raw[1]
                self.locpot_gpu = cp.stack([v_total + 0.5 * v_diff, v_total - 0.5 * v_diff])
            else:
                self.locpot_gpu = locpot_raw
                
            self.inv_lv_gpu = cp.array(inv(self.lv), dtype=cp.float32)
            self.locpot_dims_gpu = cp.array(self.locpot_gpu.shape[-3:], dtype=cp.float32)
            self.num_total_atoms = sum(self.atomnums)
            self.dos_up_gpu = cp.array(dos_parser.get_dos_for_simulator(spin='up'), dtype=cp.float32)
            self.dos_up_collapsed = cp.ascontiguousarray(cp.sum(self.dos_up_gpu, axis=2))
            if self.is_polarized:
                self.dos_dn_gpu = cp.array(dos_parser.get_dos_for_simulator(spin='down'), dtype=cp.float32)
                self.dos_dn_collapsed = cp.ascontiguousarray(cp.sum(self.dos_dn_gpu, axis=2))
            else:
                self.dos_dn_gpu = None
                self.dos_dn_collapsed = None
            
            inv_lv = inv(self.lv)
            frac_coords = np.dot(self.coord, inv_lv)
            z_filter_mask = frac_coords[:, 2] < 0.9
            self.z_highest_atom = np.max(self.coord[z_filter_mask, 2])
            
            coords, idx_list = [], []; base_idx = np.arange(len(self.coord))
            for i in range(-self.unit_cell_num, self.unit_cell_num + 1):
                for j in range(-self.unit_cell_num, self.unit_cell_num + 1):
                    coords.append(self.coord + self.lv[0] * i + self.lv[1] * j); idx_list.append(base_idx)
            self.periodic_coord_gpu = cp.array(np.concatenate(coords), dtype=cp.float32)
            self.atom_indices_periodic_gpu = cp.array(np.concatenate(idx_list))
            self.map_mat_gpu = cp.zeros((self.num_total_atoms, len(self.atom_indices_periodic_gpu)), dtype=cp.float32)
            self.map_mat_gpu[self.atom_indices_periodic_gpu, cp.arange(len(self.atom_indices_periodic_gpu))] = 1.0

    def _calculate_ldos_at_points_gpu(self, tip_positions, emin, emax, use_energy_decay=False, preserve_orbitals=False, global_bias=None):
        estart, eend = np.searchsorted(self.energies, emin), np.searchsorted(self.energies, emax, side='right')
        energy_indices = cp.arange(estart, eend); calc_energies_gpu = cp.array(self.energies[estart:eend], dtype=cp.float32)
        num_pts, num_e = tip_positions.shape[0], len(calc_energies_gpu)
        tip_pos_gpu = cp.array(tip_positions, dtype=cp.float32); frac_coords = cp.dot(tip_pos_gpu, self.inv_lv_gpu)
        
        wrapped_frac_coords = cp.empty_like(frac_coords)
        wrapped_frac_coords[:, :2] = frac_coords[:, :2] % 1.0
        wrapped_frac_coords[:, 2] = frac_coords[:, 2]
        
        lv_gpu = cp.array(self.lv, dtype=cp.float32)
        wrapped_tip_pos_gpu = cp.dot(wrapped_frac_coords, lv_gpu)
        
        grid_indices = (frac_coords % 1.0).T * self.locpot_dims_gpu[:, None]
        dists = cp.sqrt(cp.sum((self.periodic_coord_gpu[:, None, :] - wrapped_tip_pos_gpu[None, :, :])**2, axis=2))

        def _compute_channel(pot, dos_gpu, dos_collapsed):
            phi_local = cp_ndimage.map_coordinates(pot, grid_indices, order=1, mode='wrap') - self.ef
            if not preserve_orbitals:
                output_ldos = cp.zeros((num_pts, num_e), dtype=cp.float32)
                dos_periodic = dos_collapsed[self.atom_indices_periodic_gpu, :]
                dos_active = dos_periodic[:, energy_indices]
                if use_energy_decay:
                    bias_v = cp.array(global_bias if global_bias is not None else (emax - emin), dtype=cp.float32)
                    for e_idx in range(num_e):
                        K = gpu_chen_tunneling_factor(bias_v, calc_energies_gpu[e_idx], phi_local)
                        sf = cp.exp(-1.0 * dists * K[None, :] * 1e-10)
                        output_ldos[:, e_idx] = cp.dot(sf.T, dos_active[:, e_idx])
                else:
                    kappa = 0.512 * cp.sqrt(cp.maximum(0.1, phi_local))
                    sf = cp.exp(-2.0 * kappa[None, :] * dists)
                    output_ldos = cp.dot(sf.T, dos_active)
            else:
                output_ldos = cp.zeros((num_pts, num_e, self.num_total_atoms, dos_gpu.shape[2]), dtype=cp.float32)
                if use_energy_decay:
                    bias_v = cp.array(global_bias if global_bias is not None else (emax - emin), dtype=cp.float32)
                    for e_idx in range(num_e):
                        K = gpu_chen_tunneling_factor(bias_v, calc_energies_gpu[e_idx], phi_local)
                        sf = cp.exp(-1.0 * dists * K[None, :] * 1e-10)
                        w_atom = cp.dot(self.map_mat_gpu, sf)
                        output_ldos[:, e_idx, :, :] = w_atom.T[:, :, None] * dos_gpu[:, energy_indices[e_idx], :][None, :, :]
                else:
                    kappa = 0.512 * cp.sqrt(cp.maximum(0.1, phi_local))
                    sf = cp.exp(-2.0 * kappa[None, :] * dists)
                    w_atom = cp.dot(self.map_mat_gpu, sf)
                    output_ldos = w_atom.T[:, None, :, None] * dos_gpu[:, energy_indices, :][None, :, :, :]
            return output_ldos

        if self.is_polarized:
            return _compute_channel(self.locpot_gpu[0], self.dos_up_gpu, self.dos_up_collapsed), _compute_channel(self.locpot_gpu[1], self.dos_dn_gpu, self.dos_dn_collapsed), calc_energies_gpu
        return _compute_channel(self.locpot_gpu, self.dos_up_gpu, self.dos_up_collapsed), None, calc_energies_gpu

class Interactive_STM_Simulator(Unified_STM_Simulator):
    def __init__(self, filepath, erange, ldos_height, cmap_topo):
        super().__init__(filepath)
        self.parse_vasp_outputs("LOCPOT")
        self.p1, self.p2 = np.array([0.0, 0.0]), self.lv[0, :2] + self.lv[1, :2]
        self.erange, self.ldos_height, self.cmap_topo = list(erange), ldos_height, cmap_topo
        self.npts = 72; self.is_running, self.normalize, self.show_mag = False, False, False
        self.show_atoms, self.show_unit_cell = True, False
        self.use_decay_topo, self.use_decay_ldos = True, True; self.display_cells = 1
        self.mode = 'Single Point'
        self.show_decomp, self.show_dcmp_norm = False, False
        
        self.m_colors = ['#1f77b4', '#2ca02c', '#9467bd', '#00ced1', '#e377c2', '#17becf', '#bcbd22', '#7f7f7f', '#8c564b', '#d62728']
        self.marker_ratios = [0.25, 0.75]
        self.marker_coords = [[self.lv[0,0]*0.2, self.lv[1,1]*0.2], [self.lv[0,0]*0.8, self.lv[1,1]*0.8]]
        self.active_obj = None; self.active_marker_idx = 0
        self.plot_level = 0; self.active_element, self.active_atom = None, None
        self._type_color_map = {'Au': 'orange', 'N': 'blue', 'C': 'brown', 'H': 'grey'}

        self.cached_p1, self.cached_p2 = None, None
        self.cached_emin, self.cached_emax = None, None
        self.cached_d_topo_line, self.cached_d_topo_map, self.cached_d_ldos = None, None, None
        self.cached_bias_energy_line, self.cached_bias_energy_map, self.cached_nepts = None, None, None
        self.cached_ld_up, self.cached_ld_dn, self.cached_eg = None, None, None
        self.cached_marker_coords, self.cached_spec_ldos = None, None

    def run_interactive(self, grid_res=64, topo_bias=0.2, topo_height=2.5, ldos_bias_sign='neg', use_decay_topo=True, use_decay_ldos=True):
        self.ldos_bias_sign, self.use_decay_topo, self.use_decay_ldos = ldos_bias_sign, use_decay_topo, use_decay_ldos
        self.global_topo_bias = topo_bias
        self.topo_height = topo_height
        print("\n--- Phase 1: Global Topography Pre-Calculation ---")
        grid_xy = (np.meshgrid(np.linspace(0,1,grid_res), np.linspace(0,1,grid_res))[0].ravel()[:, None] * self.lv[0, :2]) + (np.meshgrid(np.linspace(0,1,grid_res), np.linspace(0,1,grid_res))[1].ravel()[:, None] * self.lv[1, :2])
        grid_xy_gpu = cp.array(grid_xy, dtype=cp.float32); z_fixed = cp.full(grid_xy_gpu.shape[0], self.z_highest_atom + topo_height, dtype=cp.float32)
        t_emin, t_emax = sorted([0.0, topo_bias])
        ld_up, ld_dn, init_engs = self._calculate_ldos_at_points_gpu(cp.hstack([grid_xy_gpu, z_fixed[:, None]]), t_emin, t_emax, use_energy_decay=self.use_decay_topo, preserve_orbitals=False)
        target_setp = cp.max(gpu_simpson(ld_up + ld_dn if ld_dn is not None else ld_up, init_engs))
        print(f"[*] Global Setpoint LDOS: {float(target_setp):.6e}")
        z_map_gpu = self._converge_tip_height(z_fixed, grid_xy_gpu, t_emin, t_emax, target_setp, use_decay=self.use_decay_topo)
        self.current_z_map, self.grid_xy = cp.asnumpy(z_map_gpu), grid_xy
        self.global_z_map = self.current_z_map.copy()
        self.grid_xy_gpu = grid_xy_gpu
        self.fig = plt.figure(figsize=(20, 14))
        self._build_ui()
        plt.show()

    def _get_partitions(self, f_ldos_raw):
            partitions = []
            if f_ldos_raw is not None:
                if not self.show_decomp or self.plot_level == 0:
                    partitions.append(("Total", np.sum(f_ldos_raw, axis=(2, 3))))
                elif self.plot_level == 1:
                    atom_types_exp = np.repeat(self.atomtypes, self.atomnums)
                    au_idx = [i for i, t in enumerate(atom_types_exp) if t == 'Au']
                    mol_idx = [i for i, t in enumerate(atom_types_exp) if t != 'Au']
                    au_data = np.sum(f_ldos_raw[:, :, au_idx, :], axis=(2, 3)) if au_idx else np.zeros_like(f_ldos_raw[:, :, 0, 0])
                    mol_data = np.sum(f_ldos_raw[:, :, mol_idx, :], axis=(2, 3)) if mol_idx else np.zeros_like(f_ldos_raw[:, :, 0, 0])
                    partitions.extend([("Au", au_data), ("Molecule", mol_data)])
                elif self.plot_level == 2:
                    atom_types_exp = np.repeat(self.atomtypes, self.atomnums)
                    for t in self.atomtypes:
                        t_idx = [i for i, x in enumerate(atom_types_exp) if x == t]
                        t_data = np.sum(f_ldos_raw[:, :, t_idx, :], axis=(2, 3)) if t_idx else np.zeros_like(f_ldos_raw[:, :, 0, 0])
                        partitions.append((t, t_data))
                elif self.plot_level >= 3:
                    atom_types_exp = np.repeat(self.atomtypes, self.atomnums)
                    if getattr(self, 'active_element', None) is not None:
                        e_idx = [i for i, x in enumerate(atom_types_exp) if x == self.active_element]
                        for col_idx, orb in enumerate(self.orbitals):
                            if e_idx:
                                D = cp.sum(self.dos_up_gpu[e_idx, :, col_idx], axis=0)
                                dD = cp.gradient(D)
                                d2D = cp.gradient(dD)
                                if not (float(cp.max(cp.abs(D))) < 1e-5 and float(cp.max(cp.abs(dD))) < 1e-5 and float(cp.max(cp.abs(d2D))) < 1e-5):
                                    orb_data = np.sum(f_ldos_raw[:, :, e_idx, col_idx], axis=2)
                                    partitions.append((f"{self.active_element} {orb}", orb_data))
            return partitions

    def _build_ui(self):
        self.fig.clf()
        ax_radio = plt.axes([0.02, 0.90, 0.1, 0.08], facecolor='lightgray')
        self.ui_radio = RadioButtons(ax_radio, ('Single Point', 'Line', 'Map'), active=['Single Point', 'Line', 'Map'].index(self.mode))
        self.ui_radio.on_clicked(self._on_mode_change)

        self.btn_run = Button(plt.axes([0.02, 0.02, 0.05, 0.06]), 'RUN', color='lightgray', hovercolor='lime')
        self.chk = CheckButtons(plt.axes([0.08, 0.02, 0.12, 0.08]), ['Atoms', 'Decay', 'Norm', 'Mag', 'Cell', 'Decomp', 'Dcmp Norm'], [self.show_atoms, self.use_decay_ldos, self.normalize, self.show_mag, self.show_unit_cell, self.show_decomp, self.show_dcmp_norm])
        self.s_cell = Slider(plt.axes([0.22, 0.02, 0.1, 0.03]), 'Cells', 0, 4, valinit=self.display_cells, valstep=1)
        self.s_emin = Slider(plt.axes([0.50, 0.05, 0.20, 0.02]), 'E Min', -5.0, 5.0, valinit=self.erange[0])
        self.s_emax = Slider(plt.axes([0.50, 0.02, 0.20, 0.02]), 'E Max', -5.0, 5.0, valinit=self.erange[1])

        if self.mode == 'Single Point':
            self.gs = gridspec.GridSpec(1, 2, width_ratios=[1, 1], wspace=0.25)
            self.ax_map, self.ax_spec = self.fig.add_subplot(self.gs[0, 0]), self.fig.add_subplot(self.gs[0, 1])
            self.s_num_marks = Slider(plt.axes([0.35, 0.02, 0.1, 0.03]), 'Points', 1, 10, valinit=len(self.marker_coords), valstep=1)
            self.marks = self.ax_map.scatter([], [], s=150, edgecolors='black', zorder=15, picker=5)
            self.s_num_marks.on_changed(self._on_ui_change)

        elif self.mode == 'Line':
            self.gs = gridspec.GridSpec(3, 2, height_ratios=[2.5, 1, 0.25], hspace=0.35, wspace=0.25)
            self.ax_map, self.ax_prof, self.ax_spec = self.fig.add_subplot(self.gs[0, 0]), self.fig.add_subplot(self.gs[1, 0]), self.fig.add_subplot(self.gs[1, 1])
            lgs = gridspec.GridSpecFromSubplotSpec(1, 4, subplot_spec=self.gs[0, 1], width_ratios=[0.08, 2, 0.03, 0.05], wspace=0.0)
            self.ax_stripe, self.ax_ldos, self.cax = self.fig.add_subplot(lgs[0]), self.fig.add_subplot(lgs[1]), self.fig.add_subplot(lgs[3])
            self.line_art, = self.ax_map.plot([], [], 'r--', lw=2.5, zorder=5)
            self.ends = self.ax_map.scatter([], [], c='white', edgecolors='red', s=100, zorder=10, picker=5)
            self.marks = self.ax_map.scatter([], [], s=150, edgecolors='black', zorder=15, picker=5)
            self.s_num_marks = Slider(plt.axes([0.35, 0.02, 0.1, 0.03]), 'Points', 1, 10, valinit=len(self.marker_ratios), valstep=1)
            self.s_num_marks.on_changed(self._on_ui_change)

        elif self.mode == 'Map':
            self.gs = gridspec.GridSpec(2, 3, height_ratios=[2.5, 1], width_ratios=[1, 1, 1.2], hspace=0.35, wspace=0.25)
            self.ax_map_global, self.ax_map, self.ax_spec = self.fig.add_subplot(self.gs[0, 0]), self.fig.add_subplot(self.gs[0, 1]), self.fig.add_subplot(self.gs[0, 2])
            self.map_axes = []
            self.marks = self.ax_map.scatter([], [], s=150, edgecolors='black', zorder=15, picker=5)
            self.s_num_marks = Slider(plt.axes([0.35, 0.02, 0.1, 0.03]), 'Points', 1, 10, valinit=len(self.marker_coords), valstep=1)
            self.s_nepts = Slider(plt.axes([0.75, 0.035, 0.15, 0.02]), 'E Pts', 1, 20, valinit=5 if self.cached_nepts is None else self.cached_nepts, valstep=1)
            self.s_num_marks.on_changed(self._on_ui_change); self.s_nepts.on_changed(self._on_ui_change)

        self.fig.canvas.mpl_connect('pick_event', self._on_pick)
        self.fig.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.fig.canvas.mpl_connect('button_release_event', self._on_rel)
        self.fig.canvas.mpl_connect('button_press_event', self._on_press)
        self.btn_run.on_clicked(self._toggle_run); self.chk.on_clicked(self._on_ui_change)
        for s in [self.s_cell, self.s_emin, self.s_emax]: s.on_changed(self._on_ui_change)
        
        self.cached_emin, self.cached_emax = None, None 
        self._update_all(full_refresh=True)

    def _on_mode_change(self, label):
        self.mode = label; self.active_marker_idx = 0; self.plot_level = 0
        self._build_ui()

    def _toggle_run(self, event):
        self.is_running = not self.is_running
        self.btn_run.label.set_text('STOP' if self.is_running else 'RUN')
        if self.is_running: self._update_all()

    def _update_all(self, full_refresh=False):
        if full_refresh:
            self.ax_map.clear(); n = int(self.s_cell.val)
            if self.mode == 'Map': self.ax_map_global.clear()
            t_ax = self.ax_map_global if self.mode == 'Map' else self.ax_map
            z_data = self.global_z_map #if self.mode == 'Map' else self.current_z_map
            
            for i in range(-n, n + 1):
                for j in range(-n, n + 1):
                    off = i * self.lv[0, :2] + j * self.lv[1, :2]
                    t_ax.tricontourf(self.grid_xy[:, 0] + off[0], self.grid_xy[:, 1] + off[1], z_data, levels=60, cmap=self.cmap_topo, zorder=1)
                    if self.mode == 'Map':
                        self.ax_map.tricontourf(self.grid_xy[:, 0] + off[0], self.grid_xy[:, 1] + off[1], self.current_z_map, levels=60, cmap=self.cmap_topo, zorder=1)
                    if self.show_atoms:
                        tr = np.repeat(self.atomtypes, self.atomnums)
                        for t_idx, t_name in enumerate(self.atomtypes):
                            m = (tr == t_name)
                            t_ax.scatter(self.coord[m, 0] + off[0], self.coord[m, 1] + off[1], s=10, color=plt.cm.tab10(t_idx/10), alpha=0.3, zorder=2)
                            if self.mode == 'Map':
                                self.ax_map.scatter(self.coord[m, 0] + off[0], self.coord[m, 1] + off[1], s=10, color=plt.cm.tab10(t_idx/10), alpha=0.3, zorder=2)
            if self.show_unit_cell:
                v0, v1, v2, v3 = np.array([0,0]), self.lv[0, :2], self.lv[0, :2] + self.lv[1, :2], self.lv[1, :2]
                cell_pts = np.array([v0, v1, v2, v3, v0])
                t_ax.plot(cell_pts[:, 0], cell_pts[:, 1], color='cyan', lw=2.0, ls='-', zorder=4, label='Unit Cell')
                if self.mode == 'Map': self.ax_map.plot(cell_pts[:, 0], cell_pts[:, 1], color='cyan', lw=2.0, ls='-', zorder=4, label='Unit Cell')
            
            t_ax.set_aspect('equal')
            t_ax.set_title(f"Global Topo | Bias: {self.global_topo_bias} V | Height: {self.topo_height} Å")
            if self.mode == 'Map':
                self.ax_map.set_aspect('equal')
                self.ax_map.set_title(f"Map Topo | Bias: {self.s_emin.val if str(self.ldos_bias_sign).lower() in ['neg', '-', 'negative'] else self.s_emax.val} V | Height: {self.ldos_height} Å")
            if self.mode == 'Line':
                self.ax_map.add_line(self.line_art); self.ax_map.add_collection(self.ends)
            self.ax_map.add_collection(self.marks)

        if self.mode == 'Line':
            v = self.p2 - self.p1; p_len = norm(v); p_dist = np.linspace(0, p_len, self.npts); p_xy = np.array([self.p1 + r * v for r in np.linspace(0, 1, self.npts)])
            self.line_art.set_data([self.p1[0], self.p2[0]], [self.p1[1], self.p2[1]]); self.ends.set_offsets([self.p1, self.p2])
            self.marks.set_offsets(np.array([self.p1 + r * v for r in self.marker_ratios])); self.marks.set_facecolors(self.m_colors[:len(self.marker_ratios)])
        else:
            self.marks.set_offsets(self.marker_coords); self.marks.set_facecolors(self.m_colors[:len(self.marker_coords)])

        if not self.is_running: self.fig.canvas.draw_idle(); return

        bias_e = self.s_emin.val if str(self.ldos_bias_sign).lower() in ['neg', '-', 'negative'] else self.s_emax.val
        nepts = int(self.s_nepts.val) if hasattr(self, 's_nepts') else None
        needs_topo = ((self.mode == 'Line' and (self.cached_p1 is None or not np.array_equal(self.p1, self.cached_p1) or not np.array_equal(self.p2, self.cached_p2) or self.cached_bias_energy_line != bias_e or self.cached_d_topo_line != self.use_decay_topo)) or 
                      (self.mode == 'Map' and (self.cached_bias_energy_map != bias_e or self.cached_d_topo_map != self.use_decay_topo)))
        needs_ldos = (needs_topo or self.cached_emin != self.s_emin.val or self.cached_emax != self.s_emax.val or self.cached_d_ldos != self.use_decay_ldos or (self.mode == 'Map' and self.cached_nepts != nepts))
        needs_spec = (needs_ldos or (self.mode in ['Single Point', 'Map'] and (self.cached_marker_coords is None or not np.array_equal(self.marker_coords, self.cached_marker_coords))))

        if needs_topo:
            if self.mode == 'Line':
                l_emin, l_emax = sorted([0.0, bias_e]); p_xy_gpu = cp.array(p_xy, dtype=cp.float32)
                ld_up, ld_dn, l_engs = self._calculate_ldos_at_points_gpu(cp.hstack([p_xy_gpu, cp.full((self.npts,1), self.z_highest_atom + self.ldos_height, dtype=cp.float32)]), l_emin, l_emax, use_energy_decay=self.use_decay_topo, preserve_orbitals=False)
                target = cp.max(gpu_simpson(ld_up + ld_dn if ld_dn is not None else ld_up, l_engs))
                print(f"[*] Path Setpoint LDOS: {float(target):.6e}")
                z_line = self._converge_tip_height(cp.full(self.npts, self.z_highest_atom + self.ldos_height, dtype=cp.float32), p_xy_gpu, l_emin, l_emax, target, use_decay=self.use_decay_topo)
                self.current_z_line = cp.asnumpy(z_line)
                self.cached_p1, self.cached_p2, self.cached_bias_energy_line, self.cached_d_topo_line = self.p1.copy(), self.p2.copy(), bias_e, self.use_decay_topo
            elif self.mode == 'Map':
                t_emin, t_emax = sorted([0.0, bias_e])
                z_fixed = cp.full(self.grid_xy_gpu.shape[0], self.z_highest_atom + self.ldos_height, dtype=cp.float32)
                ld_up, ld_dn, init_engs = self._calculate_ldos_at_points_gpu(cp.hstack([self.grid_xy_gpu, z_fixed[:, None]]), t_emin, t_emax, use_energy_decay=self.use_decay_topo, preserve_orbitals=False)
                target_setp = cp.max(gpu_simpson(ld_up + ld_dn if ld_dn is not None else ld_up, init_engs))
                print(f"[*] Map Local Setpoint LDOS: {float(target_setp):.6e}")
                z_map_gpu = self._converge_tip_height(z_fixed, self.grid_xy_gpu, t_emin, t_emax, target_setp, use_decay=self.use_decay_topo)
                self.current_z_map = cp.asnumpy(z_map_gpu)
                self.cached_bias_energy_map, self.cached_d_topo_map = bias_e, self.use_decay_topo
                
                self.ax_map.clear(); n = int(self.s_cell.val)
                for i in range(-n, n + 1):
                    for j in range(-n, n + 1):
                        off = i * self.lv[0, :2] + j * self.lv[1, :2]
                        self.ax_map.tricontourf(self.grid_xy[:, 0] + off[0], self.grid_xy[:, 1] + off[1], self.current_z_map, levels=60, cmap=self.cmap_topo, zorder=1)
                        if self.show_atoms:
                            tr = np.repeat(self.atomtypes, self.atomnums)
                            for t_idx, t_name in enumerate(self.atomtypes):
                                m = (tr == t_name)
                                self.ax_map.scatter(self.coord[m, 0] + off[0], self.coord[m, 1] + off[1], s=10, color=plt.cm.tab10(t_idx/10), alpha=0.3, zorder=2)
                if self.show_unit_cell:
                    v0, v1, v2, v3 = np.array([0,0]), self.lv[0, :2], self.lv[0, :2] + self.lv[1, :2], self.lv[1, :2]
                    cell_pts = np.array([v0, v1, v2, v3, v0])
                    self.ax_map.plot(cell_pts[:, 0], cell_pts[:, 1], color='cyan', lw=2.0, ls='-', zorder=4, label='Unit Cell')
                self.ax_map.set_aspect('equal')
                self.ax_map.set_title(f"Map Topo | Bias: {bias_e} V | Height: {self.ldos_height} Å")
                self.ax_map.add_collection(self.marks)

        if needs_ldos:
            if self.mode == 'Line':
                ld_up, ld_dn, eg = self._calculate_ldos_at_points_gpu(np.hstack([p_xy, self.current_z_line[:, None]]), self.s_emin.val, self.s_emax.val, use_energy_decay=self.use_decay_ldos, preserve_orbitals=True)
            elif self.mode == 'Map':
                self.map_e_targets = np.linspace(self.s_emin.val, self.s_emax.val, nepts)
                eg = cp.array(self.energies[np.searchsorted(self.energies, self.s_emin.val):np.searchsorted(self.energies, self.s_emax.val, side='right')])
                ld_up_list, ld_dn_list = [], []
                grid_z = cp.hstack([self.grid_xy_gpu, cp.array(self.current_z_map)[:, None]])
                for t_e in self.map_e_targets:
                    e_idx = np.searchsorted(self.energies, t_e)
                    if e_idx >= len(self.energies): e_idx = len(self.energies) - 1
                    t_up, t_dn, _ = self._calculate_ldos_at_points_gpu(grid_z, self.energies[e_idx], self.energies[min(e_idx+1, len(self.energies)-1)] + 1e-6, use_energy_decay=self.use_decay_ldos, preserve_orbitals=True, global_bias=abs(self.s_emax.val - self.s_emin.val))
                    ld_up_list.append(t_up[:, 0:1])
                    if t_dn is not None: ld_dn_list.append(t_dn[:, 0:1])
                ld_up = cp.concatenate(ld_up_list, axis=1)
                ld_dn = cp.concatenate(ld_dn_list, axis=1) if ld_dn_list else None
            else:
                eg = cp.array(self.energies[np.searchsorted(self.energies, self.s_emin.val):np.searchsorted(self.energies, self.s_emax.val, side='right')])
                ld_up, ld_dn = None, None
            
            if ld_up is not None:
                self.cached_ld_up, self.cached_ld_dn = cp.asnumpy(ld_up), (cp.asnumpy(ld_dn) if ld_dn is not None else None)
            self.cached_eg = cp.asnumpy(eg)
            self.cached_emin, self.cached_emax, self.cached_d_ldos, self.cached_nepts = self.s_emin.val, self.s_emax.val, self.use_decay_ldos, nepts

        if needs_spec and self.mode in ['Single Point', 'Map']:
            m_coords = np.array(self.marker_coords)
            z_marks = []
            inv_lv_np = inv(self.lv)
            for pt in m_coords:
                pt_3d = np.array([pt[0], pt[1], 0.0])
                f_pt = np.dot(pt_3d, inv_lv_np)
                f_pt[:2] = f_pt[:2] % 1.0
                wrapped_pt = np.dot(f_pt, self.lv)
                dist_sq = (self.grid_xy[:, 0] - wrapped_pt[0])**2 + (self.grid_xy[:, 1] - wrapped_pt[1])**2
                z_marks.append(self.current_z_map[np.argmin(dist_sq)])
            z_marks = np.array(z_marks)
            
            pt_gpu = cp.array(np.hstack([m_coords, z_marks[:, None]]), dtype=cp.float32)
            s_up, s_dn, _ = self._calculate_ldos_at_points_gpu(pt_gpu, self.s_emin.val, self.s_emax.val, use_energy_decay=self.use_decay_ldos, preserve_orbitals=True)
            s_up_np, s_dn_np = cp.asnumpy(s_up), (cp.asnumpy(s_dn) if s_dn is not None else None)
            
            spec_ldos = (s_up_np - s_dn_np) if (self.show_mag and s_dn_np is not None) else (s_up_np + s_dn_np if s_dn_np is not None else s_up_np)
            self.cached_spec_ldos = spec_ldos
            self.cached_marker_coords = np.array(self.marker_coords).copy()
            
        if self.mode in ['Line', 'Map'] and self.cached_ld_up is not None:
            f_up, f_dn = self.cached_ld_up.copy(), (self.cached_ld_dn.copy() if self.cached_ld_dn is not None else None)
            f_ldos_raw = (f_up - f_dn) if (self.show_mag and f_dn is not None) else (f_up + f_dn if f_dn is not None else f_up)
            partitions = self._get_partitions(f_ldos_raw)
        else:
            f_ldos_raw = None
            partitions = []

        if self.mode == 'Line':
            active_idx = int(self.marker_ratios[min(self.active_marker_idx, len(self.marker_ratios)-1)] * (self.npts - 1))
            active_ldos = f_ldos_raw[active_idx] if f_ldos_raw is not None else np.zeros_like(self.cached_spec_ldos[0])
        else:
            active_idx = min(self.active_marker_idx, len(self.marker_coords)-1)
            active_ldos = self.cached_spec_ldos[active_idx]

        self.ax_spec.clear()
        def _orbit_base(orb):
            if orb.endswith('_up'): return orb[:-3]
            elif orb.endswith('_down'): return orb[:-5]
            return orb
        def _lighten_color(color, amount=0.3):
            c = mc.to_rgb(color)
            h, l, s = colorsys.rgb_to_hls(*c)
            return colorsys.hls_to_rgb(h, min(1, l + amount * (1 - l)), s)

        unique_bases = sorted(set(_orbit_base(o) for o in self.orbitals))
        styles = ['-', '--', ':', '-.'] + [(0, (3+i, 2)) for i in range(max(0, len(unique_bases)-4))]
        linestyle_map = dict(zip(unique_bases, styles))
        S = 0.25
        
        active_ldos_norm = active_ldos.copy()
        if self.normalize:
             total_for_norm = np.sum(active_ldos_norm, axis=(1, 2))
             norm_factor = (np.trapezoid(total_for_norm, x=self.cached_eg) + 1e-15)
             active_ldos_norm /= norm_factor

        if self.plot_level == 0:
            if self.mode == 'Line':
                for i, r in enumerate(self.marker_ratios):
                    idx = int(r * (self.npts - 1))
                    c_ldos = f_ldos_raw[idx].copy()
                    if self.normalize:
                        c_total = np.sum(c_ldos, axis=(1, 2))
                        c_ldos /= (np.trapezoid(c_total, x=self.cached_eg) + 1e-15)
                    t_y = np.sum(c_ldos, axis=(1, 2))
                    color = self.m_colors[i % len(self.m_colors)]
                    self.ax_spec.plot(self.cached_eg, t_y, color=color, lw=2.5, picker=True, pickradius=5, label=f'marker_{i}')
            else:
                for i, pt in enumerate(self.marker_coords):
                    c_ldos = self.cached_spec_ldos[i].copy()
                    if self.normalize:
                        c_total = np.sum(c_ldos, axis=(1, 2))
                        c_ldos /= (np.trapezoid(c_total, x=self.cached_eg) + 1e-15)
                    t_y = np.sum(c_ldos, axis=(1, 2))
                    color = self.m_colors[i % len(self.m_colors)]
                    self.ax_spec.plot(self.cached_eg, t_y, color=color, lw=2.5, picker=True, pickradius=5, label=f'marker_{i}')
            self.ax_spec.legend(loc='upper right', frameon=False)
        elif self.plot_level == 1:
            self.ax_spec.plot(self.cached_eg, np.sum(active_ldos_norm, axis=(1, 2)), color='black', lw=1.5, alpha=0.1, zorder=1)
            atom_types_exp = np.repeat(self.atomtypes, self.atomnums)
            au_idx = [i for i, t in enumerate(atom_types_exp) if t == 'Au']
            mol_idx = [i for i, t in enumerate(atom_types_exp) if t != 'Au']
            au_y = np.sum(active_ldos_norm[:, au_idx, :], axis=(1, 2)) if au_idx else np.zeros_like(self.cached_eg)
            mol_y = np.sum(active_ldos_norm[:, mol_idx, :], axis=(1, 2)) if mol_idx else np.zeros_like(self.cached_eg)
            self.ax_spec.plot(self.cached_eg, au_y, color='orange', lw=2, label='Au', picker=True, pickradius=5, zorder=2)
            self.ax_spec.plot(self.cached_eg, mol_y, color='black', lw=2, label='Molecule', picker=True, pickradius=5, zorder=2)
            self.ax_spec.legend([Line2D([0], [0], color='orange', lw=2), Line2D([0], [0], color='black', lw=2)], ['Au', 'Molecule'], title="Partition", loc='upper right', frameon=False)
        elif self.plot_level == 2:
            atom_types_exp = np.repeat(self.atomtypes, self.atomnums)
            for t in self.atomtypes:
                t_idx = [i for i, x in enumerate(atom_types_exp) if x == t]
                t_y = np.sum(active_ldos_norm[:, t_idx, :], axis=(1, 2)) if t_idx else np.zeros_like(self.cached_eg)
                self.ax_spec.plot(self.cached_eg, t_y, color=self._type_color_map.get(t, 'grey'), lw=2, label=t, picker=True, pickradius=3)
            self.ax_spec.legend([Line2D([0], [0], color=self._type_color_map.get(t, 'grey'), lw=2) for t in self.atomtypes], self.atomtypes, title="Atom Types", loc='upper right', frameon=False)
        elif self.plot_level == 3:
            atom_types_exp = np.repeat(self.atomtypes, self.atomnums)
            e_idx = [i for i, x in enumerate(atom_types_exp) if x == self.active_element]
            e_y = np.sum(active_ldos_norm[:, e_idx, :], axis=(1, 2)) if e_idx else np.zeros_like(self.cached_eg)
            self.ax_spec.plot(self.cached_eg, e_y, color=self._type_color_map.get(self.active_element, 'grey'), lw=2, alpha=0.15, zorder=1)
            for a_idx, label in self.vesta_label_map.items():
                if label.startswith(self.active_element):
                    y_sum = np.sum(active_ldos_norm[:, a_idx-1, :], axis=1)
                    self.ax_spec.plot(self.cached_eg, y_sum, color=self._type_color_map.get(self.active_element, 'grey'), lw=2, label=label, picker=True, pickradius=3, zorder=2)
            self.ax_spec.legend([Line2D([0], [0], color=self._type_color_map.get(t, 'grey'), lw=2) for t in self.atomtypes], self.atomtypes, title="Atom Types", loc='upper right', frameon=False)
        elif self.plot_level == 4:
            for a_idx, label in self.vesta_label_map.items():
                if label.startswith(self.active_element):
                    y_sum = np.sum(active_ldos_norm[:, a_idx-1, :], axis=1)
                    element_color = self._type_color_map.get(self.active_element, 'grey')
                    if a_idx == self.active_atom:
                        self.ax_spec.plot(self.cached_eg, y_sum, color=element_color, lw=2.5, alpha=1.0, zorder=5)
                        orb_artists = []
                        for orb in self.orbitals:
                            col_idx = self.orbitals.index(orb)
                            y_orb = active_ldos_norm[:, a_idx-1, col_idx]
                            ls = linestyle_map[_orbit_base(orb)]
                            p_color = _lighten_color(element_color, 0.3) if orb.endswith('_up') else element_color
                            o_line, = self.ax_spec.plot(self.cached_eg, y_orb, color=p_color, linestyle=ls, lw=1.2, label=f"{label} – {orb}", zorder=10)
                            orb_artists.append(o_line)
                        if hasattr(self, 'cursor'): self.cursor.remove()
                        self.cursor = mplcursors.cursor(orb_artists, hover=True)
                        self.cursor.connect("add", lambda sel: sel.annotation.set_text(sel.artist.get_label()))
                    else:
                        orig = mc.to_rgb(element_color)
                        lumi = 0.299*orig[0] + 0.587*orig[1] + 0.114*orig[2]
                        faded_color = (S * np.array(orig)) + ((1 - S) * lumi)
                        self.ax_spec.plot(self.cached_eg, y_sum, color=faded_color, lw=1.5, alpha=0.05, zorder=2)
            leg1 = self.ax_spec.legend([Line2D([0], [0], color=self._type_color_map.get(t, 'grey'), lw=2) for t in self.atomtypes], self.atomtypes, title="Atom Types", loc='upper right', frameon=False)
            self.ax_spec.add_artist(leg1)
            self.ax_spec.legend([Line2D([0], [0], color='black', linestyle=linestyle_map[b], lw=1.5) for b in unique_bases], unique_bases, title="Orbitals", loc='upper left', frameon=False)

        lines = self.ax_spec.get_lines()
        active_maxes = [np.max(l.get_ydata()) for l in lines if l.get_visible() and l.get_alpha() == 1.0]
        if active_maxes: self.ax_spec.set_ylim(0, max(active_maxes) * 1.1)
        
        p_dist_val = p_dist[active_idx] if self.mode == 'Line' else f"[{self.marker_coords[active_idx][0]:.1f}, {self.marker_coords[active_idx][1]:.1f}]"
        self.ax_spec.set(title=f"Partitioned LDOS (Marker {self.active_marker_idx}: {p_dist_val} Å)", xlabel="Energy (eV)")

        if self.mode == 'Map':
            if not hasattr(self, 'map_e_targets') or len(self.map_e_targets) != nepts or full_refresh:
                self.map_e_targets = np.linspace(self.s_emin.val, self.s_emax.val, nepts)
            for i, e_val in enumerate(self.map_e_targets):
                self.ax_spec.axvline(x=e_val, color='black', linestyle='-', lw=2, picker=5, label=f'emarker_{i}')

        if self.mode == 'Line':
            if hasattr(self, 'line_decomp_axes'): 
                for ax in self.line_decomp_axes:
                    if ax in self.fig.axes: ax.remove()
            self.ax_prof.clear()
            if getattr(self, 'ax_ldos', None) and self.ax_ldos in self.fig.axes: self.ax_ldos.remove(); self.ax_ldos = None
            if getattr(self, 'ax_stripe', None) and self.ax_stripe in self.fig.axes: self.ax_stripe.remove(); self.ax_stripe = None
            if getattr(self, 'cax', None) and self.cax in self.fig.axes: self.cax.remove(); self.cax = None
            import matplotlib.ticker as ticker
            self.line_decomp_axes = []
            
            processed_partitions = []
            global_vmax = 0.0
            for p_label, p_data in partitions:
                t_data = p_data.copy()
                if self.normalize: t_data /= (np.trapezoid(t_data, x=self.cached_eg, axis=1)[:, None] + 1e-15)
                processed_partitions.append((p_label, t_data))
                v_max = np.max(np.abs(t_data))
                if v_max > global_vmax: global_vmax = v_max
            if global_vmax == 0: global_vmax = 1e-15
            
            num_p = max(1, len(partitions))
            if self.show_dcmp_norm:
                w_ratios = [0.08 * num_p] + [2, 0.05 * num_p, 0.15 * num_p] * num_p
            else:
                w_ratios = [0.08 * num_p] + [2, 0.0, 0.0] * (num_p - 1) + [2, 0.05 * num_p, 0.0]
            lgs = gridspec.GridSpecFromSubplotSpec(1, 1 + num_p*3, subplot_spec=self.gs[0, 1], width_ratios=w_ratios, wspace=0.1)
            
            self.ax_stripe = self.fig.add_subplot(lgs[0])
            self.line_decomp_axes.append(self.ax_stripe)
            
            lc = LineCollection(np.array([np.array([np.zeros_like(p_dist), p_dist]).T[:-1], np.array([np.zeros_like(p_dist), p_dist]).T[1:]]).transpose(1, 0, 2), cmap=self.cmap_topo, norm=plt.Normalize(self.current_z_line.min(), self.current_z_line.max()), linewidth=40)
            lc.set_array(self.current_z_line[:-1]); self.ax_stripe.add_collection(lc)
            self.ax_stripe.set(xlim=(-0.1, 0.1), ylim=(0, p_len)); self.ax_stripe.set_xticks([])
            self.ax_prof.plot(p_dist, self.current_z_line, 'k-', lw=1.5); self.ax_prof.set(ylabel="Height (Å)", title="Tip Height", xlabel="Dist (Å)")
            
            for p_idx, (p_label, t_data) in enumerate(processed_partitions):
                ax_l = self.fig.add_subplot(lgs[1 + p_idx*3])
                cax_l = self.fig.add_subplot(lgs[2 + p_idx*3])
                self.line_decomp_axes.extend([ax_l, cax_l])
                
                v_max = np.max(np.abs(t_data)) if self.show_dcmp_norm else global_vmax
                if v_max == 0: v_max = 1e-15
                
                if self.show_mag and f_dn is not None:
                    mesh = ax_l.pcolormesh(self.cached_eg, p_dist, t_data, cmap='bwr', shading='auto', vmin=-v_max, vmax=v_max)
                else:
                    mesh = ax_l.pcolormesh(self.cached_eg, p_dist, t_data, cmap='jet', shading='auto', vmin=0, vmax=v_max)
                
                if self.plot_level >= 3:
                    ax_l.set_title(p_label.split()[-1], fontsize=10)
                else:
                    ax_l.set_title(f"LDOS: {p_label}", fontsize=10)
                ax_l.set_yticks([])
                
                if self.show_dcmp_norm or p_idx == num_p - 1:
                    cb = self.fig.colorbar(mesh, cax=cax_l)
                    exp = int(np.floor(np.log10(v_max)))
                    cb.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos, e=exp: f"{x / (10**e):.1f}"))
                    cax_l.set_title(f"1e{exp}", fontsize=10)
                else:
                    cax_l.axis('off')
                
                for i, r in enumerate(self.marker_ratios):
                    idx = int(r * (self.npts - 1)); color = self.m_colors[i % len(self.m_colors)]
                    if p_idx == 0: self.ax_prof.axvline(x=p_dist[idx], color=color, ls='--', lw=2, alpha=0.7, picker=5, label=f'marker_{i}')
                    ax_l.axhline(y=p_dist[idx], color=color, ls='--', lw=2, alpha=0.7, picker=5, label=f'marker_{i}')
                    if p_idx == 0: self.ax_stripe.axhline(y=p_dist[idx], color=color, ls='--', lw=2, alpha=0.7, picker=5, label=f'marker_{i}')

            if self.plot_level >= 3:
                ax_super = self.fig.add_subplot(self.gs[0, 1])
                ax_super.axis('off')
                ax_super.set_title(f"LDOS: {self.active_element}", fontsize=12, pad=20)
                self.line_decomp_axes.append(ax_super)

        elif self.mode == 'Map':
            num_p = max(1, len(partitions))
            h_topo = max(1.0, 2.5 - 0.5 * (num_p - 1))
            self.gs.set_height_ratios([h_topo, float(num_p)])

            if getattr(self, 'cax_list', None):
                for cax in self.cax_list:
                    if cax in self.fig.axes: cax.remove()
            self.cax_list = []

            if len(self.map_axes) != nepts * num_p or full_refresh:
                for ax in self.map_axes:
                    if ax in self.fig.axes: ax.remove()
                self.map_axes.clear()
                
                w_ratios = [1.0] * nepts + [0.08]
                sub_gs = gridspec.GridSpecFromSubplotSpec(num_p, nepts + 1, subplot_spec=self.gs[1, :], width_ratios=w_ratios, wspace=0.1, hspace=0.2)
                
                for r in range(num_p):
                    for c in range(nepts): 
                        self.map_axes.append(self.fig.add_subplot(sub_gs[r, c]))
                    if self.show_dcmp_norm:
                        self.cax_list.append(self.fig.add_subplot(sub_gs[r, nepts]))
                if not self.show_dcmp_norm:
                    self.cax_list.append(self.fig.add_subplot(sub_gs[:, nepts]))

            if not hasattr(self, 'map_e_targets') or len(self.map_e_targets) != nepts or full_refresh:
                self.map_e_targets = np.linspace(self.s_emin.val, self.s_emax.val, nepts)
            m_coords_np = np.array(self.marker_coords)

            processed_partitions = []
            global_vmax = 0.0
            for p_label, p_data in partitions:
                t_data = p_data.copy()
                if self.normalize:
                    s_idx = np.argsort(self.map_e_targets)
                    t_data /= (np.trapezoid(t_data[:, s_idx], x=self.map_e_targets[s_idx], axis=1)[:, None] + 1e-15)
                t_data = np.nan_to_num(t_data, nan=0.0, posinf=0.0, neginf=0.0)
                processed_partitions.append((p_label, t_data))
                v_max = np.max(np.abs(t_data))
                if v_max > global_vmax: global_vmax = v_max
            if global_vmax == 0: global_vmax = 1e-15
            
            import matplotlib.ticker as ticker
            for p_idx, (p_label, t_data) in enumerate(processed_partitions):
                v_max = np.max(np.abs(t_data)) if self.show_dcmp_norm else global_vmax
                if v_max == 0: v_max = 1e-15
                
                for i, target_e in enumerate(self.map_e_targets):
                    ax = self.map_axes[p_idx * nepts + i]
                    ax.clear()
                    slice_data = t_data[:, i]

                    for nx in range(2):
                        for ny in range(2):
                            off = nx * self.lv[0, :2] + ny * self.lv[1, :2]
                            if self.show_mag and f_dn is not None:
                                mesh = ax.tricontourf(self.grid_xy[:,0] + off[0], self.grid_xy[:,1] + off[1], slice_data, levels=40, cmap='bwr', vmin=-v_max, vmax=v_max)
                            else:
                                mesh = ax.tricontourf(self.grid_xy[:,0] + off[0], self.grid_xy[:,1] + off[1], slice_data, levels=40, cmap='jet', vmin=0, vmax=v_max)
                    
                    ax.scatter(m_coords_np[:, 0], m_coords_np[:, 1], color=self.m_colors[:len(m_coords_np)], s=30, edgecolors='white', zorder=5)
                    title_str = f"E = {target_e:.3f} eV" if p_idx == 0 else ""
                    if self.plot_level >= 3:
                        ylabel_str = p_label.split()[-1] if i == 0 else ""
                    else:
                        ylabel_str = p_label if i == 0 else ""
                    ax.set_title(title_str, fontsize=10)
                    if ylabel_str: ax.set_ylabel(ylabel_str, fontsize=10)
                    ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])

                if self.show_dcmp_norm:
                    cax = self.cax_list[p_idx]
                    cax.clear()
                    cb = self.fig.colorbar(mesh, cax=cax)
                    exp = int(np.floor(np.log10(v_max)))
                    cb.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos, e=exp: f"{x / (10**e):.1f}"))
                    cax.set_title(f"1e{exp}", fontsize=10)
                    
            if not self.show_dcmp_norm and len(processed_partitions) > 0:
                cax = self.cax_list[0]
                cax.clear()
                cb = self.fig.colorbar(mesh, cax=cax)
                exp = int(np.floor(np.log10(global_vmax)))
                cb.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos, e=exp: f"{x / (10**e):.1f}"))
                cax.set_title(f"1e{exp}", fontsize=10)

            if self.plot_level >= 3:
                ax_super = self.fig.add_subplot(self.gs[1, :])
                ax_super.axis('off')
                ax_super.set_title(f"LDOS: {self.active_element}", fontsize=12, pad=25)
                self.map_axes.append(ax_super)

        self.fig.canvas.draw_idle()

    def _redraw_map_slice(self, i):
        target_e = self.map_e_targets[i]
        f_up, f_dn = self.cached_ld_up.copy(), (self.cached_ld_dn.copy() if self.cached_ld_dn is not None else None)
        f_ldos_raw = (f_up - f_dn) if (self.show_mag and f_dn is not None) else (f_up + f_dn if f_dn is not None else f_up)
        partitions = self._get_partitions(f_ldos_raw)
        
        m_coords_np = np.array(self.marker_coords)
        nepts = len(self.map_e_targets)
        
        processed_partitions = []
        global_vmax = 0.0
        for p_label, p_data in partitions:
            t_data = p_data.copy()
            if self.normalize:
                s_idx = np.argsort(self.map_e_targets)
                t_data /= (np.trapezoid(t_data[:, s_idx], x=self.map_e_targets[s_idx], axis=1)[:, None] + 1e-15)
            t_data = np.nan_to_num(t_data, nan=0.0, posinf=0.0, neginf=0.0)
            processed_partitions.append((p_label, t_data))
            v_max = np.max(np.abs(t_data))
            if v_max > global_vmax: global_vmax = v_max
        if global_vmax == 0: global_vmax = 1e-15
        
        for p_idx, (p_label, t_data) in enumerate(processed_partitions):
            ax = self.map_axes[p_idx * nepts + i]
            ax.clear()
            v_max = np.max(np.abs(t_data)) if self.show_dcmp_norm else global_vmax
            if v_max == 0: v_max = 1e-15
            slice_data = t_data[:, i]

            for nx in range(2):
                for ny in range(2):
                    off = nx * self.lv[0, :2] + ny * self.lv[1, :2]
                    if self.show_mag and f_dn is not None:
                        ax.tricontourf(self.grid_xy[:,0] + off[0], self.grid_xy[:,1] + off[1], slice_data, levels=40, cmap='bwr', vmin=-v_max, vmax=v_max)
                    else:
                        ax.tricontourf(self.grid_xy[:,0] + off[0], self.grid_xy[:,1] + off[1], slice_data, levels=40, cmap='jet', vmin=0, vmax=v_max)
            
            ax.scatter(m_coords_np[:, 0], m_coords_np[:, 1], color=self.m_colors[:len(m_coords_np)], s=30, edgecolors='white', zorder=5)
            title_str = f"E = {target_e:.3f} eV" if p_idx == 0 else ""
            if self.plot_level >= 3:
                ylabel_str = p_label.split()[-1] if i == 0 else ""
            else:
                ylabel_str = p_label if i == 0 else ""
            ax.set_title(title_str, fontsize=10)
            if ylabel_str: ax.set_ylabel(ylabel_str, fontsize=10)
            ax.set_aspect('equal'); ax.set_xticks([]); ax.set_yticks([])

    def _on_pick(self, event):
        if event.artist == getattr(self, 'ends', None) and self.mode == 'Line': self.active_obj = ('end', event.ind[0])
        elif event.artist == getattr(self, 'marks', None): self.active_obj = ('mark_map', event.ind[0])
        elif getattr(event, 'mouseevent', None) and event.mouseevent.inaxes == self.ax_spec:
            label = event.artist.get_label()
            if label.startswith('emarker_'):
                if self.plot_level == 0:
                    self.active_obj = ('emarker', int(label.split('_')[1]))
                return
            if self.plot_level == 0:
                if label.startswith('marker_'): self.active_marker_idx = int(label.split('_')[1])
                self.plot_level = 1
            elif self.plot_level == 1: self.plot_level = 2
            elif self.plot_level == 2:
                self.active_element = label
                self.plot_level = 3
            elif self.plot_level == 3:
                self.active_atom = next(k for k, v in self.vesta_label_map.items() if v == label)
                self.active_element = self._get_element_by_index_helper(self.active_atom)
                self.plot_level = 4
            self._update_all()
        elif isinstance(event.artist, plt.Line2D) and event.artist.get_label().startswith('marker_'):
            self.active_obj = ('mark_dynamic', int(event.artist.get_label().split('_')[1]))
            self.active_marker_idx = int(event.artist.get_label().split('_')[1])
            self._update_all()

    def _on_press(self, event):
        if event.inaxes == self.ax_spec:
            if self.fig.canvas.manager.toolbar.mode == "" and not event.dblclick:
                hit = any(l.contains(event)[0] for l in self.ax_spec.get_lines() if l.get_picker())
                if not hit:
                    self.plot_level = max(0, self.plot_level - 1)
                    self._update_all()

    def _get_element_by_index_helper(self, a):
        curr = 0
        for idx, count in enumerate(self.atomnums):
            if a <= curr + count: return self.atomtypes[idx]
            curr += count
        return 'grey'

    def _on_motion(self, event):
        if self.active_obj is None or event.xdata is None: return
        t_obj, idx = self.active_obj
        p_len = norm(self.p2 - self.p1) if self.mode == 'Line' else 1.0
        
        if t_obj == 'end' and self.mode == 'Line':
            if idx == 0: self.p1 = np.array([event.xdata, event.ydata])
            else: self.p2 = np.array([event.xdata, event.ydata])
        elif t_obj == 'mark_map' and event.inaxes == self.ax_map:
            if self.mode == 'Line':
                v = self.p2 - self.p1; v_sq = np.dot(v, v)
                if v_sq > 1e-9: self.marker_ratios[idx] = np.clip(np.dot(np.array([event.xdata, event.ydata]) - self.p1, v) / v_sq, 0, 1)
            else:
                self.marker_coords[idx] = [event.xdata, event.ydata]
        elif t_obj == 'mark_dynamic' and self.mode == 'Line':
            if hasattr(self, 'ax_prof') and event.inaxes == self.ax_prof:
                if p_len > 1e-9: self.marker_ratios[idx] = np.clip(event.xdata / p_len, 0, 1)
            elif hasattr(self, 'ax_ldos') and event.inaxes in [self.ax_ldos, self.ax_stripe]:
                if p_len > 1e-9: self.marker_ratios[idx] = np.clip(event.ydata / p_len, 0, 1)
        elif t_obj == 'emarker' and event.inaxes == self.ax_spec:
            self.map_e_targets[idx] = np.clip(event.xdata, self.s_emin.val, self.s_emax.val)
            for line in self.ax_spec.get_lines():
                if line.get_label() == f'emarker_{idx}': line.set_xdata([self.map_e_targets[idx], self.map_e_targets[idx]])
            self.fig.canvas.draw_idle()
            return
        self._update_all()

    def _on_ui_change(self, val):
        states = self.chk.get_status()
        self.show_atoms, self.use_decay_ldos, self.normalize, self.show_mag, self.show_unit_cell, self.show_decomp, self.show_dcmp_norm = states[:7]
        self.display_cells = int(self.s_cell.val)
        
        new_count = int(self.s_num_marks.val)
        if self.mode == 'Line':
            if new_count != len(self.marker_ratios):
                if new_count > len(self.marker_ratios): self.marker_ratios = list(np.linspace(0.1, 0.9, new_count))
                else: self.marker_ratios = self.marker_ratios[:new_count]
        else:
            if new_count != len(self.marker_coords):
                if new_count > len(self.marker_coords):
                    for _ in range(new_count - len(self.marker_coords)):
                        self.marker_coords.append([self.lv[0,0]*0.5 + (np.random.rand()-0.5)*0.1, self.lv[1,1]*0.5 + (np.random.rand()-0.5)*0.1])
                else:
                    self.marker_coords = self.marker_coords[:new_count]
                self.cached_marker_coords = None
                
        self._update_all(full_refresh=True)

    def _on_rel(self, event):
        if self.active_obj is not None and self.active_obj[0] == 'emarker':
            idx = self.active_obj[1]
            target_e = self.map_e_targets[idx]
            e_idx = np.searchsorted(self.energies, target_e)
            if e_idx >= len(self.energies): e_idx = len(self.energies) - 1
            grid_z = cp.hstack([self.grid_xy_gpu, cp.array(self.current_z_map)[:, None]])
            t_up, t_dn, _ = self._calculate_ldos_at_points_gpu(grid_z, self.energies[e_idx], self.energies[min(e_idx+1, len(self.energies)-1)] + 1e-6, use_energy_decay=self.use_decay_ldos, preserve_orbitals=True, global_bias=abs(self.s_emax.val - self.s_emin.val))
            self.cached_ld_up[:, idx:idx+1] = cp.asnumpy(t_up[:, 0:1])
            if t_dn is not None and self.cached_ld_dn is not None:
                self.cached_ld_dn[:, idx:idx+1] = cp.asnumpy(t_dn[:, 0:1])
            self._redraw_map_slice(idx)
            self.fig.canvas.draw_idle()
        self.active_obj = None

if __name__ == "__main__":
    v_dir = r'C:/dir'
    # Initialized without hardcoded path or marker indices
    sim = Interactive_STM_Simulator(v_dir, [-2.525, -1.3], 1.3, LinearSegmentedColormap.from_list("t", ["black", "firebrick", "yellow"]))
    sim.run_interactive(grid_res=64, topo_bias=0.2, topo_height=2.5, ldos_bias_sign='neg', use_decay_topo=True)
