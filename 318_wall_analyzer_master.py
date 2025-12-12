#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RC Wall Design Tool – ACI 318-19 with FEM Analysis (Revised)
=============================================================

Key Features:
- Engineering title block on every page (TOP of page)
- Full mathematical derivations with ACI 318-19 citations
- Combined in-plane AND out-of-plane load analysis
- Proper 0.80Po cutoff with full Pn curve plotted
- Total displacement magnitude calculation
- Complete FEM solver (plane stress + plate bending)

Units: in, lb, psi (output in kip, kip-ft)
Origin: Bottom-left (0,0)
"""

import math
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Image, PageBreak, KeepTogether,
                                Flowable, Frame, PageTemplate)
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

# Try to import sparse matrices for memory-efficient FEM
try:
    from scipy.sparse import lil_matrix, csr_matrix
    from scipy.sparse.linalg import spsolve
    _SCIPY_AVAILABLE = True
except Exception:
    _SCIPY_AVAILABLE = False

# =============================================================================
# CONSTANTS & MATERIAL PROPERTIES (ACI 318-19)
# =============================================================================
BAR_AREAS_IN2 = {
    3: 0.11, 4: 0.20, 5: 0.31, 6: 0.44, 7: 0.60,
    8: 0.79, 9: 1.00, 10: 1.27, 11: 1.56, 14: 2.25, 18: 4.00
}
BAR_DIAMETERS_IN = {
    3: 0.375, 4: 0.500, 5: 0.625, 6: 0.750, 7: 0.875,
    8: 1.000, 9: 1.128, 10: 1.270, 11: 1.410, 14: 1.693, 18: 2.257
}
Es_psi = 29_000_000.0  # Steel modulus (ACI 20.2.2.2)
LAMBDA_NW = 1.0  # Normal weight concrete (ACI 19.2.4.2)
EPS_CU = 0.003  # Ultimate concrete strain (ACI 22.2.2.1)


def Ec_psi(fc: float) -> float:
    """ACI 19.2.2.1(b): Ec = 57000*sqrt(f'c) for normal weight concrete."""
    return 57_000.0 * math.sqrt(fc)


def fr_psi(fc: float) -> float:
    """ACI 19.2.3.1: Modulus of rupture fr = 7.5*lambda*sqrt(f'c)."""
    return 7.5 * LAMBDA_NW * math.sqrt(fc)


def beta1(fc: float) -> float:
    """ACI 22.2.2.4.3: Beta_1 factor for Whitney stress block."""
    if fc <= 4000:
        return 0.85
    if fc >= 8000:
        return 0.65
    return 0.85 - 0.05 * (fc - 4000) / 1000


def phi_flexure(eps_t: float) -> float:
    """
    ACI 21.2.2: Strength reduction factor phi for flexure.
    Transition from compression-controlled to tension-controlled.
    """
    eps_ty = 60000 / Es_psi  # Yield strain for Grade 60
    if eps_t <= eps_ty:
        return 0.65  # Compression-controlled
    if eps_t >= 0.005:
        return 0.90  # Tension-controlled
    # Transition zone (ACI Table 21.2.2)
    return 0.65 + 0.25 * (eps_t - eps_ty) / (0.005 - eps_ty)

# =============================================================================
# DEVELOPMENT LENGTH (ACI 25.4)
# =============================================================================
def development_length_tension(fy: float, fc: float, bar_no: int,
                               lambda_nw: float = LAMBDA_NW,
                               epoxy_coated: bool = False,
                               top_bar: bool = False) -> float:
    """Compute basic tension development length (ld) per ACI 318-19.

    This function implements the basic expression from ACI 25.4.2.3 for
    tension development of deformed bars:

        ld = (3/40) * (fy / (lambda * sqrt(fc))) * db

    The result is adjusted for top reinforcement (25.4.3.1) and epoxy
    coated bars (25.4.3.2). A minimum length of 12 in is enforced.

    Parameters
    ----------
    fy : float
        Yield strength of reinforcement (psi).
    fc : float
        Concrete compressive strength (psi).
    bar_no : int
        Bar number (#3, #4, #5, etc.).
    lambda_nw : float, optional
        Concrete density factor (λ), by default 1.0 for normal weight.
    epoxy_coated : bool, optional
        Whether the bar is epoxy coated, by default False.
    top_bar : bool, optional
        Whether the bar is located near the top of a member cast in place,
        requiring the 1.3 factor (ACI 25.4.3.1), by default False.

    Returns
    -------
    float
        Required development length ld (in).
    """
    db = BAR_DIAMETERS_IN.get(bar_no, 0.0)
    if db <= 0 or fy <= 0 or fc <= 0:
        return 0.0

    # Basic ld expression (25.4.2.3)
    ld = (3.0 / 40.0) * (fy / (lambda_nw * math.sqrt(fc))) * db

    # Modification factors (25.4.3)
    if epoxy_coated:
        ld *= 1.2  # factor for epoxy-coated bars
    if top_bar:
        ld *= 1.3  # factor for top bars cast in place

    # Minimum length (25.4.2.9)
    return max(ld, 12.0)


# =============================================================================
# DATA CLASSES
# =============================================================================
@dataclass
class Opening:
    """Rectangular opening in wall."""
    width_in: float
    height_in: float
    cx_in: float
    cy_in: float

    @property
    def left(self) -> float:
        return self.cx_in - 0.5 * self.width_in

    @property
    def right(self) -> float:
        return self.cx_in + 0.5 * self.width_in

    @property
    def bottom(self) -> float:
        return self.cy_in - 0.5 * self.height_in

    @property
    def top(self) -> float:
        return self.cy_in + 0.5 * self.height_in


@dataclass
class ProjectInfo:
    """Project information for title block."""
    project_name: str = "RC Wall Design"
    project_number: str = ""
    client_name: str = ""
    designer: str = ""
    checker: str = ""
    date: str = ""
    logo_path: Optional[str] = None
    company_name: str = ""
    company_address: str = ""


@dataclass
class WallInput:
    """Wall geometry, materials, reinforcement, and loads."""
    # Geometry
    Lw_in: float  # Wall length (horizontal)
    h_in: float   # Wall thickness
    hw_in: float  # Wall height
    
    # Materials
    fc_psi: float
    fy_psi: float
    cover_in: float
    
    # Vertical (longitudinal) reinforcement
    vert_bar_no: int
    vert_bar_spacing_in: float
    vert_faces: int  # 1 or 2
    
    # Horizontal (transverse) reinforcement
    horiz_bar_no: int
    horiz_bar_spacing_in: float
    horiz_faces: int
    
    # Shear ties (optional)
    tie_bar_no: int = 3
    tie_spacing_in: float = 12.0
    tie_legs: int = 0  # 0 = no ties provided
    
    # Factored loads - IN-PLANE
    Pu_kip: float = 0.0      # Axial load (compression positive)
    Vu_ip_kip: float = 0.0   # In-plane shear
    Mu_ip_kip_ft: float = 0.0  # In-plane moment (optional override)
    
    # Factored loads - OUT-OF-PLANE
    Vu_oop_kip: float = 0.0  # Out-of-plane shear
    wu_oop_psf: float = 0.0  # Out-of-plane pressure (lateral load)
    Mu_oop_kip_ft: float = 0.0  # Out-of-plane moment (optional override)
    
    # Boundary conditions
    wall_type: str = "bearing"  # "bearing", "nonbearing", "basement"
    bracing: str = "braced_restrained"  # k factor selection
    
    # Openings
    openings: List[Opening] = field(default_factory=list)
    
    # Project info
    project_info: ProjectInfo = field(default_factory=ProjectInfo)


@dataclass
class PMPoint:
    """Point on P-M interaction diagram."""
    c: float       # Neutral axis depth (in)
    eps_t: float   # Tensile strain
    Pn: float      # Nominal axial (lb) - FULL, not capped
    Mn: float      # Nominal moment (lb-in)
    phi: float     # Strength reduction factor
    phi_Pn: float  # Design axial (lb)
    phi_Mn: float  # Design moment (lb-in)
    label: str = ""
    
    # For 0.80Po cap tracking
    Pn_capped: float = 0.0  # Capped at 0.80Po
    phi_Pn_capped: float = 0.0  # phi * Pn_capped


# =============================================================================
# FEM ANALYSIS
# =============================================================================
class FEMAnalysis:
    """
    Finite Element Analysis for RC Walls with Openings.
    
    Two analysis types:
    1. Plane Stress (2D): For in-plane forces
    2. Plate Bending: For out-of-plane deflections
    
    Pressures derived from loads/area.
    """
    
    def __init__(self, inp: WallInput):
        self.inp = inp
        self.Ec = Ec_psi(inp.fc_psi)
        self.nu = 0.20  # Poisson's ratio for concrete
        
        # Plate stiffness
        self.D = self.Ec * inp.h_in**3 / (12 * (1 - self.nu**2))
        
        # Calculate pressures from loads
        area_plan = inp.Lw_in * inp.hw_in  # Plan area (in^2)
        area_elev = inp.Lw_in * inp.h_in   # Elevation area for axial
        
        # Axial pressure (compression, distributed over plan area)
        self.p_ax_psi = (inp.Pu_kip * 1000) / area_plan if area_plan > 0 else 0
        
        # In-plane lateral pressure
        self.p_lat_ip_psi = (inp.Vu_ip_kip * 1000) / area_plan if area_plan > 0 else 0
        
        # Out-of-plane pressure (from wu_oop_psf or Vu_oop)
        self.p_oop_psi = inp.wu_oop_psf / 144.0  # Convert psf to psi
        
        # Generate mesh
        self.nodes, self.elements, self.node_grid = self._generate_mesh()
        self.n_nodes = len(self.nodes)
    
    def _generate_mesh(self, target_size: float = 8.0):
        """Generate triangular mesh for wall with openings."""
        Lw = self.inp.Lw_in
        hw = self.inp.hw_in
        openings = self.inp.openings
        
        # Create grid lines aligned with opening edges
        x_lines = [0.0, Lw]
        y_lines = [0.0, hw]
        
        for op in openings:
            x_lines.extend([op.left, op.right])
            y_lines.extend([op.bottom, op.top])
        
        x_lines = sorted(set(x for x in x_lines if 0 <= x <= Lw))
        y_lines = sorted(set(y for y in y_lines if 0 <= y <= hw))
        
        # Subdivide intervals
        def subdivide(a, b, h):
            n = max(1, int(math.ceil((b - a) / h)))
            return [a + i * (b - a) / n for i in range(n + 1)]
        
        x_ref = []
        for i in range(len(x_lines) - 1):
            x_ref.extend(subdivide(x_lines[i], x_lines[i+1], target_size)[:-1])
        x_ref.append(x_lines[-1])
        x_ref = sorted(set(round(x, 6) for x in x_ref))
        
        y_ref = []
        for i in range(len(y_lines) - 1):
            y_ref.extend(subdivide(y_lines[i], y_lines[i+1], target_size)[:-1])
        y_ref.append(y_lines[-1])
        y_ref = sorted(set(round(y, 6) for y in y_ref))
        
        # Create nodes (skip opening interiors)
        nodes = []
        node_grid = {}
        
        for j, y in enumerate(y_ref):
            for i, x in enumerate(x_ref):
                # Check if inside any opening
                inside = any(
                    (op.left < x < op.right) and (op.bottom < y < op.top)
                    for op in openings
                )
                if inside:
                    continue
                
                idx = len(nodes)
                nodes.append((x, y))
                node_grid[(i, j)] = idx
        
        # Create triangular elements
        elements = []
        nx, ny = len(x_ref), len(y_ref)
        
        for j in range(ny - 1):
            for i in range(nx - 1):
                n1 = node_grid.get((i, j))
                n2 = node_grid.get((i+1, j))
                n3 = node_grid.get((i+1, j+1))
                n4 = node_grid.get((i, j+1))
                
                if None in (n1, n2, n3, n4):
                    continue
                
                # Two triangles per quad
                elements.append((n1, n2, n3))
                elements.append((n1, n3, n4))
        
        return nodes, elements, node_grid
    
    def solve_plane_stress(self):
        """
        Solve 2D plane stress problem for in-plane analysis.
        Returns displacement field (ux, uy at each node).
        """
        n_dof = 2 * self.n_nodes
        # Use sparse or dense stiffness matrix depending on availability
        if _SCIPY_AVAILABLE:
            K = lil_matrix((n_dof, n_dof), dtype=float)
        else:
            K = np.zeros((n_dof, n_dof))
        F = np.zeros(n_dof)
        
        # Plane stress constitutive matrix
        t = self.inp.h_in
        D = self.Ec * t / (1 - self.nu**2) * np.array([
            [1, self.nu, 0],
            [self.nu, 1, 0],
            [0, 0, (1 - self.nu) / 2]
        ])
        
        for elem in self.elements:
            n1, n2, n3 = elem
            x1, y1 = self.nodes[n1]
            x2, y2 = self.nodes[n2]
            x3, y3 = self.nodes[n3]
            
            # Element area
            detJ = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
            A = 0.5 * abs(detJ)
            if A < 1e-12:
                continue
            
            # B matrix (strain-displacement)
            b1, b2, b3 = y2 - y3, y3 - y1, y1 - y2
            c1, c2, c3 = x3 - x2, x1 - x3, x2 - x1
            
            B = (1 / (2 * A)) * np.array([
                [b1, 0, b2, 0, b3, 0],
                [0, c1, 0, c2, 0, c3],
                [c1, b1, c2, b2, c3, b3]
            ])
            
            Ke = A * (B.T @ D @ B)
            
            # Assembly
            dofs = [2*n1, 2*n1+1, 2*n2, 2*n2+1, 2*n3, 2*n3+1]
            for ii in range(6):
                for jj in range(6):
                    if _SCIPY_AVAILABLE:
                        K[dofs[ii], dofs[jj]] += Ke[ii, jj]
                    else:
                        K[dofs[ii], dofs[jj]] += Ke[ii, jj]
            
            # Load vector (body forces from pressures)
            if self.p_lat_ip_psi != 0:
                fx = self.p_lat_ip_psi * A / 3
                F[2*n1] += fx
                F[2*n2] += fx
                F[2*n3] += fx
            
            if self.p_ax_psi != 0:
                fy = -self.p_ax_psi * A / 3  # Compression is negative y
                F[2*n1+1] += fy
                F[2*n2+1] += fy
                F[2*n3+1] += fy
        
        # -------------------------------------------------------------------
        # Boundary conditions
        #
        # The original implementation assumed a cantilever wall by fixing
        # both degrees of freedom (ux, uy) for all nodes along the base
        # (y ≈ 0). This behaviour only captures a cantilevered support. To
        # capture braced walls per ACI 318-19, we apply boundary conditions
        # depending on the bracing type specified by the user:
        #   - "cantilever":       only fix ux and uy at the base.
        #   - "braced_unrestrained": fix ux and uy at the base and fix ux
        #     along the top edge to simulate a diaphragm that braces
        #     horizontal movement but allows vertical displacement.
        #   - "braced_restrained": fix ux and uy at both the base and top
        #     edges to simulate a diaphragm that braces both horizontal and
        #     vertical movement.
        # See ACI 11.5.3.2 for effective length factors; the FEM model now
        # reflects those bracing conditions.
        fixed_dofs = []
        bracing = self.inp.bracing
        for idx, (x, y) in enumerate(self.nodes):
            on_base = abs(y) < 1e-6
            on_top  = abs(y - self.inp.hw_in) < 1e-6
            if bracing == 'cantilever':
                # fix base only (ux, uy)
                if on_base:
                    fixed_dofs.extend([2*idx, 2*idx+1])
            elif bracing == 'braced_unrestrained':
                # fix base (ux, uy) and fix horizontal displacement at top (ux)
                if on_base:
                    fixed_dofs.extend([2*idx, 2*idx+1])
                elif on_top:
                    fixed_dofs.append(2*idx)  # fix ux
            elif bracing == 'braced_restrained':
                # fix base (ux, uy) and fix top (ux, uy)
                if on_base or on_top:
                    fixed_dofs.extend([2*idx, 2*idx+1])
            else:
                # default: treat as cantilever
                if on_base:
                    fixed_dofs.extend([2*idx, 2*idx+1])
        
        free_dofs = [d for d in range(n_dof) if d not in fixed_dofs]
        
        # Convert sparse matrix to CSR if using SciPy for solving
        u = np.zeros(n_dof)
        if free_dofs:
            if _SCIPY_AVAILABLE:
                # Convert to CSR for efficient arithmetic/slicing
                K = K.tocsr()
                Kff = K[free_dofs, :][:, free_dofs]
                Ff = F[free_dofs]
                # Add small regularization to diagonal to avoid singularity
                diag_vals = Kff.diagonal()
                if diag_vals.size > 0:
                    reg = 1e-12 * np.max(np.abs(diag_vals) + 1)
                    Kff = Kff + csr_matrix((np.full(len(free_dofs), reg), (np.arange(len(free_dofs)), np.arange(len(free_dofs)))), shape=Kff.shape)
                try:
                    uf = spsolve(Kff, Ff)
                    u[free_dofs] = uf
                except Exception:
                    # fallback: try dense solve if SciPy solver fails
                    try:
                        Kff_dense = Kff.toarray()
                        uf = np.linalg.solve(Kff_dense, Ff)
                        u[free_dofs] = uf
                    except Exception:
                        pass
            else:
                Kff = K[np.ix_(free_dofs, free_dofs)]
                Ff = F[free_dofs]
                # Add regularization
                diag_vals = np.diag(Kff)
                if diag_vals.size > 0:
                    Kff += np.eye(len(free_dofs)) * 1e-12 * np.max(np.abs(diag_vals) + 1)
                try:
                    uf = np.linalg.solve(Kff, Ff)
                    u[free_dofs] = uf
                except np.linalg.LinAlgError:
                    pass
        
        # Extract displacements
        ux_all = u[0::2]
        uy_all = u[1::2]
        
        # Total displacement magnitude
        u_mag = np.sqrt(ux_all**2 + uy_all**2)
        
        # Average top deflection
        top_nodes = [i for i, (x, y) in enumerate(self.nodes) if abs(y - self.inp.hw_in) < 1e-6]
        avg_top_ux = np.mean([ux_all[i] for i in top_nodes]) if top_nodes else 0.0
        avg_top_uy = np.mean([uy_all[i] for i in top_nodes]) if top_nodes else 0.0
        avg_top_mag = np.mean([u_mag[i] for i in top_nodes]) if top_nodes else 0.0
        
        # Max displacements
        i_ux_max = int(np.argmax(np.abs(ux_all))) if ux_all.size else 0
        i_uy_max = int(np.argmax(np.abs(uy_all))) if uy_all.size else 0
        i_mag_max = int(np.argmax(u_mag)) if u_mag.size else 0
        
        return {
            'ux': ux_all,
            'uy': uy_all,
            'u_mag': u_mag,
            'nodes': self.nodes,
            'elements': self.elements,
            'avg_top_ux': float(avg_top_ux),
            'avg_top_uy': float(avg_top_uy),
            'avg_top_mag': float(avg_top_mag),
            'max_ux': float(ux_all[i_ux_max]) if ux_all.size else 0.0,
            'max_uy': float(uy_all[i_uy_max]) if uy_all.size else 0.0,
            'max_mag': float(u_mag[i_mag_max]) if u_mag.size else 0.0,
            'loc_ux': self.nodes[i_ux_max] if ux_all.size else (0, 0),
            'loc_uy': self.nodes[i_uy_max] if uy_all.size else (0, 0),
            'loc_mag': self.nodes[i_mag_max] if u_mag.size else (0, 0),
        }
    
    def solve_plate_bending(self):
        """
        Solve plate bending problem for out-of-plane deflection (uz).
        Uses p_oop_psi from input.
        """
        if abs(self.p_oop_psi) < 1e-12:
            return {
                'uz': np.zeros(self.n_nodes),
                'nodes': self.nodes,
                'elements': self.elements,
                'max_uz': 0.0,
                'max_uz_loc': (0, 0),
                'avg_top_uz': 0.0
            }
        
        n_dof = self.n_nodes
        if _SCIPY_AVAILABLE:
            K = lil_matrix((n_dof, n_dof), dtype=float)
        else:
            K = np.zeros((n_dof, n_dof))
        F = np.zeros(n_dof)
        
        # Plate flexural rigidity
        D = self.D
        
        for elem in self.elements:
            n1, n2, n3 = elem
            x1, y1 = self.nodes[n1]
            x2, y2 = self.nodes[n2]
            x3, y3 = self.nodes[n3]
            
            # Element area
            A = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
            if A < 1e-12:
                continue
            
            # Characteristic length
            h_e = math.sqrt(A)
            
            # Simplified plate stiffness
            k_elem = D / (h_e**2) * A / 9
            
            # Simple stiffness matrix
            Ke = k_elem * np.array([
                [2, -1, -1],
                [-1, 2, -1],
                [-1, -1, 2]
            ])
            
            # Assembly
            dofs = [n1, n2, n3]
            for ii in range(3):
                for jj in range(3):
                    if _SCIPY_AVAILABLE:
                        K[dofs[ii], dofs[jj]] += Ke[ii, jj]
                    else:
                        K[dofs[ii], dofs[jj]] += Ke[ii, jj]
            
            # Load vector (uniform pressure)
            f_node = self.p_oop_psi * A / 3
            F[n1] += f_node
            F[n2] += f_node
            F[n3] += f_node
        
        # -------------------------------------------------------------------
        # Boundary conditions
        #
        # For out-of-plane bending the only degree of freedom per node is uz.
        # The original implementation fixed the base (y ≈ 0), modelling a
        # cantilever. We now apply boundary conditions consistent with the
        # bracing type in the input:
        #   - "cantilever":       fix uz at the base only.
        #   - "braced_unrestrained": fix uz at both the base and top edges
        #     to represent a diaphragm that braces vertical bending but still
        #     allows in-plane deformation through the plane stress model.
        #   - "braced_restrained": fix uz at both the base and top edges
        #     (same as unrestrained for plate bending because rotation DOF
        #     isn’t modelled here).
        bracing = self.inp.bracing
        fixed_dofs = []
        for idx, (x, y) in enumerate(self.nodes):
            on_base = abs(y) < 1e-6
            on_top  = abs(y - self.inp.hw_in) < 1e-6
            if bracing == 'cantilever':
                if on_base:
                    fixed_dofs.append(idx)
            elif bracing in ('braced_unrestrained', 'braced_restrained'):
                # fix both base and top edges for uz
                if on_base or on_top:
                    fixed_dofs.append(idx)
            else:
                if on_base:
                    fixed_dofs.append(idx)
        free_dofs = [d for d in range(n_dof) if d not in fixed_dofs]
        
        # Solve
        uz = np.zeros(n_dof)
        if free_dofs:
            if _SCIPY_AVAILABLE:
                K = K.tocsr()
                Kff = K[free_dofs, :][:, free_dofs]
                Ff = F[free_dofs]
                diag_vals = Kff.diagonal()
                if diag_vals.size > 0:
                    reg = 1e-10 * np.max(diag_vals + 1)
                    Kff = Kff + csr_matrix((np.full(len(free_dofs), reg), (np.arange(len(free_dofs)), np.arange(len(free_dofs)))), shape=Kff.shape)
                try:
                    uf = spsolve(Kff, Ff)
                    uz[free_dofs] = uf
                except Exception:
                    try:
                        Kff_dense = Kff.toarray()
                        uf = np.linalg.solve(Kff_dense, Ff)
                        uz[free_dofs] = uf
                    except Exception:
                        pass
            else:
                Kff = K[np.ix_(free_dofs, free_dofs)]
                Ff = F[free_dofs]
                diag_vals = np.diag(Kff)
                if diag_vals.size > 0:
                    Kff += np.eye(len(free_dofs)) * 1e-10 * np.max(diag_vals + 1)
                try:
                    uf = np.linalg.solve(Kff, Ff)
                    uz[free_dofs] = uf
                except np.linalg.LinAlgError:
                    pass
        
        max_uz = float(np.max(np.abs(uz))) if uz.size else 0.0
        i_max = int(np.argmax(np.abs(uz))) if uz.size else 0
        
        # Average top deflection
        top_nodes = [i for i, (x, y) in enumerate(self.nodes) if abs(y - self.inp.hw_in) < 1e-6]
        avg_top_uz = np.mean([uz[i] for i in top_nodes]) if top_nodes else 0.0
        
        return {
            'uz': uz,
            'nodes': self.nodes,
            'elements': self.elements,
            'max_uz': max_uz,
            'max_uz_loc': self.nodes[i_max] if uz.size else (0, 0),
            'avg_top_uz': float(avg_top_uz)
        }


# =============================================================================
# ACI 318-19 CHAPTER 11 CHECKS
# =============================================================================
class ACI318Chapter11:
    """All ACI 318-19 Chapter 11 wall design checks with derivations."""
    
    @staticmethod
    def min_thickness(wall_type: str, Lw: float, hw: float) -> Tuple[float, str]:
        """ACI 11.3.1.1 Table 11.3.1.1 - Minimum wall thickness."""
        unsup = min(Lw, hw)
        if wall_type == "bearing":
            h_min = max(4.0, unsup / 25.0)
            eq = f"h_min = max(4\", L_unsup/25) = max(4\", {unsup:.1f}/25) = {h_min:.2f} in"
        elif wall_type == "nonbearing":
            h_min = max(4.0, unsup / 30.0)
            eq = f"h_min = max(4\", L_unsup/30) = max(4\", {unsup:.1f}/30) = {h_min:.2f} in"
        elif wall_type == "basement":
            h_min = 7.5
            eq = "h_min = 7.5 in (basement wall)"
        else:
            h_min = 4.0
            eq = "h_min = 4 in (default)"
        return h_min, eq
    
    @staticmethod
    def effective_length_factor(bracing: str) -> Tuple[float, str]:
        """ACI 11.5.3.2 Table 11.5.3.2 - Effective length factor k."""
        factors = {
            "braced_restrained": (0.8, "Braced, restrained top/bottom -> k = 0.8"),
            "braced_unrestrained": (1.0, "Braced, unrestrained -> k = 1.0"),
            "cantilever": (2.0, "Cantilever (unbraced) -> k = 2.0")
        }
        return factors.get(bracing, (1.0, "Default -> k = 1.0"))
    
    @staticmethod
    def min_reinf_ratios(fc: float, fy: float, bar_no: int, 
                         high_shear: bool) -> Tuple[float, float, str]:
        """ACI 11.6.1/11.6.2 - Minimum reinforcement ratios."""
        if high_shear:
            rho_L = 0.0025
            rho_t = 0.0025
            eq = "High shear (Vu > 0.5*phi*Vc): rho_L,min = rho_t,min = 0.0025 (ACI 11.6.2)"
        else:
            if bar_no <= 5 and fy >= 60000:
                rho_L = 0.0012
                rho_t = 0.0020
                eq = f"Bar <= #5, fy >= 60 ksi: rho_L,min = 0.0012, rho_t,min = 0.0020 (ACI 11.6.1)"
            elif fy < 60000:
                rho_L = 0.0015
                rho_t = 0.0025
                eq = f"fy < 60 ksi: rho_L,min = 0.0015, rho_t,min = 0.0025 (ACI 11.6.1)"
            else:
                rho_L = 0.0015
                rho_t = 0.0025
                eq = "Default: rho_L,min = 0.0015, rho_t,min = 0.0025 (ACI 11.6.1)"
        return rho_L, rho_t, eq
    
    @staticmethod
    def max_spacing(h: float, shear_required: bool, Lw: float) -> Tuple[float, float, str]:
        """ACI 11.7.2.1, 11.7.3.1 - Maximum bar spacing."""
        base = min(3 * h, 18.0)
        if shear_required:
            s_v = min(base, Lw / 3)
            s_h = min(base, Lw / 5)
            eq = f"Shear reinf req'd: s_v,max = min(3h, 18\", Lw/3) = {s_v:.1f} in; s_h,max = min(3h, 18\", Lw/5) = {s_h:.1f} in"
        else:
            s_v = base
            s_h = base
            eq = f"No shear reinf: s_max = min(3h, 18\") = min({3*h:.1f}\", 18\") = {base:.1f} in (ACI 11.7.2.1)"
        return s_v, s_h, eq
    
    @staticmethod
    def requires_two_layers(h: float) -> Tuple[bool, str]:
        """ACI 11.7.2.3 - Two layers required if h > 10 in."""
        req = h > 10.0
        eq = f"h = {h:.1f} in {'>' if req else '<='} 10 in -> Two layers {'required' if req else 'not required'} (ACI 11.7.2.3)"
        return req, eq
    
    @staticmethod
    def Vn_inplane(fc: float, Lw: float, h: float, hw: float,
                   rho_t: float, fy: float, Nu: float = 0) -> Dict[str, Any]:
        """
        ACI 11.5.4 - In-plane shear strength.
        Returns capacity and full derivation.
        """
        Acv = Lw * h
        ratio = hw / Lw
        
        # Alpha_c per ACI 11.5.4.3
        if ratio <= 1.5:
            alpha_c = 3.0
        elif ratio >= 2.0:
            alpha_c = 2.0
        else:
            alpha_c = 3.0 - (ratio - 1.5) * 2.0
        
        # Tension effect
        if Nu < 0:
            Ag = Lw * h
            alpha_c_mod = max(0, 2 * (1 + Nu / (500 * Ag)))
            alpha_c = min(alpha_c, alpha_c_mod)
        
        # ACI 11.5.4.4: Vn = Acv(alpha_c*lambda*sqrt(f'c) + rho_t*fy)
        Vc = alpha_c * LAMBDA_NW * math.sqrt(fc) * Acv
        Vs = rho_t * fy * Acv
        Vn = Vc + Vs
        
        # ACI 11.5.4.5: Maximum Vn
        Vn_max = 8 * math.sqrt(fc) * Acv
        
        # Derivation string (using ASCII-friendly notation)
        deriv = f"""ACI 11.5.4 In-Plane Shear:
        
        Acv = Lw x h = {Lw:.1f} x {h:.1f} = {Acv:.0f} in^2
        
        hw/Lw = {hw:.1f}/{Lw:.1f} = {ratio:.2f}
        alpha_c = {alpha_c:.2f} (ACI 11.5.4.3: alpha_c = 3 for hw/Lw <= 1.5, alpha_c = 2 for hw/Lw >= 2.0)
        
        Vc = alpha_c*lambda*sqrt(f'c)*Acv = {alpha_c:.2f}x{LAMBDA_NW:.1f}xsqrt({fc:.0f})x{Acv:.0f}
           = {Vc/1000:.1f} kip
        
        Vs = rho_t*fy*Acv = {rho_t:.4f}x{fy:.0f}x{Acv:.0f}
           = {Vs/1000:.1f} kip
        
        Vn = Vc + Vs = {Vc/1000:.1f} + {Vs/1000:.1f} = {Vn/1000:.1f} kip
        
        Vn,max = 8*sqrt(f'c)*Acv = 8xsqrt({fc:.0f})x{Acv:.0f} = {Vn_max/1000:.1f} kip
        
        phi*Vn = 0.75 x min({Vn/1000:.1f}, {Vn_max/1000:.1f}) = {0.75*min(Vn, Vn_max)/1000:.1f} kip"""
        
        return {
            'Vn': min(Vn, Vn_max),
            'Vc': Vc,
            'Vs': Vs,
            'Vn_max': Vn_max,
            'alpha_c': alpha_c,
            'Acv': Acv,
            'derivation': deriv
        }
    
    @staticmethod
    def Vn_outofplane(fc: float, bw: float, d: float, 
                      Av: float = 0, s: float = 12, fy: float = 60000,
                      Nu: float = 0, Ag: float = 0) -> Dict[str, Any]:
        """
        ACI 11.5.5, 22.5 - Out-of-plane (one-way) shear strength.
        """
        # ACI 22.5.5.1: Vc = 2*lambda*sqrt(f'c)*bw*d
        Vc_base = 2 * LAMBDA_NW * math.sqrt(fc) * bw * d
        
        # Axial load modification (ACI 22.5.6)
        if Nu > 0 and Ag > 0:
            Vc = min(Vc_base * (1 + Nu / (2000 * Ag)), 
                     3.5 * LAMBDA_NW * math.sqrt(fc) * bw * d)
        elif Nu < 0 and Ag > 0:
            Vc = max(0, Vc_base * (1 + Nu / (500 * Ag)))
        else:
            Vc = Vc_base
        
        # Shear reinforcement
        if Av > 0 and s > 0:
            Vs = Av * fy * d / s
        else:
            Vs = 0
        
        Vn_max = 8 * math.sqrt(fc) * bw * d
        Vn = min(Vc + Vs, Vn_max)
        
        deriv = f"""ACI 22.5 Out-of-Plane Shear:
        
        bw = {bw:.1f} in (critical width)
        d = {d:.2f} in (effective depth)
        
        Vc = 2*lambda*sqrt(f'c)*bw*d = 2x{LAMBDA_NW:.1f}xsqrt({fc:.0f})x{bw:.1f}x{d:.2f}
           = {Vc/1000:.1f} kip (ACI 22.5.5.1)
        
        Vs = Av*fy*d/s = {Av:.2f}x{fy:.0f}x{d:.2f}/{s:.1f}
           = {Vs/1000:.1f} kip (ACI 22.5.8.5)
        
        Vn = Vc + Vs = {Vc/1000:.1f} + {Vs/1000:.1f} = {(Vc+Vs)/1000:.1f} kip
        
        Vn,max = 8*sqrt(f'c)*bw*d = {Vn_max/1000:.1f} kip
        
        phi*Vn = 0.75 x {min(Vc+Vs, Vn_max)/1000:.1f} = {0.75*Vn/1000:.1f} kip"""
        
        return {
            'Vn': Vn,
            'Vc': Vc,
            'Vs': Vs,
            'Vn_max': Vn_max,
            'derivation': deriv
        }


# =============================================================================
# P-M INTERACTION DIAGRAM
# =============================================================================
class PMDiagram:
    """Generate P-M interaction diagram per ACI 318-19 with full derivations."""
    
    def __init__(self, fc: float, fy: float, b: float, h: float,
                 layers: List[Tuple[float, float]], direction: str = ""):
        """
        Args:
            fc: Concrete strength (psi)
            fy: Steel yield strength (psi)
            b: Section width (in)
            h: Section depth (in)
            layers: List of (d_from_compression_face, As) tuples
            direction: "in-plane" or "out-of-plane" for labeling
        """
        self.fc = fc
        self.fy = fy
        self.b = b
        self.h = h
        self.layers = layers
        self.direction = direction
        self.beta1_val = beta1(fc)
        self.eps_y = fy / Es_psi
        
        # Section properties
        self.Ag = b * h
        self.Ast = sum(As for _, As in layers)
        self.dt = max(d for d, _ in layers) if layers else h - 2.0
        
        # Key c values
        self.c_balanced = EPS_CU / (EPS_CU + self.eps_y) * self.dt
        self.c_tension_controlled = EPS_CU / (EPS_CU + 0.005) * self.dt
        
        # Pure compression capacity (ACI 22.4.2.1)
        self.Po = 0.85 * fc * (self.Ag - self.Ast) + fy * self.Ast
        self.Po_cap = 0.80 * self.Po  # ACI 22.4.2.1 limit
    
    def get_derivation(self) -> str:
        """Generate full mathematical derivation (ASCII-friendly)."""
        deriv = f"""P-M Interaction Diagram Derivation ({self.direction})
        
SECTION PROPERTIES (ACI 22.2):
        b = {self.b:.1f} in
        h = {self.h:.1f} in
        Ag = b x h = {self.Ag:.0f} in^2
        Ast = Sum(As) = {self.Ast:.2f} in^2
        dt = {self.dt:.2f} in (depth to extreme tension steel)
        
MATERIAL PROPERTIES:
        f'c = {self.fc:.0f} psi
        fy = {self.fy:.0f} psi
        Es = {Es_psi/1e6:.0f} x 10^6 psi (ACI 20.2.2.2)
        eps_cu = {EPS_CU:.3f} (ACI 22.2.2.1)
        eps_y = fy/Es = {self.eps_y:.5f}
        
WHITNEY STRESS BLOCK (ACI 22.2.2.4):
        Beta_1 = 0.85 - 0.05(f'c - 4000)/1000 for 4000 < f'c < 8000
        Beta_1 = {self.beta1_val:.3f}
        a = Beta_1 x c
        
PURE COMPRESSION (ACI 22.4.2.1):
        Po = 0.85*f'c*(Ag - Ast) + fy*Ast
           = 0.85x{self.fc:.0f}x({self.Ag:.0f} - {self.Ast:.2f}) + {self.fy:.0f}x{self.Ast:.2f}
           = {self.Po/1000:.0f} kip
        
        Po,cap = 0.80*Po = {self.Po_cap/1000:.0f} kip (ACI 22.4.2.1 maximum)
        
KEY POINTS ON DIAGRAM:
        
1. BALANCED CONDITION (eps_t = eps_y):
        c_bal = eps_cu/(eps_cu + eps_y) x dt
              = {EPS_CU:.3f}/({EPS_CU:.3f} + {self.eps_y:.5f}) x {self.dt:.2f}
              = {self.c_balanced:.2f} in
        
2. TENSION-CONTROLLED LIMIT (eps_t = 0.005, phi = 0.90):
        c_tc = eps_cu/(eps_cu + 0.005) x dt
             = {EPS_CU:.3f}/({EPS_CU:.3f} + 0.005) x {self.dt:.2f}
             = {self.c_tension_controlled:.2f} in
        
STRENGTH REDUCTION FACTOR (ACI 21.2.2):
        phi = 0.65 for eps_t <= eps_y (compression-controlled)
        phi = 0.90 for eps_t >= 0.005 (tension-controlled)
        phi = 0.65 + 0.25(eps_t - eps_y)/(0.005 - eps_y) for eps_y < eps_t < 0.005 (transition)
        
FORCE EQUILIBRIUM AT ANY c:
        Cc = 0.85*f'c*b*a = 0.85*f'c*b*Beta_1*c
        
        For each steel layer at depth di:
            eps_si = eps_cu x (c - di)/c
            fsi = Es x eps_si (limited to fy)
            Fsi = Asi x fsi
        
        Pn = Cc + Sum(Fsi)
        Mn = Cc*(h/2 - a/2) + Sum(Fsi*(h/2 - di))
"""
        return deriv
    
    def _forces_at_c(self, c: float) -> Tuple[float, float, float]:
        """Compute Pn, Mn, eps_t for given neutral axis depth c."""
        if c <= 1e-6:
            Pn = -self.Ast * self.fy
            Mn = 0.0
            return Pn, Mn, 1.0
        
        # Concrete compression block
        a = min(self.beta1_val * c, self.h)
        Cc = 0.85 * self.fc * self.b * a
        
        # Steel forces
        Fs_total = 0.0
        Mn = 0.0
        eps_t_max = 0.0
        
        for d_i, As_i in self.layers:
            eps_i = EPS_CU * (c - d_i) / c
            fs_i = max(-self.fy, min(Es_psi * eps_i, self.fy))
            Fs_i = As_i * fs_i
            Fs_total += Fs_i
            
            # Moment about plastic centroid (h/2)
            Mn += Fs_i * (self.h / 2 - d_i)
            
            # Track max tensile strain
            if eps_i < 0:
                eps_t_max = max(eps_t_max, abs(eps_i))
        
        Pn = Cc + Fs_total
        Mn += Cc * (self.h / 2 - a / 2)
        
        return Pn, abs(Mn), eps_t_max
    
    def _find_c_for_Pn_zero(self) -> float:
        """Find c where Pn = 0 (pure bending)."""
        c_lo, c_hi = 0.01, self.h * 2
        for _ in range(50):
            c_mid = (c_lo + c_hi) / 2
            Pn, _, _ = self._forces_at_c(c_mid)
            if abs(Pn) < 10:
                return c_mid
            if Pn > 0:
                c_hi = c_mid
            else:
                c_lo = c_mid
        return (c_lo + c_hi) / 2
    
    def generate_curve(self, n_points: int = 100) -> Tuple[List[PMPoint], List[PMPoint]]:
        """
        Generate P-M curve from pure compression to pure tension.
        Returns (all_points, key_points).
        
        IMPORTANT: Pn is NOT capped - full curve is generated.
        Pn_capped and phi_Pn_capped track the 0.80Po limit separately.
        """
        points = []
        key_points = []
        
        # Pure compression (Po) - FULL value
        Pn_max = self.Po
        
        # Key c values
        c_pure_bending = self._find_c_for_Pn_zero()
        
        # Create c values array
        c_vals = np.concatenate([
            np.linspace(self.h * 5, self.dt, 20),
            np.linspace(self.dt, self.c_balanced, 20),
            np.linspace(self.c_balanced, self.c_tension_controlled, 15),
            np.linspace(self.c_tension_controlled, c_pure_bending, 20),
            np.linspace(c_pure_bending, 0.1, 15),
        ])
        c_vals = np.unique(np.sort(c_vals)[::-1])
        
        for c in c_vals:
            Pn, Mn, eps_t = self._forces_at_c(c)
            phi = phi_flexure(eps_t)
            
            # Full Pn (not capped)
            Pn_full = Pn
            
            # Capped values for design
            Pn_capped = min(Pn, self.Po_cap)
            phi_Pn_capped = phi * Pn_capped
            
            pt = PMPoint(
                c=c, eps_t=eps_t,
                Pn=Pn_full, Mn=Mn,
                phi=phi,
                phi_Pn=phi * Pn_full,  # Full phi*Pn
                phi_Mn=phi * Mn,
                label="",
                Pn_capped=Pn_capped,
                phi_Pn_capped=phi_Pn_capped
            )
            points.append(pt)
        
        # Pure compression point (full Po)
        pt_comp = PMPoint(
            c=float('inf'), eps_t=0.0,
            Pn=Pn_max, Mn=0.0,
            phi=0.65, 
            phi_Pn=0.65 * Pn_max,
            phi_Mn=0.0,
            label="Po (max compression)",
            Pn_capped=self.Po_cap,
            phi_Pn_capped=0.65 * self.Po_cap
        )
        points.insert(0, pt_comp)
        key_points.append(pt_comp)
        
        # 0.80Po point
        pt_080 = PMPoint(
            c=float('inf'), eps_t=0.0,
            Pn=self.Po_cap, Mn=0.0,
            phi=0.65,
            phi_Pn=0.65 * self.Po_cap,
            phi_Mn=0.0,
            label="0.80Po (design max)",
            Pn_capped=self.Po_cap,
            phi_Pn_capped=0.65 * self.Po_cap
        )
        key_points.append(pt_080)
        
        # Pure tension point
        Pn_tension = -self.Ast * self.fy
        pt_tension = PMPoint(
            c=0.0, eps_t=1.0,
            Pn=Pn_tension, Mn=0.0,
            phi=0.90, 
            phi_Pn=0.90 * Pn_tension, 
            phi_Mn=0.0,
            label="Pure Tension",
            Pn_capped=Pn_tension,
            phi_Pn_capped=0.90 * Pn_tension
        )
        points.append(pt_tension)
        key_points.append(pt_tension)
        
        # Find key points
        idx_et0 = np.argmin([abs(pt.c - self.dt) for pt in points])
        points[idx_et0].label = "eps_t = 0 (c = dt)"
        key_points.append(points[idx_et0])
        
        idx_bal = np.argmin([abs(pt.c - self.c_balanced) for pt in points])
        points[idx_bal].label = f"Balanced (eps_t = eps_y = {self.eps_y:.5f})"
        key_points.append(points[idx_bal])
        
        idx_tc = np.argmin([abs(pt.c - self.c_tension_controlled) for pt in points])
        points[idx_tc].label = "Tension-Controlled (eps_t = 0.005, phi = 0.90)"
        key_points.append(points[idx_tc])
        
        idx_pb = np.argmin([abs(pt.Pn) for pt in points])
        points[idx_pb].label = "Pure Bending (Pn = 0)"
        key_points.append(points[idx_pb])
        
        # Remove duplicates while preserving order
        seen_labels = set()
        unique_keys = []
        for kp in key_points:
            if kp.label not in seen_labels:
                seen_labels.add(kp.label)
                unique_keys.append(kp)
        key_points = sorted(unique_keys, key=lambda p: p.Pn, reverse=True)
        
        return points, key_points
    
    def get_capacity_at_Pu(self, points: List[PMPoint], Pu: float, 
                           use_factored: bool = True, use_capped: bool = True) -> float:
        """Interpolate Mn capacity at given Pu."""
        if use_factored:
            if use_capped:
                Pn_vals = [pt.phi_Pn_capped for pt in points]
            else:
                Pn_vals = [pt.phi_Pn for pt in points]
            Mn_vals = [pt.phi_Mn for pt in points]
        else:
            if use_capped:
                Pn_vals = [pt.Pn_capped for pt in points]
            else:
                Pn_vals = [pt.Pn for pt in points]
            Mn_vals = [pt.Mn for pt in points]
        
        sorted_idx = np.argsort(Pn_vals)[::-1]
        Pn_sorted = [Pn_vals[i] for i in sorted_idx]
        Mn_sorted = [Mn_vals[i] for i in sorted_idx]
        
        for i in range(len(Pn_sorted) - 1):
            if Pn_sorted[i+1] <= Pu <= Pn_sorted[i]:
                ratio = (Pu - Pn_sorted[i+1]) / (Pn_sorted[i] - Pn_sorted[i+1] + 1e-9)
                return Mn_sorted[i+1] + ratio * (Mn_sorted[i] - Mn_sorted[i+1])
        
        if Pu >= Pn_sorted[0]:
            return Mn_sorted[0]
        return Mn_sorted[-1]


# =============================================================================
# WALL DESIGNER
# =============================================================================
class WallDesigner:
    """Complete RC wall design per ACI 318-19 Chapter 11 with FEM."""
    
    def __init__(self, inp: WallInput):
        self.inp = inp
        self.Pu = inp.Pu_kip * 1000  # lb
        self.Vu_ip = inp.Vu_ip_kip * 1000  # lb (in-plane)
        self.Vu_oop = inp.Vu_oop_kip * 1000  # lb (out-of-plane)
        
        # Moments
        self.Mu_ip = inp.Mu_ip_kip_ft * 12000 if inp.Mu_ip_kip_ft > 0 else 0  # lb-in
        self.Mu_oop = inp.Mu_oop_kip_ft * 12000 if inp.Mu_oop_kip_ft > 0 else 0  # lb-in
        
        # Critical net width (accounting for openings)
        self.crit_width = self._calc_critical_width()
        self.Ag_net = self.crit_width * inp.h_in
    
    def _calc_critical_width(self) -> float:
        """Minimum net width over wall height."""
        def net_at_y(y):
            deductions = sum(
                op.width_in for op in self.inp.openings
                if op.bottom <= y <= op.top
            )
            return max(0, self.inp.Lw_in - deductions)
        
        probe_y = [0, self.inp.hw_in]
        for op in self.inp.openings:
            probe_y.extend([op.bottom, op.top, op.cy_in])
        
        return max(self.inp.Lw_in * 0.5, min(net_at_y(y) for y in probe_y))
    
    def run_design(self) -> Dict[str, Any]:
        """Execute all design checks."""
        inp = self.inp
        results = {}
        derivations = {}  # Store all derivations
        
        # ─────────────────────────────────────────────────────────────────────
        # GEOMETRY CHECKS (ACI 11.3.1.1)
        # ─────────────────────────────────────────────────────────────────────
        h_min, h_min_deriv = ACI318Chapter11.min_thickness(inp.wall_type, inp.Lw_in, inp.hw_in)
        results['h_min'] = h_min
        results['h_provided'] = inp.h_in
        results['h_ok'] = inp.h_in >= h_min
        derivations['h_min'] = h_min_deriv
        
        two_req, two_deriv = ACI318Chapter11.requires_two_layers(inp.h_in)
        results['two_layers_req'] = two_req
        results['two_layers_prov'] = inp.vert_faces == 2
        derivations['two_layers'] = two_deriv
        
        # ─────────────────────────────────────────────────────────────────────
        # FEM ANALYSIS (Both in-plane and out-of-plane)
        # ─────────────────────────────────────────────────────────────────────
        fem = FEMAnalysis(inp)
        
        # In-plane (plane stress) analysis
        fem_inplane = fem.solve_plane_stress()
        results['fem_inplane'] = fem_inplane
        
        # Out-of-plane (plate bending) analysis
        fem_oop = fem.solve_plate_bending()
        results['fem_oop'] = fem_oop
        
        # Combined total displacement magnitude
        ux_ip = fem_inplane['ux']
        uy_ip = fem_inplane['uy']
        uz_oop = fem_oop['uz']
        
        # Total 3D displacement
        u_total = np.sqrt(ux_ip**2 + uy_ip**2 + uz_oop**2)
        results['max_total_disp'] = float(np.max(u_total))
        i_max_total = int(np.argmax(u_total))
        results['loc_max_total'] = fem.nodes[i_max_total] if u_total.size else (0, 0)
        
        # Average top total displacement
        top_nodes = [i for i, (x, y) in enumerate(fem.nodes) if abs(y - inp.hw_in) < 1e-6]
        results['avg_top_total'] = float(np.mean([u_total[i] for i in top_nodes])) if top_nodes else 0.0
        
        # Calculate moments from FEM if not provided
        if self.Mu_ip == 0 and self.Vu_ip > 0:
            # Simplified: M = V x h/2 (triangular distribution)
            self.Mu_ip = self.Vu_ip * (inp.hw_in / 2.0)
        
        if self.Mu_oop == 0 and inp.wu_oop_psf > 0:
            # Cantilever: M = w x h^2 / 2
            w_pli = inp.wu_oop_psf / 144.0 * inp.Lw_in  # lb/in
            self.Mu_oop = w_pli * inp.hw_in**2 / 2.0
        
        results['Mu_ip'] = self.Mu_ip
        results['Mu_oop'] = self.Mu_oop
        
        # Store FEM object and derived pressures
        results['fem'] = fem
        results['p_ax_psi'] = fem.p_ax_psi
        results['p_lat_ip_psi'] = fem.p_lat_ip_psi
        results['p_oop_psi'] = fem.p_oop_psi
        
        derivations['fem_pressures'] = f"""FEM-Derived Pressures:
        
        Plan area = Lw x hw = {inp.Lw_in:.1f} x {inp.hw_in:.1f} = {inp.Lw_in*inp.hw_in:.0f} in^2
        
        Axial pressure: p_ax = Pu / (Lw x hw) = {inp.Pu_kip:.1f}x1000 / {inp.Lw_in*inp.hw_in:.0f} = {fem.p_ax_psi:.4f} psi
        
        In-plane lateral: p_lat,ip = Vu,ip / (Lw x hw) = {inp.Vu_ip_kip:.1f}x1000 / {inp.Lw_in*inp.hw_in:.0f} = {fem.p_lat_ip_psi:.4f} psi
        
        Out-of-plane: p_oop = wu,oop / 144 = {inp.wu_oop_psf:.1f} / 144 = {fem.p_oop_psi:.4f} psi"""
        
        # ─────────────────────────────────────────────────────────────────────
        # CRITICAL WIDTH CALCULATION DERIVATION
        # ─────────────────────────────────────────────────────────────────────
        crit_width_deriv_lines = [
            f"Critical Net Width Calculation (for shear and reinforcement):",
            f"",
            f"  Gross wall length: Lw = {inp.Lw_in:.2f} in",
            f"",
            f"  The critical net width is the minimum net width of the wall",
            f"  at any horizontal section over the height, accounting for",
            f"  any opening deductions at that level.",
            f"",
            f"  Calculation procedure:",
            f"  1. Identify all openings and their vertical extents",
            f"  2. At each horizontal level y, compute:",
            f"     net_width(y) = Lw - sum(opening widths where bottom <= y <= top)",
            f"  3. Critical width = max(0.5*Lw, min(net_width(y) for all y))",
            f"",
        ]
        
        if inp.openings:
            crit_width_deriv_lines.append("  Openings present:")
            for i, op in enumerate(inp.openings, 1):
                crit_width_deriv_lines.append(
                    f"    Opening {i}: width = {op.width_in:.2f} in, "
                    f"height = {op.height_in:.2f} in, "
                    f"center = ({op.cx_in:.2f}, {op.cy_in:.2f}) in"
                )
                crit_width_deriv_lines.append(
                    f"               bottom = {op.bottom:.2f} in, top = {op.top:.2f} in"
                )
            crit_width_deriv_lines.append("")
            
            # Show calculation at various probe heights
            probe_y = sorted(set([0, inp.hw_in] + 
                                 [op.bottom for op in inp.openings] + 
                                 [op.top for op in inp.openings] + 
                                 [op.cy_in for op in inp.openings]))
            crit_width_deriv_lines.append("  Net width at probe elevations:")
            for y in probe_y:
                deductions = sum(op.width_in for op in inp.openings if op.bottom <= y <= op.top)
                net = max(0, inp.Lw_in - deductions)
                crit_width_deriv_lines.append(
                    f"    y = {y:.2f} in: deductions = {deductions:.2f} in, net = {inp.Lw_in:.2f} - {deductions:.2f} = {net:.2f} in"
                )
            crit_width_deriv_lines.append("")
        else:
            crit_width_deriv_lines.append("  No openings present.")
            crit_width_deriv_lines.append(f"  Net width = Lw = {inp.Lw_in:.2f} in at all elevations.")
            crit_width_deriv_lines.append("")
        
        crit_width_deriv_lines.extend([
            f"  Minimum net width over height = {self.crit_width:.2f} in",
            f"  Lower bound (0.5*Lw) = 0.5 x {inp.Lw_in:.2f} = {0.5*inp.Lw_in:.2f} in",
            f"",
            f"  Critical width = max({0.5*inp.Lw_in:.2f}, {self.crit_width:.2f}) = {self.crit_width:.2f} in",
            f"",
            f"  Net area Ag,net = crit_width x h = {self.crit_width:.2f} x {inp.h_in:.2f} = {self.Ag_net:.2f} in^2"
        ])
        
        derivations['critical_width'] = "\n".join(crit_width_deriv_lines)
        results['crit_width'] = self.crit_width
        results['Ag_net'] = self.Ag_net
        
        # ─────────────────────────────────────────────────────────────────────
        # FEM BOUNDARY CONDITIONS DERIVATION
        # ─────────────────────────────────────────────────────────────────────
        # Get k factor early for use in boundary condition documentation
        k_bc, _ = ACI318Chapter11.effective_length_factor(inp.bracing)
        
        bc_deriv_lines = [
            f"FEM Boundary Conditions (per ACI 318-19 Table 11.5.3.2):",
            f"",
            f"  Wall bracing type: {inp.bracing}",
            f"  Effective length factor k = {k_bc:.2f}",
            f"",
            f"  ═══════════════════════════════════════════════════════════════",
            f"  IN-PLANE ANALYSIS (Plane Stress - 2 DOF per node: ux, uy)",
            f"  ═══════════════════════════════════════════════════════════════",
            f"",
        ]
        
        if inp.bracing == 'cantilever':
            bc_deriv_lines.extend([
                f"  Cantilever wall (k = 2.0):",
                f"    - BASE (y = 0): Fixed in both directions",
                f"        ux = 0 (no horizontal translation)",
                f"        uy = 0 (no vertical translation)",
                f"    - TOP (y = hw): Free (unrestricted)",
                f"        ux = free",
                f"        uy = free",
                f"",
                f"  Physical interpretation: Wall is rigidly attached at base,",
                f"  with no lateral support at top. Free to deflect and rotate",
                f"  at the top edge.",
            ])
        elif inp.bracing == 'braced_unrestrained':
            bc_deriv_lines.extend([
                f"  Braced, unrestrained at ends (k = 1.0):",
                f"    - BASE (y = 0): Fixed in both directions",
                f"        ux = 0 (no horizontal translation)",
                f"        uy = 0 (no vertical translation)",
                f"    - TOP (y = hw): Braced horizontally, free vertically",
                f"        ux = 0 (horizontal bracing by diaphragm)",
                f"        uy = free (vertical displacement allowed)",
                f"",
                f"  Physical interpretation: Wall is supported at base and",
                f"  braced laterally at top by floor diaphragm. Top edge can",
                f"  move vertically under axial load but cannot sway laterally.",
            ])
        elif inp.bracing == 'braced_restrained':
            bc_deriv_lines.extend([
                f"  Braced, restrained at both ends (k = 0.8):",
                f"    - BASE (y = 0): Fixed in both directions",
                f"        ux = 0 (no horizontal translation)",
                f"        uy = 0 (no vertical translation)",
                f"    - TOP (y = hw): Fixed in both directions",
                f"        ux = 0 (no horizontal translation)",
                f"        uy = 0 (no vertical translation)",
                f"",
                f"  Physical interpretation: Wall is rigidly connected at both",
                f"  base and top. Both ends are restrained against rotation and",
                f"  translation by floor/roof diaphragms or other structural",
                f"  elements.",
            ])
        
        bc_deriv_lines.extend([
            f"",
            f"  ═══════════════════════════════════════════════════════════════",
            f"  OUT-OF-PLANE ANALYSIS (Plate Bending - 1 DOF per node: uz)",
            f"  ═══════════════════════════════════════════════════════════════",
            f"",
        ])
        
        if inp.bracing == 'cantilever':
            bc_deriv_lines.extend([
                f"  Cantilever wall (k = 2.0):",
                f"    - BASE (y = 0): Fixed",
                f"        uz = 0 (no out-of-plane deflection)",
                f"    - TOP (y = hw): Free",
                f"        uz = free",
                f"",
                f"  Physical interpretation: Wall acts as vertical cantilever",
                f"  spanning from foundation to roof level. Maximum deflection",
                f"  expected at top edge.",
            ])
        else:
            bc_deriv_lines.extend([
                f"  Braced wall (k = {k_bc:.1f}):",
                f"    - BASE (y = 0): Fixed",
                f"        uz = 0 (no out-of-plane deflection)",
                f"    - TOP (y = hw): Fixed",
                f"        uz = 0 (no out-of-plane deflection)",
                f"",
                f"  Physical interpretation: Wall is restrained against",
                f"  out-of-plane deflection at both base and top by floor/roof",
                f"  diaphragms. Maximum deflection expected at mid-height.",
            ])
        
        bc_deriv_lines.extend([
            f"",
            f"  ═══════════════════════════════════════════════════════════════",
            f"  BOUNDARY CONDITION SUMMARY TABLE",
            f"  ═══════════════════════════════════════════════════════════════",
            f"",
            f"  Location    | In-Plane (ux)  | In-Plane (uy)  | Out-of-Plane (uz)",
            f"  ------------|----------------|----------------|------------------",
        ])
        
        if inp.bracing == 'cantilever':
            bc_deriv_lines.extend([
                f"  Base (y=0)  | Fixed (=0)     | Fixed (=0)     | Fixed (=0)",
                f"  Top (y=hw)  | Free           | Free           | Free",
            ])
        elif inp.bracing == 'braced_unrestrained':
            bc_deriv_lines.extend([
                f"  Base (y=0)  | Fixed (=0)     | Fixed (=0)     | Fixed (=0)",
                f"  Top (y=hw)  | Fixed (=0)     | Free           | Fixed (=0)",
            ])
        elif inp.bracing == 'braced_restrained':
            bc_deriv_lines.extend([
                f"  Base (y=0)  | Fixed (=0)     | Fixed (=0)     | Fixed (=0)",
                f"  Top (y=hw)  | Fixed (=0)     | Fixed (=0)     | Fixed (=0)",
            ])
        
        bc_deriv_lines.extend([
            f"",
            f"  Note: Fixed = prescribed zero displacement; Free = unconstrained",
        ])
        
        derivations['fem_boundary_conditions'] = "\n".join(bc_deriv_lines)
        
        # ─────────────────────────────────────────────────────────────────────
        # REINFORCEMENT PROPERTIES
        # ─────────────────────────────────────────────────────────────────────
        results['Pu'] = self.Pu
        results['Vu_ip'] = self.Vu_ip
        results['Vu_oop'] = self.Vu_oop
        
        # Vertical reinforcement
        Ab_v = BAR_AREAS_IN2.get(inp.vert_bar_no, 0.31)
        n_v = max(1, int(self.crit_width / inp.vert_bar_spacing_in))
        As_v = n_v * Ab_v * inp.vert_faces
        rho_L = As_v / self.Ag_net
        
        # Horizontal reinforcement
        Ab_h = BAR_AREAS_IN2.get(inp.horiz_bar_no, 0.31)
        n_h = max(1, int(inp.hw_in / inp.horiz_bar_spacing_in))
        As_h = n_h * Ab_h * inp.horiz_faces
        rho_t = As_h / (inp.hw_in * inp.h_in)
        
        results['As_v'] = As_v
        results['As_h'] = As_h
        results['rho_L'] = rho_L
        results['rho_t'] = rho_t
        
        derivations['reinforcement'] = f"""Reinforcement Calculation:
        
        VERTICAL (Longitudinal):
        Ab = {Ab_v:.2f} in^2 (#{inp.vert_bar_no})
        n = Lw_crit / s = {self.crit_width:.1f} / {inp.vert_bar_spacing_in:.1f} = {n_v} bars
        As,v = n x Ab x faces = {n_v} x {Ab_v:.2f} x {inp.vert_faces} = {As_v:.2f} in^2
        rho_L = As,v / Ag,net = {As_v:.2f} / {self.Ag_net:.0f} = {rho_L:.4f}
        
        HORIZONTAL (Transverse):
        Ab = {Ab_h:.2f} in^2 (#{inp.horiz_bar_no})
        n = hw / s = {inp.hw_in:.1f} / {inp.horiz_bar_spacing_in:.1f} = {n_h} bars
        As,h = n x Ab x faces = {n_h} x {Ab_h:.2f} x {inp.horiz_faces} = {As_h:.2f} in^2
        rho_t = As,h / (hw x h) = {As_h:.2f} / ({inp.hw_in:.1f} x {inp.h_in:.1f}) = {rho_t:.4f}"""
        
        # ─────────────────────────────────────────────────────────────────────
        # IN-PLANE SHEAR CHECK (ACI 11.5.4)
        # ─────────────────────────────────────────────────────────────────────
        # Use the critical net width for in-plane shear calculations
        shear_ip = ACI318Chapter11.Vn_inplane(
            inp.fc_psi,
            self.crit_width,  # effective wall length considering openings
            inp.h_in,
            inp.hw_in,
            rho_t,
            inp.fy_psi,
            -self.Pu if self.Pu < 0 else 0
        )
        phi_v = 0.75
        
        results['phi_Vn_ip'] = phi_v * shear_ip['Vn']
        results['phi_Vc_ip'] = phi_v * shear_ip['Vc']
        results['phi_Vs_ip'] = phi_v * shear_ip['Vs']
        results['Vn_max_ip'] = shear_ip['Vn_max']
        results['UC_V_ip'] = self.Vu_ip / max(1, results['phi_Vn_ip'])
        results['V_ip_ok'] = results['UC_V_ip'] <= 1.0
        results['shear_ip_details'] = shear_ip
        derivations['shear_ip'] = shear_ip['derivation']
        
        # Check if shear reinforcement required
        Vc_threshold_ip = 0.5 * phi_v * shear_ip['alpha_c'] * LAMBDA_NW * math.sqrt(inp.fc_psi) * shear_ip['Acv']
        results['shear_reinf_req_ip'] = self.Vu_ip > Vc_threshold_ip
        
        # ─────────────────────────────────────────────────────────────────────
        # OUT-OF-PLANE SHEAR CHECK (ACI 22.5)
        # ─────────────────────────────────────────────────────────────────────
        d_eff = inp.h_in - inp.cover_in - BAR_DIAMETERS_IN.get(inp.vert_bar_no, 0.625) / 2
        Av = inp.tie_legs * BAR_AREAS_IN2.get(inp.tie_bar_no, 0.11) if inp.tie_legs > 0 else 0
        
        shear_oop = ACI318Chapter11.Vn_outofplane(
            inp.fc_psi, self.crit_width, d_eff,
            Av, inp.tie_spacing_in, inp.fy_psi,
            self.Pu, self.Ag_net
        )
        
        results['phi_Vn_oop'] = phi_v * shear_oop['Vn']
        results['phi_Vc_oop'] = phi_v * shear_oop['Vc']
        results['phi_Vs_oop'] = phi_v * shear_oop['Vs']
        results['Vn_max_oop'] = shear_oop['Vn_max']
        results['UC_V_oop'] = self.Vu_oop / max(1, results['phi_Vn_oop']) if self.Vu_oop > 0 else 0
        results['V_oop_ok'] = results['UC_V_oop'] <= 1.0
        results['shear_oop_details'] = shear_oop
        derivations['shear_oop'] = shear_oop['derivation']
        
        # Check if ties required
        Vc_threshold_oop = 0.5 * phi_v * shear_oop['Vc']
        results['shear_reinf_req_oop'] = self.Vu_oop > Vc_threshold_oop
        
        # ─────────────────────────────────────────────────────────────────────
        # REINFORCEMENT CHECKS (ACI 11.6, 11.7)
        # ─────────────────────────────────────────────────────────────────────
        high_shear = results['shear_reinf_req_ip'] or results['shear_reinf_req_oop']
        rho_L_min, rho_t_min, rho_deriv = ACI318Chapter11.min_reinf_ratios(
            inp.fc_psi, inp.fy_psi, inp.vert_bar_no, high_shear
        )
        
        results['rho_L_min'] = rho_L_min
        results['rho_L_ok'] = rho_L >= rho_L_min
        results['rho_t_min'] = rho_t_min
        results['rho_t_ok'] = rho_t >= rho_t_min
        derivations['rho_min'] = rho_deriv
        
        # Spacing checks (use critical width for bar spacing limits when openings reduce the effective width)
        s_v_max, s_h_max, s_deriv = ACI318Chapter11.max_spacing(
            inp.h_in, high_shear, self.crit_width
        )
        results['s_v_max'] = s_v_max
        results['s_v_ok'] = inp.vert_bar_spacing_in <= s_v_max
        results['s_h_max'] = s_h_max
        results['s_h_ok'] = inp.horiz_bar_spacing_in <= s_h_max
        derivations['spacing'] = s_deriv

        # ─────────────────────────────────────────────────────────────────────
        # DEVELOPMENT LENGTH & OPENING REINFORCEMENT
        # ─────────────────────────────────────────────────────────────────────
        # For walls with openings, provide additional reinforcement around
        # each opening. ACI 25.4 requires reinforcing bars to be developed
        # beyond the opening edges. In this implementation we assume
        # two #5 bars at each edge of every opening (top, bottom, left and
        # right) and compute the required development length using the
        # basic ACI 25.4.2.3 equation. The provided reinforcement area and
        # bar lengths are reported, but no embedment check is enforced.

        opening_reinf: List[Dict[str, Any]] = []
        total_opening_As = 0.0
        if inp.openings:
            # Development length for #5 bars in tension (ACI 25.4.2.3)
            Ld_open = development_length_tension(inp.fy_psi, inp.fc_psi, 5)
            bar_area = BAR_AREAS_IN2.get(5, 0.31)

            for i, op in enumerate(inp.openings, start=1):
                # Length of straight bars that extend beyond the opening by Ld on each side
                horiz_bar_len = op.width_in + 2 * Ld_open
                vert_bar_len = op.height_in + 2 * Ld_open
                # Two bars per edge (4 bars total per opening)
                As_open = 4 * bar_area
                total_opening_As += As_open
                opening_reinf.append({
                    'opening_index': i,
                    'width_in': op.width_in,
                    'height_in': op.height_in,
                    'Ld_req_in': Ld_open,
                    'horiz_bar_len_in': horiz_bar_len,
                    'vert_bar_len_in': vert_bar_len,
                    'bar_area_in2': bar_area,
                    'As_opening_in2': As_open
                })

        # Store opening reinforcement summary
        results['openings_reinf'] = opening_reinf
        results['As_openings'] = total_opening_As

        # Build derivation text summarising development length and bar sizing
        if opening_reinf:
            op_deriv_lines = [
                f"Development length Ld (ACI 25.4.2.3) for #5 bars:" \
                f" Ld = 3/40 * fy / (λ*√fc) * db = 3/40 * {inp.fy_psi:.0f}/(1*√{inp.fc_psi:.0f})" \
                f"*{BAR_DIAMETERS_IN[5]:.3f} = {Ld_open:.2f} in (≥12 in)"
            ]
            for rec in opening_reinf:
                op_deriv_lines.append(
                    f"Opening {rec['opening_index']}: width={rec['width_in']:.1f} in, height={rec['height_in']:.1f} in\n" \
                    f"  Provide two #5 bars at each edge. " \
                    f"Horizontal bar length = w + 2Ld = {rec['horiz_bar_len_in']:.1f} in, " \
                    f"vertical bar length = h + 2Ld = {rec['vert_bar_len_in']:.1f} in.\n" \
                    f"  Total area provided = {rec['As_opening_in2']:.2f} in^2"
                )
            derivations['openings_reinf'] = "\n\n".join(op_deriv_lines)
        
        # ─────────────────────────────────────────────────────────────────────
        # SLENDERNESS (ACI 11.4.1.3, 11.8)
        # ─────────────────────────────────────────────────────────────────────
        k, k_deriv = ACI318Chapter11.effective_length_factor(inp.bracing)
        kLc_h = k * inp.hw_in / inp.h_in
        results['k'] = k
        results['kLc_h'] = kLc_h
        results['slender_ok'] = kLc_h <= 100
        derivations['slenderness'] = f"""Slenderness Check (ACI 11.8):
        
        {k_deriv}
        
        kLc/h = {k:.1f} x {inp.hw_in:.1f} / {inp.h_in:.1f} = {kLc_h:.1f}
        
        Limit: kLc/h <= 100
        Status: {kLc_h:.1f} {'<=' if kLc_h <= 100 else '>'} 100 -> {'OK' if kLc_h <= 100 else 'NG'}"""
        
        # Serviceability deflection limit
        results['delta_limit'] = inp.hw_in / 150
        results['delta_service'] = results['max_total_disp']
        results['delta_service_ok'] = abs(results['delta_service']) <= results['delta_limit']
        
        # ─────────────────────────────────────────────────────────────────────
        # P-M INTERACTION DIAGRAMS
        # ─────────────────────────────────────────────────────────────────────
        # Out-of-plane P-M
        layers_oop = self._get_layers_outofplane()
        pm_oop = PMDiagram(inp.fc_psi, inp.fy_psi, self.crit_width, inp.h_in, 
                          layers_oop, "Out-of-Plane")
        pts_oop, keys_oop = pm_oop.generate_curve()
        
        results['pm_oop'] = {
            'points': pts_oop, 
            'keys': keys_oop,
            'diagram': pm_oop
        }
        derivations['pm_oop'] = pm_oop.get_derivation()
        
        # In-plane P-M
        layers_ip = self._get_layers_inplane()
        # For in-plane bending, use the critical net width as the section height (depth)
        pm_ip = PMDiagram(inp.fc_psi, inp.fy_psi, inp.h_in, self.crit_width,
                         layers_ip, "In-Plane")
        pts_ip, keys_ip = pm_ip.generate_curve()
        
        results['pm_ip'] = {
            'points': pts_ip, 
            'keys': keys_ip,
            'diagram': pm_ip
        }
        derivations['pm_ip'] = pm_ip.get_derivation()
        
        # Capacities at Pu (using capped values for design)
        phi_Mn_oop = pm_oop.get_capacity_at_Pu(pts_oop, self.Pu, use_factored=True, use_capped=True)
        phi_Mn_ip = pm_ip.get_capacity_at_Pu(pts_ip, self.Pu, use_factored=True, use_capped=True)
        
        results['phi_Mn_oop'] = phi_Mn_oop
        results['phi_Mn_ip'] = phi_Mn_ip
        
        # Full Po and 0.80Po
        results['Po_oop'] = pm_oop.Po
        results['Po_080_oop'] = pm_oop.Po_cap
        results['phi_Po_080_oop'] = 0.65 * pm_oop.Po_cap
        
        results['Po_ip'] = pm_ip.Po
        results['Po_080_ip'] = pm_ip.Po_cap
        results['phi_Po_080_ip'] = 0.65 * pm_ip.Po_cap
        
        # Unity checks
        results['UC_PM_oop'] = abs(self.Mu_oop) / max(1, phi_Mn_oop) if self.Pu <= results['phi_Po_080_oop'] else 999
        results['UC_PM_ip'] = abs(self.Mu_ip) / max(1, phi_Mn_ip) if self.Pu <= results['phi_Po_080_ip'] else 999
        results['PM_oop_ok'] = results['UC_PM_oop'] <= 1.0 and self.Pu <= results['phi_Po_080_oop']
        results['PM_ip_ok'] = results['UC_PM_ip'] <= 1.0 and self.Pu <= results['phi_Po_080_ip']
        
        # ─────────────────────────────────────────────────────────────────────
        # OVERALL STATUS
        # ─────────────────────────────────────────────────────────────────────
        results['all_ok'] = all([
            results['h_ok'],
            results['V_ip_ok'],
            results['V_oop_ok'],
            results['rho_L_ok'],
            results['rho_t_ok'],
            results['s_v_ok'],
            results['s_h_ok'],
            results['slender_ok'],
            results['delta_service_ok'],
            results['PM_oop_ok'],
            results['PM_ip_ok'],
        ])
        
        # Compile issues
        issues = []
        if not results['h_ok']:
            issues.append(f"Thickness {inp.h_in:.1f}\" < min {h_min:.1f}\"")
        if not results['PM_oop_ok']:
            issues.append(f"Out-of-plane P-M UC = {results['UC_PM_oop']:.2f} > 1.0")
        if not results['PM_ip_ok']:
            issues.append(f"In-plane P-M UC = {results['UC_PM_ip']:.2f} > 1.0")
        if not results['V_ip_ok']:
            issues.append(f"In-plane shear UC = {results['UC_V_ip']:.2f} > 1.0")
        if not results['V_oop_ok']:
            issues.append(f"Out-of-plane shear UC = {results['UC_V_oop']:.2f} > 1.0")
        if not results['rho_L_ok']:
            issues.append(f"rho_L = {rho_L:.4f} < min {rho_L_min:.4f}")
        if not results['rho_t_ok']:
            issues.append(f"rho_t = {rho_t:.4f} < min {rho_t_min:.4f}")
        if not results['s_v_ok']:
            issues.append(f"Vert spacing {inp.vert_bar_spacing_in:.1f}\" > max {s_v_max:.1f}\"")
        if not results['s_h_ok']:
            issues.append(f"Horiz spacing {inp.horiz_bar_spacing_in:.1f}\" > max {s_h_max:.1f}\"")
        if results['two_layers_req'] and not results['two_layers_prov']:
            issues.append("Two layers required (h > 10 in)")
        if results['shear_reinf_req_oop'] and inp.tie_legs == 0:
            issues.append("Shear ties required for out-of-plane (Vu > 0.5*phi*Vc)")
        if not results['delta_service_ok']:
            issues.append(f"Deflection {results['delta_service']:.4f}\" > limit {results['delta_limit']:.4f}\"")

        # Opening reinforcement is always provided as two #5 bars at each edge.
        # No embedment check is enforced here because bar lengths include the
        # required development length beyond the opening.  The designer
        # should verify embedment as needed.
        
        results['issues'] = issues
        results['derivations'] = derivations
        
        return results
    
    def _get_layers_outofplane(self) -> List[Tuple[float, float]]:
        """Reinforcement layers for out-of-plane bending."""
        h = self.inp.h_in
        cover = self.inp.cover_in
        db = BAR_DIAMETERS_IN.get(self.inp.vert_bar_no, 0.625)
        
        d1 = cover + db / 2
        d2 = h - cover - db / 2
        
        Ab = BAR_AREAS_IN2.get(self.inp.vert_bar_no, 0.31)
        n = max(1, int(self.crit_width / self.inp.vert_bar_spacing_in))
        As = n * Ab
        
        layers = [(d1, As)]
        if self.inp.vert_faces == 2:
            layers.append((d2, As))
        return layers
    
    def _get_layers_inplane(self) -> List[Tuple[float, float]]:
        """Reinforcement layers for in-plane bending.

        For in-plane bending, the effective span of reinforcement along the wall
        should correspond to the critical net width (minimum pier width)
        rather than the full wall length when openings are present. This
        ensures that reinforcement layers are placed within the most
        critical section of the wall.
        """
        # Use the critical net width for distributing reinforcement layers
        Lw_eff = self.crit_width
        cover = self.inp.cover_in
        db = BAR_DIAMETERS_IN.get(self.inp.vert_bar_no, 0.625)

        Ab = BAR_AREAS_IN2.get(self.inp.vert_bar_no, 0.31)
        n_height = max(1, int(self.inp.hw_in / self.inp.vert_bar_spacing_in))
        As_total = n_height * Ab * self.inp.vert_faces

        # Number of layers along the effective width (at least 2 to define neutral axis depth)
        n_along = max(2, int(Lw_eff / self.inp.vert_bar_spacing_in))
        layers = []
        for i in range(n_along):
            # Linear spacing of reinforcement along Lw_eff, measured from compression face
            d_i = cover + db/2 + i * (Lw_eff - 2*cover - db) / max(1, n_along - 1)
            layers.append((d_i, As_total / n_along))

        return layers


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================
def plot_pm_diagram(
    points: List[PMPoint],
    key_points: List[PMPoint],
    Pu: float, Mu: float,
    title: str, outpath: str,
    Po: float,
    Po_080: float,
    figsize: Tuple[float, float] = (11, 9),
    dpi: int = 300
):
    """
    Plot P-M interaction diagram with:
    - Full Pn curve (not capped at 0.80Po)
    - Clear 0.80Po and 0.80*phi*Po cutoff lines
    - All 4 quadrants
    - Key control points
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Extract FULL values (not capped)
    Pn = np.array([pt.Pn / 1000 for pt in points])
    Mn = np.array([pt.Mn / 12000 for pt in points])
    phi_Pn = np.array([pt.phi_Pn / 1000 for pt in points])
    phi_Mn = np.array([pt.phi_Mn / 12000 for pt in points])
    
    # Sort by c
    c_vals = np.array([pt.c if pt.c != float('inf') else 1e9 for pt in points])
    sort_idx = np.argsort(c_vals)[::-1]
    
    Pn_sorted = Pn[sort_idx]
    Mn_sorted = Mn[sort_idx]
    phi_Pn_sorted = phi_Pn[sort_idx]
    phi_Mn_sorted = phi_Mn[sort_idx]
    
    # Mirror for all 4 quadrants
    Mn_full = np.concatenate([Mn_sorted, -Mn_sorted[::-1]])
    Pn_full = np.concatenate([Pn_sorted, Pn_sorted[::-1]])
    phi_Mn_full = np.concatenate([phi_Mn_sorted, -phi_Mn_sorted[::-1]])
    phi_Pn_full = np.concatenate([phi_Pn_sorted, phi_Pn_sorted[::-1]])
    
    # Plot FULL curves (not capped)
    ax.plot(Mn_full, Pn_full, 'b--', linewidth=1.5, label='Nominal (Pn, Mn)')
    ax.plot(phi_Mn_full, phi_Pn_full, 'b-', linewidth=2.5, label='Design (phi*Pn, phi*Mn)')
    
    # Po and 0.80Po cutoff lines
    Po_kip = Po / 1000
    Po_080_kip = Po_080 / 1000
    phi_Po_080_kip = 0.65 * Po_080_kip
    
    Mn_range = max(abs(Mn_sorted.max()), abs(Mu/12000)) * 1.15
    
    # Full Po line (dashed gray)
    ax.axhline(y=Po_kip, color='gray', linestyle=':', linewidth=1.5, alpha=0.7,
               label=f'Po = {Po_kip:.0f} kip')
    
    # 0.80Po line (solid red)
    ax.axhline(y=Po_080_kip, color='red', linestyle='-', linewidth=2,
               label=f'0.80Po = {Po_080_kip:.0f} kip (ACI 22.4.2.1)')
    
    # phi(0.80Po) line (solid darkred)
    ax.axhline(y=phi_Po_080_kip, color='darkred', linestyle='-', linewidth=2,
               label=f'phi(0.80Po) = {phi_Po_080_kip:.0f} kip')
    
    # Shade region above 0.80Po
    ax.fill_between([-Mn_range, Mn_range], Po_080_kip, Po_kip * 1.1,
                    color='red', alpha=0.1, label='Above design limit')
    
    # Plot key points (on FULL curve)
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*', 'X']
    colors_key = ['red', 'orange', 'green', 'purple', 'brown', 'darkblue', 'magenta', 'cyan', 'olive']
    
    for i, kp in enumerate(key_points):
        kp_Mn = kp.phi_Mn / 12000
        kp_Pn = kp.phi_Pn / 1000  # Use FULL phi_Pn
        marker = markers[i % len(markers)]
        color = colors_key[i % len(colors_key)]
        
        ax.plot(kp_Mn, kp_Pn, marker, markersize=12, color=color,
                markeredgecolor='black', markeredgewidth=1.5,
                label=f"{kp.label}" if kp.label else f"Pt {i+1}")
        ax.plot(-kp_Mn, kp_Pn, marker, markersize=12, color=color,
                markeredgecolor='black', markeredgewidth=1.5)
    
    # Plot demand point
    Mu_kft = Mu / 12000
    Pu_kip = Pu / 1000
    ax.plot(Mu_kft, Pu_kip, '*', markersize=22, color='lime',
            markeredgecolor='darkgreen', markeredgewidth=2,
            label=f'DEMAND ({Mu_kft:.0f} k-ft, {Pu_kip:.0f} kip)', zorder=10)
    
    # Axes
    ax.axhline(y=0, color='k', linewidth=1)
    ax.axvline(x=0, color='k', linewidth=1)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    ax.set_xlabel('Moment (kip-ft)', fontsize=12)
    ax.set_ylabel('Axial Force (kip)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Set limits
    Pn_max_plot = max(Po_kip, Pu_kip) * 1.15
    Pn_min_plot = min(Pn_sorted.min(), 0) * 1.15
    
    ax.set_xlim(-Mn_range, Mn_range)
    ax.set_ylim(Pn_min_plot, Pn_max_plot)
    
    # Legend - outside plot
    ax.legend(loc='upper right', fontsize=8, framealpha=0.95, ncol=1)
    
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def _auto_scale_disp(max_disp, span, target_frac=0.10, min_scale=10.0, max_scale=2000.0):
    """Auto-scale deflection for visualization."""
    if max_disp <= 1e-9:
        return min_scale
    s = target_frac * span / max_disp
    return float(max(min_scale, min(s, max_scale)))


def plot_deformed_mesh(
    fem_result: Dict[str, Any],
    openings: List[Opening],
    title: str,
    outpath: str,
    component: str = 'mag',  # 'ux', 'uy', 'mag', 'uz'
    figsize: Tuple[float, float] = (10, 8),
    dpi: int = 300
):
    """Plot deformed mesh with displacement contour."""
    import matplotlib.tri as mtri
    
    nodes = fem_result['nodes']
    elements = fem_result['elements']
    
    if component == 'uz':
        u_plot = fem_result.get('uz', np.zeros(len(nodes)))
        label = 'Out-of-Plane Displacement uz (in)'
    elif component == 'ux':
        u_plot = fem_result.get('ux', np.zeros(len(nodes)))
        label = 'In-Plane Displacement ux (in)'
    elif component == 'uy':
        u_plot = fem_result.get('uy', np.zeros(len(nodes)))
        label = 'In-Plane Displacement uy (in)'
    else:  # magnitude
        ux = fem_result.get('ux', np.zeros(len(nodes)))
        uy = fem_result.get('uy', np.zeros(len(nodes)))
        u_plot = np.sqrt(ux**2 + uy**2)
        label = 'Total Displacement Magnitude (in)'
    
    u_mag = np.abs(u_plot)
    
    if np.max(u_mag) < 1e-12:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, f'No {component} displacement', 
                ha='center', va='center', fontsize=14, transform=ax.transAxes)
        ax.set_title(title)
        fig.savefig(outpath, dpi=dpi)
        plt.close(fig)
        return
    
    x = np.array([n[0] for n in nodes])
    y = np.array([n[1] for n in nodes])
    triangles = np.array(elements)
    
    # Deformation scale
    max_u = np.max(u_mag)
    max_dim = max(np.max(x) - np.min(x), np.max(y) - np.min(y))
    scale = _auto_scale_disp(max_u, max_dim)
    
    # For in-plane, deform mesh; for out-of-plane, just show contour
    if component in ['ux', 'uy', 'mag']:
        ux = fem_result.get('ux', np.zeros(len(nodes)))
        uy = fem_result.get('uy', np.zeros(len(nodes)))
        x_def = x + ux * scale
        y_def = y + uy * scale
    else:
        x_def = x
        y_def = y
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Undeformed mesh
    triang = mtri.Triangulation(x, y, triangles)
    ax.triplot(triang, 'k-', linewidth=0.3, alpha=0.3)
    
    # Deformed mesh with contour
    triang_def = mtri.Triangulation(x_def, y_def, triangles)
    levels = np.linspace(0, np.max(u_mag), 20) if component != 'uz' else np.linspace(np.min(u_plot), np.max(u_plot), 20)
    
    if np.max(u_mag) > 0:
        contour = ax.tricontourf(triang_def, u_plot if component == 'uz' else u_mag, 
                                 levels=levels, cmap='jet', alpha=0.8)
        cbar = fig.colorbar(contour, ax=ax, shrink=0.8)
        cbar.set_label(label, fontsize=11)
    
    ax.triplot(triang_def, 'b-', linewidth=0.5, alpha=0.6)
    
    # Openings
    for op in openings:
        rect = Rectangle(
            (op.left, op.bottom), op.width_in, op.height_in,
            fill=False, edgecolor='red', linestyle='--', linewidth=2
        )
        ax.add_patch(rect)
    
    ax.set_xlabel('Wall Length (in)', fontsize=11)
    ax.set_ylabel('Wall Height (in)', fontsize=11)
    ax.set_title(f'{title}\n(Scale = {scale:.0f}x, Max = {max_u:.4f} in)', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(outpath, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
# PDF REPORT WITH TITLE BLOCK AT TOP
# =============================================================================
class TitleBlockCanvas(canvas.Canvas):
    """Custom canvas that adds title block to TOP of every page."""
    
    def __init__(self, *args, project_info: ProjectInfo = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.project_info = project_info or ProjectInfo()
        self.pages = []
    
    def showPage(self):
        self.pages.append(dict(self.__dict__))
        self._startPage()
    
    def save(self):
        page_count = len(self.pages)
        for i, page in enumerate(self.pages):
            self.__dict__.update(page)
            self._draw_title_block(i + 1, page_count)
            super().showPage()
        super().save()
    
    def _draw_title_block(self, page_num: int, total_pages: int):
        """Draw engineering title block at TOP of each page - clean layout."""
        width, height = LETTER
        
        # Title block dimensions - compact but readable
        tb_height = 0.6 * inch
        margin = 0.5 * inch
        tb_y = height - margin - tb_height  # Position at top
        tb_width = width - 2 * margin
        
        # Draw outer border
        self.setStrokeColor(colors.black)
        self.setLineWidth(1.5)
        self.rect(margin, tb_y, tb_width, tb_height)
        
        # Column positions (5 columns)
        logo_width = 1.0 * inch
        col1 = margin + logo_width  # After logo
        col2 = col1 + 2.2 * inch    # Project info
        col3 = col2 + 1.5 * inch    # Date
        col4 = col3 + 1.4 * inch    # Designer/Checker
        col5 = col4 + 1.0 * inch    # Sheet number
        
        # Draw vertical dividers
        self.setLineWidth(0.5)
        for x in [col1, col2, col3, col4]:
            self.line(x, tb_y, x, tb_y + tb_height)
        
        # Logo area (left cell)
        logo_x = margin + 2
        logo_y = tb_y + 2
        logo_w = logo_width - 4
        logo_h = tb_height - 4
        
        if self.project_info.logo_path and os.path.exists(self.project_info.logo_path):
            try:
                self.drawImage(self.project_info.logo_path, logo_x, logo_y, 
                              width=logo_w, height=logo_h, preserveAspectRatio=True)
            except:
                self._draw_logo_placeholder(logo_x, logo_y, logo_w, logo_h)
        else:
            self._draw_logo_placeholder(logo_x, logo_y, logo_w, logo_h)
        
        # Text padding
        pad = 4
        top_line = tb_y + tb_height - 12
        mid_line = tb_y + tb_height/2 - 3
        bot_line = tb_y + 8
        
        # Column 1: Project Name and Number
        self.setFont('Helvetica-Bold', 7)
        self.drawString(col1 + pad, top_line, "PROJECT:")
        self.setFont('Helvetica', 7)
        proj_name = (self.project_info.project_name or "N/A")[:28]
        self.drawString(col1 + pad + 45, top_line, proj_name)
        
        self.setFont('Helvetica-Bold', 7)
        self.drawString(col1 + pad, bot_line, "NO:")
        self.setFont('Helvetica', 7)
        proj_no = (self.project_info.project_number or "N/A")[:20]
        self.drawString(col1 + pad + 20, bot_line, proj_no)
        
        # Column 2: Client and Date
        self.setFont('Helvetica-Bold', 7)
        self.drawString(col2 + pad, top_line, "CLIENT:")
        self.setFont('Helvetica', 7)
        client = (self.project_info.client_name or "N/A")[:18]
        self.drawString(col2 + pad + 40, top_line, client)
        
        self.setFont('Helvetica-Bold', 7)
        self.drawString(col2 + pad, bot_line, "DATE:")
        self.setFont('Helvetica', 7)
        date_str = self.project_info.date or datetime.now().strftime('%Y-%m-%d')
        self.drawString(col2 + pad + 30, bot_line, date_str)
        
        # Column 3: Designer and Checker
        self.setFont('Helvetica-Bold', 7)
        self.drawString(col3 + pad, top_line, "BY:")
        self.setFont('Helvetica', 7)
        designer = (self.project_info.designer or "N/A")[:15]
        self.drawString(col3 + pad + 18, top_line, designer)
        
        self.setFont('Helvetica-Bold', 7)
        self.drawString(col3 + pad, bot_line, "CHK:")
        self.setFont('Helvetica', 7)
        checker = (self.project_info.checker or "N/A")[:15]
        self.drawString(col3 + pad + 25, bot_line, checker)
        
        # Column 4: Sheet number and code reference
        self.setFont('Helvetica-Bold', 8)
        self.drawString(col4 + pad, top_line, "SHEET:")
        self.setFont('Helvetica-Bold', 10)
        sheet_str = f"{page_num} of {total_pages}"
        self.drawString(col4 + pad + 38, top_line - 1, sheet_str)
        
        self.setFont('Helvetica', 7)
        self.drawString(col4 + pad, bot_line, "ACI 318-19")
    
    def _draw_logo_placeholder(self, x, y, w, h):
        """Draw placeholder if no logo provided."""
        self.setStrokeColor(colors.grey)
        self.setFillColor(colors.Color(0.95, 0.95, 0.95))
        self.rect(x, y, w, h, fill=1, stroke=1)
        self.setFillColor(colors.black)
        self.setFont('Helvetica-Bold', 8)
        company = (self.project_info.company_name or "LOGO")[:10]
        self.drawCentredString(x + w/2, y + h/2 - 3, company)


def get_styles():
    """Create styles for the report - handles duplicate style names."""
    styles = getSampleStyleSheet()
    
    # Use try/except or check before adding
    custom_styles = {
        'Title2': ParagraphStyle('Title2', parent=styles['Title'], fontSize=16),
        'Section': ParagraphStyle('Section', parent=styles['Heading2'], 
                                  textColor=colors.HexColor('#1a5276')),
        'Subsection': ParagraphStyle('Subsection', parent=styles['Heading3'],
                                     textColor=colors.HexColor('#2874a6')),
        'Pass': ParagraphStyle('Pass', parent=styles['Normal'], textColor=colors.green),
        'Fail': ParagraphStyle('Fail', parent=styles['Normal'], textColor=colors.red),
        'CodeBlock': ParagraphStyle('CodeBlock', parent=styles['Normal'], 
                                    fontName='Courier', fontSize=8,
                                    leftIndent=20, spaceBefore=6, spaceAfter=6),
    }
    
    for name, style in custom_styles.items():
        try:
            styles.add(style)
        except KeyError:
            # Style already exists, update it
            pass
    
    return styles, custom_styles


def generate_report(inp: WallInput, results: Dict, plot_paths: Dict, outpath: str):
    """Generate comprehensive PDF design report with title block at TOP of every page."""
    
    # Ensure date is set
    if not inp.project_info.date:
        inp.project_info.date = datetime.now().strftime('%Y-%m-%d')
    
    styles, custom = get_styles()
    
    story = []
    
    # ─── COVER PAGE ───
    story.append(Spacer(1, 1.5*inch))
    story.append(Paragraph("RC WALL DESIGN REPORT", styles['Title']))
    story.append(Paragraph("Per ACI 318-19 with FEM Analysis", custom['Title2']))
    story.append(Spacer(1, 0.3*inch))
    story.append(Paragraph(f"Project: {inp.project_info.project_name}", styles['Heading2']))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", styles['Normal']))
    story.append(Spacer(1, 0.3*inch))
    
    # Status
    status = "ALL CHECKS PASS" if results['all_ok'] else "DESIGN INADEQUATE"
    st = custom['Pass'] if results['all_ok'] else custom['Fail']
    story.append(Paragraph(f"<b>Design Status: {status}</b>", st))
    
    if results['issues']:
        story.append(Spacer(1, 0.15*inch))
        story.append(Paragraph("<b>Design Issues:</b>", custom['Fail']))
        for iss in results['issues']:
            story.append(Paragraph(f"  - {iss}", custom['Fail']))
    
    story.append(PageBreak())
    
    # ─── TABLE OF CONTENTS ───
    story.append(Paragraph("TABLE OF CONTENTS", custom['Section']))
    toc = [
        "1. Input Data",
        "2. FEM Analysis and Derived Pressures",
        "   2.1 FEM Boundary Conditions",
        "   2.2 Critical Width Calculation",
        "   2.3 Pressure Derivation",
        "3. ACI 318-19 Design Checks Summary",
        "4. Shear Capacity Derivation",
        "5. P-M Interaction Diagrams",
        "6. FEM Displacement Results",
        "7. Appendix: Full Mathematical Derivations"
    ]
    for item in toc:
        story.append(Paragraph(item, styles['Normal']))
    story.append(PageBreak())
    
    # ─── 1. INPUTS ───
    story.append(Paragraph("1. INPUT DATA", custom['Section']))
    
    data = [
        ["Parameter", "Value", "Units"],
        ["Wall Length (Lw)", f"{inp.Lw_in:.1f}", "in"],
        ["Wall Thickness (h)", f"{inp.h_in:.1f}", "in"],
        ["Wall Height (hw)", f"{inp.hw_in:.1f}", "in"],
        ["f'c", f"{inp.fc_psi:.0f}", "psi"],
        ["fy", f"{inp.fy_psi:.0f}", "psi"],
        ["Cover", f"{inp.cover_in:.2f}", "in"],
        ["Vert Reinf", f"#{inp.vert_bar_no} @ {inp.vert_bar_spacing_in:.1f}\" ({inp.vert_faces} face)", ""],
        ["Horiz Reinf", f"#{inp.horiz_bar_no} @ {inp.horiz_bar_spacing_in:.1f}\" ({inp.horiz_faces} face)", ""],
        ["Pu (axial)", f"{inp.Pu_kip:.1f}", "kip"],
        ["Vu,ip (in-plane shear)", f"{inp.Vu_ip_kip:.1f}", "kip"],
        ["Vu,oop (out-of-plane shear)", f"{inp.Vu_oop_kip:.1f}", "kip"],
        ["wu,oop (out-of-plane pressure)", f"{inp.wu_oop_psf:.1f}", "psf"],
        ["Wall Type", inp.wall_type, ""],
        ["Bracing", inp.bracing, ""],
        ["Openings", f"{len(inp.openings)}" if inp.openings else "None", ""],
    ]
    
    tbl = Table(data, colWidths=[3.0*inch, 2.5*inch, 1.0*inch])
    tbl.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
    ]))
    story.append(tbl)
    story.append(PageBreak())
    
    # ─── 2. FEM PRESSURES ───
    story.append(Paragraph("2. FEM ANALYSIS AND DERIVED PRESSURES", custom['Section']))
    
    story.append(Paragraph("2.1 FEM Boundary Conditions", custom['Subsection']))
    story.append(Paragraph(
        "The finite element model applies boundary conditions based on the wall bracing type "
        "per ACI 318-19 Table 11.5.3.2. These conditions determine how the wall is restrained "
        "at its base and top edges for both in-plane and out-of-plane analysis.",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph(results['derivations']['fem_boundary_conditions'].replace('\n', '<br/>'), custom['CodeBlock']))
    story.append(Spacer(1, 0.15*inch))
    
    story.append(Paragraph("2.2 Critical Width Calculation", custom['Subsection']))
    story.append(Paragraph(
        "The critical width is the minimum net width of the wall section over its height, "
        "accounting for any opening deductions. This value is used for shear capacity and "
        "reinforcement calculations per ACI 318-19.",
        styles['Normal']
    ))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph(results['derivations']['critical_width'].replace('\n', '<br/>'), custom['CodeBlock']))
    
    # Critical width summary table
    crit_width_data = [
        ["Parameter", "Value", "Units"],
        ["Gross Wall Length (Lw)", f"{inp.Lw_in:.2f}", "in"],
        ["Critical Net Width", f"{results['crit_width']:.2f}", "in"],
        ["Wall Thickness (h)", f"{inp.h_in:.2f}", "in"],
        ["Net Area (Ag,net)", f"{results['Ag_net']:.2f}", "in²"],
    ]
    tbl_crit = Table(crit_width_data, colWidths=[2.5*inch, 1.5*inch, 1.0*inch])
    tbl_crit.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a5276')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
    ]))
    story.append(Spacer(1, 0.1*inch))
    story.append(tbl_crit)
    story.append(PageBreak())
    
    story.append(Paragraph("2.3 Pressure Derivation", custom['Subsection']))
    story.append(Paragraph(results['derivations']['fem_pressures'].replace('\n', '<br/>'), custom['CodeBlock']))
    
    press_data = [
        ["Pressure Type", "Formula", "Value (psi)"],
        ["Axial (gravity)", "p_ax = Pu / (Lw x hw)", f"{results['p_ax_psi']:.4f}"],
        ["In-Plane Lateral", "p_lat,ip = Vu,ip / (Lw x hw)", f"{results['p_lat_ip_psi']:.4f}"],
        ["Out-of-Plane", "p_oop = wu,oop / 144", f"{results['p_oop_psi']:.4f}"],
    ]
    
    tbl_press = Table(press_data, colWidths=[2.0*inch, 2.5*inch, 1.5*inch])
    tbl_press.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
    ]))
    story.append(tbl_press)
    story.append(PageBreak())
    
    # ─── 3. DESIGN CHECKS SUMMARY ───
    story.append(Paragraph("3. ACI 318-19 DESIGN CHECKS SUMMARY", custom['Section']))
    
    def status_str(ok):
        return "OK" if ok else "NG"
    
    checks = [
        ["Check", "Reference", "Required", "Provided", "Status"],
        ["Min Thickness", "11.3.1.1", f">= {results['h_min']:.2f} in", 
         f"{results['h_provided']:.2f} in", status_str(results['h_ok'])],
        ["Out-of-Plane P-M", "11.5.1", f"UC <= 1.0", f"UC = {results['UC_PM_oop']:.3f}", 
         status_str(results['PM_oop_ok'])],
        ["In-Plane P-M", "11.5.1", f"UC <= 1.0", f"UC = {results['UC_PM_ip']:.3f}",
         status_str(results['PM_ip_ok'])],
        ["In-Plane Shear", "11.5.4", f"UC <= 1.0", f"UC = {results['UC_V_ip']:.3f}",
         status_str(results['V_ip_ok'])],
        ["Out-of-Plane Shear", "22.5", f"UC <= 1.0", f"UC = {results['UC_V_oop']:.3f}",
         status_str(results['V_oop_ok'])],
        ["rho_L (vertical)", "11.6.1", f">= {results['rho_L_min']:.4f}", f"{results['rho_L']:.4f}",
         status_str(results['rho_L_ok'])],
        ["rho_t (horizontal)", "11.6.2", f">= {results['rho_t_min']:.4f}", f"{results['rho_t']:.4f}",
         status_str(results['rho_t_ok'])],
        ["Vert Spacing", "11.7.2.1", f"<= {results['s_v_max']:.1f} in", 
         f"{inp.vert_bar_spacing_in:.1f} in", status_str(results['s_v_ok'])],
        ["Horiz Spacing", "11.7.3.1", f"<= {results['s_h_max']:.1f} in",
         f"{inp.horiz_bar_spacing_in:.1f} in", status_str(results['s_h_ok'])],
        ["Slenderness", "11.8", f"kLc/h <= 100", f"{results['kLc_h']:.1f}",
         status_str(results['slender_ok'])],
        ["Deflection", "Service", f"<= {results['delta_limit']:.4f} in",
         f"{results['delta_service']:.4f} in", status_str(results['delta_service_ok'])],
    ]
    
    tbl2 = Table(checks, colWidths=[1.6*inch, 0.9*inch, 1.5*inch, 1.5*inch, 0.8*inch])
    tbl2.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
    ]))
    story.append(tbl2)
    # ─── DEVELOPMENT LENGTH & OPENING REINFORCEMENT SUMMARY ───
    # Insert this summary inside section 3 if openings are present
    if results.get('openings_reinf'):
        story.append(Spacer(1, 0.15*inch))
        story.append(Paragraph("3.1 Development Length & Opening Reinforcement", custom['Subsection']))
        # Table header
        # Build table summarising development length and reinforcement around openings
        open_table_data = [[
            "Op", "w (in)", "h (in)", "Ld_req (in)",
            "Horiz bar len (in)", "Vert bar len (in)", "As (in^2)"
        ]]
        for rec in results['openings_reinf']:
            open_table_data.append([
                str(rec.get('opening_index', '')),
                f"{rec['width_in']:.2f}",
                f"{rec['height_in']:.2f}",
                f"{rec['Ld_req_in']:.2f}",
                f"{rec['horiz_bar_len_in']:.2f}",
                f"{rec['vert_bar_len_in']:.2f}",
                f"{rec['As_opening_in2']:.2f}"
            ])
        # Define column widths to fit table within page (7 columns)
        col_widths = [0.4*inch, 0.8*inch, 0.8*inch, 0.8*inch, 1.2*inch, 1.2*inch, 0.8*inch]
        tbl_open = Table(open_table_data, colWidths=col_widths)
        tbl_open.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a5276')),
            ('TEXTCOLOR', (0,0), (-1,0), colors.white),
            ('FONTSIZE', (0,0), (-1,-1), 8),
            ('ALIGN', (1,0), (-1,-1), 'CENTER'),
        ]))
        story.append(tbl_open)
        story.append(Spacer(1, 0.1*inch))
        story.append(Paragraph(
            f"<b>Total Additional Reinforcement around Openings:</b> {results['As_openings']:.2f} in^2",
            styles['Normal']
        ))
    story.append(PageBreak())
    
    # ─── 4. SHEAR CAPACITY DERIVATION ───
    story.append(Paragraph("4. SHEAR CAPACITY DERIVATION", custom['Section']))
    
    story.append(Paragraph("4.1 In-Plane Shear (ACI 11.5.4)", custom['Subsection']))
    story.append(Paragraph(results['derivations']['shear_ip'].replace('\n', '<br/>'), custom['CodeBlock']))
    
    shear_ip_data = [
        ["Parameter", "Value", "Units"],
        ["Vu,ip (demand)", f"{results['Vu_ip']/1000:.2f}", "kip"],
        ["Vc", f"{results['shear_ip_details']['Vc']/1000:.2f}", "kip"],
        ["Vs", f"{results['shear_ip_details']['Vs']/1000:.2f}", "kip"],
        ["phi*Vc", f"{results['phi_Vc_ip']/1000:.2f}", "kip"],
        ["phi*Vs", f"{results['phi_Vs_ip']/1000:.2f}", "kip"],
        ["phi*Vn", f"{results['phi_Vn_ip']/1000:.2f}", "kip"],
        ["phi*Vn,max", f"{0.75*results['Vn_max_ip']/1000:.2f}", "kip"],
        ["UC = Vu/phi*Vn", f"{results['UC_V_ip']:.3f}", ""],
    ]
    tbl_shear_ip = Table(shear_ip_data, colWidths=[2.5*inch, 1.5*inch, 1.0*inch])
    tbl_shear_ip.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a5276')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
    ]))
    story.append(tbl_shear_ip)
    story.append(Spacer(1, 0.2*inch))
    
    story.append(Paragraph("4.2 Out-of-Plane Shear (ACI 22.5)", custom['Subsection']))
    story.append(Paragraph(results['derivations']['shear_oop'].replace('\n', '<br/>'), custom['CodeBlock']))
    
    shear_oop_data = [
        ["Parameter", "Value", "Units"],
        ["Vu,oop (demand)", f"{results['Vu_oop']/1000:.2f}", "kip"],
        ["Vc", f"{results['shear_oop_details']['Vc']/1000:.2f}", "kip"],
        ["Vs", f"{results['shear_oop_details']['Vs']/1000:.2f}", "kip"],
        ["phi*Vc", f"{results['phi_Vc_oop']/1000:.2f}", "kip"],
        ["phi*Vs", f"{results['phi_Vs_oop']/1000:.2f}", "kip"],
        ["phi*Vn", f"{results['phi_Vn_oop']/1000:.2f}", "kip"],
        ["UC = Vu/phi*Vn", f"{results['UC_V_oop']:.3f}", ""],
    ]
    tbl_shear_oop = Table(shear_oop_data, colWidths=[2.5*inch, 1.5*inch, 1.0*inch])
    tbl_shear_oop.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a5276')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
    ]))
    story.append(tbl_shear_oop)
    story.append(PageBreak())
    
    # ─── 5. P-M INTERACTION DIAGRAMS ───
    story.append(Paragraph("5. P-M INTERACTION DIAGRAMS", custom['Section']))
    
    # Out-of-plane
    story.append(Paragraph("5.1 Out-of-Plane P-M Interaction", custom['Subsection']))
    story.append(Paragraph(
        f"<b>Design Demand:</b> Pu = {results['Pu']/1000:.1f} kip, Mu,oop = {results['Mu_oop']/12000:.1f} kip-ft",
        styles['Normal']
    ))
    story.append(Paragraph(
        f"<b>Capacity:</b> phi*Pn,max = {results['phi_Po_080_oop']/1000:.0f} kip (at 0.80Po), "
        f"phi*Mn @ Pu = {results['phi_Mn_oop']/12000:.0f} kip-ft",
        styles['Normal']
    ))
    story.append(Paragraph(
        f"<b>Unity Check:</b> UC = {results['UC_PM_oop']:.3f} {'<= 1.0 OK' if results['PM_oop_ok'] else '> 1.0 NG'}",
        custom['Pass'] if results['PM_oop_ok'] else custom['Fail']
    ))
    
    keys_oop = results['pm_oop']['keys']
    kp_data = [["Pt", "Description", "c (in)", "eps_t", "Pn (kip)", "phi*Pn (kip)", "phi*Mn (k-ft)"]]
    for i, kp in enumerate(keys_oop, 1):
        c_str = "inf" if kp.c == float('inf') else f"{kp.c:.2f}"
        eps_str = f"{kp.eps_t:.5f}" if kp.eps_t < 1 else ">1"
        kp_data.append([
            str(i), kp.label[:35] if kp.label else "-",
            c_str, eps_str,
            f"{kp.Pn/1000:.0f}", f"{kp.phi_Pn/1000:.0f}", f"{kp.phi_Mn/12000:.0f}"
        ])
    
    tbl_kp_oop = Table(kp_data, colWidths=[0.3*inch, 2.3*inch, 0.6*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.7*inch])
    tbl_kp_oop.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a5276')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('ALIGN', (2,0), (-1,-1), 'CENTER'),
    ]))
    story.append(tbl_kp_oop)
    
    if 'pm_oop' in plot_paths:
        story.append(Spacer(1, 0.1*inch))
        story.append(Image(plot_paths['pm_oop'], width=6.5*inch, height=5*inch))
    
    story.append(PageBreak())
    
    # In-plane
    story.append(Paragraph("5.2 In-Plane P-M Interaction", custom['Subsection']))
    story.append(Paragraph(
        f"<b>Design Demand:</b> Pu = {results['Pu']/1000:.1f} kip, Mu,ip = {results['Mu_ip']/12000:.1f} kip-ft",
        styles['Normal']
    ))
    story.append(Paragraph(
        f"<b>Capacity:</b> phi*Pn,max = {results['phi_Po_080_ip']/1000:.0f} kip (at 0.80Po), "
        f"phi*Mn @ Pu = {results['phi_Mn_ip']/12000:.0f} kip-ft",
        styles['Normal']
    ))
    story.append(Paragraph(
        f"<b>Unity Check:</b> UC = {results['UC_PM_ip']:.3f} {'<= 1.0 OK' if results['PM_ip_ok'] else '> 1.0 NG'}",
        custom['Pass'] if results['PM_ip_ok'] else custom['Fail']
    ))
    
    keys_ip = results['pm_ip']['keys']
    kp_data_ip = [["Pt", "Description", "c (in)", "eps_t", "Pn (kip)", "phi*Pn (kip)", "phi*Mn (k-ft)"]]
    for i, kp in enumerate(keys_ip, 1):
        c_str = "inf" if kp.c == float('inf') else f"{kp.c:.2f}"
        eps_str = f"{kp.eps_t:.5f}" if kp.eps_t < 1 else ">1"
        kp_data_ip.append([
            str(i), kp.label[:35] if kp.label else "-",
            c_str, eps_str,
            f"{kp.Pn/1000:.0f}", f"{kp.phi_Pn/1000:.0f}", f"{kp.phi_Mn/12000:.0f}"
        ])
    
    tbl_kp_ip = Table(kp_data_ip, colWidths=[0.3*inch, 2.3*inch, 0.6*inch, 0.7*inch, 0.7*inch, 0.7*inch, 0.7*inch])
    tbl_kp_ip.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#1a5276')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('FONTSIZE', (0,0), (-1,-1), 8),
        ('ALIGN', (2,0), (-1,-1), 'CENTER'),
    ]))
    story.append(tbl_kp_ip)
    
    if 'pm_ip' in plot_paths:
        story.append(Spacer(1, 0.1*inch))
        story.append(Image(plot_paths['pm_ip'], width=6.5*inch, height=5*inch))
    
    story.append(PageBreak())
    
    # ─── 6. FEM RESULTS ───
    story.append(Paragraph("6. FEM DISPLACEMENT RESULTS", custom['Section']))
    
    fem_ip = results['fem_inplane']
    fem_oop = results['fem_oop']
    
    fem_data = [
        ["Parameter", "In-Plane", "Out-of-Plane", "Combined"],
        ["Mesh Elements", f"{len(results['fem'].elements)}", "-", "-"],
        ["Mesh Nodes", f"{len(results['fem'].nodes)}", "-", "-"],
        ["Max |ux|", f"{abs(fem_ip['max_ux']):.5f} in", "-", "-"],
        ["Max |uy|", f"{abs(fem_ip['max_uy']):.5f} in", "-", "-"],
        ["Max |uz|", "-", f"{fem_oop['max_uz']:.5f} in", "-"],
        ["Max Total Magnitude", "-", "-", f"{results['max_total_disp']:.5f} in"],
        ["Avg Top Displacement", f"{fem_ip['avg_top_mag']:.5f} in", 
         f"{fem_oop['avg_top_uz']:.5f} in", f"{results['avg_top_total']:.5f} in"],
        ["Deflection Limit (hw/150)", "-", "-", f"{results['delta_limit']:.4f} in"],
        ["Status", "-", "-", "OK" if results['delta_service_ok'] else "NG"],
    ]
    
    tbl_fem = Table(fem_data, colWidths=[2.0*inch, 1.3*inch, 1.3*inch, 1.3*inch])
    tbl_fem.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2c3e50')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('ALIGN', (1,0), (-1,-1), 'CENTER'),
    ]))
    story.append(tbl_fem)
    
    # Deformed shape plots
    if 'inplane_mag' in plot_paths:
        story.append(Spacer(1, 0.15*inch))
        story.append(Paragraph("In-Plane Deformed Shape (Total Magnitude):", styles['Normal']))
        story.append(Image(plot_paths['inplane_mag'], width=6*inch, height=4.5*inch))
    
    if 'oop_uz' in plot_paths:
        story.append(Spacer(1, 0.15*inch))
        story.append(Paragraph("Out-of-Plane Displacement (uz):", styles['Normal']))
        story.append(Image(plot_paths['oop_uz'], width=6*inch, height=4.5*inch))
    
    story.append(PageBreak())
    
    # ─── 7. APPENDIX: DERIVATIONS ───
    story.append(Paragraph("7. APPENDIX: FULL MATHEMATICAL DERIVATIONS", custom['Section']))
    
    story.append(Paragraph("7.1 Reinforcement Calculation", custom['Subsection']))
    story.append(Paragraph(results['derivations']['reinforcement'].replace('\n', '<br/>'), custom['CodeBlock']))
    
    story.append(Paragraph("7.2 Minimum Reinforcement (ACI 11.6)", custom['Subsection']))
    story.append(Paragraph(results['derivations']['rho_min'].replace('\n', '<br/>'), custom['CodeBlock']))
    
    story.append(Paragraph("7.3 Maximum Spacing (ACI 11.7)", custom['Subsection']))
    story.append(Paragraph(results['derivations']['spacing'].replace('\n', '<br/>'), custom['CodeBlock']))
    
    story.append(Paragraph("7.4 Minimum Thickness (ACI 11.3.1.1)", custom['Subsection']))
    story.append(Paragraph(results['derivations']['h_min'].replace('\n', '<br/>'), custom['CodeBlock']))
    
    story.append(Paragraph("7.5 Two Layer Requirement", custom['Subsection']))
    story.append(Paragraph(results['derivations']['two_layers'].replace('\n', '<br/>'), custom['CodeBlock']))
    
    story.append(Paragraph("7.6 Slenderness Check", custom['Subsection']))
    story.append(Paragraph(results['derivations']['slenderness'].replace('\n', '<br/>'), custom['CodeBlock']))
    
    story.append(PageBreak())
    
    story.append(Paragraph("7.7 Out-of-Plane P-M Derivation", custom['Subsection']))
    deriv_text = results['derivations']['pm_oop'].replace('\n', '<br/>')
    story.append(Paragraph(deriv_text, custom['CodeBlock']))
    
    story.append(PageBreak())
    
    story.append(Paragraph("7.8 In-Plane P-M Derivation", custom['Subsection']))
    deriv_text = results['derivations']['pm_ip'].replace('\n', '<br/>')
    story.append(Paragraph(deriv_text, custom['CodeBlock']))

    # Development length & openings derivation (if available)
    if results['derivations'].get('openings_reinf'):
        story.append(PageBreak())
        story.append(Paragraph("7.9 Development Length & Opening Reinforcement", custom['Subsection']))
        story.append(Paragraph(
            results['derivations']['openings_reinf'].replace('\n', '<br/>'),
            custom['CodeBlock']
        ))
    
    # Build PDF with custom canvas - compact title block at TOP
    doc = SimpleDocTemplate(
        str(outpath), 
        pagesize=LETTER,
        leftMargin=0.5*inch, 
        rightMargin=0.5*inch,
        topMargin=1.5*inch,  # Space for compact title block
        bottomMargin=0.5*inch
    )
    
    # Use custom canvas with title block
    doc.build(
        story, 
        canvasmaker=lambda *args, **kwargs: TitleBlockCanvas(
            *args, 
            project_info=inp.project_info, 
            **kwargs
        )
    )


# =============================================================================
# CONSOLE I/O
# =============================================================================
def ask_float(msg, lo=None, default=None):
    while True:
        try:
            raw = input(msg).strip()
            if raw == "" and default is not None:
                return default
            x = float(raw)
            if lo is not None and x < lo:
                print(f"  Must be >= {lo}."); continue
            return x
        except:
            print("  Enter a number.")


def ask_int(msg, lo=None, hi=None, default=None):
    while True:
        try:
            raw = input(msg).strip()
            if raw == "" and default is not None:
                return default
            x = int(raw)
            if lo is not None and x < lo:
                print(f"  Must be >= {lo}."); continue
            if hi is not None and x > hi:
                print(f"  Must be <= {hi}."); continue
            return x
        except:
            print("  Enter an integer.")


def ask_string(msg, default=""):
    raw = input(msg).strip()
    return raw if raw else default


def ask_choice(msg, options, default=None):
    opts_lower = [o.lower() for o in options]
    while True:
        raw = input(msg).strip().lower()
        if raw == "" and default is not None:
            return default
        if raw in opts_lower:
            return options[opts_lower.index(raw)]
        print(f"  Choose: {', '.join(options)}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("\n" + "="*70)
    print("  RC WALL DESIGN - ACI 318-19 with FEM Analysis (Revised)")
    print("  Combined In-Plane AND Out-of-Plane Analysis")
    print("="*70)
    
    # ─── PROJECT INFO ───
    print("\n--- PROJECT INFORMATION (for title block) ---")
    proj_name = ask_string("Project Name [RC Wall Design]: ", "RC Wall Design")
    proj_num = ask_string("Project Number []: ", "")
    client = ask_string("Client Name []: ", "")
    designer = ask_string("Designer []: ", "")
    checker = ask_string("Checker []: ", "")
    company = ask_string("Company Name []: ", "")
    
    print("\nOptional: Provide path to company logo (PNG/JPG)")
    print("  (Leave blank for placeholder)")
    logo_path = ask_string("Logo path []: ", "")
    if logo_path and not os.path.exists(logo_path):
        print(f"  Warning: Logo file not found at '{logo_path}'. Using placeholder.")
        logo_path = None
    
    project_info = ProjectInfo(
        project_name=proj_name,
        project_number=proj_num,
        client_name=client,
        designer=designer,
        checker=checker,
        logo_path=logo_path if logo_path else None,
        company_name=company,
        date=datetime.now().strftime('%Y-%m-%d')
    )
    
    # ─── GEOMETRY ───
    print("\n--- WALL GEOMETRY ---")
    Lw = ask_float("Wall length Lw (in): ", lo=12)
    h = ask_float("Wall thickness h (in): ", lo=4)
    hw = ask_float("Wall height hw (in): ", lo=12)
    
    # ─── MATERIALS ───
    print("\n--- MATERIALS ---")
    fc = ask_float("f'c (psi) [4000]: ", lo=2500, default=4000)
    fy = ask_float("fy (psi) [60000]: ", lo=40000, default=60000)
    cover = ask_float("Cover (in) [1.5]: ", lo=0.5, default=1.5)
    
    # ─── REINFORCEMENT ───
    print(f"\n--- REINFORCEMENT --- (Bars: {list(BAR_AREAS_IN2.keys())})")
    vbar = ask_int("Vertical bar # [5]: ", lo=3, hi=18, default=5)
    vs = ask_float("Vertical spacing (in) [12]: ", lo=3, default=12)
    vf = ask_int("Vertical faces (1 or 2) [2]: ", lo=1, hi=2, default=2)
    hbar = ask_int("Horizontal bar # [4]: ", lo=3, hi=18, default=4)
    hs = ask_float("Horizontal spacing (in) [12]: ", lo=3, default=12)
    hf = ask_int("Horizontal faces (1 or 2) [2]: ", lo=1, hi=2, default=2)
    
    print("\n--- SHEAR TIES (for out-of-plane, if needed) ---")
    print("  Enter 0 legs if no ties provided.")
    tleg = ask_int("Tie legs (0 if none) [0]: ", lo=0, default=0)
    if tleg > 0:
        tbar = ask_int("Tie bar # [3]: ", lo=3, hi=8, default=3)
        ts = ask_float("Tie spacing (in) [12]: ", lo=3, default=12)
    else:
        tbar, ts = 3, 12.0
    
    # ─── FACTORED LOADS ───
    print("\n--- FACTORED LOADS ---")
    print("NOTE: Both in-plane AND out-of-plane loads will be analyzed")
    
    Pu = ask_float("Pu (kip, +compression) [0]: ", default=0)
    
    print("\nIn-Plane Loading:")
    Vu_ip = ask_float("  Vu,ip - In-plane shear (kip) [0]: ", lo=0, default=0)
    Mu_ip = ask_float("  Mu,ip - In-plane moment override (kip-ft, 0=auto) [0]: ", lo=0, default=0)
    
    print("\nOut-of-Plane Loading:")
    wu_oop = ask_float("  wu,oop - Out-of-plane pressure (psf) [0]: ", lo=0, default=0)
    Vu_oop = ask_float("  Vu,oop - Out-of-plane shear (kip) [0]: ", lo=0, default=0)
    Mu_oop = ask_float("  Mu,oop - Out-of-plane moment override (kip-ft, 0=auto) [0]: ", lo=0, default=0)
    
    # ─── WALL TYPE & BRACING ───
    print("\n--- WALL TYPE & BRACING ---")
    wtype = ask_choice("Wall type [bearing/nonbearing/basement] [bearing]: ", 
                       ["bearing", "nonbearing", "basement"], default="bearing")
    brace = ask_choice("Bracing [braced_restrained/braced_unrestrained/cantilever] [braced_restrained]: ",
                       ["braced_restrained", "braced_unrestrained", "cantilever"], default="braced_restrained")
    
    # ─── OPENINGS ───
    print("\n--- OPENINGS ---")
    n_op = ask_int("Number of openings [0]: ", lo=0, default=0)
    openings = []
    for i in range(n_op):
        print(f"  Opening {i+1}:")
        w = ask_float("    Width (in): ", lo=1)
        ht = ask_float("    Height (in): ", lo=1)
        cx = ask_float("    Center X (in): ", lo=0)
        cy = ask_float("    Center Y (in): ", lo=0)
        openings.append(Opening(w, ht, cx, cy))
    
    inp = WallInput(
        Lw_in=Lw, h_in=h, hw_in=hw,
        fc_psi=fc, fy_psi=fy, cover_in=cover,
        vert_bar_no=vbar, vert_bar_spacing_in=vs, vert_faces=vf,
        horiz_bar_no=hbar, horiz_bar_spacing_in=hs, horiz_faces=hf,
        tie_bar_no=tbar, tie_spacing_in=ts, tie_legs=tleg,
        Pu_kip=Pu, 
        Vu_ip_kip=Vu_ip, Mu_ip_kip_ft=Mu_ip,
        Vu_oop_kip=Vu_oop, wu_oop_psf=wu_oop, Mu_oop_kip_ft=Mu_oop,
        wall_type=wtype, bracing=brace,
        openings=openings,
        project_info=project_info
    )
    
    print("\n" + "="*70)
    print("  RUNNING ANALYSIS...")
    print("="*70)
    
    # Design
    designer_obj = WallDesigner(inp)
    results = designer_obj.run_design()
    
    # Plots
    tmpdir = tempfile.mkdtemp(prefix="rcwall_")
    plot_paths = {}
    
    # P-M diagrams
    # Out-of-plane
    pts_oop = results['pm_oop']['points']
    keys_oop = results['pm_oop']['keys']
    pm_oop_path = os.path.join(tmpdir, "pm_oop.png")
    plot_pm_diagram(
        pts_oop, keys_oop, 
        results['Pu'], results['Mu_oop'],
        "P-M Interaction Diagram (Out-of-Plane)", 
        pm_oop_path,
        Po=results['Po_oop'],
        Po_080=results['Po_080_oop']
    )
    plot_paths['pm_oop'] = pm_oop_path
    
    # In-plane
    pts_ip = results['pm_ip']['points']
    keys_ip = results['pm_ip']['keys']
    pm_ip_path = os.path.join(tmpdir, "pm_ip.png")
    plot_pm_diagram(
        pts_ip, keys_ip, 
        results['Pu'], results['Mu_ip'],
        "P-M Interaction Diagram (In-Plane)", 
        pm_ip_path,
        Po=results['Po_ip'],
        Po_080=results['Po_080_ip']
    )
    plot_paths['pm_ip'] = pm_ip_path
    
    # FEM plots
    # In-plane magnitude
    inplane_path = os.path.join(tmpdir, "inplane_mag.png")
    plot_deformed_mesh(
        results['fem_inplane'],
        inp.openings,
        "In-Plane Deformed Shape",
        inplane_path,
        component='mag'
    )
    plot_paths['inplane_mag'] = inplane_path
    
    # Out-of-plane uz
    if results['fem_oop']['max_uz'] > 0:
        oop_path = os.path.join(tmpdir, "oop_uz.png")
        plot_deformed_mesh(
            results['fem_oop'],
            inp.openings,
            "Out-of-Plane Displacement",
            oop_path,
            component='uz'
        )
        plot_paths['oop_uz'] = oop_path
    
    # PDF - Save to current working directory
    pdf_path = Path.cwd() / "RC_Wall_ACI318_Design_Report.pdf"
    generate_report(inp, results, plot_paths, pdf_path)
    
    # Console summary
    print("\n" + "="*70)
    print("  DESIGN SUMMARY")
    print("="*70)
    
    status = "PASS" if results['all_ok'] else "FAIL"
    print(f"\nStatus: {status}")
    
    if results['issues']:
        print("\nIssues:")
        for iss in results['issues']:
            print(f"  - {iss}")
    
    print(f"\n--- FEM-Derived Pressures ---")
    print(f"  Axial: p_ax = {results['p_ax_psi']:.4f} psi")
    print(f"  In-plane lateral: p_lat,ip = {results['p_lat_ip_psi']:.4f} psi")
    print(f"  Out-of-plane: p_oop = {results['p_oop_psi']:.4f} psi")
    
    print(f"\n--- P-M Interaction ---")
    print(f"  Pu = {results['Pu']/1000:.1f} kip")
    print(f"  OUT-OF-PLANE:")
    print(f"    Mu,oop = {results['Mu_oop']/12000:.1f} kip-ft")
    print(f"    Po = {results['Po_oop']/1000:.0f} kip")
    print(f"    0.80Po = {results['Po_080_oop']/1000:.0f} kip")
    print(f"    phi(0.80Po) = {results['phi_Po_080_oop']/1000:.0f} kip")
    print(f"    phi*Mn @ Pu = {results['phi_Mn_oop']/12000:.1f} kip-ft")
    print(f"    UC = {results['UC_PM_oop']:.3f}")
    print(f"  IN-PLANE:")
    print(f"    Mu,ip = {results['Mu_ip']/12000:.1f} kip-ft")
    print(f"    Po = {results['Po_ip']/1000:.0f} kip")
    print(f"    0.80Po = {results['Po_080_ip']/1000:.0f} kip")
    print(f"    phi(0.80Po) = {results['phi_Po_080_ip']/1000:.0f} kip")
    print(f"    phi*Mn @ Pu = {results['phi_Mn_ip']/12000:.1f} kip-ft")
    print(f"    UC = {results['UC_PM_ip']:.3f}")
    
    print(f"\n--- Shear ---")
    print(f"  IN-PLANE (ACI 11.5.4):")
    print(f"    Vu,ip = {results['Vu_ip']/1000:.1f} kip")
    print(f"    phi*Vc = {results['phi_Vc_ip']/1000:.1f} kip")
    print(f"    phi*Vs = {results['phi_Vs_ip']/1000:.1f} kip")
    print(f"    phi*Vn = {results['phi_Vn_ip']/1000:.1f} kip")
    print(f"    UC = {results['UC_V_ip']:.3f}")
    print(f"  OUT-OF-PLANE (ACI 22.5):")
    print(f"    Vu,oop = {results['Vu_oop']/1000:.1f} kip")
    print(f"    phi*Vc = {results['phi_Vc_oop']/1000:.1f} kip")
    print(f"    phi*Vs = {results['phi_Vs_oop']/1000:.1f} kip")
    print(f"    phi*Vn = {results['phi_Vn_oop']/1000:.1f} kip")
    print(f"    UC = {results['UC_V_oop']:.3f}")
    
    print(f"\n--- FEM Displacements ---")
    fem_ip = results['fem_inplane']
    fem_oop = results['fem_oop']
    print(f"  In-plane:")
    print(f"    Max |ux| = {abs(fem_ip['max_ux']):.5f} in at {fem_ip['loc_ux']}")
    print(f"    Max |uy| = {abs(fem_ip['max_uy']):.5f} in at {fem_ip['loc_uy']}")
    print(f"    Max magnitude = {fem_ip['max_mag']:.5f} in at {fem_ip['loc_mag']}")
    print(f"  Out-of-plane:")
    print(f"    Max |uz| = {fem_oop['max_uz']:.5f} in at {fem_oop['max_uz_loc']}")
    print(f"  COMBINED TOTAL:")
    print(f"    Max total displacement = {results['max_total_disp']:.5f} in")
    print(f"    Deflection limit (hw/150) = {results['delta_limit']:.4f} in")
    print(f"    Status: {'OK' if results['delta_service_ok'] else 'NG'}")
    
    print(f"\n--- Reinforcement ---")
    print(f"  rho_L = {results['rho_L']:.4f} (min {results['rho_L_min']:.4f})")
    print(f"  rho_t = {results['rho_t']:.4f} (min {results['rho_t_min']:.4f})")
    
    print(f"\n--- Slenderness ---")
    print(f"  k = {results['k']:.2f}")
    print(f"  kLc/h = {results['kLc_h']:.1f} (limit 100)")
    
    print(f"\n{'='*70}")
    print(f"  PDF Report saved to: {pdf_path}")
    print(f"{'='*70}\n")
    
    return pdf_path


if __name__ == "__main__":
    main()
