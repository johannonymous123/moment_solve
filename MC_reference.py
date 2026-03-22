"""
Numba-compatible 2D MC transport on a Cartesian (rectangular) grid.

Workflow:
- Build a uniform Nx x Ny grid over [0,Lx] x [0,Ly] directly in the script.
  No mesh files are loaded.
- Assign per-cell materials sigma_a, sigma_s from the same lattice / Hohlraum
  functions as the triangular-mesh version (evaluated at cell centres).
- Sample Np source particles (same as before) + find initial cell via O(1)
  integer arithmetic.
- Transport: within each rectangular cell find the ray-box exit distance,
  sample a scatter distance; if scatter < exit -> scatter inside; otherwise
  cross the boundary into the neighbouring cell (or leak).
- Tally scalar flux with track-length estimator (same formula as before).
- Apply implicit absorption, Russian roulette on cell exit.
- Plot flux (linear + log) and material maps.

Cell layout
-----------
Cell (ix, iy) occupies [ix*dx, (ix+1)*dx] x [iy*dy, (iy+1)*dy].
Flat cell index: cell = iy * Nx + ix   (row-major, y is the slow index).
Neighbours:  edge 0 -> +x  (ix+1, iy)
             edge 1 -> -x  (ix-1, iy)
             edge 2 -> +y  (ix,   iy+1)
             edge 3 -> -y  (ix,   iy-1)
A neighbour index of -1 means leakage (domain boundary).
"""

from __future__ import annotations

import numpy as np
import numba as nb
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# -----------------------------
# Numeric conventions
# -----------------------------
EPS = 1e-12
TWO_PI = 2.0 * np.pi


# ============================================================
# Grid construction
# ============================================================

def build_cartesian_grid(Nx: int, Ny: int, Lx: float, Ly: float):
    """
    Build a uniform Nx x Ny Cartesian grid.

    Returns
    -------
    dx, dy : float
        Cell dimensions.
    cell_centers : (Nc, 2) float64
        (cx, cy) for each flat cell index c = iy*Nx + ix.
    neighbors : (Nc, 4) int64
        Neighbour flat indices for edges 0(+x), 1(-x), 2(+y), 3(-y).
        -1 = leakage boundary.
    cell_areas : (Nc,) float64
        Area of every cell (= dx*dy, uniform).
    """
    Nc = Nx * Ny
    dx = Lx / Nx
    dy = Ly / Ny

    cell_centers = np.empty((Nc, 2), dtype=np.float64)
    neighbors    = np.full((Nc, 4), -1, dtype=np.int64)

    for iy in range(Ny):
        for ix in range(Nx):
            c = iy * Nx + ix
            cell_centers[c, 0] = (ix + 0.5) * dx
            cell_centers[c, 1] = (iy + 0.5) * dy

            # edge 0: +x
            if ix + 1 < Nx:
                neighbors[c, 0] = iy * Nx + (ix + 1)
            # edge 1: -x
            if ix - 1 >= 0:
                neighbors[c, 1] = iy * Nx + (ix - 1)
            # edge 2: +y
            if iy + 1 < Ny:
                neighbors[c, 2] = (iy + 1) * Nx + ix
            # edge 3: -y
            if iy - 1 >= 0:
                neighbors[c, 3] = (iy - 1) * Nx + ix

    cell_areas = np.full(Nc, dx * dy, dtype=np.float64)
    return dx, dy, cell_centers, neighbors, cell_areas


# ============================================================
# Material assignment
# ============================================================

def sigma_lattice(x: float, y: float) -> tuple[float, float]:
    """Returns (sigma_a, sigma_s) for the lattice geometry."""
    s = np.floor(x) * 10 + np.floor(y)
    if s in [11, 13, 15, 22, 24, 31, 42, 44, 51, 53, 55]:
        return 10.0, 0.0
    else:
        return 0.0, 1.0


def sigma_hohlraum(x: float, y: float) -> tuple[float, float]:
    """Returns (sigma_a, sigma_s) for the Hohlraum geometry (original 1.3x1.3 domain)."""
    if x > 1.25:
        return 100.0, 0.0    # outer shell right
    if y < 0.05 or y > 1.25:
        return 100.0, 0.0    # outer shell top/bottom
    if x < 0.05 and 0.25 < y < 1.05:
        return 5.0, 95.0     # left obstruction
    if 0.5 < x < 0.85 and 0.3 < y < 1:
        return 50.0, 50.0    # inside inner rectangle
    if 0.45 < x < 0.85 and 0.25 < y < 1.05:
        return 10.0, 90.0    # shell around inner rectangle
    return 1.0, 0.1


def sigma_hohlraum_v2(x: float, y: float) -> tuple[float, float]:
    """
    Returns (sigma_a, sigma_s) for the extended Hohlraum geometry on a 1.5x1.5 domain.

    Changes vs. the original:
      - Domain is [0, 1.5] x [0, 1.5].
      - Absorbing boundary shell (sigma_a=100) of thickness 0.05 on ALL 4 sides.
      - All interior features shifted by +0.2 in x and +0.1 in y:
          left obstruction  : 0.20 < x < 0.25,  0.35 < y < 1.15
          inner shell       : 0.65 < x < 1.05,  0.35 < y < 1.15
          inner core        : 0.70 < x < 1.05,  0.40 < y < 1.10
      - The original left boundary source is replaced by a volumetric isotropic
        source strip at 0.10 < x < 0.15,  0.10 < y < 1.40  (handled in sampler).
    """
    # -- 4-sided absorbing boundary shell (thickness 0.05) --
    if x < 0.05 or x > 1.45:
        return 100.0, 0.0    # left / right boundary
    if y < 0.05 or y > 1.45:
        return 100.0, 0.0    # bottom / top boundary
    # -- shifted left obstruction --
    if 0.20 < x < 0.25 and 0.35 < y < 1.15:
        return 5.0, 95.0
    # -- shifted inner core (checked before shell so it takes priority) --
    if 0.70 < x < 1.05 and 0.40 < y < 1.10:
        return 50.0, 50.0
    # -- shifted inner shell --
    if 0.65 < x < 1.05 and 0.35 < y < 1.15:
        return 10.0, 90.0
    # -- background --
    return 1.0, 0.1


def sigma_crossing_beams(x: float, y: float) -> tuple[float, float]:
    """
    Returns (sigma_a, sigma_s) for the crossing-beams geometry.

    Domain [0,7]x[0,7].  Two narrow source strips (handled in sampler).
    Absorbing frame (width 0.5) around boundary to kill leakage;
    uniform sigma_a=0.02, sigma_s=0 in the interior.
    """
    if x < 0.5 or x > 6.5 or y < 0.5 or y > 6.5:
        return 10.0, 0.0   # absorbing frame
    return 0.02, 0.0


def assign_materials_cartesian(cell_centers: np.ndarray, geometry: str):
    """
    Evaluate material functions at cell centres.

    Parameters
    ----------
    cell_centers : (Nc, 2)
    geometry     : "lattice" or "Hohlraum"

    Returns
    -------
    sigma_a, sigma_s : (Nc,) float64
    """
    Nc = cell_centers.shape[0]
    sigma_a = np.empty(Nc, dtype=np.float64)
    sigma_s = np.empty(Nc, dtype=np.float64)
    if geometry == "lattice":
        mat_fn = sigma_lattice
    elif geometry == "crossing_beams":
        mat_fn = sigma_crossing_beams
    elif geometry == "Hohlraum_v2":
        mat_fn = sigma_hohlraum_v2
    else:
        mat_fn = sigma_hohlraum
    for c in range(Nc):
        sigma_a[c], sigma_s[c] = mat_fn(cell_centers[c, 0], cell_centers[c, 1])
    return sigma_a, sigma_s


# ============================================================
# Source sampling  (unchanged logic)
# ============================================================

def sample_source_particle_numpy(rng: np.random.Generator, geometry: str):
    if geometry == "lattice":
        xmin, xmax, ymin, ymax = 3.0, 4.0, 3.0, 4.0
        x = np.array([rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)], dtype=np.float64)
        phi   = rng.uniform(0.0, TWO_PI)
        costh = rng.uniform(-1.0, 1.0)
        sinth = np.sqrt(1.0 - costh**2)
        u = np.array([sinth * np.cos(phi), sinth * np.sin(phi)], dtype=np.float64)
        w = 1.0
    elif geometry == "Hohlraum_v2":
        # Volumetric isotropic source in strip [0.10, 0.15] x [0.10, 1.40]
        x = np.array([rng.uniform(0.10, 0.15), rng.uniform(0.10, 1.40)], dtype=np.float64)
        costh = rng.uniform(-1.0, 1.0)
        sinth = np.sqrt(1.0 - costh**2)
        phi   = rng.uniform(0.0, TWO_PI)
        u = np.array([sinth * np.cos(phi), sinth * np.sin(phi)], dtype=np.float64)
        w = 1.0
    elif geometry == "crossing_beams":
        # Two volumetric isotropic source strips with equal area (=1.0 each):
        #   Beam A: x in [0.5,1.0], y in [2.5,4.5]  (area 1.0)
        #   Beam B: x in [2.5,4.5], y in [0.5,1.0]  (area 1.0)
        # Pick a beam uniformly at random; weight = total_area / Np normalised later.
        if rng.uniform(0.0, 1.0) < 0.5:
            x = np.array([rng.uniform(0.5, 1.0), rng.uniform(2.5, 4.5)], dtype=np.float64)
        else:
            x = np.array([rng.uniform(2.5, 4.5), rng.uniform(0.5, 1.0)], dtype=np.float64)
        costh = rng.uniform(-1.0, 1.0)
        sinth = np.sqrt(1.0 - costh**2)
        phi   = rng.uniform(0.0, TWO_PI)
        u = np.array([sinth * np.cos(phi), sinth * np.sin(phi)], dtype=np.float64)
        w = 1.0
    else:
        y_min, y_max = 0.0, 1.3
        x = np.array([1e-8, rng.uniform(y_min, y_max)], dtype=np.float64)
        mu  = np.sqrt(rng.uniform(0.0, 1.0))
        phi = rng.uniform(0.0, TWO_PI)
        u   = np.array([mu, np.sqrt(1.0 - mu**2) * np.sin(phi)], dtype=np.float64)
        w   = 1.0 / 1.3
    return x, u, w


# ============================================================
# Initial cell find: O(1) for Cartesian grid
# ============================================================

def find_initial_cell_cartesian(x: np.ndarray, Nx: int, Ny: int,
                                 dx: float, dy: float) -> int:
    """Return flat cell index, or -1 if outside the domain."""
    ix = int(x[0] / dx)
    iy = int(x[1] / dy)
    if ix < 0 or ix >= Nx or iy < 0 or iy >= Ny:
        return -1
    return iy * Nx + ix


# ============================================================
# Numba RNG: xorshift64*  (unchanged)
# ============================================================

@nb.njit(inline="always")
def rng_u01(state):
    x = state
    x ^= x >> 12
    x ^= x << 25
    x ^= x >> 27
    state = x
    out = x * np.uint64(2685821657736338717)
    u = (out >> np.uint64(11)) * (1.0 / (1 << 53))
    if u <= 0.0:
        u = 5e-324
    return u, state


@nb.njit(inline="always")
def rng_uniform(state, a, b):
    u, state = rng_u01(state)
    return a + (b - a) * u, state


# ============================================================
# Geometry (Numba): ray-AABB exit for a Cartesian cell
# ============================================================

@nb.njit(inline="always")
def distance_to_cell_exit_numba(px, py, dx_dir, dy_dir,
                                 cell, Nx, Ny, dx, dy):
    """
    Find the distance along (dx_dir, dy_dir) to the first exit face of the
    axis-aligned cell `cell = iy*Nx + ix`.

    Returns
    -------
    t_exit : float   – distance to exit
    edge_id : int    – 0(+x), 1(-x), 2(+y), 3(-y)
    """
    iy = cell // Nx
    ix = cell  - iy * Nx

    x_lo = ix * dx
    x_hi = x_lo + dx
    y_lo = iy * dy
    y_hi = y_lo + dy

    t_min = 1e300
    edge  = -1

    # +x face
    if dx_dir > 1e-15:
        t = (x_hi - px) / dx_dir
        if t > 1e-12 and t < t_min:
            t_min = t
            edge  = 0
    # -x face
    elif dx_dir < -1e-15:
        t = (x_lo - px) / dx_dir
        if t > 1e-12 and t < t_min:
            t_min = t
            edge  = 1
    # +y face
    if dy_dir > 1e-15:
        t = (y_hi - py) / dy_dir
        if t > 1e-12 and t < t_min:
            t_min = t
            edge  = 2
    # -y face
    elif dy_dir < -1e-15:
        t = (y_lo - py) / dy_dir
        if t > 1e-12 and t < t_min:
            t_min = t
            edge  = 3

    return t_min, edge


# ============================================================
# Physics pieces (Numba) — unchanged from triangular version
# ============================================================

@nb.njit(inline="always")
def sample_scatter_distance_numba(state, sigma_s):
    if sigma_s <= 0.0:
        return 1e300, state
    u, state = rng_u01(state)
    s = -np.log(u) / sigma_s
    return s, state


@nb.njit(inline="always")
def sample_scatter_dir_2d_numba(state):
    """
    Sample a uniformly random direction on the 3D unit sphere and return
    the in-plane (x, y) components.  This matches the projected-3D convention
    used by the source samplers:
        cos(theta) ~ Uniform[-1, 1]
        phi        ~ Uniform[0, 2*pi]
        Omega_x = sin(theta)*cos(phi),  Omega_y = sin(theta)*sin(phi)
    The returned vector is NOT a unit 2D vector; |u|=sin(theta) in [0,1].
    """
    costh, state = rng_uniform(state, -1.0, 1.0)
    sinth = np.sqrt(1.0 - costh * costh)
    phi, state   = rng_uniform(state,  0.0, TWO_PI)
    return sinth * np.cos(phi), sinth * np.sin(phi), state


@nb.njit(inline="always")
def tracklength_flux_contrib(w_in, sigma_a, L):
    if L <= 0.0:
        return 0.0
    if sigma_a > 0.0:
        return w_in * (1.0 - np.exp(-sigma_a * L)) / sigma_a
    else:
        return w_in * L


@nb.njit(inline="always")
def tracklength_moment_contrib(w_in, sigma_a, L, omega):
    """
    Track-length estimator weight for one direction component (or product).
    Returns  omega * int_0^L w(s) ds  where w(s) = w_in * exp(-sigma_a * s).
    """
    if L <= 0.0:
        return 0.0
    if sigma_a > 0.0:
        return omega * w_in * (1.0 - np.exp(-sigma_a * L)) / sigma_a
    else:
        return omega * w_in * L


@nb.njit(inline="always")
def attenuate_weight_numba(w, sigma_a, L):
    if sigma_a <= 0.0:
        return w
    return w * np.exp(-sigma_a * L)


@nb.njit(inline="always")
def russian_roulette_numba(state, w, w_cut, w_survive):
    if w >= w_cut:
        return True, w, state
    if w <= 0.0:
        return False, 0.0, state
    ps = w / w_survive
    if ps <= 0.0:
        return False, 0.0, state
    u, state = rng_u01(state)
    if u < ps:
        return True, w_survive, state
    else:
        return False, 0.0, state


# ============================================================
# Core: move within a Cartesian cell until boundary exit
# ============================================================

@nb.njit
def move_to_boundary_cartesian(px, py, dx_dir, dy_dir, cell,
                                neighbors,
                                sigma_a, sigma_s,
                                Nx, Ny, dx, dy,
                                rng_state,
                                w_cur):
    """
    Simulate within-cell scattering until the particle exits the current cell.

    Parameters mirror the triangular version; tri_verts is replaced by the
    grid parameters (Nx, Ny, dx, dy).
    w_cur is the current particle weight (before entering this cell), used to
    accumulate the angular-moment tallies for each straight sub-segment.

    Returns
    -------
    px, py          : exit position
    dx_dir, dy_dir  : exit direction
    next_cell       : neighbouring cell index, or -1 (leakage)
    L_total         : total path length in this cell
    rng_state       : updated RNG state
    edge_id         : which face was crossed (0-3)
    Jx_contrib      : sum_segments  Omega_x * w_eff(L_seg)
    Jy_contrib      : sum_segments  Omega_y * w_eff(L_seg)
    Pxx_contrib     : sum_segments  Omega_x^2 * w_eff(L_seg)
    Pxy_contrib     : sum_segments  Omega_x*Omega_y * w_eff(L_seg)
    Pyy_contrib     : sum_segments  Omega_y^2 * w_eff(L_seg)
    """
    L_total     = 0.0
    Jx_contrib  = 0.0
    Jy_contrib  = 0.0
    Pxx_contrib = 0.0
    Pxy_contrib = 0.0
    Pyy_contrib = 0.0
    sa = sigma_a[cell]
    ss = sigma_s[cell]

    # Running weight inside the cell: decrements with implicit absorption
    # along each sub-segment so that J/P get the same attenuation as phi.
    w_seg = w_cur

    while True:
        t_exit, edge_id = distance_to_cell_exit_numba(
            px, py, dx_dir, dy_dir, cell, Nx, Ny, dx, dy)

        if edge_id < 0 or t_exit > 1e299:
            # degenerate numerical case; treat as leakage
            return (px, py, dx_dir, dy_dir, -1, L_total, rng_state, -1,
                    Jx_contrib, Jy_contrib,
                    Pxx_contrib, Pxy_contrib, Pyy_contrib)

        s_scatter, rng_state = sample_scatter_distance_numba(rng_state, ss)

        if s_scatter < t_exit - 1e-12:
            # scatter happens inside cell — accumulate this sub-segment
            phi0 = tracklength_flux_contrib(w_seg, sa, s_scatter)
            Jx_contrib  += dx_dir * phi0
            Jy_contrib  += dy_dir * phi0
            Pxx_contrib += dx_dir * dx_dir * phi0
            Pxy_contrib += dx_dir * dy_dir * phi0
            Pyy_contrib += dy_dir * dy_dir * phi0

            px      += s_scatter * dx_dir
            py      += s_scatter * dy_dir
            L_total += s_scatter
            # update running weight for next sub-segment
            w_seg    = attenuate_weight_numba(w_seg, sa, s_scatter)
            dx_dir, dy_dir, rng_state = sample_scatter_dir_2d_numba(rng_state)
            continue

        # exit the cell at the boundary — accumulate final sub-segment
        phi0 = tracklength_flux_contrib(w_seg, sa, t_exit)
        Jx_contrib  += dx_dir * phi0
        Jy_contrib  += dy_dir * phi0
        Pxx_contrib += dx_dir * dx_dir * phi0
        Pxy_contrib += dx_dir * dy_dir * phi0
        Pyy_contrib += dy_dir * dy_dir * phi0

        px      += t_exit * dx_dir
        py      += t_exit * dy_dir
        L_total += t_exit

        next_cell = neighbors[cell, edge_id]
        return (px, py, dx_dir, dy_dir, next_cell, L_total, rng_state, edge_id,
                Jx_contrib, Jy_contrib,
                Pxx_contrib, Pxy_contrib, Pyy_contrib)


# ============================================================
# Transport kernel (Numba, SoA particles)  — same structure
# ============================================================

@nb.njit
def run_mc_cartesian(Np,
                     init_x, init_y, init_dx, init_dy, init_w, init_cell,
                     neighbors,
                     sigma_a, sigma_s,
                     w_cut, w_survive,
                     max_cell_crossings,
                     Nx, Ny, dx, dy,
                     rng_states,
                     flux_tally,  flux_tally_sq,
                     J_tally,     J_tally_sq,
                     P_tally,     P_tally_sq):
    """
    MC transport kernel on a Cartesian grid.

    Tally arrays (all indexed by flat cell index c)
    ------------------------------------------------
    flux_tally    : (Nc,)   — sum of phi contributions  (0th moment)
    flux_tally_sq : (Nc,)   — sum of squared phi contributions
    J_tally       : (Nc,2)  — sum of (Jx, Jy) contributions
    J_tally_sq    : (Nc,2)  — sum of squared (Jx², Jy²) contributions
    P_tally       : (Nc,3)  — sum of (Pxx, Pxy, Pyy) contributions
    P_tally_sq    : (Nc,3)  — sum of squared (Pxx², Pxy², Pyy²) contributions

    Variance of the mean is estimated per cell as:
        Var[c] = ( sq[c]/N - (sum[c]/N)^2 ) / N
    using the per-crossing contributions as independent samples
    (standard track-length variance estimator).
    """
    for i in range(Np):
        cell = init_cell[i]
        if cell < 0:
            continue

        px    = init_x[i]
        py    = init_y[i]
        dx_p  = init_dx[i]
        dy_p  = init_dy[i]
        w     = init_w[i]
        state = rng_states[i]

        alive     = True
        crossings = 0

        while alive and crossings < max_cell_crossings:
            crossings += 1

            (px_exit, py_exit, dx_exit, dy_exit, next_cell, L_total, state, edge_id,
             Jx_c, Jy_c, Pxx_c, Pxy_c, Pyy_c) = \
                move_to_boundary_cartesian(
                    px, py, dx_p, dy_p, cell,
                    neighbors, sigma_a, sigma_s,
                    Nx, Ny, dx, dy, state, w)

            # Track-length flux tally (0th moment)
            phi_c = tracklength_flux_contrib(w, sigma_a[cell], L_total)
            flux_tally[cell]    += phi_c
            flux_tally_sq[cell] += phi_c * phi_c

            # 1st moment tally  J = int Omega psi dOmega
            J_tally[cell, 0]    += Jx_c
            J_tally[cell, 1]    += Jy_c
            J_tally_sq[cell, 0] += Jx_c * Jx_c
            J_tally_sq[cell, 1] += Jy_c * Jy_c

            # 2nd moment tally  P = int Omega⊗Omega psi dOmega  (xx, xy, yy)
            P_tally[cell, 0]    += Pxx_c
            P_tally[cell, 1]    += Pxy_c
            P_tally[cell, 2]    += Pyy_c
            P_tally_sq[cell, 0] += Pxx_c * Pxx_c
            P_tally_sq[cell, 1] += Pxy_c * Pxy_c
            P_tally_sq[cell, 2] += Pyy_c * Pyy_c

            # Implicit absorption
            w = attenuate_weight_numba(w, sigma_a[cell], L_total)

            if next_cell < 0:
                alive = False
                break

            # Russian roulette on cell exit
            alive, w, state = russian_roulette_numba(state, w, w_cut, w_survive)
            if not alive:
                break

            # Advance into next cell with a small nudge
            px   = px_exit + 1e-10 * dx_exit
            py   = py_exit + 1e-10 * dy_exit
            dx_p = dx_exit
            dy_p = dy_exit
            cell = next_cell

        rng_states[i] = state


# ============================================================
# Plotting
# ============================================================

def plot_flux_heatmap_cartesian(phi_2d: np.ndarray,
                                 sigma_a_2d: np.ndarray,
                                 sigma_s_2d: np.ndarray,
                                 Lx: float, Ly: float,
                                 title: str = "Scalar flux"):
    """
    phi_2d, sigma_a_2d, sigma_s_2d are (Ny, Nx) arrays (row = y, col = x).
    """
    extent = [0, Lx, 0, Ly]
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    def _ishow(ax, data, label, log=False):
        if log:
            d = np.copy(data)
            d[d < 1e-6] = 1e-6
            im = ax.imshow(d, origin="lower", extent=extent, aspect="equal",
                           norm=mcolors.LogNorm(vmin=1e-6, vmax=d.max()),
                           interpolation="nearest")
        else:
            im = ax.imshow(data, origin="lower", extent=extent, aspect="equal",
                           interpolation="nearest")
        fig.colorbar(im, ax=ax, label=label)
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    _ishow(axs[0, 0], phi_2d,    "flux",          log=False)
    axs[0, 0].set_title(title)

    _ishow(axs[0, 1], phi_2d,    "flux (log)",     log=True)
    axs[0, 1].set_title(title + " (log scale)")

    _ishow(axs[1, 0], sigma_a_2d, "σ_a",           log=False)
    axs[1, 0].set_title("Absorption cross-section σ_a")

    _ishow(axs[1, 1], sigma_s_2d, "σ_s",           log=False)
    axs[1, 1].set_title("Scattering cross-section σ_s")

    plt.tight_layout()
    plt.show()


# ============================================================
# Driver
# ============================================================

def main():
    # ------------------------------------------------------------------
    # Choose geometry
    # ------------------------------------------------------------------
    Lattice = False
    Crossing = True

    if Lattice:
        geometry = "lattice"
        Lx, Ly   = 7.0, 7.0
        Nx, Ny   = 70, 70      # 10 cells per unit length -> 0.05 cm resolution
    elif Crossing:
        geometry = "crossing_beams"
        Lx, Ly   = 7.0, 7.0
        Nx, Ny   = 70, 70
    else:
        geometry = "Hohlraum_v2"
        Lx, Ly   = 1.5, 1.5
        Nx, Ny   = 75, 75      # 50 cells per unit length -> 0.02 cm resolution

    # ------------------------------------------------------------------
    # Build grid
    # ------------------------------------------------------------------
    dx, dy, cell_centers, neighbors, cell_areas = build_cartesian_grid(Nx, Ny, Lx, Ly)
    Nc = Nx * Ny
    print(f"Grid: {Nx}x{Ny} = {Nc} cells, dx={dx:.4f}, dy={dy:.4f}")

    # ------------------------------------------------------------------
    # Materials
    # ------------------------------------------------------------------
    sigma_a, sigma_s = assign_materials_cartesian(cell_centers, geometry)

    # ------------------------------------------------------------------
    # MC parameters
    # ------------------------------------------------------------------
    Np               = 5000000
    w_cut            = 1e-6 / Np
    w_survive        = 1e-2
    max_cell_crossings = Np * 1000

    # ------------------------------------------------------------------
    # Initialise particles
    # ------------------------------------------------------------------
    rng    = np.random.default_rng(1234)
    init_x  = np.empty(Np, dtype=np.float64)
    init_y  = np.empty(Np, dtype=np.float64)
    init_dx = np.empty(Np, dtype=np.float64)
    init_dy = np.empty(Np, dtype=np.float64)
    init_w  = np.empty(Np, dtype=np.float64)
    init_cell = np.empty(Np, dtype=np.int64)

    for i in range(Np):
        x, u, w = sample_source_particle_numpy(rng, geometry)
        cell     = find_initial_cell_cartesian(x, Nx, Ny, dx, dy)
        init_cell[i] = cell
        init_x[i]    = x[0]
        init_y[i]    = x[1]
        init_dx[i]   = u[0]
        init_dy[i]   = u[1]
        init_w[i]    = w

    # ------------------------------------------------------------------
    # Per-particle RNG seeds
    # ------------------------------------------------------------------
    base       = np.uint64(0x9E3779B97F4A7C15)
    rng_states = np.empty(Np, dtype=np.uint64)
    for i in range(Np):
        rng_states[i] = base ^ np.uint64(i + 1) ^ np.uint64(0xD1B54A32D192ED03)

    # ------------------------------------------------------------------
    # Tally
    # ------------------------------------------------------------------
    flux_tally    = np.zeros(Nc,       dtype=np.float64)
    flux_tally_sq = np.zeros(Nc,       dtype=np.float64)
    J_tally       = np.zeros((Nc, 2),  dtype=np.float64)
    J_tally_sq    = np.zeros((Nc, 2),  dtype=np.float64)
    P_tally       = np.zeros((Nc, 3),  dtype=np.float64)
    P_tally_sq    = np.zeros((Nc, 3),  dtype=np.float64)

    # ------------------------------------------------------------------
    # Run Numba kernel
    # ------------------------------------------------------------------
    print("Running MC kernel …")
    run_mc_cartesian(Np,
                     init_x, init_y, init_dx, init_dy, init_w, init_cell,
                     neighbors,
                     sigma_a, sigma_s,
                     w_cut, w_survive,
                     max_cell_crossings,
                     Nx, Ny, dx, dy,
                     rng_states,
                     flux_tally, flux_tally_sq,
                     J_tally,    J_tally_sq,
                     P_tally,    P_tally_sq)
    print("Done.")

    # ------------------------------------------------------------------
    # Normalise: divide by cell area and number of particles
    # Variance of the mean:  Var[c] = ( <x²> - <x>² ) / N
    #   where <x>  = tally[c]    / N
    #         <x²> = tally_sq[c] / N
    # ------------------------------------------------------------------
    norm = cell_areas * Np   # shape (Nc,)

    phi  = flux_tally / norm
    # variance: (sq/N - mean²) / N  — each divided by norm then by Np
    phi_var = (flux_tally_sq / norm - phi**2) / Np

    Jx = J_tally[:, 0] / norm;  Jy = J_tally[:, 1] / norm
    Jx_var = (J_tally_sq[:, 0] / norm - Jx**2) / Np
    Jy_var = (J_tally_sq[:, 1] / norm - Jy**2) / Np

    Pxx = P_tally[:, 0] / norm
    Pxy = P_tally[:, 1] / norm
    Pyy = P_tally[:, 2] / norm
    Pxx_var = (P_tally_sq[:, 0] / norm - Pxx**2) / Np
    Pxy_var = (P_tally_sq[:, 1] / norm - Pxy**2) / Np
    Pyy_var = (P_tally_sq[:, 2] / norm - Pyy**2) / Np

    # Relative standard deviation (%) for a quick sanity check
    def rel_std(mean, var):
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(np.abs(mean) > 0, 100.0 * np.sqrt(np.maximum(var, 0.0)) / np.abs(mean), 0.0)

    # Reshape to 2D (Ny, Nx) for imshow (row = y, col = x)
    def r2d(a): return a.reshape(Ny, Nx)

    phi_2d     = r2d(phi);      phi_std_2d  = r2d(np.sqrt(np.maximum(phi_var, 0.0)))
    Jx_2d      = r2d(Jx);      Jx_std_2d   = r2d(np.sqrt(np.maximum(Jx_var, 0.0)))
    Jy_2d      = r2d(Jy);      Jy_std_2d   = r2d(np.sqrt(np.maximum(Jy_var, 0.0)))
    Pxx_2d     = r2d(Pxx);     Pxx_std_2d  = r2d(np.sqrt(np.maximum(Pxx_var, 0.0)))
    Pxy_2d     = r2d(Pxy);     Pxy_std_2d  = r2d(np.sqrt(np.maximum(Pxy_var, 0.0)))
    Pyy_2d     = r2d(Pyy);     Pyy_std_2d  = r2d(np.sqrt(np.maximum(Pyy_var, 0.0)))
    sigma_a_2d = r2d(sigma_a)
    sigma_s_2d = r2d(sigma_s)

    print(f"phi  min/max: {phi.min():.4e}  {phi.max():.4e}  "
          f"| rel-std median: {np.median(rel_std(phi, phi_var)):.1f}%")
    print(f"|J|  min/max: {np.hypot(Jx, Jy).min():.4e}  {np.hypot(Jx, Jy).max():.4e}")
    print(f"Pxx  min/max: {Pxx.min():.4e}  {Pxx.max():.4e}")
    print(f"Pyy  min/max: {Pyy.min():.4e}  {Pyy.max():.4e}")

    # ------------------------------------------------------------------
    # Plot scalar flux + material maps
    # ------------------------------------------------------------------
    plot_flux_heatmap_cartesian(phi_2d, sigma_a_2d, sigma_s_2d, Lx, Ly,
                                title=f"Scalar flux — {geometry} ({Nx}×{Ny} grid)")

    # ------------------------------------------------------------------
    # Plot angular moments (mean)
    # ------------------------------------------------------------------
    extent = [0, Lx, 0, Ly]

    def _ishow(ax, fig, data, label, log=False):
        if log:
            d = np.copy(data)
            d[d < 1e-6] = 1e-6
            im = ax.imshow(d, origin="lower", extent=extent, aspect="equal",
                           norm=mcolors.LogNorm(vmin=1e-6, vmax=d.max()),
                           interpolation="nearest")
        else:
            im = ax.imshow(data, origin="lower", extent=extent, aspect="equal",
                           interpolation="nearest")
        fig.colorbar(im, ax=ax, label=label)
        ax.set_xlabel("x"); ax.set_ylabel("y")

    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    _ishow(axs[0, 0], fig, phi_2d,  "φ",    log=True);  axs[0, 0].set_title("φ = ∫ ψ dΩ  (log)")
    _ishow(axs[0, 1], fig, Jx_2d,   "Jx");              axs[0, 1].set_title("Jx = ∫ Ωx ψ dΩ")
    _ishow(axs[0, 2], fig, Jy_2d,   "Jy");              axs[0, 2].set_title("Jy = ∫ Ωy ψ dΩ")
    _ishow(axs[1, 0], fig, Pxx_2d,  "Pxx"); axs[1, 0].set_title("Pxx = ∫ Ωx² ψ dΩ")
    _ishow(axs[1, 1], fig, Pxy_2d,  "Pxy"); axs[1, 1].set_title("Pxy = ∫ Ωx Ωy ψ dΩ")
    _ishow(axs[1, 2], fig, Pyy_2d,  "Pyy"); axs[1, 2].set_title("Pyy = ∫ Ωy² ψ dΩ")
    fig.suptitle(f"Angular moments (mean) — {geometry} ({Nx}×{Ny} grid)")
    plt.tight_layout(); plt.show()

    # ------------------------------------------------------------------
    # Plot standard deviations of the mean (σ = sqrt(Var))
    # ------------------------------------------------------------------
    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    _ishow(axs[0, 0], fig, phi_std_2d,  "σ(φ)",   log=False); axs[0, 0].set_title("std(φ)")
    _ishow(axs[0, 1], fig, Jx_std_2d,   "σ(Jx)");             axs[0, 1].set_title("std(Jx)")
    _ishow(axs[0, 2], fig, Jy_std_2d,   "σ(Jy)");             axs[0, 2].set_title("std(Jy)")
    _ishow(axs[1, 0], fig, Pxx_std_2d,  "σ(Pxx)");            axs[1, 0].set_title("std(Pxx)")
    _ishow(axs[1, 1], fig, Pxy_std_2d,  "σ(Pxy)");            axs[1, 1].set_title("std(Pxy)")
    _ishow(axs[1, 2], fig, Pyy_std_2d,  "σ(Pyy)");            axs[1, 2].set_title("std(Pyy)")
    fig.suptitle(f"Standard deviation of the mean — {geometry} ({Nx}×{Ny} grid)")
    plt.tight_layout(); plt.show()


if __name__ == "__main__":
    main()
