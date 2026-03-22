"""
Numba-compatible 2D time-dependent MC transport on a Cartesian (rectangular) grid.

Workflow:
- Build a uniform Nx x Ny grid over [0,Lx] x [0,Ly] directly in the script.
- Assign per-cell materials sigma_a, sigma_s (time-independent for now).
- Sample Np source particles; each particle is assigned a birth time
  t_birth ~ Uniform[0, T_f].  The source is treated as continuously emitting
  at constant rate Q(t) = Q(0), so the uniform birth-time sampling is exact.
- Each particle carries an internal clock  t = t_birth + path_length / c.
- Tally arrays are time-resolved: shape (Nc, N_t).
  Every straight sub-segment of a track is clipped to each time bin it
  overlaps and its contribution (track-length × attenuated weight / dt) is
  accumulated into that bin.
- Normalization: phi[cell, it] = tally[cell, it] / (cell_area * Np * dt)
  giving the time-averaged scalar flux over each bin.
- Plots show the snapshot at the final bin [T_f - dt, T_f].

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
    cell_centers : (Nc, 2) float64
    neighbors : (Nc, 4) int64   — -1 = leakage boundary
    cell_areas : (Nc,) float64
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

            if ix + 1 < Nx:
                neighbors[c, 0] = iy * Nx + (ix + 1)
            if ix - 1 >= 0:
                neighbors[c, 1] = iy * Nx + (ix - 1)
            if iy + 1 < Ny:
                neighbors[c, 2] = (iy + 1) * Nx + ix
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
        return 100.0, 0.0
    if y < 0.05 or y > 1.25:
        return 100.0, 0.0
    if x < 0.05 and 0.25 < y < 1.05:
        return 5.0, 95.0
    if 0.5 < x < 0.85 and 0.3 < y < 1:
        return 50.0, 50.0
    if 0.45 < x < 0.85 and 0.25 < y < 1.05:
        return 10.0, 90.0
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
      - Source is volumetric isotropic strip at 0.10 < x < 0.15, 0.10 < y < 1.40.
    """
    if x < 0.05 or x > 1.45:
        return 100.0, 0.0
    if y < 0.05 or y > 1.45:
        return 100.0, 0.0
    if 0.20 < x < 0.25 and 0.35 < y < 1.15:
        return 5.0, 95.0
    if 0.70 < x < 1.05 and 0.40 < y < 1.10:
        return 50.0, 50.0
    if 0.65 < x < 1.05 and 0.35 < y < 1.15:
        return 10.0, 90.0
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
# Source sampling
# ============================================================

def sample_source_particle_numpy(rng: np.random.Generator, geometry: str,
                                  T_f: float):
    """
    Sample a source particle.  Returns position x, direction u, weight w,
    and birth time t_birth ~ Uniform[0, T_f].
    """
    if geometry == "lattice":
        xmin, xmax, ymin, ymax = 3.0, 4.0, 3.0, 4.0
        x = np.array([rng.uniform(xmin, xmax), rng.uniform(ymin, ymax)], dtype=np.float64)
        phi_ang = rng.uniform(0.0, TWO_PI)
        costh   = rng.uniform(-1.0, 1.0)
        sinth   = np.sqrt(1.0 - costh**2)
        u = np.array([sinth * np.cos(phi_ang), sinth * np.sin(phi_ang)], dtype=np.float64)
        w = T_f*1.0
    elif geometry == "Hohlraum_v2":
        # Volumetric isotropic source in strip [0.10, 0.15] x [0.10, 1.40]
        x = np.array([rng.uniform(0.10, 0.15), rng.uniform(0.10, 1.40)], dtype=np.float64)
        costh   = rng.uniform(-1.0, 1.0)
        sinth   = np.sqrt(1.0 - costh**2)
        phi_ang = rng.uniform(0.0, TWO_PI)
        u = np.array([sinth * np.cos(phi_ang), sinth * np.sin(phi_ang)], dtype=np.float64)
        w = T_f * 1.0
    elif geometry == "crossing_beams":
        # Two volumetric isotropic source strips with equal area (=1.0 each):
        #   Beam A: x in [0.5,1.0], y in [2.5,4.5]  (area 1.0)
        #   Beam B: x in [2.5,4.5], y in [0.5,1.0]  (area 1.0)
        if rng.uniform(0.0, 1.0) < 0.5:
            x = np.array([rng.uniform(0.5, 1.0), rng.uniform(2.5, 4.5)], dtype=np.float64)
        else:
            x = np.array([rng.uniform(2.5, 4.5), rng.uniform(0.5, 1.0)], dtype=np.float64)
        costh   = rng.uniform(-1.0, 1.0)
        sinth   = np.sqrt(1.0 - costh**2)
        phi_ang = rng.uniform(0.0, TWO_PI)
        u = np.array([sinth * np.cos(phi_ang), sinth * np.sin(phi_ang)], dtype=np.float64)
        w = T_f * 1.0
    else:
        y_min, y_max = 0.0, 1.3
        x   = np.array([1e-8, rng.uniform(y_min, y_max)], dtype=np.float64)
        mu  = np.sqrt(rng.uniform(0.0, 1.0))
        phi_ang = rng.uniform(0.0, TWO_PI)
        u   = np.array([mu, np.sqrt(1.0 - mu**2) * np.sin(phi_ang)], dtype=np.float64)
        w   = T_f*1.0 / 1.3

    t_birth = rng.uniform(0.0, T_f)
    return x, u, w, t_birth


# ============================================================
# Initial cell find
# ============================================================

def find_initial_cell_cartesian(x: np.ndarray, Nx: int, Ny: int,
                                 dx: float, dy: float) -> int:
    ix = int(x[0] / dx)
    iy = int(x[1] / dy)
    if ix < 0 or ix >= Nx or iy < 0 or iy >= Ny:
        return -1
    return iy * Nx + ix


# ============================================================
# Numba RNG: xorshift64*
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
    iy = cell // Nx
    ix = cell  - iy * Nx

    x_lo = ix * dx;  x_hi = x_lo + dx
    y_lo = iy * dy;  y_hi = y_lo + dy

    t_min = 1e300
    edge  = -1

    if dx_dir > 1e-15:
        t = (x_hi - px) / dx_dir
        if t > 1e-12 and t < t_min:
            t_min = t;  edge = 0
    elif dx_dir < -1e-15:
        t = (x_lo - px) / dx_dir
        if t > 1e-12 and t < t_min:
            t_min = t;  edge = 1

    if dy_dir > 1e-15:
        t = (y_hi - py) / dy_dir
        if t > 1e-12 and t < t_min:
            t_min = t;  edge = 2
    elif dy_dir < -1e-15:
        t = (y_lo - py) / dy_dir
        if t > 1e-12 and t < t_min:
            t_min = t;  edge = 3

    return t_min, edge


# ============================================================
# Physics pieces (Numba)
# ============================================================

@nb.njit(inline="always")
def sample_scatter_distance_numba(state, sigma_s):
    if sigma_s <= 0.0:
        return 1e300, state
    u, state = rng_u01(state)
    return -np.log(u) / sigma_s, state


@nb.njit(inline="always")
def sample_scatter_dir_2d_numba(state):
    costh, state = rng_uniform(state, -1.0, 1.0)
    sinth = np.sqrt(1.0 - costh * costh)
    phi, state   = rng_uniform(state,  0.0, TWO_PI)
    return sinth * np.cos(phi), sinth * np.sin(phi), state


@nb.njit(inline="always")
def tracklength_flux_contrib(w_in, sigma_a, L):
    """Integral of attenuated weight over path of length L."""
    if L <= 0.0:
        return 0.0
    if sigma_a > 0.0:
        return w_in * (1.0 - np.exp(-sigma_a * L)) / sigma_a
    else:
        return w_in * L


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
    return False, 0.0, state


# ============================================================
# Time-bin tally helper
# ============================================================

@nb.njit(inline="always")
def tally_segment_into_bins(
        flux_tally, J_tally, P_tally,
        cell,
        t_seg_start, t_seg_end,
        w_seg_start, sigma_a,
        dx_dir, dy_dir,
        dt, N_t, T_f, c):
    """
    Accumulate a straight sub-segment [t_seg_start, t_seg_end] into all
    time bins it overlaps.

    The particle travels at speed c, so path length ds = c * dt_overlap.
    The attenuated weight at time t within the segment is:
        w(t) = w_seg_start * exp(-sigma_a * (t - t_seg_start) * c)

    The tally contribution to bin [t_lo, t_hi] is:
        integral_{t_lo}^{t_hi} w(t) c dt  (= tracklength_flux_contrib for that slice)
    clipped to the actual segment extent.

    Moment tallies use the same integral weighted by Omega components.
    """
    if t_seg_end <= t_seg_start:
        return

    # First and last bin indices that this segment overlaps
    it_lo = int(t_seg_start / dt)
    it_hi = int(t_seg_end   / dt)
    if it_lo >= N_t:
        return
    if it_hi >= N_t:
        it_hi = N_t - 1

    for it in range(it_lo, it_hi + 1):
        t_bin_lo = it * dt
        t_bin_hi = t_bin_lo + dt

        # Clip segment to this bin
        t_lo = t_seg_start if t_seg_start > t_bin_lo else t_bin_lo
        t_hi = t_seg_end   if t_seg_end   < t_bin_hi else t_bin_hi
        if t_hi <= t_lo:
            continue

        # Path length covered in this bin slice
        L_slice = (t_hi - t_lo) * c

        # Weight at the start of this slice (attenuated from t_seg_start)
        L_before = (t_lo - t_seg_start) * c
        w_slice_start = attenuate_weight_numba(w_seg_start, sigma_a, L_before)

        # Integral of attenuated weight over the slice
        phi_contrib = tracklength_flux_contrib(w_slice_start, sigma_a, L_slice)

        flux_tally[cell, it]    += phi_contrib
        J_tally[cell, it, 0]    += dx_dir * phi_contrib
        J_tally[cell, it, 1]    += dy_dir * phi_contrib
        P_tally[cell, it, 0]    += dx_dir * dx_dir * phi_contrib
        P_tally[cell, it, 1]    += dx_dir * dy_dir * phi_contrib
        P_tally[cell, it, 2]    += dy_dir * dy_dir * phi_contrib


# ============================================================
# Core: move within a Cartesian cell until boundary exit
# ============================================================

@nb.njit
def move_to_boundary_cartesian_timed(
        px, py, dx_dir, dy_dir, cell,
        neighbors,
        sigma_a, sigma_s,
        Nx, Ny, dx, dy,
        rng_state,
        w_cur, t_cur,
        flux_tally, J_tally, P_tally,
        dt, N_t, T_f, c):
    """
    Simulate within-cell scattering until the particle exits the current cell,
    tallying contributions into time-resolved arrays.

    Parameters
    ----------
    w_cur, t_cur : weight and clock at cell entry.

    Returns
    -------
    px, py, dx_dir, dy_dir : exit position and direction
    next_cell              : neighbour index or -1 (leakage)
    L_total                : total path length in this cell
    t_exit                 : particle clock at cell exit
    w_exit                 : attenuated weight at cell exit
    rng_state              : updated RNG
    edge_id                : face crossed (0-3)
    """
    sa = sigma_a[cell]
    ss = sigma_s[cell]

    L_total = 0.0
    w_seg   = w_cur
    t_seg   = t_cur

    while True:
        t_space_exit, edge_id = distance_to_cell_exit_numba(
            px, py, dx_dir, dy_dir, cell, Nx, Ny, dx, dy)

        if edge_id < 0 or t_space_exit > 1e299:
            return (px, py, dx_dir, dy_dir, -1, L_total,
                    t_seg, w_seg, rng_state, -1)

        # Time at which particle would exit the cell (no scatter)
        t_space_exit_time = t_seg + t_space_exit / c

        # Particle is killed if it has passed T_f already
        if t_seg >= T_f:
            return (px, py, dx_dir, dy_dir, -1, L_total,
                    t_seg, w_seg, rng_state, -1)

        # Clip the spatial step by T_f (particle dies at T_f)
        t_end = t_space_exit_time if t_space_exit_time < T_f else T_f

        s_scatter, rng_state = sample_scatter_distance_numba(rng_state, ss)
        t_scatter = t_seg + s_scatter / c

        if s_scatter < t_space_exit - 1e-12 and t_scatter < T_f:
            # Scatter happens inside the cell before T_f
            t_seg_end = t_scatter
            tally_segment_into_bins(
                flux_tally, J_tally, P_tally,
                cell, t_seg, t_seg_end,
                w_seg, sa, dx_dir, dy_dir,
                dt, N_t, T_f, c)

            w_seg    = attenuate_weight_numba(w_seg, sa, s_scatter)
            px      += s_scatter * dx_dir
            py      += s_scatter * dy_dir
            L_total += s_scatter
            t_seg    = t_seg_end
            dx_dir, dy_dir, rng_state = sample_scatter_dir_2d_numba(rng_state)
            continue

        # Exit the cell (or reach T_f)
        # Tally up to min(cell exit, T_f)
        L_to_end = (t_end - t_seg) * c
        tally_segment_into_bins(
            flux_tally, J_tally, P_tally,
            cell, t_seg, t_end,
            w_seg, sa, dx_dir, dy_dir,
            dt, N_t, T_f, c)

        w_exit   = attenuate_weight_numba(w_seg, sa, L_to_end)
        px      += L_to_end * dx_dir
        py      += L_to_end * dy_dir
        L_total += L_to_end
        t_exit   = t_end

        if t_exit >= T_f:
            # Particle reached final time — kill it
            return (px, py, dx_dir, dy_dir, -1, L_total,
                    t_exit, w_exit, rng_state, -1)

        next_cell = neighbors[cell, edge_id]
        return (px, py, dx_dir, dy_dir, next_cell, L_total,
                t_exit, w_exit, rng_state, edge_id)


# ============================================================
# Transport kernel (Numba, SoA particles)
# ============================================================

@nb.njit
def run_mc_cartesian(Np,
                     init_x, init_y, init_dx, init_dy,
                     init_w, init_cell, init_t,
                     neighbors,
                     sigma_a, sigma_s,
                     w_cut, w_survive,
                     max_cell_crossings,
                     Nx, Ny, dx, dy,
                     rng_states,
                     flux_tally,
                     J_tally,
                     P_tally,
                     dt, N_t, T_f, c):
    """
    Time-dependent MC transport kernel on a Cartesian grid.

    Tally arrays (indexed [cell, time_bin])
    ----------------------------------------
    flux_tally : (Nc, N_t)   — cumulative track-length phi contributions
    J_tally    : (Nc, N_t, 2) — Jx, Jy contributions
    P_tally    : (Nc, N_t, 3) — Pxx, Pxy, Pyy contributions

    Each contribution is the integral of attenuated weight over the portion
    of the track that falls in each (cell, time_bin) voxel.
    Normalization in Python: divide by (cell_area * Np * dt).
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
        t     = init_t[i]
        state = rng_states[i]

        if t >= T_f:
            continue

        alive     = True
        crossings = 0

        while alive and crossings < max_cell_crossings:
            crossings += 1

            (px_exit, py_exit, dx_exit, dy_exit,
             next_cell, L_total, t_exit, w_exit,
             state, edge_id) = \
                move_to_boundary_cartesian_timed(
                    px, py, dx_p, dy_p, cell,
                    neighbors, sigma_a, sigma_s,
                    Nx, Ny, dx, dy,
                    state, w, t,
                    flux_tally, J_tally, P_tally,
                    dt, N_t, T_f, c)

            # Advance particle state
            w = w_exit
            t = t_exit

            if next_cell < 0 or t >= T_f:
                alive = False
                break

            # Russian roulette on cell exit (spatial leakage only)
            alive, w, state = russian_roulette_numba(state, w, w_cut, w_survive)
            if not alive:
                break

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
    extent = [0, Lx, 0, Ly]
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))

    def _ishow(ax, data, label, log=False):
        if log:
            d = np.copy(data)
            d[d < 1e-10] = 1e-10
            im = ax.imshow(d, origin="lower", extent=extent, aspect="equal",
                           norm=mcolors.LogNorm(vmin=1e-10, vmax=d.max()),
                           interpolation="nearest")
        else:
            im = ax.imshow(data, origin="lower", extent=extent, aspect="equal",
                           interpolation="nearest")
        fig.colorbar(im, ax=ax, label=label)
        ax.set_xlabel("x");  ax.set_ylabel("y")

    _ishow(axs[0, 0], phi_2d,     "flux",     log=False);  axs[0, 0].set_title(title)
    _ishow(axs[0, 1], phi_2d,     "flux",     log=True);   axs[0, 1].set_title(title + " (log)")
    _ishow(axs[1, 0], sigma_a_2d, "σ_a",      log=False);  axs[1, 0].set_title("σ_a")
    _ishow(axs[1, 1], sigma_s_2d, "σ_s",      log=False);  axs[1, 1].set_title("σ_s")
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

    if Lattice:
        geometry = "lattice"
        Lx, Ly   = 7.0, 7.0
        Nx, Ny   = 70, 70
    elif True:   # crossing_beams
        geometry = "crossing_beams"
        Lx, Ly   = 7.0, 7.0
        Nx, Ny   = 70, 70
    else:
        geometry = "Hohlraum_v2"
        Lx, Ly   = 1.5, 1.5
        Nx, Ny   = 75, 75

    # ------------------------------------------------------------------
    # Time parameters
    # ------------------------------------------------------------------
    c   = 1.0       # speed of light (change units here)
    T_f = 2.5       # final time
    N_t = 10        # number of time steps
    dt  = T_f / N_t

    # ------------------------------------------------------------------
    # Build grid
    # ------------------------------------------------------------------
    dx, dy, cell_centers, neighbors, cell_areas = build_cartesian_grid(Nx, Ny, Lx, Ly)
    Nc = Nx * Ny
    print(f"Grid: {Nx}x{Ny} = {Nc} cells, dx={dx:.4f}, dy={dy:.4f}")
    print(f"Time: T_f={T_f}, N_t={N_t}, dt={dt:.4f}, c={c}")

    # ------------------------------------------------------------------
    # Materials
    # ------------------------------------------------------------------
    sigma_a, sigma_s = assign_materials_cartesian(cell_centers, geometry)

    # ------------------------------------------------------------------
    # MC parameters
    # ------------------------------------------------------------------
    Np                 = 5000
    w_cut              = 1e-6 / Np
    w_survive          = 1e-2
    max_cell_crossings = 100_000   # per particle (not Np*1000 — time kills them)

    # ------------------------------------------------------------------
    # Initialise particles (birth times sampled uniformly over [0, T_f])
    # ------------------------------------------------------------------
    rng       = np.random.default_rng(1234)
    init_x    = np.empty(Np, dtype=np.float64)
    init_y    = np.empty(Np, dtype=np.float64)
    init_dx   = np.empty(Np, dtype=np.float64)
    init_dy   = np.empty(Np, dtype=np.float64)
    init_w    = np.empty(Np, dtype=np.float64)
    init_t    = np.empty(Np, dtype=np.float64)
    init_cell = np.empty(Np, dtype=np.int64)

    for i in range(Np):
        x, u, w, t_birth = sample_source_particle_numpy(rng, geometry, T_f)
        cell = find_initial_cell_cartesian(x, Nx, Ny, dx, dy)
        init_cell[i] = cell
        init_x[i]    = x[0]
        init_y[i]    = x[1]
        init_dx[i]   = u[0]
        init_dy[i]   = u[1]
        init_w[i]    = w
        init_t[i]    = t_birth

    # ------------------------------------------------------------------
    # Per-particle RNG seeds
    # ------------------------------------------------------------------
    base       = np.uint64(0x9E3779B97F4A7C15)
    rng_states = np.empty(Np, dtype=np.uint64)
    for i in range(Np):
        rng_states[i] = base ^ np.uint64(i + 1) ^ np.uint64(0xD1B54A32D192ED03)

    # ------------------------------------------------------------------
    # Tally arrays: (Nc, N_t) for phi, (Nc, N_t, 2) for J, (Nc, N_t, 3) for P
    # ------------------------------------------------------------------
    flux_tally = np.zeros((Nc, N_t),    dtype=np.float64)
    J_tally    = np.zeros((Nc, N_t, 2), dtype=np.float64)
    P_tally    = np.zeros((Nc, N_t, 3), dtype=np.float64)

    # ------------------------------------------------------------------
    # Run Numba kernel
    # ------------------------------------------------------------------
    print("Running time-dependent MC kernel …")
    run_mc_cartesian(Np,
                     init_x, init_y, init_dx, init_dy,
                     init_w, init_cell, init_t,
                     neighbors,
                     sigma_a, sigma_s,
                     w_cut, w_survive,
                     max_cell_crossings,
                     Nx, Ny, dx, dy,
                     rng_states,
                     flux_tally, J_tally, P_tally,
                     dt, N_t, T_f, c)
    print("Done.")

    # ------------------------------------------------------------------
    # Normalise: phi[cell, it] = tally / (cell_area * Np * dt)
    # This gives the time-averaged scalar flux over each bin.
    # ------------------------------------------------------------------
    norm = cell_areas * Np * dt   # shape (Nc,) — broadcast over time axis

    phi = flux_tally / norm[:, np.newaxis]          # (Nc, N_t)
    Jx  = J_tally[:, :, 0] / norm[:, np.newaxis]
    Jy  = J_tally[:, :, 1] / norm[:, np.newaxis]
    Pxx = P_tally[:, :, 0] / norm[:, np.newaxis]
    Pxy = P_tally[:, :, 1] / norm[:, np.newaxis]
    Pyy = P_tally[:, :, 2] / norm[:, np.newaxis]

    # ------------------------------------------------------------------
    # Extract the final time bin (snapshot at T_f)
    # ------------------------------------------------------------------
    it_final = N_t - 1
    phi_f   = phi[:, it_final]
    Jx_f    = Jx[:,  it_final]
    Jy_f    = Jy[:,  it_final]
    Pxx_f   = Pxx[:, it_final]
    Pxy_f   = Pxy[:, it_final]
    Pyy_f   = Pyy[:, it_final]

    def r2d(a): return a.reshape(Ny, Nx)

    phi_2d     = r2d(phi_f)
    Jx_2d      = r2d(Jx_f)
    Jy_2d      = r2d(Jy_f)
    Pxx_2d     = r2d(Pxx_f)
    Pxy_2d     = r2d(Pxy_f)
    Pyy_2d     = r2d(Pyy_f)
    sigma_a_2d = r2d(sigma_a)
    sigma_s_2d = r2d(sigma_s)

    t_lo = it_final * dt
    t_hi = T_f
    print(f"\n--- Final time bin [{t_lo:.3f}, {t_hi:.3f}] ---")
    print(f"phi  min/max: {phi_f.min():.4e}  {phi_f.max():.4e}")
    print(f"|J|  min/max: {np.hypot(Jx_f, Jy_f).min():.4e}  {np.hypot(Jx_f, Jy_f).max():.4e}")
    print(f"Pxx  min/max: {Pxx_f.min():.4e}  {Pxx_f.max():.4e}")
    print(f"Pyy  min/max: {Pyy_f.min():.4e}  {Pyy_f.max():.4e}")

    # ------------------------------------------------------------------
    # Plot scalar flux + material maps at final time
    # ------------------------------------------------------------------
    title_str = (f"Scalar flux — {geometry} ({Nx}×{Ny}), "
                 f"t ∈ [{t_lo:.2f}, {t_hi:.2f}]")
    plot_flux_heatmap_cartesian(phi_2d, sigma_a_2d, sigma_s_2d, Lx, Ly,
                                title=title_str)

    # ------------------------------------------------------------------
    # Plot angular moments at final time
    # ------------------------------------------------------------------
    extent = [0, Lx, 0, Ly]

    def _ishow(ax, fig, data, label, log=False):
        if log:
            d = np.copy(data)
            d[d < 1e-10] = 1e-10
            im = ax.imshow(d, origin="lower", extent=extent, aspect="equal",
                           norm=mcolors.LogNorm(vmin=1e-10, vmax=d.max()),
                           interpolation="nearest")
        else:
            im = ax.imshow(data, origin="lower", extent=extent, aspect="equal",
                           interpolation="nearest")
        fig.colorbar(im, ax=ax, label=label)
        ax.set_xlabel("x");  ax.set_ylabel("y")

    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    _ishow(axs[0, 0], fig, phi_2d,  "φ",    log=True);  axs[0, 0].set_title("φ (log)")
    _ishow(axs[0, 1], fig, Jx_2d,   "Jx");              axs[0, 1].set_title("Jx")
    _ishow(axs[0, 2], fig, Jy_2d,   "Jy");              axs[0, 2].set_title("Jy")
    _ishow(axs[1, 0], fig, Pxx_2d,  "Pxx"); axs[1, 0].set_title("Pxx")
    _ishow(axs[1, 1], fig, Pxy_2d,  "Pxy"); axs[1, 1].set_title("Pxy")
    _ishow(axs[1, 2], fig, Pyy_2d,  "Pyy"); axs[1, 2].set_title("Pyy")
    fig.suptitle(f"Angular moments — {geometry} ({Nx}×{Ny}), "
                 f"t ∈ [{t_lo:.2f}, {t_hi:.2f}]")
    plt.tight_layout();  plt.show()

    # ------------------------------------------------------------------
    # Optional: time series of domain-integrated phi (useful sanity check)
    # ------------------------------------------------------------------
    phi_integrated = np.array([
        np.sum(phi[:, it] * cell_areas) for it in range(N_t)
    ])
    t_centers = (np.arange(N_t) + 0.5) * dt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_centers, phi_integrated, marker="o", ms=3)
    ax.set_xlabel("t");  ax.set_ylabel("∫ φ dA")
    ax.set_title(f"Domain-integrated flux vs. time — {geometry}")
    ax.grid(True)
    plt.tight_layout();  plt.show()


if __name__ == "__main__":
    main()
