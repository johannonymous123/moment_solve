#!/usr/bin/env python
"""
Quick comparison of steady-state MC vs moments solver for the crossing_beams
geometry.  Runs both, computes domain-integrated phi and plots side-by-side.

Purpose: verify that the source normalization, material definitions, and
boundary conditions are consistent between the two codes.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ============================================================
# 1) Run the moments solver (steady-state, px=0)
# ============================================================
from moments_solve import Grid, solve_m1_dg, D_from_phiJ_levermore

grid = Grid(x0=0.0, x1=7.0, y0=0.0, y1=7.0, nx=70, ny=70)

def Q(x, y):
    beam_a = (0.5 < x < 1.0) and (2.5 < y < 4.5)
    beam_b = (2.5 < x < 4.5) and (0.5 < y < 1.0)
    return 1.0 if (beam_a or beam_b) else 0.0

def sigma_a(x, y):
    if x < 0.5 or x > 6.5 or y < 0.5 or y > 6.5:
        return 10.0
    return 0.02

def sigma_s(x, y):
    return 0.0

print("=" * 60)
print("  Running moments solver (px=0, steady-state)")
print("=" * 60)
sol_m1 = solve_m1_dg(
    grid=grid, px=0, py=0,
    Q_cell=Q, sigma_a_cell=sigma_a, sigma_s_cell=sigma_s,
    D_func=D_from_phiJ_levermore,
    use_face_speed=True, amin_face=1e-3,
    enforce_realizability=True, phi_floor=1e-12, realiz_delta=1e-10,
    stabilize_D_tensor=False,
    use_modal_filter=True, modal_alpha=18.0, modal_s=8,
    max_picard=80, relax=0.3,
    adaptive_relax=True, adapt_max_halvings=10,
    tol=1e-5,
    stall_tol=1e-2, stall_window=10,
    verbose=True,
    bc_type="vacuum_marshak", marshak_beta=0.25,
)

phi_m1 = sol_m1.phi[:, 0]  # cell-average φ
Jx_m1  = sol_m1.Jx[:, 0]
Jy_m1  = sol_m1.Jy[:, 0]

# ============================================================
# 2) Run the MC solver (steady-state)
# ============================================================
from MC_reference import (
    build_cartesian_grid, assign_materials_cartesian,
    sample_source_particle_numpy, find_initial_cell_cartesian,
    run_mc_cartesian,
)

Nx, Ny = 70, 70
Lx, Ly = 7.0, 7.0
geometry = "crossing_beams"

dx, dy, cell_centers, neighbors, cell_areas = build_cartesian_grid(Nx, Ny, Lx, Ly)
Nc = Nx * Ny
mc_sigma_a, mc_sigma_s = assign_materials_cartesian(cell_centers, geometry)

Np = 500_000   # enough for a decent comparison; bump to 5M for cleaner results
w_cut = 1e-6 / Np
w_survive = 1e-2
max_cell_crossings = Np * 1000

rng = np.random.default_rng(42)
init_x    = np.empty(Np, dtype=np.float64)
init_y    = np.empty(Np, dtype=np.float64)
init_dx   = np.empty(Np, dtype=np.float64)
init_dy   = np.empty(Np, dtype=np.float64)
init_w    = np.empty(Np, dtype=np.float64)
init_cell = np.empty(Np, dtype=np.int64)

for i in range(Np):
    x, u, w = sample_source_particle_numpy(rng, geometry)
    cell = find_initial_cell_cartesian(x, Nx, Ny, dx, dy)
    init_cell[i] = cell
    init_x[i] = x[0]; init_y[i] = x[1]
    init_dx[i] = u[0]; init_dy[i] = u[1]
    init_w[i] = w

base = np.uint64(0x9E3779B97F4A7C15)
rng_states = np.empty(Np, dtype=np.uint64)
for i in range(Np):
    rng_states[i] = base ^ np.uint64(i + 1) ^ np.uint64(0xD1B54A32D192ED03)

flux_tally    = np.zeros(Nc, dtype=np.float64)
flux_tally_sq = np.zeros(Nc, dtype=np.float64)
J_tally       = np.zeros((Nc, 2), dtype=np.float64)
J_tally_sq    = np.zeros((Nc, 2), dtype=np.float64)
P_tally       = np.zeros((Nc, 3), dtype=np.float64)
P_tally_sq    = np.zeros((Nc, 3), dtype=np.float64)

print("\n" + "=" * 60)
print(f"  Running MC solver (Np={Np:,}, steady-state)")
print("=" * 60)

run_mc_cartesian(
    Np, init_x, init_y, init_dx, init_dy,
    init_w, init_cell,
    neighbors, mc_sigma_a, mc_sigma_s,
    w_cut, w_survive, max_cell_crossings,
    Nx, Ny, dx, dy,
    rng_states,
    flux_tally, flux_tally_sq,
    J_tally, J_tally_sq,
    P_tally, P_tally_sq,
)

norm_mc = cell_areas * Np
phi_mc = flux_tally / norm_mc
Jx_mc  = J_tally[:, 0] / norm_mc
Jy_mc  = J_tally[:, 1] / norm_mc

print("MC done.")

# ============================================================
# 3) Compare
# ============================================================
hx, hy = grid.hx, grid.hy
cell_area = hx * hy

int_phi_m1 = np.sum(phi_m1) * cell_area
int_phi_mc = np.sum(phi_mc) * cell_area

# Integral of source: Q=1 over two strips each of area 0.5*2.0 = 1.0, total = 2.0
int_Q = 2.0

print(f"\n{'='*60}")
print(f"  Comparison summary")
print(f"{'='*60}")
print(f"∫ Q dA (analytic)       = {int_Q:.4f}")
print(f"∫ φ dA (moments, px=0)  = {int_phi_m1:.4f}")
print(f"∫ φ dA (MC, Np={Np:,})   = {int_phi_mc:.4f}")
print(f"Ratio  moments/MC       = {int_phi_m1 / int_phi_mc:.4f}")
print(f"Ratio  MC*2/moments    = {2*int_phi_mc / int_phi_m1:.4f}  (sanity: old w=1 bug)")
print(f"")
print(f"φ max  (moments) = {phi_m1.max():.4e}")
print(f"φ max  (MC)      = {phi_mc.max():.4e}")
print(f"φ min  (moments) = {phi_m1.min():.4e}")
print(f"φ min  (MC)      = {phi_mc.min():.4e}")

# L2 relative error (excluding absorbing frame where both are ~0)
interior = np.array([
    (0.5 <= cell_centers[c, 0] <= 6.5) and (0.5 <= cell_centers[c, 1] <= 6.5)
    for c in range(Nc)
])
diff = phi_m1[interior] - phi_mc[interior]
l2_rel = np.linalg.norm(diff) / (np.linalg.norm(phi_mc[interior]) + 1e-30)
linf_rel = np.max(np.abs(diff)) / (np.max(np.abs(phi_mc[interior])) + 1e-30)
print(f"L2 rel error  (interior) = {l2_rel:.4f}")
print(f"Linf rel error (interior) = {linf_rel:.4f}")

# ============================================================
# 4) Plots
# ============================================================
phi_m1_2d = phi_m1.reshape((Ny, Nx))
phi_mc_2d = phi_mc.reshape((Ny, Nx))

extent = [0, Lx, 0, Ly]
vmin = max(1e-6, min(phi_m1[phi_m1 > 0].min() if np.any(phi_m1 > 0) else 1e-6,
                     phi_mc[phi_mc > 0].min() if np.any(phi_mc > 0) else 1e-6))
vmax = max(phi_m1.max(), phi_mc.max())

fig, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)

im0 = axes[0].imshow(np.maximum(phi_m1_2d, vmin), origin="lower", extent=extent,
                      aspect="auto", norm=mcolors.LogNorm(vmin=vmin, vmax=vmax))
axes[0].set_title("Moments (M1, px=0)")
axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
fig.colorbar(im0, ax=axes[0], label="φ")

im1 = axes[1].imshow(np.maximum(phi_mc_2d, vmin), origin="lower", extent=extent,
                      aspect="auto", norm=mcolors.LogNorm(vmin=vmin, vmax=vmax))
axes[1].set_title(f"MC (Np={Np:,})")
axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
fig.colorbar(im1, ax=axes[1], label="φ")

# Ratio plot
ratio = np.where(phi_mc_2d > 1e-8, phi_m1_2d / phi_mc_2d, np.nan)
im2 = axes[2].imshow(ratio, origin="lower", extent=extent,
                      aspect="auto", cmap="RdBu_r", vmin=0.5, vmax=1.5)
axes[2].set_title("Ratio  M1 / MC")
axes[2].set_xlabel("x"); axes[2].set_ylabel("y")
fig.colorbar(im2, ax=axes[2], label="φ_M1 / φ_MC")

fig.suptitle("Crossing beams — steady-state comparison", fontsize=14)
plt.show()

# Line cuts through the crossing region
iy_mid = Ny // 2  # y ≈ 3.5
ix_mid = Nx // 2  # x ≈ 3.5

fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)

x_coords = np.linspace(hx/2, Lx - hx/2, Nx)
y_coords = np.linspace(hy/2, Ly - hy/2, Ny)

axes[0].semilogy(x_coords, phi_m1_2d[iy_mid, :], 'b-', label="Moments (M1)")
axes[0].semilogy(x_coords, phi_mc_2d[iy_mid, :], 'r--', label="MC")
axes[0].set_xlabel("x"); axes[0].set_ylabel("φ")
axes[0].set_title(f"Line cut at y ≈ {y_coords[iy_mid]:.2f}")
axes[0].legend(); axes[0].grid(True)

axes[1].semilogy(y_coords, phi_m1_2d[:, ix_mid], 'b-', label="Moments (M1)")
axes[1].semilogy(y_coords, phi_mc_2d[:, ix_mid], 'r--', label="MC")
axes[1].set_xlabel("y"); axes[1].set_ylabel("φ")
axes[1].set_title(f"Line cut at x ≈ {x_coords[ix_mid]:.2f}")
axes[1].legend(); axes[1].grid(True)

fig.suptitle("Crossing beams — line cuts", fontsize=14)
plt.show()
