"""
compare_mc_m1.py
----------------
Run both the MC (Run_cartesian) and M1-DG (moments_solve) on the same
lattice problem and grid, then compare φ, J, and the Eddington ratio P/φ.

Uses a reduced particle count (500k) for speed; the MC will be noisy but
good enough to check normalization / systematic offsets.
"""

import numpy as np
import sys, time
sys.path.insert(0, ".")

# ── MC imports ──────────────────────────────────────────────────────────────
from Run_cartesian import (
    build_cartesian_grid,
    assign_materials_cartesian,
    sample_source_particle_numpy,
    find_initial_cell_cartesian,
    run_mc_cartesian,
    TWO_PI,
)

# ── M1 imports ──────────────────────────────────────────────────────────────
from moments_solve import (
    Grid,
    solve_m1_dg,
    D_from_phiJ_levermore,
)


def main():
    # ================================================================
    # Common parameters
    # ================================================================
    Lx, Ly = 7.0, 7.0
    Nx, Ny = 70, 70          # same grid for both
    geometry = "lattice"

    # ================================================================
    # 1) MC run
    # ================================================================
    print("=" * 60)
    print("  MC  (Run_cartesian)")
    print("=" * 60)

    dx, dy, cell_centers, neighbors, cell_areas = build_cartesian_grid(Nx, Ny, Lx, Ly)
    Nc = Nx * Ny

    sigma_a_mc, sigma_s_mc = assign_materials_cartesian(cell_centers, geometry)

    Np = 2_000_000
    w_cut = 1e-6 / Np
    w_survive = 1e-2
    max_crossings = Np * 1000

    rng = np.random.default_rng(42)
    init_x    = np.empty(Np, dtype=np.float64)
    init_y    = np.empty(Np, dtype=np.float64)
    init_dx   = np.empty(Np, dtype=np.float64)
    init_dy   = np.empty(Np, dtype=np.float64)
    init_w    = np.empty(Np, dtype=np.float64)
    init_cell = np.empty(Np, dtype=np.int64)

    for i in range(Np):
        x, u, w = sample_source_particle_numpy(rng, geometry)
        init_x[i]    = x[0]
        init_y[i]    = x[1]
        init_dx[i]   = u[0]
        init_dy[i]   = u[1]
        init_w[i]    = w
        init_cell[i] = find_initial_cell_cartesian(x, Nx, Ny, dx, dy)

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

    print(f"Grid {Nx}x{Ny}, Np={Np:,}")
    t0 = time.perf_counter()
    run_mc_cartesian(
        Np, init_x, init_y, init_dx, init_dy, init_w, init_cell,
        neighbors, sigma_a_mc, sigma_s_mc,
        w_cut, w_survive, max_crossings,
        Nx, Ny, dx, dy, rng_states,
        flux_tally, flux_tally_sq,
        J_tally, J_tally_sq,
        P_tally, P_tally_sq,
    )
    print(f"MC done in {time.perf_counter()-t0:.1f}s")

    norm = cell_areas * Np
    phi_mc = flux_tally / norm
    Jx_mc  = J_tally[:, 0] / norm
    Jy_mc  = J_tally[:, 1] / norm
    Pxx_mc = P_tally[:, 0] / norm
    Pyy_mc = P_tally[:, 2] / norm

    # ================================================================
    # 2) M1-DG run
    # ================================================================
    print()
    print("=" * 60)
    print("  M1-DG  (moments_solve)")
    print("=" * 60)

    grid = Grid(x0=0, x1=Lx, y0=0, y1=Ly, nx=Nx, ny=Ny)

    def Q(x, y):
        return 1.0 if (3 < x < 4) and (3 < y < 4) else 0.0

    def sa(x, y):
        s = np.floor(x) * 10 + np.floor(y)
        return 10.0 if s in [11,13,15,22,24,31,42,44,51,53,55] else 0.0

    def ss(x, y):
        s = np.floor(x) * 10 + np.floor(y)
        return 0.0 if s in [11,13,15,22,24,31,42,44,51,53,55] else 1.0

    t0 = time.perf_counter()
    sol = solve_m1_dg(
        grid=grid, px=0, py=0,
        Q_cell=Q, sigma_a_cell=sa, sigma_s_cell=ss,
        D_func=D_from_phiJ_levermore,
        use_face_speed=True, amin_face=1e-3,
        enforce_realizability=True, phi_floor=1e-12, realiz_delta=1e-10,
        stabilize_D_tensor=False,
        use_modal_filter=True, modal_alpha=18.0, modal_s=8,
        max_picard=200, relax=1.0, adaptive_relax=True, tol=1e-8,
        bc_type="vacuum_marshak", marshak_beta=0.25,
        verbose=False,
    )
    print(f"M1 done in {time.perf_counter()-t0:.1f}s")

    phi_m1 = sol.phi[:, 0]
    Jx_m1  = sol.Jx[:, 0]
    Jy_m1  = sol.Jy[:, 0]

    # ================================================================
    # 3) Compare
    # ================================================================
    print()
    print("=" * 60)
    print("  COMPARISON")
    print("=" * 60)

    hx, hy = Lx / Nx, Ly / Ny

    # Source-cell index (center of [3,4]x[3,4])
    ix_src = int(3.5 / hx);  iy_src = int(3.5 / hy)
    c_src = iy_src * Nx + ix_src

    # A cell in the scattering region away from source: (1.5, 3.5)
    ix_scat = int(1.5 / hx);  iy_scat = int(3.5 / hy)
    c_scat = iy_scat * Nx + ix_scat

    # A cell near absorber: (1.5, 1.5) -> region 11 (absorber)
    ix_abs = int(1.5 / hx);  iy_abs = int(1.5 / hy)
    c_abs = iy_abs * Nx + ix_abs

    print(f"\n{'Location':<25} {'MC φ':>12} {'M1 φ':>12} {'MC/M1':>8}")
    print("-" * 60)
    for label, c in [("Source center (3.5,3.5)", c_src),
                     ("Scatter (1.5,3.5)", c_scat),
                     ("Absorber (1.5,1.5)", c_abs)]:
        ratio = phi_mc[c] / phi_m1[c] if phi_m1[c] > 1e-14 else float('nan')
        print(f"{label:<25} {phi_mc[c]:12.4e} {phi_m1[c]:12.4e} {ratio:8.3f}")

    print(f"\n{'Statistic':<25} {'MC':>12} {'M1':>12} {'MC/M1':>8}")
    print("-" * 60)
    ratio_max = phi_mc.max() / phi_m1.max()
    print(f"{'max(φ)':<25} {phi_mc.max():12.4e} {phi_m1.max():12.4e} {ratio_max:8.3f}")

    # Global balance: total absorbed + leaked = total source
    sa_arr = np.array([sa(cell_centers[c, 0], cell_centers[c, 1]) for c in range(Nc)])
    cell_a = hx * hy

    abs_mc = np.sum(sa_arr * phi_mc * cell_a)
    abs_m1 = np.sum(sa_arr * phi_m1 * cell_a)
    src    = np.sum(np.array([Q(cell_centers[c,0], cell_centers[c,1]) for c in range(Nc)]) * cell_a)
    print(f"{'∫σ_a φ dA (absorbed)':<25} {abs_mc:12.4e} {abs_m1:12.4e} {abs_mc/abs_m1:8.3f}")
    print(f"{'∫Q dA (source)':<25} {src:12.4e}")
    print(f"{'absorbed/source':<25} {abs_mc/src:12.4f} {abs_m1/src:12.4f}")

    # Eddington ratio at source center: P_xx / phi  (should be ~1/3 if isotropic)
    D_mc = Pxx_mc[c_src] / phi_mc[c_src] if phi_mc[c_src] > 1e-14 else float('nan')
    D_mc_yy = Pyy_mc[c_src] / phi_mc[c_src] if phi_mc[c_src] > 1e-14 else float('nan')
    print(f"\nMC Eddington tensor at source center:")
    print(f"  Dxx = Pxx/φ = {D_mc:.4f}   (1/3 = {1/3:.4f})")
    print(f"  Dyy = Pyy/φ = {D_mc_yy:.4f}")
    print(f"  Dxx + Dyy   = {D_mc + D_mc_yy:.4f}   (2/3 = {2/3:.4f})")
    # Note: in 3D, tr(D) = (Pxx+Pyy+Pzz)/phi = 1.
    # We only see xx+yy; the missing Pzz/phi = 1 - Dxx - Dyy.
    Dzz = 1.0 - D_mc - D_mc_yy
    print(f"  Dzz (inferred) = 1 - Dxx - Dyy = {Dzz:.4f}   (1/3 = {1/3:.4f})")

    # Line-out along y=3.5 (middle row through source)
    iy_mid = int(3.5 / hy)
    phi_mc_line  = np.array([phi_mc[iy_mid * Nx + ix] for ix in range(Nx)])
    phi_m1_line  = np.array([phi_m1[iy_mid * Nx + ix] for ix in range(Nx)])
    x_line = np.array([(ix + 0.5) * hx for ix in range(Nx)])

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)
    extent = [0, Lx, 0, Ly]

    # Heatmaps
    import matplotlib.colors as mcolors
    for ax, data, title in [(axes[0, 0], phi_mc.reshape(Ny, Nx), "MC φ"),
                            (axes[0, 1], phi_m1.reshape(Ny, Nx), "M1 φ")]:
        d = np.maximum(data, 1e-8)
        im = ax.imshow(d, origin="lower", extent=extent, aspect="equal",
                       norm=mcolors.LogNorm(vmin=1e-6, vmax=max(phi_mc.max(), phi_m1.max())),
                       interpolation="nearest", cmap="inferno")
        ax.set_title(title)
        ax.set_xlabel("x"); ax.set_ylabel("y")
        fig.colorbar(im, ax=ax)

    # Line-out comparison (linear)
    axes[1, 0].plot(x_line, phi_mc_line, "b-", label="MC", linewidth=1.5)
    axes[1, 0].plot(x_line, phi_m1_line, "r--", label="M1", linewidth=1.5)
    axes[1, 0].set_xlabel("x"); axes[1, 0].set_ylabel("φ")
    axes[1, 0].set_title("φ along y = 3.5")
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # Line-out comparison (log)
    axes[1, 1].semilogy(x_line, np.maximum(phi_mc_line, 1e-12), "b-", label="MC", linewidth=1.5)
    axes[1, 1].semilogy(x_line, np.maximum(phi_m1_line, 1e-12), "r--", label="M1", linewidth=1.5)
    axes[1, 1].set_xlabel("x"); axes[1, 1].set_ylabel("φ (log)")
    axes[1, 1].set_title("φ along y = 3.5 (log)")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f"MC vs M1-DG comparison — lattice {Nx}×{Ny}, MC Np={Np:,}", fontsize=14)
    plt.show()


if __name__ == "__main__":
    main()
