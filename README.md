# moment_solve

A 2D M1 moment radiation transport solver using the Modal Discontinuous Galerkin (DG) method on a structured rectangular mesh.

## Physics

Solves the steady-state M1 two-moment system:

```
σ_a φ + ∇·J = Q
σ_t J + D(φ,J) ∇φ = 0
```

where **D** is the nonlinear Eddington tensor from the Levermore M1 closure.

## Method

- **Modal DG** with tensor-product Legendre basis P_i(ξ)P_j(η) of degrees `(px, py)`
- **Exact** (quadrature-free) element matrices via analytic Legendre identities
- **Rusanov / local Lax-Friedrichs** numerical flux with face-local wave speed from D
- **Picard iteration** with lagged D, optional adaptive relaxation
- **Vacuum BCs**: `vacuum_zero` (U_ext=0) or `vacuum_marshak` (Marshak-like inflow condition)
- Optional **realizability enforcement**, **D-tensor stabilization**, and **exponential modal filter**

## Requirements

```
numpy
scipy
matplotlib
```

Install via:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy scipy matplotlib
```

## Usage

```bash
python moments_solve.py
```

Edit the `__main__` block to configure:
- `Grid(x0, x1, y0, y1, nx, ny)` — domain and mesh resolution
- `Q(x, y)` — source term
- `sigma_a(x, y)`, `sigma_s(x, y)` — absorption / scattering coefficients
- `px, py` — polynomial degree per cell
- Solver knobs: `max_picard`, `relax`, `tol`, `bc_type`, etc.

## Structure

| Section | Content |
|---|---|
| §1 | 1D Legendre mass / stiffness matrices (analytic) |
| §2 | 2D element matrices (volume + face trace/lift operators) |
| §3 | Levermore M1 closure `D(φ, J)` |
| §4 | `Grid` dataclass and helpers |
| §5 | Stabilization utilities (realizability, modal filter, D stabilization) |
| §6 | Rusanov flux matrices `A±` |
| §7 | Main solver `solve_m1_dg()` |
| §8 | Driver / example |

## Note on the closure

The `D_from_phiJ_levermore` function in §3 is intended to be replaced by an ML-based closure. The solver calls it as `D_func(phi_avg, J_avg)` — swap in any callable with the same signature.
