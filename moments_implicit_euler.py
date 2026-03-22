import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple, Optional
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve

# ============================================================
# 1) 1D Legendre analytic matrices (NO quadrature)
# ============================================================

def legendre_mass_1d(p: int) -> np.ndarray:
    """
    M_ij = ∫_{-1}^1 P_i(x) P_j(x) dx = 2/(2i+1) δ_ij
    """
    M = np.zeros((p + 1, p + 1))
    for i in range(p + 1):
        M[i, i] = 2.0 / (2 * i + 1)
    return M

def legendre_dmass_1d(p: int) -> np.ndarray:
    """
    S_ij = ∫_{-1}^1 P_i'(x) P_j(x) dx

    Analytic identity:
      S_ij = 2  if i > j and (i + j) is odd
           = 0  otherwise
    """
    S = np.zeros((p + 1, p + 1))
    for i in range(p + 1):
        for j in range(p + 1):
            if i > j and ((i + j) % 2 == 1):
                S[i, j] = 2.0
    return S

def legendre_endpoint_values(p: int, sign: int) -> np.ndarray:
    """
    e_i = P_i(±1). For Legendre:
      P_i(1) = 1
      P_i(-1) = (-1)^i
    sign = +1 -> x=+1
    sign = -1 -> x=-1
    """
    if sign not in (-1, +1):
        raise ValueError("sign must be ±1")
    if sign == +1:
        return np.ones(p + 1, dtype=float)
    return np.array([(-1.0) ** i for i in range(p + 1)], dtype=float)

# ============================================================
# 2) Element matrices on a physical rectangle, tensor-product basis
#    FIX: faces now use trace (restriction) + lift operators (no "e e^T" bug)
# ============================================================

@dataclass
class ElementMatrices:
    # volume
    M: np.ndarray     # Nloc x Nloc
    Gx: np.ndarray    # Nloc x Nloc  (∫ (∂x test) * trial )
    Gy: np.ndarray    # Nloc x Nloc
    # 1D mass in transverse directions (for exact face integration)
    Mx: np.ndarray    # (px+1)x(px+1)
    My: np.ndarray    # (py+1)x(py+1)
    # trace operators:
    # vertical faces: map vol coeffs (Nloc) -> y-face coeffs (py+1)
    Rx_minus: np.ndarray  # (py+1) x Nloc
    Rx_plus:  np.ndarray  # (py+1) x Nloc
    # horizontal faces: map vol coeffs (Nloc) -> x-face coeffs (px+1)
    Ry_minus: np.ndarray  # (px+1) x Nloc
    Ry_plus:  np.ndarray  # (px+1) x Nloc
    # lifted (test) operators:
    # map face coeffs -> volume test residual (Nloc)
    Lx_minus: np.ndarray  # Nloc x (py+1)
    Lx_plus:  np.ndarray  # Nloc x (py+1)
    Ly_minus: np.ndarray  # Nloc x (px+1)
    Ly_plus:  np.ndarray  # Nloc x (px+1)

def build_element_matrices(px: int, py: int, hx: float, hy: float) -> ElementMatrices:
    """
    Basis: φ_{i,j}(ξ,η) = P_i(ξ) P_j(η), i=0..px, j=0..py on reference [-1,1]^2
    Map: x = x_c + (hx/2) ξ, y = y_c + (hy/2) η

    Volume (exact):
      M  = (hx*hy/4) (Mx ⊗ My)
      Gx = (hy/2)    (Sx ⊗ My)   for ∫ (∂x v) u dA
      Gy = (hx/2)    (Mx ⊗ Sy)

    Faces (exact, modal, no quadrature):
      For vertical face ξ=±1, restriction:
         u_face_y(j) = Σ_i u_{i,j} P_i(±1)
      So Rx± is (py+1)xNloc.
      Face integration uses My (in η) with weight hy/2.

      Lift operator for test:
         ∫_{face} v(±1,η) f(η) ds = (hy/2) (Rx± v)^T My f
      Thus Lx± = Rx±^T * (hy/2) * My  maps face coeffs to volume residual.

      Analogous for horizontal faces with Mx and hx/2.
    """
    Mx = legendre_mass_1d(px)
    My = legendre_mass_1d(py)
    Sx = legendre_dmass_1d(px)
    Sy = legendre_dmass_1d(py)

    Nloc = (px + 1) * (py + 1)

    # ordering consistent with kron(Mx,My): idx(i,j) = i*(py+1) + j (y fastest)
    def idx(i: int, j: int) -> int:
        return i * (py + 1) + j

    exm = legendre_endpoint_values(px, -1)
    exp = legendre_endpoint_values(px, +1)
    eym = legendre_endpoint_values(py, -1)
    eyp = legendre_endpoint_values(py, +1)

    # Restriction matrices
    Rx_minus = np.zeros((py + 1, Nloc), dtype=float)
    Rx_plus  = np.zeros((py + 1, Nloc), dtype=float)
    for j in range(py + 1):
        for i in range(px + 1):
            Rx_minus[j, idx(i, j)] = exm[i]
            Rx_plus[j,  idx(i, j)] = exp[i]

    Ry_minus = np.zeros((px + 1, Nloc), dtype=float)
    Ry_plus  = np.zeros((px + 1, Nloc), dtype=float)
    for i in range(px + 1):
        for j in range(py + 1):
            Ry_minus[i, idx(i, j)] = eym[j]
            Ry_plus[i,  idx(i, j)] = eyp[j]

    # Volume matrices (exact)
    M  = (hx * hy / 4.0) * np.kron(Mx, My)
    Gx = (hy / 2.0)      * np.kron(Sx, My)
    Gy = (hx / 2.0)      * np.kron(Mx, Sy)

    # Lift operators (exact face integration)
    Wx = (hy / 2.0) * My   # vertical faces integrate over η
    Wy = (hx / 2.0) * Mx   # horizontal faces integrate over ξ

    Lx_minus = Rx_minus.T @ Wx
    Lx_plus  = Rx_plus.T  @ Wx
    Ly_minus = Ry_minus.T @ Wy
    Ly_plus  = Ry_plus.T  @ Wy

    return ElementMatrices(
        M=M, Gx=Gx, Gy=Gy,
        Mx=Mx, My=My,
        Rx_minus=Rx_minus, Rx_plus=Rx_plus,
        Ry_minus=Ry_minus, Ry_plus=Ry_plus,
        Lx_minus=Lx_minus, Lx_plus=Lx_plus,
        Ly_minus=Ly_minus, Ly_plus=Ly_plus
    )

# ============================================================
# 3) Levermore/M1 D(φ,J) (cellwise constant)
# ============================================================
#This section should be replaced by the ML-closure
#As inputs will change, calls to these functions in the main solver should be updated accordingly. The current structure is for the Levermore/M1 closure, which computes D from the cell-average φ and J. The new ML-closure will likely take different inputs (e.g., local coefficients or features) and produce a D tensor, so the interface will need to be adapted.
def levermore_chi(f: float) -> float:
    f = float(np.clip(f, 0.0, 1.0))
    return (3.0 + 4.0 * f * f) / (5.0 + 2.0 * np.sqrt(max(0.0, 4.0 - 3.0 * f * f)))

def D_from_phiJ_levermore(phi_avg: float, J_avg: np.ndarray, eps: float = 1e-14) -> np.ndarray:
    if phi_avg <= eps:
        return (1.0/3.0) * np.eye(2)

    Jn = np.linalg.norm(J_avg)
    f = min(Jn / phi_avg, 1.0 - 1e-12)
    chi = levermore_chi(f)

    if Jn <= eps:
        return (1.0/3.0) * np.eye(2)

    fhat = J_avg / Jn
    I = np.eye(2)
    return 0.5*(1.0-chi)*I + 0.5*(3.0*chi-1.0)*np.outer(fhat, fhat)

# ============================================================
# 4) Grid + global assembly helpers
# ============================================================

@dataclass
class Grid:
    x0: float
    x1: float
    y0: float
    y1: float
    nx: int
    ny: int

    @property
    def hx(self) -> float:
        return (self.x1 - self.x0) / self.nx

    @property
    def hy(self) -> float:
        return (self.y1 - self.y0) / self.ny

    def cell_center(self, ix: int, iy: int) -> Tuple[float, float]:
        xc = self.x0 + (ix + 0.5) * self.hx
        yc = self.y0 + (iy + 0.5) * self.hy
        return xc, yc

def cell_id(ix: int, iy: int, nx: int) -> int:
    return iy * nx + ix

def dof_offset(cell: int, comp: int, Nloc: int) -> int:
    """
    comp: 0 -> phi, 1 -> Jx, 2 -> Jy
    """
    return cell * (3 * Nloc) + comp * Nloc

# ============================================================
# 5) Stabilization / limiters
# ============================================================

def face_speed(D: np.ndarray, n: np.ndarray, amin: float = 1e-3) -> float:
    n = np.asarray(n, dtype=float)
    s = float(n @ (D @ n))
    return max(np.sqrt(max(s, 0.0)), amin)

def enforce_realizable(
    phi_c: np.ndarray,
    Jx_c: np.ndarray,
    Jy_c: np.ndarray,
    phi_floor: float = 1e-12,
    delta: float = 1e-10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    phi0 = float(phi_c[0])
    J0 = np.array([float(Jx_c[0]), float(Jy_c[0])], dtype=float)

    phi0_new = max(phi0, phi_floor)
    Jn = float(np.linalg.norm(J0))
    Jmax = (1.0 - delta) * phi0_new

    if Jn > Jmax and Jn > 0.0:
        s = Jmax / Jn
        Jx_c = s * Jx_c
        Jy_c = s * Jy_c

    phi_c = phi_c.copy()
    phi_c[0] = phi0_new
    return phi_c, Jx_c, Jy_c

def stabilize_D(D: np.ndarray, lam_min: float = 1e-12, lam_max: float = 1.0) -> np.ndarray:
    Ds = 0.5 * (D + D.T)
    w, V = np.linalg.eigh(Ds)
    w = np.clip(w, lam_min, lam_max)
    return (V * w) @ V.T

def exp_modal_filter(vec: np.ndarray, px: int, py: int, alpha: float = 18.0, s: int = 8) -> np.ndarray:
    """
    Simple exponential modal filter for tensor-product Legendre modes.
    vec is length Nloc with idx(i,j)=i*(py+1)+j (y fastest).
    """
    out = vec.copy()
    for i in range(px + 1):
        for j in range(py + 1):
            if i == 0 and j == 0:
                continue
            rx = i / max(px, 1)
            ry = j / max(py, 1)
            r = max(rx, ry)
            sigma = np.exp(-alpha * (r ** s))
            out[i * (py + 1) + j] *= sigma
    return out

# ============================================================
# 6) Flux matrices
# ============================================================

def build_Bn_and_A(n: np.ndarray, Dcell: np.ndarray, a: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build 3x3 matrices:
      Bn = [[0, nx, ny],
            [(D n)_x, 0, 0],
            [(D n)_y, 0, 0]]

      Aminus = 0.5*(Bn + a I)
      Aplus  = 0.5*(Bn - a I)
    """
    n = np.asarray(n, dtype=float)
    c = Dcell @ n  # 2-vector
    Bn = np.array([[0.0, n[0], n[1]],
                   [c[0], 0.0, 0.0],
                   [c[1], 0.0, 0.0]], dtype=float)
    I3 = np.eye(3)
    Aminus = 0.5*(Bn + a * I3)
    Aplus  = 0.5*(Bn - a * I3)
    return Aminus, Aplus

# ============================================================
# 7) Single time-step DG solver (implicit Euler, lagged D / Picard)
#
#  Solves one implicit-Euler step of the time-dependent M1 system:
#
#    (1/(c dt) + sigma_a) phi  -  div J         = Q  +  phi_prev / (c dt)
#    (1/(c dt) + sigma_t) J   -  div(D phi)     =        J_prev  / (c dt)
#
#  i.e. sigma_a -> sigma_a_hat = sigma_a + 1/(c*dt)
#       sigma_t -> sigma_t_hat = sigma_t + 1/(c*dt)
#       RHS gains  M @ (prev / (c*dt))  for each component.
#
#  For a steady-state solve pass phi_prev=Jx_prev=Jy_prev=None and dt=inf
#  (equivalently inv_cdt=0), which recovers the original behaviour.
# ============================================================

@dataclass
class CoeffField:
    phi: np.ndarray  # (ncells, Nloc)
    Jx: np.ndarray
    Jy: np.ndarray

def solve_m1_dg_timestep(
    grid: Grid,
    px: int,
    py: int,
    Q_cell: Callable[[float, float], float],
    sigma_a_cell: Callable[[float, float], float],
    sigma_s_cell: Callable[[float, float], float],
    # ---- time-stepping inputs (None => steady-state) ----
    phi_prev: Optional[np.ndarray] = None,   # (ncells, Nloc) previous time step
    Jx_prev:  Optional[np.ndarray] = None,
    Jy_prev:  Optional[np.ndarray] = None,
    inv_cdt:  float = 0.0,                   # = 1/(c*dt); 0 => steady-state
    # ---- initial guess for Picard (warm start from prev step) ----
    phi_init: Optional[np.ndarray] = None,
    Jx_init:  Optional[np.ndarray] = None,
    Jy_init:  Optional[np.ndarray] = None,
    D_func: Callable[[float, np.ndarray], np.ndarray] = D_from_phiJ_levermore,
    a_rusanov: float = 1.0,                 # if use_face_speed=False
    max_picard: int = 50,
    relax: float = 1.0,
    tol: float = 1e-10,
    verbose: bool = True,
    # ---- flux / stability knobs ----
    use_face_speed: bool = True,
    amin_face: float = 1e-3,
    enforce_realizability: bool = True,
    phi_floor: float = 1e-12,
    realiz_delta: float = 1e-10,
    stabilize_D_tensor: bool = True,
    D_lam_min: float = 1e-12,
    D_lam_max: float = 1.0,
    use_modal_filter: bool = True,
    modal_alpha: float = 18.0,
    modal_s: int = 8,
    adaptive_relax: bool = True,
    adapt_max_halvings: int = 6,
    # ---- stall / limit-cycle detection ----
    stall_tol: float = 0.0,       # if >0, stop when rel_change stays below this for stall_window iters
    stall_window: int = 10,
    # ---- boundary condition ----
    bc_type: str = "vacuum_marshak",  # "vacuum_zero" or "vacuum_marshak"
    marshak_beta: float = 0.25,   # 3D Marshak: J·n = (1/4)φ at vacuum boundary
) -> CoeffField:
    nx, ny = grid.nx, grid.ny
    ncells = nx * ny
    hx, hy = grid.hx, grid.hy

    em = build_element_matrices(px, py, hx, hy)
    Nloc = (px + 1) * (py + 1)

    # face sizes
    Nfx = py + 1  # vertical face: modes along y
    Nfy = px + 1  # horizontal face: modes along x

    # Helper: cell average extraction from modal coefficients.
    def avg_from_coeff(coeff_vec: np.ndarray) -> float:
        return float(coeff_vec[0])

    # Initialize coefficients — use warm start if provided, else zero / Q/sigma_a guess
    if phi_init is not None:
        phi = phi_init.copy()
        Jx  = Jx_init.copy()
        Jy  = Jy_init.copy()
    else:
        phi = np.zeros((ncells, Nloc))
        Jx  = np.zeros((ncells, Nloc))
        Jy  = np.zeros((ncells, Nloc))
        # crude initial guess for steady-state (inv_cdt==0) only
        if inv_cdt == 0.0:
            for iy in range(ny):
                for ix in range(nx):
                    cid = cell_id(ix, iy, nx)
                    xc, yc = grid.cell_center(ix, iy)
                    sa = float(sigma_a_cell(xc, yc))
                    Qv = float(Q_cell(xc, yc))
                    phi[cid, 0] = Qv / sa if sa > 1e-14 else 0.0

    if enforce_realizability:
        for c in range(ncells):
            phi[c], Jx[c], Jy[c] = enforce_realizable(
                phi[c], Jx[c], Jy[c], phi_floor=phi_floor, delta=realiz_delta
            )

    ndof = ncells * 3 * Nloc

    # Precompute per-cell constants (cellwise homogeneous)
    # For implicit Euler: sigma_hat = sigma + 1/(c*dt) = sigma + inv_cdt
    sigma_a = np.zeros(ncells)
    sigma_t = np.zeros(ncells)
    sigma_a_hat = np.zeros(ncells)
    sigma_t_hat = np.zeros(ncells)
    Q0 = np.zeros(ncells)
    for iy in range(ny):
        for ix in range(nx):
            cid = cell_id(ix, iy, nx)
            xc, yc = grid.cell_center(ix, iy)
            sa = float(sigma_a_cell(xc, yc))
            ss = float(sigma_s_cell(xc, yc))
            sigma_a[cid] = sa
            sigma_t[cid] = sa + ss
            sigma_a_hat[cid] = sa + inv_cdt
            sigma_t_hat[cid] = sa + ss + inv_cdt
            Q0[cid] = float(Q_cell(xc, yc))

    I3 = np.eye(3)
    Inloc = np.eye(Nloc)
    Ifx = np.eye(Nfx)
    Ify = np.eye(Nfy)

    # Trace and lift operators, lifted to 3 components
    # vertical faces:
    Txm_3 = np.kron(I3, em.Rx_minus)   # (3Nfx) x (3Nloc)
    Txp_3 = np.kron(I3, em.Rx_plus)
    Lxm_3 = np.kron(I3, em.Lx_minus)   # (3Nloc) x (3Nfx)
    Lxp_3 = np.kron(I3, em.Lx_plus)
    # horizontal faces:
    Tym_3 = np.kron(I3, em.Ry_minus)   # (3Nfy) x (3Nloc)
    Typ_3 = np.kron(I3, em.Ry_plus)
    Lym_3 = np.kron(I3, em.Ly_minus)   # (3Nloc) x (3Nfy)
    Lyp_3 = np.kron(I3, em.Ly_plus)

    # Picard loop initial vector
    U_prev = np.zeros(ndof)
    for c in range(ncells):
        base = c * (3 * Nloc)
        U_prev[base:base+Nloc] = phi[c]
        U_prev[base+Nloc:base+2*Nloc] = Jx[c]
        U_prev[base+2*Nloc:base+3*Nloc] = Jy[c]

    diff_prev = np.inf
    diff_history: list[float] = []

    for it in range(max_picard):
        # Compute D per cell from previous iterate averages
        Dcell = np.zeros((ncells, 2, 2))
        for c in range(ncells):
            phi_avg = avg_from_coeff(phi[c])
            J_avg = np.array([avg_from_coeff(Jx[c]), avg_from_coeff(Jy[c])], dtype=float)
            D = D_func(phi_avg, J_avg)
            if stabilize_D_tensor:
                D = stabilize_D(D, lam_min=D_lam_min, lam_max=D_lam_max)
            Dcell[c] = D

        # Assemble global linear system A U = b
        rows: list[int] = []
        cols: list[int] = []
        data: list[float] = []
        b = np.zeros(ndof)

        def add_block(row0: int, col0: int, block: np.ndarray):
            rr, cc = np.nonzero(block)
            if rr.size == 0:
                return
            rows.extend((row0 + rr).tolist())
            cols.extend((col0 + cc).tolist())
            data.extend(block[rr, cc].tolist())

        # ----------------------------
        # Volume terms + sources
        # ----------------------------
        for iy in range(ny):
            for ix in range(nx):
                c = cell_id(ix, iy, nx)
                off = c * (3 * Nloc)

                sa = sigma_a_hat[c]
                st = sigma_t_hat[c]
                D = Dcell[c]

                M = em.M
                Gx = em.Gx
                Gy = em.Gy

                # Implicit Euler modified system:
                # phi:  sigma_a_hat M phi  - Gx Jx - Gy Jy = M Q + inv_cdt M phi_prev
                # Jx :  -(D Gx + ...) phi + sigma_t_hat M Jx = inv_cdt M Jx_prev
                # Jy :  -(D Gx + ...) phi + sigma_t_hat M Jy = inv_cdt M Jy_prev

                A_pp = sa * M
                A_pJx = -Gx
                A_pJy = -Gy

                A_Jxphi = -(D[0,0]*Gx + D[0,1]*Gy)
                A_JxJx  = st * M

                A_Jyphi = -(D[1,0]*Gx + D[1,1]*Gy)
                A_JyJy  = st * M

                # blocks into global (3Nloc x 3Nloc)
                off_phi = off + 0 * Nloc
                off_Jx  = off + 1 * Nloc
                off_Jy  = off + 2 * Nloc

                add_block(off_phi, off_phi, A_pp)
                add_block(off_phi, off_Jx,  A_pJx)
                add_block(off_phi, off_Jy,  A_pJy)

                add_block(off_Jx,  off_phi, A_Jxphi)
                add_block(off_Jx,  off_Jx,  A_JxJx)

                add_block(off_Jy,  off_phi, A_Jyphi)
                add_block(off_Jy,  off_Jy,  A_JyJy)

                # RHS: steady source + implicit-Euler inertia terms
                Qcoeff = np.zeros(Nloc)
                Qcoeff[0] = Q0[c]
                b[off_phi:off_phi+Nloc] += M @ Qcoeff
                if inv_cdt > 0.0 and phi_prev is not None:
                    b[off_phi:off_phi+Nloc] += inv_cdt * (M @ phi_prev[c])
                    b[off_Jx:off_Jx+Nloc]  += inv_cdt * (M @ Jx_prev[c])
                    b[off_Jy:off_Jy+Nloc]  += inv_cdt * (M @ Jy_prev[c])

        # ----------------------------
        # Face contributions
        # ----------------------------
        def add_face_contrib(
            cL: int,
            faceL: str,
            n: Tuple[float, float],
            cR: Optional[int],
            faceR: Optional[str],
        ):
            """
            DG face term:
              For cell L:  ∫ v_L * (Aminus U_L^tr + Aplus U_R^tr) ds
            implemented via:
              Lift_L * (Aminus⊗I) * Trace_L  + Lift_L * (Aplus⊗I) * Trace_R

            Boundary:
              - vacuum_zero: U_ext=0
              - vacuum_marshak: phi_ext = phi_int, J_ext = beta*phi_int*n (tangent 0)
                implemented in trace-space as U_ext_tr = (T3⊗I) U_int_tr
            """
            nvec = np.array(n, dtype=float)

            # choose trace/lift for L and dimensional identity
            if faceL == 'x-':
                TL_3 = Txm_3; LL_3 = Lxm_3; Nf = Nfx; If = Ifx
            elif faceL == 'x+':
                TL_3 = Txp_3; LL_3 = Lxp_3; Nf = Nfx; If = Ifx
            elif faceL == 'y-':
                TL_3 = Tym_3; LL_3 = Lym_3; Nf = Nfy; If = Ify
            elif faceL == 'y+':
                TL_3 = Typ_3; LL_3 = Lyp_3; Nf = Nfy; If = Ify
            else:
                raise ValueError(faceL)

            DL = Dcell[cL]
            if use_face_speed:
                aL = face_speed(DL, nvec, amin=amin_face)
                if cR is not None:
                    DR = Dcell[cR]
                    aR = face_speed(DR, nvec, amin=amin_face)
                    a = max(aL, aR)
                else:
                    a = aL
            else:
                a = float(a_rusanov)

            AminusL, AplusL = build_Bn_and_A(nvec, DL, a)
            AminusL_big = np.kron(AminusL, If)
            AplusL_big  = np.kron(AplusL,  If)

            offL = cL * (3 * Nloc)

            if cR is None:
                # Boundary
                if bc_type == "vacuum_zero":
                    # LL * (Aminus ⊗ I) * TL
                    add_block(offL, offL, LL_3 @ (AminusL_big @ TL_3))
                    return

                if bc_type == "vacuum_marshak":
                    beta = float(marshak_beta)
                    # component map in trace space: [phi, Jx, Jy] -> [phi, beta*phi*n_x, beta*phi*n_y]
                    T3 = np.array([
                        [1.0, 0.0, 0.0],
                        [beta * nvec[0], 0.0, 0.0],
                        [beta * nvec[1], 0.0, 0.0],
                    ], dtype=float)
                    T3_big = np.kron(T3, If)  # (3Nf)x(3Nf)

                    # LL * [ (Aminus⊗I) + (Aplus⊗I)*(T3⊗I) ] * TL
                    add_block(offL, offL, LL_3 @ ((AminusL_big + AplusL_big @ T3_big) @ TL_3))
                    return

                raise ValueError(f"Unknown bc_type={bc_type}")

            # Interior: coupling to cR
            offR = cR * (3 * Nloc)

            # Choose trace operator for R (on its faceR)
            if faceR == 'x-':
                TR_3 = Txm_3
            elif faceR == 'x+':
                TR_3 = Txp_3
            elif faceR == 'y-':
                TR_3 = Tym_3
            elif faceR == 'y+':
                TR_3 = Typ_3
            else:
                raise ValueError(faceR)

            # Consistent interior flux: use left coefficient on UL and right coefficient on UR
            DR = Dcell[cR]
            AminusR_n, AplusR_n = build_Bn_and_A(nvec, DR, a)
            AplusR_n_big = np.kron(AplusR_n, If)

            # L block: LL*(Aminus(DL,n)*TL + Aplus(DR,n)*TR)
            add_block(offL, offL, LL_3 @ (AminusL_big @ TL_3))
            add_block(offL, offR, LL_3 @ (AplusR_n_big @ TR_3))

            # R side with outward normal -n
            nR = -nvec
            AminusR, AplusR = build_Bn_and_A(nR, DR, a)
            AminusR_big = np.kron(AminusR, If)
            # cross term for UL must use left coefficient with normal nR
            AminusL_nR, AplusL_nR = build_Bn_and_A(nR, DL, a)
            AplusL_nR_big = np.kron(AplusL_nR, If)

            # Choose lift for R face
            if faceR == 'x-':
                LR_3 = Lxm_3
            elif faceR == 'x+':
                LR_3 = Lxp_3
            elif faceR == 'y-':
                LR_3 = Lym_3
            elif faceR == 'y+':
                LR_3 = Lyp_3
            else:
                raise ValueError(faceR)

            # R block: LR*(Aminus(DR,-n)*TR + Aplus(DL,-n)*TL)
            add_block(offR, offR, LR_3 @ (AminusR_big @ TR_3))
            add_block(offR, offL, LR_3 @ (AplusL_nR_big @ TL_3))

        # Loop over all faces once (vertical + horizontal)
        for iy in range(ny):
            for ix in range(nx):
                c = cell_id(ix, iy, nx)

                # Left boundary or interior vertical face
                if ix == 0:
                    add_face_contrib(cL=c, faceL='x-', n=(-1.0, 0.0), cR=None, faceR=None)
                else:
                    cL = cell_id(ix-1, iy, nx)
                    cR = c
                    add_face_contrib(cL=cL, faceL='x+', n=(+1.0, 0.0), cR=cR, faceR='x-')

                # Right boundary
                if ix == nx - 1:
                    add_face_contrib(cL=c, faceL='x+', n=(+1.0, 0.0), cR=None, faceR=None)

                # Bottom boundary or interior horizontal face
                if iy == 0:
                    add_face_contrib(cL=c, faceL='y-', n=(0.0, -1.0), cR=None, faceR=None)
                else:
                    cB = cell_id(ix, iy-1, nx)
                    cT = c
                    add_face_contrib(cL=cB, faceL='y+', n=(0.0, +1.0), cR=cT, faceR='y-')

                # Top boundary
                if iy == ny - 1:
                    add_face_contrib(cL=c, faceL='y+', n=(0.0, +1.0), cR=None, faceR=None)

        A = coo_matrix((data, (rows, cols)), shape=(ndof, ndof)).tocsr()

        # Solve linear system (one Picard step)
        U_lin = spsolve(A, b)

        # Relaxation: optional adaptive or fixed
        if adaptive_relax:
            omega = float(relax)
            U = None
            for _ in range(adapt_max_halvings + 1):
                U_try = (1.0 - omega) * U_prev + omega * U_lin
                if np.all(np.isfinite(U_try)):
                    diff_try = np.linalg.norm(U_try - U_prev) / (np.linalg.norm(U_prev) + 1e-30)
                    if it == 0 or diff_try <= 1.2 * diff_prev:
                        U = U_try
                        break
                omega *= 0.5
            if U is None:
                U = (1.0 - 0.1) * U_prev + 0.1 * U_lin
        else:
            U = (1.0 - relax) * U_prev + relax * U_lin if relax != 1.0 else U_lin

        # update fields
        for c in range(ncells):
            base = c * (3 * Nloc)
            phi[c] = U[base:base+Nloc]
            Jx[c]  = U[base+Nloc:base+2*Nloc]
            Jy[c]  = U[base+2*Nloc:base+3*Nloc]

            if use_modal_filter and (px > 0 or py > 0):
                phi[c] = exp_modal_filter(phi[c], px, py, alpha=modal_alpha, s=modal_s)
                Jx[c]  = exp_modal_filter(Jx[c],  px, py, alpha=modal_alpha, s=modal_s)
                Jy[c]  = exp_modal_filter(Jy[c],  px, py, alpha=modal_alpha, s=modal_s)

            if enforce_realizability:
                phi[c], Jx[c], Jy[c] = enforce_realizable(
                    phi[c], Jx[c], Jy[c], phi_floor=phi_floor, delta=realiz_delta
                )

        # Repack filtered/projected fields so the Picard state, closure state, and convergence test agree
        U_post = np.zeros_like(U)
        for c in range(ncells):
            base = c * (3 * Nloc)
            U_post[base:base+Nloc] = phi[c]
            U_post[base+Nloc:base+2*Nloc] = Jx[c]
            U_post[base+2*Nloc:base+3*Nloc] = Jy[c]

        diff = np.linalg.norm(U_post - U_prev) / (np.linalg.norm(U_prev) + 1e-30)
        if verbose:
            extra = " (adaptive relax)" if adaptive_relax else ""
            print(f"[Picard {it+1:02d}] rel_change={diff:.3e}{extra}")

        if diff < tol:
            if verbose:
                print(f"  -> converged at iteration {it+1}")
            U_prev = U_post.copy()
            break

        # Limit-cycle / stall detection
        diff_history.append(diff)
        if stall_tol > 0.0 and len(diff_history) >= stall_window:
            window = diff_history[-stall_window:]
            if max(window) < stall_tol:
                if verbose:
                    print(f"  -> stall detected: rel_change < {stall_tol:.1e} "
                          f"for {stall_window} iters, accepting solution")
                U_prev = U_post.copy()
                break

        U_prev = U_post.copy()
        diff_prev = diff

    return CoeffField(phi=phi, Jx=Jx, Jy=Jy)


# ============================================================
# 8) Implicit-Euler time loop
# ============================================================

def solve_m1_dg_time(
    grid: Grid,
    px: int,
    py: int,
    Q_cell: Callable[[float, float], float],
    sigma_a_cell: Callable[[float, float], float],
    sigma_s_cell: Callable[[float, float], float],
    T_f: float,
    N_t: int,
    c: float = 1.0,
    D_func: Callable[[float, np.ndarray], np.ndarray] = D_from_phiJ_levermore,
    verbose_steps: bool = True,
    **kwargs,          # forwarded verbatim to solve_m1_dg_timestep
) -> list:
    """
    Advance the M1 system from t=0 (zero initial data) to t=T_f using
    N_t uniform implicit-Euler steps of size dt = T_f / N_t.

    Returns
    -------
    snapshots : list of CoeffField, length N_t + 1
        snapshots[0]   is the zero initial condition at t = 0.
        snapshots[k+1] is the solution at t = (k+1)*dt, for k = 0..N_t-1.
    """
    dt     = T_f / N_t
    inv_cdt = 1.0 / (c * dt)

    ncells = grid.nx * grid.ny
    Nloc   = (px + 1) * (py + 1)

    # Zero initial condition
    phi_prev = np.zeros((ncells, Nloc))
    Jx_prev  = np.zeros((ncells, Nloc))
    Jy_prev  = np.zeros((ncells, Nloc))

    # Store t=0 snapshot (zero initial data)
    snapshots = [CoeffField(phi=phi_prev.copy(), Jx=Jx_prev.copy(), Jy=Jy_prev.copy())]

    for step in range(N_t):
        t_new = (step + 1) * dt
        if verbose_steps:
            print(f"\n{'='*60}")
            print(f"  Time step {step+1}/{N_t}   t = {t_new:.4f}   dt = {dt:.4f}")
            print(f"{'='*60}")

        sol = solve_m1_dg_timestep(
            grid=grid,
            px=px, py=py,
            Q_cell=Q_cell,
            sigma_a_cell=sigma_a_cell,
            sigma_s_cell=sigma_s_cell,
            phi_prev=phi_prev,
            Jx_prev=Jx_prev,
            Jy_prev=Jy_prev,
            inv_cdt=inv_cdt,
            # warm-start Picard from previous time step solution
            phi_init=phi_prev.copy(),
            Jx_init=Jx_prev.copy(),
            Jy_init=Jy_prev.copy(),
            D_func=D_func,
            **kwargs,
        )

        snapshots.append(sol)

        # Advance
        phi_prev = sol.phi
        Jx_prev  = sol.Jx
        Jy_prev  = sol.Jy

    return snapshots


# ============================================================
# 9) Example usage
# ============================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    # ------------------------------------------------------------------
    # Choose geometry: "lattice", "crossing_beams", or "Hohlraum_v2"
    # ------------------------------------------------------------------
    GEOMETRY = "crossing_beams"   # <-- switch here

    if GEOMETRY == "lattice":
        grid = Grid(x0=0.0, x1=7.0, y0=0.0, y1=7.0, nx=70, ny=70)

        def Q(x, y):
            return 1.0 if (3 < x < 4) and (3 < y < 4) else 0.0

        def sigma_a(x, y):
            s = np.floor(x) * 10 + np.floor(y)
            return 10.0 if s in [11, 13, 15, 22, 24, 31, 42, 44, 51, 53, 55] else 0.0

        def sigma_s(x, y):
            s = np.floor(x) * 10 + np.floor(y)
            return 0.0 if s in [11, 13, 15, 22, 24, 31, 42, 44, 51, 53, 55] else 1.0

        c   = 1.0
        T_f = 3.5
        N_t = 10

    elif GEOMETRY == "crossing_beams":
        # Two narrow source strips on perpendicular sides of a low-absorption box.
        # Domain [0,7]x[0,7], absorbing frame (width 0.5) with sigma_a=10.
        # Beam A: vertical strip near left wall, x in [0.5,1.0], y in [2.5,4.5]
        # Beam B: horizontal strip near bottom wall, x in [2.5,4.5], y in [0.5,1.0]
        grid = Grid(x0=0.0, x1=7.0, y0=0.0, y1=7.0, nx=70, ny=70)

        def Q(x, y):
            beam_a = (0.5 < x < 1.0) and (2.5 < y < 4.5)
            beam_b = (2.5 < x < 4.5) and (0.5 < y < 1.0)
            return 1.0 if (beam_a or beam_b) else 0.0

        def sigma_a(x, y):
            if x < 0.5 or x > 6.5 or y < 0.5 or y > 6.5:
                return 10.0   # absorbing frame
            return 0.02

        def sigma_s(x, y):
            return 0.0

        c   = 1.0
        T_f = 3.5
        N_t = 10

    else:  # Hohlraum_v2: 1.5x1.5 domain, all interior features shifted +0.2x +0.1y
        grid = Grid(x0=0.0, x1=1.5, y0=0.0, y1=1.5, nx=75, ny=75)

        def Q(x, y):
            # Volumetric isotropic source strip (proxy for the original boundary source)
            return 1.0 if (0.10 < x < 0.15) and (0.10 < y < 1.40) else 0.0

        def sigma_a(x, y):
            # 4-sided absorbing boundary shell (thickness 0.05)
            if x < 0.05 or x > 1.45 or y < 0.05 or y > 1.45:
                return 100.0
            # shifted left obstruction
            if 0.20 < x < 0.25 and 0.35 < y < 1.15:
                return 5.0
            # shifted inner core
            if 0.70 < x < 1.05 and 0.40 < y < 1.10:
                return 50.0
            # shifted inner shell
            if 0.65 < x < 1.05 and 0.35 < y < 1.15:
                return 10.0
            return 1.0

        def sigma_s(x, y):
            # 4-sided absorbing boundary shell (no scattering)
            if x < 0.05 or x > 1.45 or y < 0.05 or y > 1.45:
                return 0.0
            # shifted left obstruction
            if 0.20 < x < 0.25 and 0.35 < y < 1.15:
                return 95.0
            # shifted inner core
            if 0.70 < x < 1.05 and 0.40 < y < 1.10:
                return 50.0
            # shifted inner shell
            if 0.65 < x < 1.05 and 0.35 < y < 1.15:
                return 90.0
            return 0.1

        c   = 1.0
        T_f = 2.0
        N_t = 20

    dt = T_f / N_t

    # ------------------------------------------------------------------
    # Run implicit-Euler time integration
    # ------------------------------------------------------------------
    snapshots = solve_m1_dg_time(
        grid=grid,
        px=0, py=0,           # Tune px, py
        Q_cell=Q,
        sigma_a_cell=sigma_a,
        sigma_s_cell=sigma_s,
        T_f=T_f,
        N_t=N_t,
        c=c,
        D_func=D_from_phiJ_levermore,
        verbose_steps=True,
        # --- stability knobs ---
        use_face_speed=True,
        amin_face=1e-3,
        enforce_realizability=True,
        phi_floor=1e-12,
        realiz_delta=1e-10,
        stabilize_D_tensor=False,
        # --- higher-order stabilization ---
        use_modal_filter=True,
        modal_alpha=18.0,
        modal_s=8,
        # --- Picard controls ---
        max_picard=60,         # Tune
        relax=0.3,
        adaptive_relax=True,
        adapt_max_halvings=10,
        tol=1e-5,
        stall_tol=1e-2,
        stall_window=10,
        verbose=True,
        # --- boundary condition ---
        bc_type="vacuum_marshak",
        marshak_beta=0.25,
    )

    # ------------------------------------------------------------------
    # Plot final snapshot  (t = T_f)
    # ------------------------------------------------------------------
    sol_final = snapshots[-1]
    nx, ny    = grid.nx, grid.ny

    phi_avg  = sol_final.phi[:, 0].reshape((ny, nx))
    Jmag_img = np.sqrt(sol_final.Jx[:, 0]**2 + sol_final.Jy[:, 0]**2).reshape((ny, nx))

    print(f"\nphi at t={T_f:.3f}  min/max: {phi_avg.min():.4e}  {phi_avg.max():.4e}")

    extent = [grid.x0, grid.x1, grid.y0, grid.y1]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    im0 = axes[0].imshow(np.maximum(phi_avg, 1e-10), origin="lower", extent=extent,
                         aspect="auto", norm=mcolors.LogNorm())
    axes[0].set_title(f"Cell-avg ϕ  (t = {T_f:.2f})  [{GEOMETRY}]")
    axes[0].set_xlabel("x");  axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(Jmag_img, origin="lower", extent=extent, aspect="auto")
    axes[1].set_title(f"Cell-avg |J|  (t = {T_f:.2f})  [{GEOMETRY}]")
    axes[1].set_xlabel("x");  axes[1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[1])

    plt.show()

    # ------------------------------------------------------------------
    # Optional: time series of domain-integrated phi (sanity check)
    # ------------------------------------------------------------------
    hx, hy     = grid.hx, grid.hy
    cell_area  = hx * hy
    phi_int    = [np.sum(s.phi[:, 0]) * cell_area for s in snapshots]
    t_values   = [k * dt for k in range(N_t + 1)]   # 0, dt, 2dt, ..., T_f

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_values, phi_int, marker="o", ms=4)
    ax.set_xlabel("t");  ax.set_ylabel("∫ φ dA")
    ax.set_title(f"Domain-integrated flux vs. time  [{GEOMETRY}]  (implicit Euler M1-DG)")
    ax.grid(True)
    plt.tight_layout();  plt.show()
