"""
plot_lattice.py
---------------
Load a 2D scalar-field file (tab-separated floats, one row per line) defined
on an [x0,x1] x [y0,y1] domain, print min/max statistics, and produce
heatmaps (linear and log scale).

Usage
-----
    python plot_lattice.py                          # defaults below
    python plot_lattice.py --file my_data.txt
    python plot_lattice.py --file my_data.txt --x0 0 --x1 7 --y0 0 --y1 7
    python plot_lattice.py --file my_data.txt --log --cmap plasma --save out.png

All arguments are optional; sensible defaults are hard-coded for the lattice
problem (7x7 domain, 448x448 grid, filename 'lattice.txt').
"""

import argparse
import pathlib
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ── helpers ──────────────────────────────────────────────────────────────────

def load_grid(path: pathlib.Path) -> np.ndarray:
    """
    Load a whitespace/tab-separated text file into a 2-D float array.
    Row 0 in the file → row 0 of the returned array (index [0, :]).
    The caller is responsible for orienting the image correctly.
    """
    data = np.loadtxt(path, dtype=np.float64)
    if data.ndim == 1:
        # single-row file — treat as a 1-D field (unlikely but safe)
        data = data[np.newaxis, :]
    return data


def print_stats(data: np.ndarray, label: str = "field") -> None:
    finite = data[np.isfinite(data)]
    print(f"\n{'─'*50}")
    print(f"  Field : {label}")
    print(f"  Shape : {data.shape}  (rows × cols = ny × nx)")
    print(f"  Min   : {finite.min():.6e}")
    print(f"  Max   : {finite.max():.6e}")
    print(f"  Mean  : {finite.mean():.6e}")
    print(f"  Std   : {finite.std():.6e}")
    n_neg   = int((finite < 0).sum())
    n_zero  = int((finite == 0).sum())
    n_pos   = int((finite > 0).sum())
    print(f"  #neg / #zero / #pos : {n_neg} / {n_zero} / {n_pos}")
    print(f"{'─'*50}\n")


def make_heatmaps(
    data: np.ndarray,
    extent: list,          # [x0, x1, y0, y1]
    label: str,
    cmap: str,
    floor: float,
    save: str | None,
) -> None:
    """
    Plot two side-by-side heatmaps: linear scale (left) and log scale (right).
    The log panel clips values below `floor` to avoid log(0).
    """
    ny, nx = data.shape
    x0, x1, y0, y1 = extent

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    fig.suptitle(f"{label}  —  {ny}×{nx} grid  [{x0},{x1}]×[{y0},{y1}]",
                 fontsize=13)

    # ── linear ──────────────────────────────────────────────────────────────
    im0 = axes[0].imshow(
        data,
        origin="lower",
        extent=extent,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
    )
    axes[0].set_title("Linear scale")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(im0, ax=axes[0], label=label)

    # ── log ─────────────────────────────────────────────────────────────────
    data_log = np.where(np.isfinite(data) & (data > 0), data, np.nan)
    vmin_log = max(data_log[np.isfinite(data_log)].min(), floor)
    vmax_log = data_log[np.isfinite(data_log)].max()

    im1 = axes[1].imshow(
        np.clip(data_log, vmin_log, None),
        origin="lower",
        extent=extent,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=mcolors.LogNorm(vmin=vmin_log, vmax=vmax_log),
    )
    axes[1].set_title(f"Log scale  (floor={floor:.0e})")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(im1, ax=axes[1], label=f"log {label}")

    if save:
        fig.savefig(save, dpi=150)
        print(f"Figure saved to: {save}")

    plt.show()


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Heatmap + min/max stats for a 2-D tab-separated data file."
    )
    p.add_argument("--file",  default="lattice.txt",
                   help="Path to the data file  (default: lattice.txt)")
    p.add_argument("--x0",   type=float, default=0.0,  help="x domain start (default 0)")
    p.add_argument("--x1",   type=float, default=7.0,  help="x domain end   (default 7)")
    p.add_argument("--y0",   type=float, default=0.0,  help="y domain start (default 0)")
    p.add_argument("--y1",   type=float, default=7.0,  help="y domain end   (default 7)")
    p.add_argument("--label", default="φ",
                   help="Field label used in titles and colorbars (default: φ)")
    p.add_argument("--cmap",  default="inferno",
                   help="Matplotlib colormap name  (default: inferno)")
    p.add_argument("--floor", type=float, default=1e-12,
                   help="Log-scale floor value     (default: 1e-12)")
    p.add_argument("--save",  default=None,
                   help="If given, save the figure to this path (e.g. out.png)")
    return p.parse_args()


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    path = pathlib.Path(args.file)
    if not path.exists():
        sys.exit(f"ERROR: file not found: {path}")

    print(f"Loading {path} …")
    data = load_grid(path)

    # The file has row 0 = y=0 (bottom) by convention.
    # np.loadtxt preserves file order; imshow with origin='lower' maps
    # array row 0 to the bottom of the plot — consistent.

    print_stats(data, label=args.label)

    extent = [args.x0, args.x1, args.y0, args.y1]
    make_heatmaps(
        data,
        extent=extent,
        label=args.label,
        cmap=args.cmap,
        floor=args.floor,
        save=args.save,
    )


if __name__ == "__main__":
    main()
