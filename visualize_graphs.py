"""
Visualize a subset of ligand graphs from a built PyG dataset (data.pt).

For each selected graph, this script:
- draws the molecular graph using node positions from (x, y, z) features
- styles bonds according to bond type scalar:
  - 1.0  -> single bond   (one solid line)
  - 2.0  -> double bond   (two parallel solid lines)
  - 3.0  -> triple bond   (three parallel solid lines)
  - 1.5  -> aromatic bond (one dashed line)
- saves the image as <PDB_ID>.png under report/input/graphs
- writes a small HTML report <PDB_ID>.html in the same folder containing:
  - PDB ID
  - category label (class index)
  - optional pIC50 value (if present)
  - node table: index, atomic_number, x, y, z
  - edge table (unique undirected bonds): src, dst, bond_scalar, bond_type

Usage (examples):
    uv run python visualize_graphs.py
    uv run python visualize_graphs.py --num_graphs 16
    uv run python visualize_graphs.py --pdb_ids 5R83 6LU7
    uv run python visualize_graphs.py --indices 0 1 2 3
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import DEFAULT_DATA_ROOT, DEFAULT_PYG_DATASET_NAME
from dataset import MProV3Dataset, load_dataset_pdb_order


@dataclass(frozen=True)
class BondVisual:
    """Visual style for a bond."""

    n_lines: int           # 1, 2, or 3
    linestyle: str         # "-" or "--"
    label: str             # human-readable type


def bond_scalar_to_visual(value: float) -> BondVisual:
    """Map stored scalar bond type to a visual representation."""
    # Values are produced by dataset._bond_type_to_scalar:
    # Single=1.0, Double=2.0, Triple=3.0, Aromatic=1.5 (default 1.0).
    if np.isclose(value, 1.0):
        return BondVisual(n_lines=1, linestyle="-", label="single")
    if np.isclose(value, 2.0):
        return BondVisual(n_lines=2, linestyle="-", label="double")
    if np.isclose(value, 3.0):
        return BondVisual(n_lines=3, linestyle="-", label="triple")
    if np.isclose(value, 1.5):
        return BondVisual(n_lines=1, linestyle="--", label="aromatic")
    # Fallback: treat as single bond.
    return BondVisual(n_lines=1, linestyle="-", label=f"unknown({value:.2f})")


def _normalize_positions(pos2d: np.ndarray) -> np.ndarray:
    """
    Normalize 2D positions to roughly [0, 1] range for stable plotting.

    pos2d: (N, 2) array.
    """
    if pos2d.size == 0:
        return pos2d
    min_vals = pos2d.min(axis=0, keepdims=True)
    max_vals = pos2d.max(axis=0, keepdims=True)
    span = np.clip(max_vals - min_vals, 1e-6, None)
    return (pos2d - min_vals) / span


def _unique_undirected_edges(
    edge_index: torch.Tensor, edge_attr: torch.Tensor
) -> List[Tuple[int, int, float]]:
    """
    Collapse bidirectional edges into a unique undirected list.

    Returns a list of (u, v, bond_scalar) with u < v.
    """
    if edge_index.numel() == 0:
        return []
    ei = edge_index.cpu().numpy()
    ea = edge_attr.view(-1).cpu().numpy()
    seen = {}
    for col in range(ei.shape[1]):
        u = int(ei[0, col])
        v = int(ei[1, col])
        if u == v:
            continue
        key = (u, v) if u < v else (v, u)
        if key not in seen:
            seen[key] = float(ea[col])
    return [(u, v, s) for (u, v), s in seen.items()]


def draw_graph(
    pos3d: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    atomic_numbers: torch.Tensor,
    title: str,
    out_path: Path,
) -> None:
    """
    Draw a single molecular graph and save as a PNG image.

    - pos3d: (N, 3) node positions (x, y, z)
    - edge_index: (2, E) edges (directed)
    - edge_attr: (E, 1) bond scalars
    - atomic_numbers: (N,) atomic numbers used for node labels
    """
    pos3d_np = pos3d.detach().cpu().numpy()
    # Use (x, y) coordinates and normalize for visualization.
    pos2d = pos3d_np[:, :2]
    pos2d = _normalize_positions(pos2d)

    unique_edges = _unique_undirected_edges(edge_index, edge_attr)

    fig, ax = plt.subplots(figsize=(4, 4))

    # Draw bonds with appropriate styles.
    for u, v, scalar in unique_edges:
        x1, y1 = pos2d[u]
        x2, y2 = pos2d[v]
        visual = bond_scalar_to_visual(scalar)

        # Direction vector and perpendicular for multi-line bonds.
        dx, dy = x2 - x1, y2 - y1
        length = np.hypot(dx, dy)
        if length < 1e-6:
            # Degenerate; draw a point-like edge.
            ax.plot([x1], [y1], color="black", linestyle=visual.linestyle, linewidth=1.0)
            continue
        # Perpendicular unit vector for offset.
        px, py = -dy / length, dx / length
        offset = 0.02  # small offset for double/triple bonds

        if visual.n_lines == 1:
            ax.plot(
                [x1, x2],
                [y1, y2],
                color="black",
                linestyle=visual.linestyle,
                linewidth=1.5,
            )
        elif visual.n_lines == 2:
            for sign in (-1, 1):
                ox = px * offset * sign
                oy = py * offset * sign
                ax.plot(
                    [x1 + ox, x2 + ox],
                    [y1 + oy, y2 + oy],
                    color="black",
                    linestyle=visual.linestyle,
                    linewidth=1.2,
                )
        else:  # triple bond
            # Center line plus two offset lines.
            ax.plot(
                [x1, x2],
                [y1, y2],
                color="black",
                linestyle=visual.linestyle,
                linewidth=1.4,
            )
            for sign in (-1, 1):
                ox = px * offset * sign
                oy = py * offset * sign
                ax.plot(
                    [x1 + ox, x2 + ox],
                    [y1 + oy, y2 + oy],
                    color="black",
                    linestyle=visual.linestyle,
                    linewidth=1.1,
                )

    # Draw atoms as points with atomic number labels.
    xs, ys = pos2d[:, 0], pos2d[:, 1]
    ax.scatter(xs, ys, s=80, c="white", edgecolors="black", zorder=3)
    for idx, (x, y, zn) in enumerate(zip(xs, ys, atomic_numbers.tolist())):
        ax.text(
            x,
            y,
            str(int(zn)),
            ha="center",
            va="center",
            fontsize=8,
            zorder=4,
        )

    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def write_html_report(
    out_dir: Path,
    image_filename: str,
    pdb_id: str,
    category: Optional[int],
    pIC50: Optional[float],
    pos3d: torch.Tensor,
    atomic_numbers: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
) -> None:
    """
    Write an HTML report for a single graph including:
    - header with PDB ID, category, optional pIC50
    - embedded image
    - node and edge tables
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    html_path = out_dir / f"{pdb_id}.html"

    pos_np = pos3d.detach().cpu().numpy()
    atomic_np = atomic_numbers.detach().cpu().numpy()
    unique_edges = _unique_undirected_edges(edge_index, edge_attr)

    lines: List[str] = []
    lines.append("<!DOCTYPE html>")
    lines.append("<html lang='en'>")
    lines.append("<head>")
    lines.append("<meta charset='utf-8' />")
    lines.append(f"<title>MPro ligand graph: {_html_escape(pdb_id)}</title>")
    lines.append(
        "<style>"
        "body { font-family: sans-serif; } "
        "table { border-collapse: collapse; margin-bottom: 1.5em; } "
        "th, td { border: 1px solid #ccc; padding: 4px 8px; font-size: 12px; } "
        "th { background: #f0f0f0; }"
        "</style>"
    )
    lines.append("</head>")
    lines.append("<body>")
    lines.append(f"<h1>MPro ligand graph: {_html_escape(pdb_id)}</h1>")

    lines.append("<p>")
    lines.append(f"<strong>PDB ID</strong>: {_html_escape(pdb_id)}<br/>")
    if category is not None:
        lines.append(f"<strong>Category (class index)</strong>: {category}<br/>")
    if pIC50 is not None:
        lines.append(f"<strong>pIC50</strong>: {pIC50:.3f}<br/>")
    lines.append("</p>")

    lines.append(f"<p><img src='{_html_escape(image_filename)}' alt='Graph {pdb_id}'/></p>")

    # Node table.
    lines.append("<h2>Nodes (atoms)</h2>")
    lines.append("<table>")
    lines.append("<tr><th>Index</th><th>Atomic number</th><th>x</th><th>y</th><th>z</th></tr>")
    for idx in range(pos_np.shape[0]):
        x, y, z = pos_np[idx]
        zn = int(atomic_np[idx])
        lines.append(
            "<tr>"
            f"<td>{idx}</td>"
            f"<td>{zn}</td>"
            f"<td>{x:.4f}</td>"
            f"<td>{y:.4f}</td>"
            f"<td>{z:.4f}</td>"
            "</tr>"
        )
    lines.append("</table>")

    # Edge table.
    lines.append("<h2>Edges (bonds)</h2>")
    lines.append("<table>")
    lines.append("<tr><th>Source</th><th>Target</th><th>bond_scalar</th><th>bond_type</th></tr>")
    for u, v, scalar in unique_edges:
        visual = bond_scalar_to_visual(scalar)
        lines.append(
            "<tr>"
            f"<td>{u}</td>"
            f"<td>{v}</td>"
            f"<td>{scalar:.2f}</td>"
            f"<td>{_html_escape(visual.label)}</td>"
            "</tr>"
        )
    lines.append("</table>")

    lines.append("</body></html>")
    html_path.write_text("\n".join(lines), encoding="utf-8")


def _select_indices_from_args(
    ds: MProV3Dataset,
    pdb_order: Optional[Sequence[str]],
    num_graphs: int,
    indices: Optional[Sequence[int]],
    pdb_ids: Optional[Sequence[str]],
) -> List[int]:
    """Resolve which dataset indices to visualize based on CLI arguments."""
    n = len(ds)
    if indices:
        return [i for i in indices if 0 <= i < n]

    if pdb_ids:
        if pdb_order is None:
            raise ValueError(
                "pdb_ids were provided but pdb_order.txt is missing. "
                "Please ensure the dataset was built with build_dataset.py."
            )
        mapping = {p: idx for idx, p in enumerate(pdb_order)}
        result: List[int] = []
        for p in pdb_ids:
            if p in mapping:
                result.append(mapping[p])
        return result

    # Default: first num_graphs graphs.
    return list(range(min(num_graphs, n)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Draw a subset of ligand graphs from a built PyG dataset (data.pt) and "
            "save images plus HTML reports to report/input/graphs."
        )
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help=f"Path to MPro-URV_Version3_snapshot (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=DEFAULT_PYG_DATASET_NAME,
        help=f"Name of the PyG dataset folder under data_root (default: {DEFAULT_PYG_DATASET_NAME})",
    )
    parser.add_argument(
        "--num_graphs",
        type=int,
        default=16,
        help="Number of graphs to visualize if --indices/--pdb_ids are not provided (default: 16).",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=None,
        help="Explicit dataset indices to visualize (overrides --num_graphs).",
    )
    parser.add_argument(
        "--pdb_ids",
        type=str,
        nargs="+",
        default=None,
        help="PDB IDs to visualize (requires pdb_order.txt built by build_dataset.py).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root or DEFAULT_DATA_ROOT)
    dataset_name = args.dataset_name

    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {data_root}")

    ds = MProV3Dataset(root=str(data_root), dataset_name=dataset_name)
    pdb_order = load_dataset_pdb_order(data_root, dataset_name)

    selected_indices = _select_indices_from_args(
        ds=ds,
        pdb_order=pdb_order,
        num_graphs=args.num_graphs,
        indices=args.indices,
        pdb_ids=args.pdb_ids,
    )

    project_root = Path(__file__).resolve().parent
    output_dir = project_root / "report" / "input" / "graphs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loaded dataset with {len(ds)} graphs from {data_root / dataset_name / 'data.pt'}")
    print(f"Writing {len(selected_indices)} graphs to {output_dir}")

    for idx in selected_indices:
        g = ds[idx]
        # x: [x, y, z, atomic_number]
        x = g.x
        pos3d = x[:, :3]
        atomic_numbers = x[:, 3].round().to(torch.long)
        edge_index = g.edge_index
        edge_attr = g.edge_attr

        pdb_id = getattr(g, "pdb_id", f"idx_{idx}")
        pdb_id_str = str(pdb_id)
        category = None
        if hasattr(g, "category"):
            try:
                category = int(g.category.view(-1)[0].item())
            except Exception:
                category = None
        pIC50 = None
        if hasattr(g, "pIC50"):
            try:
                pIC50 = float(g.pIC50.view(-1)[0].item())
            except Exception:
                pIC50 = None

        img_filename = f"{pdb_id_str}.png"
        img_path = output_dir / img_filename

        draw_graph(
            pos3d=pos3d,
            edge_index=edge_index,
            edge_attr=edge_attr,
            atomic_numbers=atomic_numbers,
            title=pdb_id_str,
            out_path=img_path,
        )

        write_html_report(
            out_dir=output_dir,
            image_filename=img_filename,
            pdb_id=pdb_id_str,
            category=category,
            pIC50=pIC50,
            pos3d=pos3d,
            atomic_numbers=atomic_numbers,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )

    print("Done.")


if __name__ == "__main__":
    main()

