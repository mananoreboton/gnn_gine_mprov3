"""
Visualize a subset of ligand graphs from a built PyG dataset (data.pt).

Uses RDKit's 2D drawer (MolDraw2D) for publication-quality graphics. For each
selected graph this script:
- Uses (x, y) coordinates only (z is dropped, no projection).
- Builds an RDKit molecule from the graph and draws it with correct bond styles:
  - single: one central line
  - double: two shifted parallel lines
  - triple: two shifted lines plus one central line
  - aromatic: dashed line
- Saves PNG and SVG (vector) under report/input/graphs.
- Writes an HTML report with PDB ID, category, and node/edge tables.

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
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import BondType as RkBondType
from rdkit.Geometry import Point3D

from config import DEFAULT_DATA_ROOT, DEFAULT_PYG_DATASET_NAME
from dataset import MProV3Dataset, load_dataset_pdb_order

# Image size in pixels (RDKit drawer uses this for PNG/SVG canvas).
_DRAW_SIZE = 500


@dataclass(frozen=True)
class BondVisual:
    """Human-readable bond type for reports."""

    label: str


def bond_scalar_to_visual(value: float) -> BondVisual:
    """Map stored scalar bond type to a label (single/double/triple/aromatic)."""
    if np.isclose(value, 1.0):
        return BondVisual(label="single")
    if np.isclose(value, 2.0):
        return BondVisual(label="double")
    if np.isclose(value, 3.0):
        return BondVisual(label="triple")
    if np.isclose(value, 1.5):
        return BondVisual(label="aromatic")
    return BondVisual(label=f"unknown({value:.2f})")


def _bond_scalar_to_rdkit(value: float) -> RkBondType:
    """Map stored bond scalar to RDKit BondType for correct drawing."""
    if np.isclose(value, 2.0):
        return RkBondType.DOUBLE
    if np.isclose(value, 3.0):
        return RkBondType.TRIPLE
    if np.isclose(value, 1.5):
        return RkBondType.AROMATIC
    return RkBondType.SINGLE


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


def _mol_from_graph(
    atomic_numbers: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    pos_xy: np.ndarray,
) -> Chem.Mol:
    """
    Build an RDKit molecule from the PyG graph using (x, y) only for the 2D conformer.

    - pos_xy: (N, 2) array (x, y); z is not used.
    """
    anum = atomic_numbers.detach().cpu().numpy()
    n = len(anum)
    mol = Chem.RWMol()
    for i in range(n):
        mol.AddAtom(Chem.Atom(int(anum[i])))
    for u, v, scalar in _unique_undirected_edges(edge_index, edge_attr):
        bt = _bond_scalar_to_rdkit(scalar)
        mol.AddBond(int(u), int(v), bt)
    mol = mol.GetMol()
    if n == 0:
        return mol
    conf = Chem.Conformer(n)
    for i in range(n):
        conf.SetAtomPosition(i, Point3D(float(pos_xy[i, 0]), float(pos_xy[i, 1]), 0.0))
    mol.AddConformer(conf, assignId=True)
    return mol


def draw_graph(
    pos3d: torch.Tensor,
    edge_index: torch.Tensor,
    edge_attr: torch.Tensor,
    atomic_numbers: torch.Tensor,
    out_path_png: Path,
    out_path_svg: Optional[Path] = None,
) -> None:
    """
    Draw a single molecular graph with RDKit (MolDraw2D) and save PNG (and optionally SVG).

    Uses (x, y) only from pos3d; z is dropped. Bond drawing follows chemistry conventions:
    single = one central line, double = two shifted lines, triple = two shifted + central,
    aromatic = dashed.
    """
    pos_np = pos3d.detach().cpu().numpy()
    pos_xy = pos_np[:, :2].copy()  # (N, 2), drop z

    mol = _mol_from_graph(atomic_numbers, edge_index, edge_attr, pos_xy)
    if mol.GetNumAtoms() == 0:
        out_path_png.parent.mkdir(parents=True, exist_ok=True)
        out_path_png.write_bytes(b"")
        if out_path_svg:
            out_path_svg.write_text("<!-- empty molecule -->", encoding="utf-8")
        return

    w = h = _DRAW_SIZE
    out_path_png.parent.mkdir(parents=True, exist_ok=True)

    # Prefer MolDraw2DCairo (best quality); fall back to MolToImage if Cairo not built.
    try:
        drawer = Draw.rdMolDraw2D.MolDraw2DCairo(w, h)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        drawer.WriteDrawingText(str(out_path_png))
    except (AttributeError, OSError):
        img = Draw.MolToImage(mol, size=(w, h))
        img.save(out_path_png)

    if out_path_svg is not None:
        drawer_svg = Draw.rdMolDraw2D.MolDraw2DSVG(w, h)
        drawer_svg.DrawMolecule(mol)
        drawer_svg.FinishDrawing()
        out_path_svg.write_text(drawer_svg.GetDrawingText(), encoding="utf-8")


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
    svg_filename: Optional[str] = None,
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
    if svg_filename:
        lines.append(
            f"<p><a href='{_html_escape(svg_filename)}'>Vector (SVG)</a></p>"
        )

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
    parser.add_argument(
        "--svg",
        action="store_true",
        help="Also save vector SVG files for publication-quality figures.",
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
        svg_path = (output_dir / f"{pdb_id_str}.svg") if args.svg else None

        draw_graph(
            pos3d=pos3d,
            edge_index=edge_index,
            edge_attr=edge_attr,
            atomic_numbers=atomic_numbers,
            out_path_png=img_path,
            out_path_svg=svg_path,
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
            svg_filename=f"{pdb_id_str}.svg" if args.svg else None,
        )

    print("Done.")


if __name__ == "__main__":
    main()

