#!/usr/bin/env python3
"""
2D Material Element Substitution Screener — 启发式筛选
======================================================
读取 POSCAR，基于离子半径、电负性、氧化态兼容性
对候选替换元素进行评分排序，并可选输出替换后 POSCAR。

用法:
  python screen.py POSCAR --site 2 --candidates all
  python screen.py POSCAR --element Mo --candidates Cr,Mn,Fe,Co,Ni,W
  python screen.py POSCAR --element O  --candidates S,Se,Te --top 10
  python screen.py POSCAR --element S  --candidates all --output-dir ./candidates
  python screen.py POSCAR --element S  --candidates all --json results.json
  python screen.py POSCAR --element Mo --candidates all --exclude Tc --ox-state 4 --coord 6
"""

import argparse
import json
import os
import sys
from math import exp
from dataclasses import dataclass, asdict
from typing import List, Optional

try:
    from pymatgen.core import Element, Structure
    from pymatgen.io.vasp import Poscar
except ImportError:
    print("Need pymatgen: pip install pymatgen", file=sys.stderr)
    sys.exit(1)

# ═══════════════════════════════════════════════════════════════
#  Constants
# ═══════════════════════════════════════════════════════════════

# Elements to skip in "all" mode
EXCLUDE = {
    "He", "Ne", "Ar", "Kr", "Xe", "Rn",
    "Po", "At", "Fr", "Ra", "Ac", "Pa",
    "Np", "Pu", "Am", "Cm", "Bk", "Cf",
}

# All reasonably available elements
ALL_EL = [
    "Li", "Be", "B", "C", "N", "O", "F", "Na", "Mg", "Al", "Si", "P",
    "S", "Cl", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co",
    "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Rb", "Sr", "Y",
    "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Sm", "Eu",
    "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",
    "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi",
]

# Named groups
GROUPS = {
    "all":        ALL_EL,
    "tm":         ["Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu",
                   "Zn", "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd",
                   "Ag", "Cd", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au"],
    "main":       ["B", "C", "N", "O", "F", "Na", "Mg", "Al", "Si", "P",
                   "S", "Cl", "Ga", "Ge", "As", "Se", "Br", "In", "Sn",
                   "Sb", "Te", "I", "Tl", "Pb", "Bi"],
    "chalcogen":  ["O", "S", "Se", "Te"],
    "pnictogen":  ["N", "P", "As", "Sb", "Bi"],
    "halogen":    ["F", "Cl", "Br", "I"],
    "alkali":     ["Li", "Na", "K", "Rb", "Cs"],
    "alkaline":   ["Mg", "Ca", "Sr", "Ba"],
    "lanthanide": ["La", "Ce", "Pr", "Nd", "Sm", "Eu", "Gd",
                   "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu"],
    "early-tm":   ["Sc", "Ti", "V", "Cr", "Y", "Zr", "Nb", "Mo",
                   "Hf", "Ta", "W", "Re"],
    "late-tm":    ["Mn", "Fe", "Co", "Ni", "Cu", "Zn",
                   "Tc", "Ru", "Rh", "Pd", "Ag", "Cd",
                   "Os", "Ir", "Pt", "Au"],
}

DEFAULT_WEIGHTS = [0.40, 0.30, 0.20, 0.10]  # radius, EN, OS, struct


# ═══════════════════════════════════════════════════════════════
#  Core scoring
# ═══════════════════════════════════════════════════════════════

def _get_ionic_radius(el: Element, ox: Optional[int], cn: int) -> Optional[float]:
    """Best ionic radius (Angstrom) for element at given oxidation state."""
    radii = el.ionic_radii
    if not radii:
        return None
    # Prefer matching oxidation state
    if ox is not None and ox in radii:
        return float(radii[ox])
    # Fallback to any available
    return float(next(iter(radii.values())))


def guess_oxidation(structure: Structure, site_idx: int) -> Optional[int]:
    """Guess oxidation state for the species at site_idx."""
    sp = list(structure[site_idx].species.keys())[0]
    try:
        s = structure.copy()
        s.add_oxidation_state_by_guess()
        for p, amt in s[site_idx].species.items():
            if p.symbol == sp.symbol:
                return int(p.oxi_state)
    except Exception:
        pass
    try:
        el = Element(sp.symbol)
        if el.common_oxidation_states:
            return el.common_oxidation_states[0]
    except Exception:
        pass
    return None


def score_radius(host_el: Element, cand_el: Element,
                 ox: Optional[int], cn: int) -> float:
    """Ionic / atomic radius compatibility (1 = perfect)."""
    rh = _get_ionic_radius(host_el, ox, cn) or host_el.atomic_radius
    rc = _get_ionic_radius(cand_el, ox, cn) or cand_el.atomic_radius
    if not rh or not rc or rh <= 0:
        return 0.50
    return exp(-2.0 * abs(rc - rh) / rh)


def score_en(host_el: Element, cand_el: Element) -> float:
    """Electronegativity compatibility (1 = same EN)."""
    try:
        return exp(-abs(host_el.X - cand_el.X) / 1.2)
    except Exception:
        return 0.50


def score_os(host_el: Element, cand_el: Element,
             target_ox: Optional[int]) -> float:
    """Oxidation-state compatibility."""
    css = cand_el.common_oxidation_states
    if not css:
        return 0.30
    if target_ox is not None:
        if target_ox in css:
            return 1.0
        for s in css:
            if abs(s - target_ox) <= 1:
                return 0.8
            if abs(s - target_ox) <= 2:
                return 0.5
        return 0.20
    return min(1.0, len(css) / 4.0)


def score_struct(host_el: Element, cand_el: Element) -> float:
    """Periodic-table proximity (same group/row → similar chemistry)."""
    try:
        gd = abs(host_el.group - cand_el.group)
        rd = abs(host_el.row - cand_el.row)
    except Exception:
        return 0.50
    if gd == 0:
        return 0.90
    if rd == 0:
        return 0.70
    if rd <= 1 and gd <= 2:
        return 0.60
    return exp(-0.3 * (gd + rd))


# ═══════════════════════════════════════════════════════════════
#  Results
# ═══════════════════════════════════════════════════════════════

@dataclass
class Hit:
    symbol: str
    total: float
    radius: float
    en: float
    os: float
    struct: float
    r_host: Optional[float]
    r_cand: Optional[float]
    en_host: float
    en_cand: float
    note: str = ""

    @property
    def as_dict(self):
        return asdict(self)


def _annotate(r: float, en: float, os_: float) -> str:
    parts = []
    if r > 0.80:
        parts.append("Radius OK")
    elif r < 0.30:
        parts.append("Size mismatch!")
    if en > 0.80:
        parts.append("EN OK")
    elif en < 0.30:
        parts.append("EN mismatch!")
    if os_ > 0.80:
        parts.append("OS OK")
    elif os_ < 0.30:
        parts.append("OS incompatible!")
    return " | ".join(parts) if parts else ""


# ═══════════════════════════════════════════════════════════════
#  Screen a site
# ═══════════════════════════════════════════════════════════════

def screen(structure, site_idx, host_sym, ox, candidates, weights, cn):
    host = Element(host_sym)
    hits = []
    for sym in candidates:
        if sym == host_sym:
            continue
        try:
            cand = Element(sym)
        except Exception:
            continue
        rs = score_radius(host, cand, ox, cn)
        es = score_en(host, cand)
        os_ = score_os(host, cand, ox)
        ss = score_struct(host, cand)
        total = weights[0] * rs + weights[1] * es + weights[2] * os_ + weights[3] * ss
        rh = _get_ionic_radius(host, ox, cn) or host.atomic_radius
        rc = _get_ionic_radius(cand, ox, cn) or cand.atomic_radius
        hits.append(Hit(sym, total, rs, es, os_, ss, rh, rc, host.X, cand.X, _annotate(rs, es, os_)))
    hits.sort(key=lambda h: h.total, reverse=True)
    return hits


# ═══════════════════════════════════════════════════════════════
#  I/O helpers
# ═══════════════════════════════════════════════════════════════

def write_poscars(structure, site_indices, hits, top_n, out_dir, host_sym):
    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for i, h in enumerate(hits[:top_n]):
        s = structure.copy()
        for idx in site_indices:
            s[idx] = {Element(h.symbol): 1.0}
        fname = f"POSCAR_{host_sym}to{h.symbol}_r{i+1}_{h.total:.3f}"
        path = os.path.join(out_dir, fname)
        Poscar(s).write_file(path)
        paths.append(path)
    return paths


# ═══════════════════════════════════════════════════════════════
#  Formatting
# ═══════════════════════════════════════════════════════════════

SEP = "-" * 80
HEADER_FMT = (
    "{rank:<5} {el:<8} {total:<8} {r:<8} {en:<8} {os:<8} {struct:<8} note"
)
ROW_FMT = "{rank:<5} {el:<8} {total:<7.3f} {r:<7.3f} {en:<7.3f} {os:<7.3f} {struct:<7.3f} {note}"


def table(hits, top_n):
    lines = [HEADER_FMT.format(
        rank="Rank", el="Element", total="Total",
        r="Radius", en="EN", os="OS", struct="Struct"
    )]
    lines.append(SEP)
    for i, h in enumerate(hits[:top_n], 1):
        lines.append(ROW_FMT.format(
            rank=i, el=h.symbol, total=h.total,
            r=h.radius, en=h.en, os=h.os,
            struct=h.struct, note=h.note[:40]
        ))
    return "\n".join(lines)


def to_json(hits, meta, top_n):
    return json.dumps({
        "meta": meta,
        "hits": [{"rank": i + 1, **h.as_dict} for i, h in enumerate(hits[:top_n])]
    }, indent=2, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════
#  Candidates
# ═══════════════════════════════════════════════════════════════

def resolve_candidates(spec: str) -> List[str]:
    spec_stripped = spec.strip()
    if spec_stripped.lower() in GROUPS:
        return list(GROUPS[spec_stripped.lower()])
    # comma-separated — keep original case
    return [s.strip() for s in spec_stripped.split(",") if s.strip()]


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="2D Material Element Substitution Screener",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("poscar", help="Path to POSCAR")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--site", type=int, nargs="+", help="Sites (1-indexed) to substitute")
    g.add_argument("--element", type=str, help="Substitute every site of this element")

    ap.add_argument("--candidates", default="all",
                    help=f"Built-in sets: {', '.join(GROUPS)}  or comma list (default: all)")
    ap.add_argument("--exclude", default="", help="Comma-separated elements to exclude")
    ap.add_argument("--ox-state", type=int, help="Host oxidation state (auto if omitted)")
    ap.add_argument("--coord", type=int, default=6, help="Coordination number (default: 6)")
    ap.add_argument("--weights", type=float, nargs=4, default=DEFAULT_WEIGHTS,
                    help=f"radius EN OS struct weights (default: {' '.join(map(str, DEFAULT_WEIGHTS))})")
    ap.add_argument("--top", type=int, default=15, help="Show top N (default: 15)")
    ap.add_argument("--output-dir", help="Write candidate POSCARs to DIR")
    ap.add_argument("--json", help="Save results as JSON")
    args = ap.parse_args()

    # ── Read structure ──
    if not os.path.exists(args.poscar):
        print(f"POSCAR not found: {args.poscar}", file=sys.stderr); sys.exit(1)
    try:
        struct = Structure.from_file(args.poscar)
    except Exception as e:
        print(f"Read error: {e}", file=sys.stderr); sys.exit(1)

    # ── Site(s) ──
    if args.site:
        sites = [i - 1 for i in args.site]          # 1‑→0‑based
        for s in sites:
            if not (0 <= s < len(struct)):
                print(f"Site {s + 1} out of range (0–{len(struct)})", file=sys.stderr)
                sys.exit(1)
        host_sym = list(struct[sites[0]].species.keys())[0].symbol
    else:
        host_sym = args.element.strip()
        sites = [i for i, s in enumerate(struct)
                 if any(sp.symbol == host_sym for sp in s.species)]
        if not sites:
            print(f"Element {host_sym} not found in POSCAR", file=sys.stderr)
            sys.exit(1)

    # ── Oxidation state ──
    ox = args.ox_state or guess_oxidation(struct, sites[0])
    if ox is None:
        print("⚠ Could not detect oxidation state; OS scoring uses flexibility only. "
              "Use --ox-state to set it.", file=sys.stderr)

    # ── Candidates ──
    cands = resolve_candidates(args.candidates)
    if args.exclude:
        xs = set(s.strip() for s in args.exclude.split(","))
        cands = [c for c in cands if c not in xs]
    cands = [c for c in cands if c != host_sym]

    # ── Screen ──
    hits = screen(struct, sites[0], host_sym, ox, cands, args.weights, args.coord)

    # ── Display ──
    fmt = struct.composition.reduced_formula
    print()
    print("=" * 65)
    title = f"2D Substitution Screener - {host_sym} in {fmt}"
    print(f"  {title}")
    print("=" * 65)
    print(f"  POSCAR     : {args.poscar}")
    print(f"  Sites      : {[s + 1 for s in sites]}")
    print(f"  Host       : {host_sym}  (ox {ox or '?'}, coord {args.coord})")
    print(f"  Candidates : {len(cands)} elements")
    print(f"  Weights    : R={args.weights[0]:.1f} EN={args.weights[1]:.1f} "
          f"OS={args.weights[2]:.1f} S={args.weights[3]:.1f}")
    print()
    print(table(hits, args.top))
    print()

    # ── POSCAR output ──
    if args.output_dir:
        paths = write_poscars(struct, sites, hits, args.top, args.output_dir, host_sym)
        print(f"  → {len(paths)} POSCARs written to {args.output_dir}/")
        for p in paths:
            print(f"    {p}")
        print()

    # ── JSON ──
    if args.json:
        meta = {
            "poscar": args.poscar, "formula": fmt,
            "sites": [s + 1 for s in sites],
            "host": host_sym, "oxidation": ox, "coord": args.coord,
            "num_candidates": len(cands),
            "top_n": args.top,
            "weights": {"radius": args.weights[0], "en": args.weights[1],
                        "os": args.weights[2], "struct": args.weights[3]},
        }
        with open(args.json, "w") as f:
            f.write(to_json(hits, meta, args.top))
        print(f"  → JSON saved to {args.json}")


if __name__ == "__main__":
    main()
