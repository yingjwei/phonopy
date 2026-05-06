#!/usr/bin/env python3
"""
2D Material Element Substitution Recommender
=============================================
基于化学兼容性 + 几何兼容性的元素替换推荐系统。
不依赖具体结构类型匹配，适用于任意新结构。

用法:
  python recommend.py POSCAR --element V                  # 自动推荐 V 的替代元素
  python recommend.py POSCAR --element V --candidates Nb,Ta,Mo,W  # 指定候选
  python recommend.py POSCAR --element V --top 10         # 只看 Top 10
  python recommend.py POSCAR --element V --json result.json   # 输出 JSON
"""

import argparse
import json
import os
import sys
from math import exp
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

try:
    from pymatgen.core import Element, Structure
except ImportError:
    print("需要 pymatgen: pip install pymatgen", file=sys.stderr)
    sys.exit(1)

from phonon_data import ALL_STRUCTURES


# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

EXCLUDE = {
    "He", "Ne", "Ar", "Kr", "Xe", "Rn",
    "Po", "At", "Fr", "Ra", "Ac", "Pa",
    "Np", "Pu", "Am", "Cm", "Bk", "Cf",
}

ALL_EL = [
    "Li", "Be", "B", "C", "N", "O", "F", "Na", "Mg", "Al", "Si", "P",
    "S", "Cl", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co",
    "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Rb", "Sr", "Y",
    "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
    "Sb", "Te", "I", "Cs", "Ba", "La", "Ce", "Pr", "Nd", "Sm", "Eu",
    "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",
    "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi",
]

GROUPS = {
    "all":        ALL_EL,
    "tm":         ["Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
                   "Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
                   "Hf","Ta","W","Re","Os","Ir","Pt","Au"],
    "main":       ["B","C","N","O","F","Na","Mg","Al","Si","P",
                   "S","Cl","Ga","Ge","As","Se","Br","In","Sn",
                   "Sb","Te","I","Tl","Pb","Bi"],
    "chalcogen":  ["O","S","Se","Te"],
    "pnictogen":  ["N","P","As","Sb","Bi"],
    "halogen":    ["F","Cl","Br","I"],
    "alkali":     ["Li","Na","K","Rb","Cs"],
    "alkaline":   ["Mg","Ca","Sr","Ba"],
    "lanthanide": ["La","Ce","Pr","Nd","Sm","Eu","Gd",
                   "Tb","Dy","Ho","Er","Tm","Yb","Lu"],
    "early-tm":   ["Sc","Ti","V","Cr","Y","Zr","Nb","Mo","Hf","Ta","W","Re"],
    "late-tm":    ["Mn","Fe","Co","Ni","Cu","Zn","Tc","Ru","Rh",
                   "Pd","Ag","Cd","Os","Ir","Pt","Au"],
}


# ═══════════════════════════════════════════════════════════════
# Cross-structure database trends (general, not structure-specific)
# ═══════════════════════════════════════════════════════════════

def _build_global_trends():
    """
    From the phonon database, extract which elements have been found
    stable/unstable ACROSS all structure types. This gives a general
    indication (not structure-specific).
    """
    stable_all = set()
    unstable_all = set()
    for struct in ALL_STRUCTURES:
        for key in struct:
            if key.endswith("_substitutions"):
                stable_all.update(struct[key].get("stable", []))
                unstable_all.update(struct[key].get("unstable", []))
    # Some elements appear in both (paired stability)
    return stable_all, unstable_all

GLOBAL_STABLE, GLOBAL_UNSTABLE = _build_global_trends()


# ═══════════════════════════════════════════════════════════════
# Geometric analysis from POSCAR
# ═══════════════════════════════════════════════════════════════

def analyze_bond_lengths(structure, site_indices):
    """
    Calculate average nearest-neighbor distance for the host sites.
    This gives a real-space measure of the space available at the site.

    Returns:
        avg_bond_length: average NN distance (Å)
        n_neighbors: number of neighbors found
    """
    site = structure[site_indices[0]]
    # Find neighbors up to 4 Å or half the shortest lattice vector
    max_r = min(structure.lattice.a, structure.lattice.b,
                structure.lattice.c) * 0.6
    max_r = min(max_r, 4.0)  # cap at 4 Å for efficiency

    neighbors = structure.get_neighbors(site, max_r)
    if not neighbors:
        return None, 0

    distances = [n.distance for n in neighbors
                 if n.distance > 0.1]  # filter self
    if not distances:
        return None, 0

    return float(np.mean(distances)), len(distances)


def score_geometric(host_el, cand_el, avg_bond_length, n_neighbors):
    """
    Score based on geometric compatibility from actual POSCAR bond lengths.

    Logic:
    - avg_bond_length ≈ r_host_eff + r_neighbor_eff
    - If we replace host with cand, new bond length ≈ r_cand + (avg - r_host)
    - Strain = |new - avg| / avg
    - High strain → lattice distortion → likely phonon instability
    """
    if avg_bond_length is None:
        return 0.50, ""

    r_host = host_el.atomic_radius
    r_cand = cand_el.atomic_radius
    if not r_host or not r_cand or r_host <= 0:
        return 0.50, ""

    # Estimate the host's effective contribution to the bond
    # (crude: assume half the bond length is the host)
    # Better: r_host_eff = min(r_host, avg_bond_length * 0.6)
    r_host_eff = r_host

    # Predicted bond length with candidate
    r_cand_eff = r_cand
    predicted_bond = avg_bond_length - r_host_eff + r_cand_eff

    if predicted_bond <= 0:
        return 0.10, "半径过大"

    # Strain ratio
    strain = abs(predicted_bond - avg_bond_length) / avg_bond_length

    # Score: exp(-5 * strain) — 5% strain → 0.78, 10% → 0.61, 20% → 0.37
    score = exp(-5.0 * strain)

    # Note
    if strain < 0.03:
        note = ""
    elif strain < 0.08:
        note = f"应变 {strain*100:.1f}%"
    elif strain < 0.15:
        note = f"应变 {strain*100:.1f}%(偏高)"
    else:
        note = f"应变 {strain*100:.1f}%(高)"

    return score, note


# ═══════════════════════════════════════════════════════════════
# Scoring — chemical dimensions (structure-transferable)
# ═══════════════════════════════════════════════════════════════

def _get_ionic_radius(el, ox, cn):
    radii = el.ionic_radii
    if not radii:
        return None
    if ox is not None and ox in radii:
        return float(radii[ox])
    return float(next(iter(radii.values())))


def guess_oxidation(structure, site_idx):
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


def score_radius(host_el, cand_el, ox, cn):
    """Ionic radius compatibility — key indicator of lattice strain."""
    rh = _get_ionic_radius(host_el, ox, cn) or host_el.atomic_radius
    rc = _get_ionic_radius(cand_el, ox, cn) or cand_el.atomic_radius
    if not rh or not rc or rh <= 0:
        return 0.50
    return exp(-2.0 * abs(rc - rh) / rh)


def score_en(host_el, cand_el):
    """Electronegativity similarity → similar bond character."""
    try:
        return exp(-abs(host_el.X - cand_el.X) / 1.2)
    except Exception:
        return 0.50


def score_os(host_el, cand_el, target_ox):
    """Oxidation state compatibility → charge balance."""
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


def score_electronic(host_el, cand_el):
    """
    Electronic configuration compatibility.
    Same group = same d-electron count = similar bonding.
    """
    try:
        same_block = host_el.block == cand_el.block
        group_diff = abs(host_el.group - cand_el.group)
        row_diff = abs(host_el.row - cand_el.row)
    except Exception:
        return 0.40
    score = 0.0
    if group_diff == 0:
        score += 0.50
    if same_block:
        score += 0.20
        if group_diff <= 2:
            score += 0.15
        elif group_diff <= 4:
            score += 0.08
    if row_diff == 0:
        score += 0.10
    elif row_diff == 1:
        score += 0.05
    return min(score, 1.0)


def score_global_trend(cand_sym):
    """
    Cross-structure trend from the database:
    - Element has ONLY been found stable across all tested structures → small bonus
    - Element has ONLY been found unstable → small penalty
    - Mixed or untested → neutral
    """
    in_stable = cand_sym in GLOBAL_STABLE
    in_unstable = cand_sym in GLOBAL_UNSTABLE
    if in_stable and not in_unstable:
        return 0.20, "数据库趋势: 该元素在多结构中表现稳定"
    if in_unstable and not in_stable:
        return -0.10, "数据库趋势: 该元素在多结构中表现不稳定"
    if in_stable and in_unstable:
        return 0.05, "数据库趋势: 该元素稳定性依赖具体组合"
    return 0.0, ""


# ═══════════════════════════════════════════════════════════════
# Hit — single result
# ═══════════════════════════════════════════════════════════════

@dataclass
class Hit:
    symbol: str
    total: float
    geometric: float
    electronic: float
    radius: float
    en: float
    os: float
    trend: float
    r_host: Optional[float]
    r_cand: Optional[float]
    en_host: float
    en_cand: float
    host_group: int
    host_block: str
    cand_group: int
    cand_block: str
    bond_length: Optional[float] = None
    geo_note: str = ""
    trend_note: str = ""

    @property
    def as_dict(self):
        return asdict(self)


# ═══════════════════════════════════════════════════════════════
# Core recommendation
# ═══════════════════════════════════════════════════════════════

# Weights: [geometric, electronic, radius, EN, OS, trend]
# geometric从POSCAR实际键长分析 → 最直接反映晶格失配
DEFAULT_WEIGHTS = [0.30, 0.25, 0.15, 0.10, 0.10, 0.10]


def recommend(structure, host_sym, ox, candidates, weights,
              site_indices):
    host = Element(host_sym)
    hits = []

    # Geometric analysis from POSCAR
    avg_bond, n_nb = analyze_bond_lengths(structure, site_indices)
    if avg_bond:
        bond_info = f"平均键长 {avg_bond:.3f} Å ({n_nb} 个近邻)"
    else:
        bond_info = "无法解析键长"

    for sym in candidates:
        if sym == host_sym:
            continue
        try:
            cand = Element(sym)
        except Exception:
            continue

        # Geometric score (from POSCAR bond analysis)
        geo, geo_note = score_geometric(host, cand, avg_bond, n_nb)

        # Chemical scores (structure-transferable)
        rs = score_radius(host, cand, ox, 6)
        es = score_en(host, cand)
        os_ = score_os(host, cand, ox)
        ec = score_electronic(host, cand)

        # Global database trend (cross-structure)
        tr, trend_note = score_global_trend(sym)

        # Total = weighted sum
        total = (weights[0] * geo
                 + weights[1] * ec
                 + weights[2] * rs
                 + weights[3] * es
                 + weights[4] * os_
                 + weights[5] * tr)

        total = max(0.0, min(1.0, total))

        rh = _get_ionic_radius(host, ox, 6) or host.atomic_radius
        rc = _get_ionic_radius(cand, ox, 6) or cand.atomic_radius

        hits.append(Hit(
            symbol=sym, total=total,
            geometric=geo, electronic=ec,
            radius=rs, en=es, os=os_, trend=tr,
            r_host=rh, r_cand=rc,
            en_host=host.X, en_cand=cand.X,
            host_group=host.group, host_block=host.block,
            cand_group=cand.group, cand_block=cand.block,
            bond_length=avg_bond,
            geo_note=geo_note,
            trend_note=trend_note,
        ))

    hits.sort(key=lambda h: h.total, reverse=True)
    return hits, bond_info


# ═══════════════════════════════════════════════════════════════
# Auto-candidates
# ═══════════════════════════════════════════════════════════════

def auto_candidates(host_sym):
    host = Element(host_sym)
    hg = host.group
    hb = host.block
    hr = host.row
    cands = []
    seen = {host_sym}

    # 0. Lanthanides for d-block hosts
    if hb == 'd':
        for el in GROUPS["lanthanide"]:
            if el not in seen:
                cands.append(el)
                seen.add(el)

    # 1. Same group (most important)
    for el in ALL_EL:
        if el not in seen:
            try:
                if Element(el).group == hg:
                    cands.append(el)
                    seen.add(el)
            except:
                pass

    # 2. Same block, nearby group
    for el in ALL_EL:
        if el not in seen:
            try:
                e = Element(el)
                if e.block == hb and abs(e.group - hg) <= 4:
                    cands.append(el)
                    seen.add(el)
            except:
                pass

    # 3. Same row
    for el in ALL_EL:
        if el not in seen:
            try:
                if Element(el).row == hr:
                    cands.append(el)
                    seen.add(el)
            except:
                pass

    # 4. Common substitutions
    common = {
        'transition': ['Ru', 'Os', 'Co', 'Ni', 'Pt', 'Pd', 'Au'],
        'main': ['Al', 'Ga', 'In', 'Si', 'Ge', 'Sn'],
    }
    bucket = 'transition' if hb == 'd' else 'main'
    for el in common[bucket]:
        if el not in seen:
            cands.append(el)
            seen.add(el)

    return cands


# ═══════════════════════════════════════════════════════════════
# Formatting
# ═══════════════════════════════════════════════════════════════

def table(hits, top_n, bond_info):
    SEP = "-" * 88

    lines = [f"  键长分析: {bond_info}", ""]
    lines.append(f"  {'Rank':<4} {'元素':<6} {'总分':<7} {'几何':<7} {'电子':<7} {'半径':<7} {'电负':<6} {'氧化':<6}")
    lines.append(SEP)
    for i, h in enumerate(hits[:top_n], 1):
        lines.append(
            f"  {i:<4} {h.symbol:<6} {h.total:<7.3f} {h.geometric:<7.3f} "
            f"{h.electronic:<7.3f} {h.radius:<7.3f} {h.en:<7.3f} {h.os:<6.2f}"
        )
    return "\n".join(lines)


def to_json(hits, meta, top_n):
    return json.dumps({
        "meta": meta,
        "hits": [{"rank": i + 1, **h.as_dict}
                 for i, h in enumerate(hits[:top_n])]
    }, indent=2, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════
# Candidates parser
# ═══════════════════════════════════════════════════════════════

def resolve_candidates(spec: str):
    spec_s = spec.strip()
    if spec_s.lower() in GROUPS:
        return list(GROUPS[spec_s.lower()])
    return [s.strip() for s in spec_s.split(",") if s.strip()]


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="2D Material Element Substitution Recommender")
    ap.add_argument("poscar", help="POSCAR 文件路径")
    ap.add_argument("--top", type=int, default=15,
                    help="显示前 N 个 (默认: 15)")
    ap.add_argument("--json",
                    help="保存结果为 JSON")
    args = ap.parse_args()

    # ── Read POSCAR ──
    if not os.path.exists(args.poscar):
        print(f"错误: POSCAR 文件不存在: {args.poscar}", file=sys.stderr)
        sys.exit(1)
    try:
        struct = Structure.from_file(args.poscar)
    except Exception as e:
        print(f"读取 POSCAR 失败: {e}", file=sys.stderr)
        sys.exit(1)

    # ── Show elements ──
    elements = list(struct.composition.as_dict().keys())
    comp = struct.composition
    print()
    print("=" * 60)
    print(f"  文件: {args.poscar}")
    print(f"  成分: {comp.reduced_formula}")
    print(f"  原子: {', '.join(elements)}")
    print(f"  数量: {dict(comp.element_composition)}")
    print("=" * 60)
    print()
    host_sym = input("  输入要替换的原子: ").strip()
    if not host_sym:
        print("未输入原子", file=sys.stderr)
        sys.exit(1)

    # ── Find sites ──
    sites = [i for i, site in enumerate(struct)
             if any(sp.symbol == host_sym for sp in site.species)]
    if not sites:
        print(f"错误: POSCAR 中没有元素 {host_sym}", file=sys.stderr)
        sys.exit(1)

    host_el = Element(host_sym)

    # ── Oxidation state ──
    ox = guess_oxidation(struct, sites[0])

    # ── Auto candidates ──
    cands = auto_candidates(host_sym)

    # ── Score ──
    w = DEFAULT_WEIGHTS
    hits, bond_info = recommend(struct, host_sym, ox, cands, w, sites)

    # ── Output ──
    print()
    print("=" * 60)
    print(f"  替换 {host_sym} 的推荐候选")
    print("=" * 60)
    print(f"  电子构型: {host_el.block}-区 {host_el.group} 族"
          f"  |  氧化态: {ox or '?'}")
    print()
    print(table(hits, args.top, bond_info))
    print()

    # ── JSON ──
    if args.json:
        meta = {
            "poscar": args.poscar, "formula": comp.reduced_formula,
            "host": host_sym,
            "oxidation": ox,
            "num_candidates": len(cands),
            "top_n": args.top,
            "weights": dict(zip(
                ["geometric", "electronic", "radius", "en", "os", "trend"],
                w)),
        }
        with open(args.json, "w") as f:
            f.write(to_json(hits, meta, args.top))
        print(f"  -> JSON 保存至 {args.json}")
        print()


if __name__ == "__main__":
    main()
