#!/usr/bin/env python3
"""
2D Material Element Substitution Recommender
=============================================
推荐引擎基于 125+ 组 DFT+phonopy 声子稳定性数据 + 化学启发式评分。

用法:
  python recommend.py POSCAR --element V                  # 自动推荐 V 的替代元素
  python recommend.py POSCAR --element V --top 10         # 只看 Top 10
  python recommend.py POSCAR --element Cl --candidates F,Br,I  # 指定候选范围
  python recommend.py POSCAR --element V --json result.json    # 输出 JSON
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
    from pymatgen.io.vasp import Poscar
except ImportError:
    print("需要 pymatgen: pip install pymatgen", file=sys.stderr)
    sys.exit(1)

from phonon_data import (
    ALL_STRUCTURES, V4S9Br4_TYPE, W6CCl16_TYPE, PbNV_TYPE,
    lookup_substitutions, get_parent_by_name,
)


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
# Structure matching — identify parent from POSCAR
# ═══════════════════════════════════════════════════════════════

def match_parent_structure(structure):
    """Identify the parent structure type from lattice + elements."""
    lattice = structure.lattice
    a, b, c = lattice.a, lattice.b, lattice.c
    alpha, beta, gamma = lattice.alpha, lattice.beta, lattice.gamma
    elements = {str(spec) for site in structure for spec in site.species}
    n_atoms = len(structure)

    # V4S9Br4-type: tetragonal P4/nmm, a≈11-13, c≈24-26
    if (abs(alpha - 90) < 1 and abs(beta - 90) < 1 and abs(gamma - 90) < 1
            and abs(a - b) / a < 0.05
            and 10.5 < a < 13.5
            and 23 < c < 27
            and n_atoms == 34):
        return V4S9Br4_TYPE, "V₄S₉Br₄-type (P4/nmm) 结构"

    # W6CCl16-type
    if "C" in elements and n_atoms >= 15:
        return W6CCl16_TYPE, "W₆CCl₁₆-type 结构"

    # PbNV-type: ternary nitride
    if "N" in elements and 2 < n_atoms < 20:
        return PbNV_TYPE, "PbNV-type 结构"

    return None, "未知结构（数据库中无匹配）"


def get_site_label(parent, element_symbol, structure):
    """Determine which site the element occupies in the structure."""
    if parent is None:
        return None
    elements = [str(spec) for site in structure for spec in site.species]
    unique = sorted(set(elements))

    if parent == V4S9Br4_TYPE:
        # M (metal, group 3-12), S (chalcogen), X (halogen)
        if element_symbol in ["S", "Se", "Te"]:
            return "S"
        if element_symbol in ["F", "Cl", "Br", "I"]:
            return "X"
        return "M"  # default for metals

    if parent == W6CCl16_TYPE:
        if element_symbol == "C":
            return "C"
        if element_symbol in ["F", "Cl", "Br", "I"]:
            return "X"
        return "M"

    if parent == PbNV_TYPE:
        if element_symbol == "N":
            return "N"
        return "M"

    return None


# ═══════════════════════════════════════════════════════════════
# Scoring — five chemical dimensions
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
    rh = _get_ionic_radius(host_el, ox, cn) or host_el.atomic_radius
    rc = _get_ionic_radius(cand_el, ox, cn) or cand_el.atomic_radius
    if not rh or not rc or rh <= 0:
        return 0.50
    return exp(-2.0 * abs(rc - rh) / rh)


def score_en(host_el, cand_el):
    try:
        return exp(-abs(host_el.X - cand_el.X) / 1.2)
    except Exception:
        return 0.50


def score_os(host_el, cand_el, target_ox):
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
    """Electronic configuration compatibility."""
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


# ═══════════════════════════════════════════════════════════════
# Empirical phonon stability score
# ═══════════════════════════════════════════════════════════════

def score_empirical(cand_sym, host_sym, parent, site_label, siblings):
    """
    Score based on phonon stability database.

    Rules:
    - Tested and stable in this parent        → +1.0
    - Tested and stable but in MSX-only form  → +0.5 (less reliable)
    - Tested and unstable in this parent      → -0.5
    - Tested and unstable in MSX-only form    → -0.3
    - Untested                                →  0.0 (neutral)
    """
    if parent is None:
        return 0.0, "无数据库匹配"

    subs_key = f"{site_label}_substitutions"
    subs = parent.get(subs_key)

    if subs is None:
        return 0.0, "无该位点数据"

    stable_set = set(subs.get("stable", []))
    unstable_set = set(subs.get("unstable", []))
    msx_set = set(parent.get("MSX_only", []))

    # Check paired stability
    paired = parent.get("paired_stability", {})
    if cand_sym in paired:
        for sibling in siblings:
            if sibling in paired[cand_sym].get("stable_with", []):
                return 1.0, f"数据库: {cand_sym} 与 {sibling} 组合稳定"
            if sibling in paired[cand_sym].get("unstable_with", []):
                return -0.5, f"数据库: {cand_sym} 与 {sibling} 组合不稳定"

    if cand_sym in stable_set:
        return 1.0, "数据库: 该替换在类似结构中稳定"
    if cand_sym in msx_set:
        return 0.5, "数据库: 仅 MSX 形式测试过(不可靠)"
    if cand_sym in unstable_set:
        return -0.5, "数据库: 该替换在类似结构中不稳定"

    return 0.0, "数据库: 无该替换数据(中性)"


# ═══════════════════════════════════════════════════════════════
# Hit — single result
# ═══════════════════════════════════════════════════════════════

@dataclass
class Hit:
    symbol: str
    total: float
    empirical: float
    electronic: float
    radius: float
    en: float
    os: float
    r_host: Optional[float]
    r_cand: Optional[float]
    en_host: float
    en_cand: float
    host_group: int
    host_block: str
    cand_group: int
    cand_block: str
    emp_note: str = ""
    status: str = ""

    @property
    def as_dict(self):
        return asdict(self)


# ═══════════════════════════════════════════════════════════════
# Core recommendation
# ═══════════════════════════════════════════════════════════════

# Default weights: [empirical, electronic, radius, EN, OS]
# empirical权重最高 — 实际声子计算结果最有说服力
DEFAULT_WEIGHTS = [0.40, 0.25, 0.15, 0.10, 0.10]


def recommend(structure, host_sym, ox, candidates, weights,
              parent, site_label, siblings):
    """Main recommendation engine."""
    host = Element(host_sym)
    hits = []

    for sym in candidates:
        if sym == host_sym:
            continue
        try:
            cand = Element(sym)
        except Exception:
            continue

        # Chemical scores
        rs = score_radius(host, cand, ox, 6)
        es = score_en(host, cand)
        os_ = score_os(host, cand, ox)
        ec = score_electronic(host, cand)

        # Empirical phonon stability score
        emp_score, emp_note = score_empirical(
            sym, host_sym, parent, site_label, siblings)

        # Total = weighted sum
        total = (weights[0] * emp_score
                 + weights[1] * ec
                 + weights[2] * rs
                 + weights[3] * es
                 + weights[4] * os_)

        # Normalize to [0, 1] range for display
        # (empirical can be negative, so shift)
        # Actually just keep raw — higher = better

        rh = _get_ionic_radius(host, ox, 6) or host.atomic_radius
        rc = _get_ionic_radius(cand, ox, 6) or cand.atomic_radius

        # Status label
        if emp_score > 0.5:
            status = "★ 稳定"
        elif emp_score > 0:
            status = "◑ 部分稳定"
        elif emp_score == 0:
            status = "○ 未验证"
        else:
            status = "✗ 不稳定"

        hits.append(Hit(
            symbol=sym, total=total,
            empirical=emp_score,
            electronic=ec,
            radius=rs, en=es, os=os_,
            r_host=rh, r_cand=rc,
            en_host=host.X, en_cand=cand.X,
            host_group=host.group, host_block=host.block,
            cand_group=cand.group, cand_block=cand.block,
            emp_note=emp_note,
            status=status,
        ))

    hits.sort(key=lambda h: h.total, reverse=True)
    return hits


# ═══════════════════════════════════════════════════════════════
# Auto-candidates
# ═══════════════════════════════════════════════════════════════

def auto_candidates(host_sym, parent=None, site_label=None):
    """
    Generate candidate substitution elements.
    Priority: database-verified > same group > same block neighbor > same row.
    """
    host = Element(host_sym)
    hg = host.group
    hb = host.block
    hr = host.row
    cands = []
    seen = {host_sym}

    # 0. Database-verified candidates first (stable in this structure)
    if parent is not None and site_label is not None:
        subs_key = f"{site_label}_substitutions"
        subs = parent.get(subs_key)
        if subs:
            for el in subs.get("stable", []):
                if el not in seen:
                    cands.append(el)
                    seen.add(el)

    # 0b. Always include lanthanides for d-block hosts
    if hb == 'd':
        for el in GROUPS["lanthanide"]:
            if el not in seen:
                cands.append(el)
                seen.add(el)

    # 1. Same group
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

def table(hits, top_n, parent_name):
    """Display formatted results table."""
    SEP = "-" * 100

    header_info = (
        "  评分说明:\n"
        "  ★ 稳定  = 数据库中该替换在类似结构中声子稳定\n"
        "  ✗ 不稳定 = 数据库中该替换在类似结构中声子不稳定\n"
        "  ○ 未验证 = 数据库中无该替换数据，化学分仅供参考\n"
        "  总分 = 经验×0.40 + 电子×0.25 + 半径×0.15 + 电负×0.10 + 氧化×0.10"
    )

    fmt = (
        "{rank:<4} {el:<6} {total:<7} {emp:<7} {ec:<7} "
        "{r:<7} {en:<6} {os:<6} {status:<10} {note}"
    )
    lines = [header_info, ""]
    lines.append(f"  匹配结构: {parent_name}")
    lines.append("")
    lines.append(fmt.format(
        rank="Rank", el="元素", total="总分", emp="经验分",
        ec="电子", r="半径", en="电负", os="氧化",
        status="状态", note="说明"
    ))
    lines.append(SEP)
    for i, h in enumerate(hits[:top_n], 1):
        lines.append(
            f"{i:<4} {h.symbol:<6} {h.total:<7.3f} {h.empirical:<7.2f} "
            f"{h.electronic:<7.3f} {h.radius:<7.3f} {h.en:<7.3f} "
            f"{h.os:<6.2f} {h.status:<10} {h.emp_note[:40]}"
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
        description="2D Material Element Substitution Recommender",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("poscar", help="POSCAR 文件路径")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--site", type=int, nargs="+",
                   help="要替换的位点编号（从 1 开始）")
    g.add_argument("--element", type=str,
                   help="要替换的元素符号")

    ap.add_argument("--candidates", default="",
                    help="候选列表：逗号分隔 或 内置组名 "
                         f"({', '.join(GROUPS)})")
    ap.add_argument("--exclude", default="",
                    help="排除的元素（逗号分隔）")
    ap.add_argument("--ox-state", type=int,
                    help="目标氧化态（推荐指定）")
    ap.add_argument("--weights", type=float, nargs=5,
                    default=DEFAULT_WEIGHTS,
                    help="权重：经验 电子 半径 电负 氧化 "
                         f"(默认: {' '.join(map(str, DEFAULT_WEIGHTS))})")
    ap.add_argument("--top", type=int, default=15,
                    help="显示前 N 个 (默认: 15)")
    ap.add_argument("--json",
                    help="保存结果为 JSON")
    ap.add_argument("--no-auto", action="store_true",
                    help="不使用数据库自动候选，仅用化学启发式")
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

    # ── Determine site and host element ──
    if args.site:
        sites = [i - 1 for i in args.site]
        for s in sites:
            if not (0 <= s < len(struct)):
                print(f"位点 {s+1} 超出范围 (1–{len(struct)})",
                      file=sys.stderr)
                sys.exit(1)
        host_sym = list(struct[sites[0]].species.keys())[0].symbol
    else:
        host_sym = args.element.strip()
        sites = [i for i, site in enumerate(struct)
                 if any(sp.symbol == host_sym for sp in site.species)]
        if not sites:
            print(f"错误: POSCAR 中没有元素 {host_sym}", file=sys.stderr)
            sys.exit(1)

    host_el = Element(host_sym)

    # ── Oxidation state ──
    ox = args.ox_state or guess_oxidation(struct, sites[0])

    # ── Match parent structure ──
    parent, parent_name = match_parent_structure(struct)
    site_label = get_site_label(parent, host_sym, struct)

    # ── Sibling elements (for paired stability check) ──
    siblings = [str(spec) for site in struct
                for spec in site.species
                if str(spec) != host_sym]

    # ── Candidates ──
    if args.candidates:
        cands = resolve_candidates(args.candidates)
    elif args.no_auto:
        cands = [e for e in ALL_EL if e not in EXCLUDE and e != host_sym]
    else:
        cands = auto_candidates(host_sym, parent, site_label)
        n_db = sum(1 for c in cands
                   if parent and parent.get(f"{site_label}_substitutions", {})
                   .get("stable", []).count(c) > 0) if site_label else 0
        print(f"  (自动生成了 {len(cands)} 个候选，含 {n_db} 个数据库稳定项)")

    if args.exclude:
        xs = set(s.strip() for s in args.exclude.split(","))
        cands = [c for c in cands if c not in xs]
    cands = [c for c in cands if c != host_sym]

    # ── Score ──
    w = args.weights
    hits = recommend(struct, host_sym, ox, cands, w,
                     parent, site_label, siblings)

    # ── Output ──
    fmt = struct.composition.reduced_formula
    print()
    print("=" * 70)
    print(f"  元素替换推荐引擎 — {host_sym} in {fmt}")
    print("=" * 70)
    print(f"  POSCAR     : {args.poscar}")
    print(f"  替换元素   : {host_sym}")
    print(f"  电子构型   : {host_el.block}-区 {host_el.group} 族")
    print(f"  氧化态     : {ox or '自动检测失败(用 --ox-state 指定)'}")
    print(f"  匹配结构   : {parent_name}")
    print(f"  位点角色   : {site_label or '未知'}")
    print()
    print(table(hits, args.top, parent_name))
    print()

    # ── JSON ──
    if args.json:
        meta = {
            "poscar": args.poscar, "formula": fmt,
            "host": host_sym, "block": host_el.block,
            "group": host_el.group,
            "oxidation": ox,
            "parent_structure": parent_name,
            "site_label": site_label,
            "num_candidates": len(cands),
            "top_n": args.top,
            "weights": dict(zip(
                ["empirical", "electronic", "radius", "en", "os"], w)),
        }
        with open(args.json, "w") as f:
            f.write(to_json(hits, meta, args.top))
        print(f"  -> JSON 结果保存至 {args.json}")
        print()


if __name__ == "__main__":
    main()
