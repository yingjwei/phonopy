#!/usr/bin/env python3
"""
2D Material Element Substitution Screener
=========================================
输入 POSCAR 和要替换的元素，基于电子构型、半径、电负性、
氧化态输出最适合替代该元素的候选方案及评分。

用法:
  python screen.py POSCAR --element V        # 自动推荐 V 的替代元素
  python screen.py POSCAR --element Cl       # 自动推荐 Cl 的替代元素
  python screen.py POSCAR --element V --candidates Nb,Ta,Mo,W,Cr   # 指定候选
  python screen.py POSCAR --element V --ox-state 4 --top 10        # 指定氧化态
  python screen.py POSCAR --element V --output-dir ./candidates    # 输出 POSCAR
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

# Weights: [radius, EN, OS, struct, electronic]
# electronic = d电子数/价电子构型 权重最高
DEFAULT_WEIGHTS = [0.20, 0.15, 0.10, 0.10, 0.45]


# ═══════════════════════════════════════════════════════════════
#  Scoring — 五项评分
# ═══════════════════════════════════════════════════════════════

def _get_ionic_radius(el, ox, cn):
    radii = el.ionic_radii
    if not radii:
        return None
    if ox is not None and ox in radii:
        return float(radii[ox])
    return float(next(iter(radii.values())))


def _charge_balance(composition):
    """Try to assign oxidation states by charge balance.

    Given a dict of {element_symbol: (count, [ox_states])}, find a
    combination (one ox_state per element) that sums closest to zero.
    Returns {element: assigned_ox_state} or None.
    """
    elements = list(composition.keys())
    elements.sort(key=lambda el: len(composition[el][1]))

    best = (None, float("inf"))

    def _search(idx, assigned):
        nonlocal best
        if idx == len(elements):
            err = abs(sum(assigned.values()))
            if err < best[1]:
                best = (dict(assigned), err)
            return
        el = elements[idx]
        count, states = composition[el]
        partial = sum(composition[e][0] * assigned[e] for e in assigned)
        for ox in states:
            candidate = partial + count * ox
            if abs(candidate) > best[1]:
                continue
            assigned[el] = ox
            _search(idx + 1, assigned)
            del assigned[el]

    _search(0, {})
    return best[0]


def guess_oxidation(structure, site_idx):
    """Determine the likely oxidation state for the atom at site_idx.

    Priority:
      1. pymatgen's add_oxidation_state_by_guess() (charge-balanced)
      2. Charge-balance analysis using composition + common_oxidation_states
      3. Fallback: common_oxidation_states[0] for the target element
    """
    sp = list(structure[site_idx].species.keys())[0]
    # Priority 1: pymatgen built-in
    try:
        s = structure.copy()
        s.add_oxidation_state_by_guess()
        for p, amt in s[site_idx].species.items():
            if p.symbol == sp.symbol:
                return int(p.oxi_state)
    except Exception:
        pass
    # Priority 2: charge balance from composition
    try:
        comp = structure.composition
        elem_data = {}
        for el, amt in comp.items():
            sym = el.symbol if hasattr(el, "symbol") else str(el)
            common = el.common_oxidation_states
            if not common:
                return Element(sp.symbol).common_oxidation_states[0]
            elem_data[sym] = (int(amt), common)
        balanced = _charge_balance(elem_data)
        if balanced and sp.symbol in balanced:
            return balanced[sp.symbol]
    except Exception:
        pass
    # Priority 3: most common state for target element
    try:
        el = Element(sp.symbol)
        if el.common_oxidation_states:
            return el.common_oxidation_states[0]
    except Exception:
        pass
    return None


def score_radius(host_el, cand_el, ox, cn):
    """离子半径兼容性。"""
    rh = _get_ionic_radius(host_el, ox, cn) or host_el.atomic_radius
    rc = _get_ionic_radius(cand_el, ox, cn) or cand_el.atomic_radius
    if not rh or not rc or rh <= 0:
        return 0.50
    return exp(-2.0 * abs(rc - rh) / rh)


def score_en(host_el, cand_el):
    """电负性相似度。"""
    try:
        return exp(-abs(host_el.X - cand_el.X) / 1.2)
    except Exception:
        return 0.50


def score_os(host_el, cand_el, target_ox):
    """氧化态兼容性。"""
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


def score_struct(host_el, cand_el):
    """周期表邻近度。"""
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


def score_electronic(host_el, cand_el):
    """
    电子构型兼容性 —— 最重要的因子。

    核心逻辑：
    - 同族 (相同价电子数) → 最优: +0.50
    - 同区块 (d↔d, p↔p) →  电子轨道类型匹配: +0.20
    - 同区块邻族 (±1~2) → d 电子数相近: +0.15
    - 同行 (相同轨道能级) → +0.10
    """
    try:
        same_block = host_el.block == cand_el.block
        group_diff = abs(host_el.group - cand_el.group)
        row_diff = abs(host_el.row - cand_el.row)
    except Exception:
        return 0.40

    score = 0.0

    # 同族 → 价电子数完全相同（如 V/Nb/Ta 都是 5 族）
    if group_diff == 0:
        score += 0.50

    # 同区块（d区换d区、p区换p区⋯⋯）
    if same_block:
        score += 0.20
        # 邻族 d 电子数相近
        if group_diff <= 2:
            score += 0.15
        elif group_diff <= 4:
            score += 0.08

    # 同行 → 轨道能级相近
    if row_diff == 0:
        score += 0.10
    elif row_diff == 1:
        score += 0.05

    return min(score, 1.0)


# ═══════════════════════════════════════════════════════════════
#  Hit — 单条结果
# ═══════════════════════════════════════════════════════════════

@dataclass
class Hit:
    symbol: str
    total: float
    radius: float
    en: float
    os: float
    struct: float
    electronic: float
    r_host: Optional[float]
    r_cand: Optional[float]
    en_host: float
    en_cand: float
    host_group: int
    host_block: str
    cand_group: int
    cand_block: str
    note: str = ""

    @property
    def as_dict(self):
        return asdict(self)


# ═══════════════════════════════════════════════════════════════
#  Core
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
        ec = score_electronic(host, cand)
        total = (weights[0]*rs + weights[1]*es + weights[2]*os_
                 + weights[3]*ss + weights[4]*ec)

        rh = _get_ionic_radius(host, ox, cn) or host.atomic_radius
        rc = _get_ionic_radius(cand, ox, cn) or cand.atomic_radius

        # Generate smart note
        note_parts = []
        if ec > 0.80:
            note_parts.append(f"同族({cand.group}族)")
        elif ec > 0.50:
            note_parts.append(f"邻族")
        if rs > 0.80:
            note_parts.append("半径匹配")
        elif rs < 0.30:
            note_parts.append("半径不匹配")
        if os_ > 0.80:
            note_parts.append("氧化态兼容")
        elif os_ < 0.30:
            note_parts.append("氧化态不兼容")

        hits.append(Hit(
            symbol=sym, total=total,
            radius=rs, en=es, os=os_, struct=ss, electronic=ec,
            r_host=rh, r_cand=rc,
            en_host=host.X, en_cand=cand.X,
            host_group=host.group, host_block=host.block,
            cand_group=cand.group, cand_block=cand.block,
            note=" | ".join(note_parts) if note_parts else ""
        ))
    hits.sort(key=lambda h: h.total, reverse=True)
    return hits


# ═══════════════════════════════════════════════════════════════
#  Auto-candidates
# ═══════════════════════════════════════════════════════════════

def auto_candidates(host_sym):
    """
    智能生成候选替换元素。
    优先级：同族 > 同区块邻族 > 同行 > 常见替换对。
    """
    host = Element(host_sym)
    hg = host.group
    hb = host.block
    hr = host.row
    cands = []
    seen = {host_sym}

    # 0. 总是包含稀土（如果 host 是稀土或过渡金属）
    if hb == 'd':
        lanthanides = GROUPS["lanthanide"]
        for el in lanthanides:
            if el not in seen:
                cands.append(el)
                seen.add(el)

    # 1. 同族（最重要）
    for el in ALL_EL:
        if el not in seen:
            try:
                if Element(el).group == hg:
                    cands.append(el)
                    seen.add(el)
            except:
                pass

    # 2. 同区块邻族（±4 族）
    for el in ALL_EL:
        if el not in seen:
            try:
                e = Element(el)
                if e.block == hb and abs(e.group - hg) <= 4:
                    cands.append(el)
                    seen.add(el)
            except:
                pass

    # 3. 同行
    for el in ALL_EL:
        if el not in seen:
            try:
                if Element(el).row == hr:
                    cands.append(el)
                    seen.add(el)
            except:
                pass

    # 4. 补充常见替换元素
    common = {
        'transition': ['Ru','Os','Co','Ni','Pt','Pd','Au'],
        'main': ['Al','Ga','In','Si','Ge','Sn'],
    }
    bucket = 'transition' if hb == 'd' else 'main'
    for el in common[bucket]:
        if el not in seen:
            cands.append(el)
            seen.add(el)

    return cands


# ═══════════════════════════════════════════════════════════════
#  I/O
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

def table(hits, top_n):
    """显示格式化的结果表格。"""
    SEP = "-" * 95

    # 表头解释
    header_info = (
        "  解释: 总分=电子×0.45 + 半径×0.20 + 电负性×0.15 + 氧化态×0.10 + 周期×0.10\n"
        "  电子: 同族同区块最高 → d电子数/价电子构型匹配度\n"
        "  注意: 电子分高 ≠ 声子稳定，需 DFT 验证"
    )

    fmt = (
        "{rank:<4} {el:<7} {total:<7} {ec:<7} {r:<7} {en:<7} {os:<6} "
        "{cgroup:<6} {cblock:<6} note"
    )
    lines = [header_info, ""]
    lines.append(fmt.format(
        rank="Rank", el="元素", total="总分", ec="电子",
        r="半径", en="电负", os="氧化", cgroup="族", cblock="区块"
    ))
    lines.append(SEP)
    for i, h in enumerate(hits[:top_n], 1):
        lines.append(
            f"{i:<4} {h.symbol:<7} {h.total:<7.3f} {h.electronic:<7.3f} "
            f"{h.radius:<7.3f} {h.en:<7.3f} {h.os:<6.2f} "
            f"{h.cand_group:<6} {h.cand_block:<6} {h.note[:45]}"
        )
    return "\n".join(lines)


def to_json(hits, meta, top_n):
    return json.dumps({
        "meta": meta,
        "hits": [{"rank": i + 1, **h.as_dict} for i, h in enumerate(hits[:top_n])]
    }, indent=2, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════
#  Candidates 解析
# ═══════════════════════════════════════════════════════════════

def resolve_candidates(spec: str) -> List[str]:
    spec_s = spec.strip()
    if spec_s.lower() in GROUPS:
        return list(GROUPS[spec_s.lower()])
    return [s.strip() for s in spec_s.split(",") if s.strip()]


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

def block_cn(block):
    return {'s': 8, 'p': 6, 'd': 6, 'f': 8}.get(block, 6)


def main():
    ap = argparse.ArgumentParser(
        description="2D Material Element Substitution Screener",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("poscar", help="POSCAR 文件路径")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--site", type=int, nargs="+",
                   help="要替换的位点编号（从 1 开始）")
    g.add_argument("--element", type=str,
                   help="要替换的元素符号，自动推荐替代候选")

    ap.add_argument("--candidates", default="",
                    help="候选列表：逗号分隔 或 内置组名 "
                         f"({', '.join(GROUPS)})")
    ap.add_argument("--exclude", default="",
                    help="排除的元素（逗号分隔）")
    ap.add_argument("--ox-state", type=int,
                    help="目标氧化态（推荐指定，提高准确性）")
    ap.add_argument("--coord", type=int,
                    help="配位数（默认 auto: d区/p区=6, s区=8, f区=8）")
    ap.add_argument("--weights", type=float, nargs=5,
                    default=DEFAULT_WEIGHTS,
                    help="权重：半径 EN OS 周期 电子 "
                         f"(默认: {' '.join(map(str, DEFAULT_WEIGHTS))})")
    ap.add_argument("--top", type=int, default=15,
                    help="显示前 N 个 (默认: 15)")
    ap.add_argument("--output-dir",
                    help="为 Top N 候选输出替换后的 POSCAR 文件")
    ap.add_argument("--json",
                    help="保存结果为 JSON")
    args = ap.parse_args()

    # ── 读取 POSCAR ──
    if not os.path.exists(args.poscar):
        print(f"错误: POSCAR 文件不存在: {args.poscar}", file=sys.stderr)
        sys.exit(1)
    try:
        struct = Structure.from_file(args.poscar)
    except Exception as e:
        print(f"读取 POSCAR 失败: {e}", file=sys.stderr)
        sys.exit(1)

    # ── 确定替换位点 ──
    if args.site:
        sites = [i - 1 for i in args.site]
        for s in sites:
            if not (0 <= s < len(struct)):
                print(f"位点 {s+1} 超出范围 (1–{len(struct)})", file=sys.stderr)
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

    # ── 氧化态 ──
    ox = args.ox_state or guess_oxidation(struct, sites[0])

    # ── 配位数 ──
    cn = args.coord or block_cn(host_el.block)

    # ── 候选列表 ──
    if args.candidates:
        cands = resolve_candidates(args.candidates)
    else:
        cands = auto_candidates(host_sym)
        print(f"  (自动生成了 {len(cands)} 个候选，"
              f"可用 --candidates 自定义)")
        print()

    if args.exclude:
        xs = set(s.strip() for s in args.exclude.split(","))
        cands = [c for c in cands if c not in xs]
    cands = [c for c in cands if c != host_sym]

    # ── 评分 ──
    w = args.weights
    hits = screen(struct, sites[0], host_sym, ox, cands, w, cn)

    # ── 输出 ──
    fmt = struct.composition.reduced_formula
    print()
    print("=" * 65)
    print(f"  元素替换推荐 — {host_sym} in {fmt}")
    print("=" * 65)
    print(f"  POSCAR   : {args.poscar}")
    print(f"  位点     : {[s+1 for s in sites]}")
    print(f"  替换     : {host_sym}")
    print(f"  电子构型 : {host_el.block}-区 {host_el.group} 族")
    print(f"  氧化态   : {ox or '自动检测失败(用 --ox-state 指定)'}")
    print(f"  配位数   : {cn}")
    print()
    print(table(hits, args.top))
    print()

    # ── 输出 POSCAR ──
    if args.output_dir:
        paths = write_poscars(struct, sites, hits, args.top,
                              args.output_dir, host_sym)
        print(f"  -> {len(paths)} 个 POSCAR 文件写入 {args.output_dir}/")
        for p in paths:
            print(f"    {p}")
        print()

    # ── JSON ──
    if args.json:
        meta = {
            "poscar": args.poscar, "formula": fmt,
            "host": host_sym, "block": host_el.block,
            "group": host_el.group,
            "oxidation": ox, "coord": cn,
            "num_candidates": len(cands),
            "top_n": args.top,
            "weights": dict(zip(
                ["radius","en","os","struct","electronic"], w)),
        }
        with open(args.json, "w") as f:
            f.write(to_json(hits, meta, args.top))
        print(f"  -> JSON 结果保存至 {args.json}")
        print()


if __name__ == "__main__":
    main()
