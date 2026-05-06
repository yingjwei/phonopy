#!/usr/bin/env python3
"""
2D Material Element Substitution Replacer
=========================================
输入 POSCAR 和要替换的元素，基于电子构型、半径、电负性、
氧化态输出最适合替代该元素的候选方案及评分。

自动按配位环境分组：同元素不同位点的近邻不同 → 有效氧化态不同
→ 分别推荐替代元素。

用法:
  python replace.py POSCAR --element V        # 自动推荐 V 的替代元素（分环境）
  python replace.py POSCAR --element Cl       # 自动推荐 Cl 的替代元素（分环境）
  python replace.py POSCAR --site 1 3 5       # 按位点编号（传统模式）
  python replace.py POSCAR --element V --candidates Nb,Ta,Mo,W,Cr   # 指定候选
  python replace.py POSCAR --element V --ox-state 4 --top 10        # 指定氧化态
  python replace.py POSCAR --element V --output-dir ./candidates    # 输出 POSCAR
"""

import argparse
import json
import os
import sys
from math import exp
from dataclasses import dataclass, asdict, field
from typing import List, Optional

import numpy as np

# ── Dual-channel: pymatgen or local fallback ──
_HAS_PYMATGEN = False
try:
    from pymatgen.core import Element, Structure
    from pymatgen.io.vasp import Poscar
    _HAS_PYMATGEN = True
except Exception:
    try:
        from element_data import ELEMENT_DATA
    except ImportError:
        print("需要 pymatgen 或 element_data.py", file=sys.stderr)
        sys.exit(1)

    class Element:
        """Minimal Element class backed by embedded data."""
        __slots__ = ('_s', '_d')
        def __init__(self, symbol):
            d = ELEMENT_DATA.get(symbol)
            if d is None:
                raise ValueError(f"Unknown element: {symbol}")
            self._s = symbol
            self._d = d
        @property
        def symbol(self): return self._s
        @property
        def atomic_radius(self): return self._d['atomic_radius']
        @property
        def X(self): return self._d['X']
        @property
        def group(self): return self._d['group']
        @property
        def block(self): return self._d['block']
        @property
        def row(self): return self._d['row']
        @property
        def common_oxidation_states(self): return self._d['common_oxidation_states']
        @property
        def ionic_radii(self): return self._d['ionic_radii']

    class SimpleStructure:
        """Pure-Python structure with POSCAR I/O and neighbor analysis."""

        class _Lattice:
            def __init__(self, matrix):
                self._matrix = np.array(matrix, dtype=float)
                self.a = float(np.linalg.norm(self._matrix[0]))
                self.b = float(np.linalg.norm(self._matrix[1]))
                self.c = float(np.linalg.norm(self._matrix[2]))
            @property
            def matrix(self): return self._matrix

        class _Site:
            def __init__(self, symbol, cart_coord, frac_coord):
                self.species = {Element(symbol): 1.0}
                self.coords = cart_coord
                self.frac_coords = frac_coord

        class _Composition:
            def __init__(self, species_list):
                from collections import Counter
                self._cnt = Counter(species_list)
            def __getitem__(self, sym): return self._cnt.get(sym, 0)
            def items(self): return self._cnt.items()
            def as_dict(self): return dict(self._cnt)

            @property
            def reduced_formula(self):
                els = sorted(self._cnt.keys())
                return "".join(f"{e}{c}" if c > 1 else e for e, c in zip(els, [self._cnt[e] for e in els]))

        def __init__(self, lattice, species, frac_coords):
            self._lattice = np.array(lattice, dtype=float)
            self._species = list(species)
            self._frac = np.array(frac_coords, dtype=float) % 1.0
            self._cart = self._frac @ self._lattice
            self._n = len(self._species)

        @property
        def lattice(self): return self._Lattice(self._lattice)

        @property
        def composition(self): return self._Composition(self._species)

        def __getitem__(self, idx):
            return self._Site(self._species[idx], self._cart[idx], self._frac[idx])

        def __len__(self): return self._n

        def __iter__(self):
            for i in range(self._n):
                yield self[i]

        def copy(self):
            return SimpleStructure(self._lattice.copy(), self._species[:], self._frac.copy())

        def __setitem__(self, idx, val):
            """Set species at index idx. val should be {Element: 1.0}."""
            if isinstance(val, dict):
                self._species[idx] = list(val.keys())[0].symbol
            else:
                self._species[idx] = str(val)

        @property
        def sites(self):
            return [self[i] for i in range(self._n)]

        def get_neighbors(self, site, r):
            q = site.frac_coords
            neigh = []
            for i in range(self._n):
                d = self._frac[i] - q
                d -= np.round(d)
                dist = float(np.linalg.norm(d @ self._lattice))
                if 0.01 < dist <= r:
                    neigh.append(dist)
            return neigh

    # Alias SimpleStructure → Structure so rest of code works unchanged
    Structure = SimpleStructure

    def _parse_poscar_block(path):
        """Read POSCAR into a dict compatible with Structure.from_file."""
        with open(path) as f:
            raw = f.readlines()
        scale = float(raw[1].strip())
        lat = np.array([list(map(float, raw[i].strip().split()[:3])) for i in range(2, 5)])
        if scale < 0:
            vol = abs(scale)
            cur_vol = float(np.linalg.det(lat))
            lat *= (vol / cur_vol) ** (1.0 / 3.0)
        elif scale != 1.0:
            lat *= scale

        line6 = raw[5].strip().split()
        line7 = raw[6].strip().split()
        if line6[0].isdigit():
            counts = list(map(int, line6))
            symbols = [f"X{i+1}" for i in range(len(counts))]
            coord_start = 6
        else:
            symbols = line6
            counts = list(map(int, line7))
            coord_start = 7

        species = []
        for sym, cnt in zip(symbols, counts):
            species.extend([sym] * cnt)

        # Skip Selective dynamics line if present
        idx = coord_start
        if raw[idx].strip().lower().startswith('s'):
            idx += 1

        coord_type = raw[idx].strip().lower()
        idx += 1
        n_atoms = len(species)
        frac = np.zeros((n_atoms, 3))
        for i in range(n_atoms):
            parts = raw[idx + i].strip().split()
            frac[i] = list(map(float, parts[:3]))
        if coord_type in ('cartesian', 'cart'):
            frac = np.linalg.solve(lat.T, frac.T).T % 1.0

        return lat.tolist(), species, frac.tolist()

    # Monkey-patch Structure.from_file
    @staticmethod
    def _structure_from_file(path):
        lat, species, frac = _parse_poscar_block(path)
        return SimpleStructure(lat, species, frac)

    Structure.from_file = _structure_from_file

    def _write_poscar(struct, path):
        """Fallback POSCAR writer when pymatgen is unavailable."""
        lat = struct._lattice
        species = struct._species
        frac = struct._frac
        # Build unique species in order of first appearance
        seen = []
        for s in species:
            if s not in seen:
                seen.append(s)
        counts = [species.count(s) for s in seen]
        with open(path, "w") as f:
            f.write("Generated by replace.py (fallback)\n")
            f.write("1.0\n")
            for row in lat:
                f.write(f"  {row[0]:20.15f} {row[1]:20.15f} {row[2]:20.15f}\n")
            f.write("  ".join(seen) + "\n")
            f.write("  ".join(str(c) for c in counts) + "\n")
            f.write("Direct\n")
            for row in frac:
                f.write(f"  {row[0]:20.15f} {row[1]:20.15f} {row[2]:20.15f}\n")

    Poscar = None  # marker for write_poscars to use fallback

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

# 强阴性元素固定价态（用于电荷平衡）
ANION_FIXED = {
    "F": -1, "Cl": -1, "Br": -1, "I": -1,
    "O": -2, "S": -2, "Se": -2, "Te": -2,
    "N": -3, "P": -3, "As": -3,
}

NEIGHBOR_CUTOFF = 3.5


# ═══════════════════════════════════════════════════════════════
#  Scoring — 五项评分
# ═══════════════════════════════════════════════════════════════

def _get_ionic_radius(el, ox, cn):
    radii = el.ionic_radii
    if not radii:
        return None
    if ox is not None:
        # Exact match
        if ox in radii:
            return float(radii[ox])
        # Fractional: interpolate between nearest integer states
        int_ox = [int(k) for k in radii.keys()]
        if int_ox and min(int_ox) <= ox <= max(int_ox):
            below = max(k for k in int_ox if k <= ox)
            above = min(k for k in int_ox if k >= ox)
            if below == above:
                return float(radii[str(below)] if isinstance(list(radii.keys())[0], str) else radii[below])
            r_below = float(radii[str(below)] if isinstance(list(radii.keys())[0], str) else radii[below])
            r_above = float(radii[str(above)] if isinstance(list(radii.keys())[0], str) else radii[above])
            t = (ox - below) / (above - below)
            return r_below + t * (r_above - r_below)
        return None
    return None


def _fractional_charge_balance(composition, target_sym):
    """Solve for target element's fractional oxidation state by charge balance.

    composition: dict {sym: count} — full composition
    target_sym: the element to solve for

    Fixed: anions (O=-2, N=-3, halogens=-1), non-target cations use common_ox_states[0].
    Returns: float — the target's effective oxidation state (fractional if needed).
    """
    # 目标元素如果是已知阴离子，直接取固定值
    # （N、O、卤素等在固体中氧化态几乎不变）
    if target_sym in ANION_FIXED:
        return ANION_FIXED[target_sym]

    total_charge = 0.0
    for sym, count in composition.items():
        if sym == target_sym:
            continue
        if sym in ANION_FIXED:
            ox = ANION_FIXED[sym]
        else:
            el = Element(sym)
            ox = el.common_oxidation_states[0] if el.common_oxidation_states else 0
        total_charge += count * ox
    # Solve: total_charge + count * x = 0
    target_count = composition[target_sym]
    x = -total_charge / target_count
    return round(x, 2)


def guess_oxidation(structure, site_idx=None):
    """Determine likely oxidation state, supporting fractional values.

    Priority:
      1. pymatgen's add_oxidation_state_by_guess()
      2. Fractional charge balance from full composition
      3. common_oxidation_states[0] fallback
    """
    sp_sym = None
    # Try single site approach (priority 1)
    if site_idx is not None:
        sp_sym = list(structure[site_idx].species.keys())[0].symbol
        try:
            s = structure.copy()
            s.add_oxidation_state_by_guess()
            for p in s[site_idx].species:
                if p.symbol == sp_sym:
                    return float(p.oxi_state)
        except Exception:
            pass

    if sp_sym is None:
        sp_sym = list(structure[0].species.keys())[0].symbol

    # Priority 2: fractional charge balance
    try:
        comp = structure.composition
        comp_dict = {}
        for el, amt in comp.items():
            sym = el.symbol if hasattr(el, "symbol") else str(el)
            comp_dict[sym] = int(amt)
        return _fractional_charge_balance(comp_dict, sp_sym)
    except Exception:
        pass

    # Priority 3: most common state
    try:
        el = Element(sp_sym)
        if el.common_oxidation_states:
            return float(el.common_oxidation_states[0])
    except Exception:
        pass
    return None


def score_radius(host_el, cand_el, ox, cn):
    """离子半径兼容性。

    保证比较在同一种半径类型上进行：
      - 双方都有目标价态的离子半径 → 用离子半径比
      - 任何一方缺失 → 都用共价半径比（避免 N³⁻ 1.32 vs P 原子 1.00 这种不一致）
    """
    rh_ion = _get_ionic_radius(host_el, ox, cn)
    rc_ion = _get_ionic_radius(cand_el, ox, cn)
    if rh_ion is not None and rc_ion is not None:
        rh, rc = rh_ion, rc_ion
    else:
        rh = host_el.atomic_radius
        rc = cand_el.atomic_radius
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
    """氧化态兼容性（支持分数氧化态）。

    如果候选元素的任一经验价态接近目标价态 → 高分。
    经验价态越常见（排越前）应得越高是错的，排后面的也是合理价态。
    """
    css = cand_el.common_oxidation_states
    if not css:
        return 0.30
    if target_ox is not None:
        exact_match = any(abs(s - target_ox) < 0.5 for s in css)
        if exact_match:
            return 1.0
        closest = min(css, key=lambda s: abs(s - target_ox))
        diff = abs(closest - target_ox)
        if diff <= 1.0:
            return 0.85
        if diff <= 2.0:
            return 0.60
        return 0.30
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

def replace(structure, site_idx, host_sym, ox, candidates, weights, cn):
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
            note_parts.append("同族")
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
#  Site classification by coordination environment
# ═══════════════════════════════════════════════════════════════

@dataclass
class SiteGroup:
    label: str
    sites: List[int]
    neighbor_counts: dict
    avg_ox: float
    avg_cn: float


def _signature(neighbor_counts):
    """Create a canonical label from neighbor counts, e.g. 'V1W3'."""
    keys = sorted(neighbor_counts.keys())
    return "".join(f"{k}{neighbor_counts[k]}" for k in keys)


def classify_sites(struct, element_sym, cutoff=NEIGHBOR_CUTOFF):
    """Group sites of element_sym by their nearest-neighbor coordination.

    Returns list of SiteGroup objects, one per distinct environment.
    """
    from itertools import product

    sites_of_el = [i for i, site in enumerate(struct)
                   if any(sp.symbol == element_sym for sp in site.species)]
    if not sites_of_el:
        return []

    # Pre-compute Cartesian coordinates + lattice for PBC
    lattice = struct.lattice.matrix
    frac_all = [site.frac_coords for site in struct]
    cart_all = [site.coords for site in struct]
    symbols_all = [list(site.species.keys())[0].symbol for site in struct]

    groups = {}  # signature -> {"indices": [], "ncounts": {}}
    for idx in sites_of_el:
        pos = cart_all[idx]
        counts = {}
        for j, pos_j in enumerate(cart_all):
            if j == idx:
                continue
            frac_d = [frac_all[j][k] - frac_all[idx][k] for k in range(3)]
            frac_d = [f - round(f) for f in frac_d]
            cart_d = [sum(frac_d[k] * lattice[k][i] for k in range(3)) for i in range(3)]
            dist = sum(d * d for d in cart_d) ** 0.5
            if dist < cutoff:
                sym = symbols_all[j]
                # For signature, only count DIFFERENT element neighbors
                # (we care about coordination by other elements)
                counts[sym] = counts.get(sym, 0) + 1

        # Remove the target element itself from neighbor counts
        if element_sym in counts:
            del counts[element_sym]

        sig = _signature(counts)
        if sig not in groups:
            groups[sig] = {"indices": [], "ncounts": counts}
        groups[sig]["indices"].append(idx)

    # Global charge balance for the target element's oxidation state
    comp_dict = {}
    for el_sym, count in struct.composition.items():
        s = el_sym if isinstance(el_sym, str) else str(el_sym)
        comp_dict[s] = int(count)

    global_ox = _fractional_charge_balance(comp_dict, element_sym)

    result = []
    for sig, data in groups.items():
        # Coordination number = total neighbor count for this environment
        avg_cn = sum(data["ncounts"].values())

        result.append(SiteGroup(
            label=f"{element_sym}@{sig}",
            sites=data["indices"],
            neighbor_counts=data["ncounts"],
            avg_ox=global_ox,
            avg_cn=round(avg_cn, 1),
        ))

    return result


def _local_ox(sym):
    """Default oxidation for a neighbor element in local charge balance."""
    if sym in ANION_FIXED:
        return ANION_FIXED[sym]
    try:
        el = Element(sym)
        if el.common_oxidation_states:
            return el.common_oxidation_states[0]
        return 0
    except Exception:
        return 0
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
        if Poscar is not None:
            Poscar(s).write_file(path)
        else:
            _write_poscar(s, path)
        paths.append(path)
    return paths


# ═══════════════════════════════════════════════════════════════
#  Formatting
# ═══════════════════════════════════════════════════════════════

def table(hits, top_n, header_prefix=""):
    """显示格式化的结果表格。"""
    SEP = "-" * 95
    lines = []
    if header_prefix:
        lines.append(header_prefix)
    fmt = (
        "{rank:<4} {el:<7} {total:<7} {ec:<7} {r:<7} {en:<7} {os:<6} "
        "{cgroup:<6} {cblock:<6} note"
    )
    lines.append(fmt.format(
        rank="Rank", el="元素", total="总分", ec="电子",
        r="半径", en="电负", os="氧化", cgroup="族号", cblock="区块"
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
        description="2D Material Element Substitution Replacer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("poscar", help="POSCAR 文件路径")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--site", type=int, nargs="+",
                   help="要替换的位点编号（从 1 开始）")
    g.add_argument("--element", type=str,
                   help="要替换的元素符号，自动推荐替代候选")

    ap.add_argument("--candidates", default="",
                    help="候选列表：逗号分隔 或 内置组名 "
                         f"({', '.join(GROUPS)})")
    ap.add_argument("--exclude", default="",
                    help="排除的元素（逗号分隔）")
    ap.add_argument("--ox-state", type=float,
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

    fmt = struct.composition.reduced_formula

    # ── 交互模式（未指定 --element 或 --site） ──
    if not args.element and not args.site:
        all_eles = sorted(set(
            list(site.species.keys())[0].symbol for site in struct
        ))
        comp_dict = {}
        for el_sym, cnt in struct.composition.items():
            s = el_sym if isinstance(el_sym, str) else str(el_sym)
            comp_dict[s] = int(cnt)

        # Pre-compute environment map per element
        env_map = {}
        for el in all_eles:
            groups = classify_sites(struct, el)
            if groups:
                env_map[el] = groups

        print()
        print("=" * 60)
        print(f"  文件: {args.poscar}")
        print(f"  成分: {fmt}")
        print(f"  原子: {', '.join(all_eles)}")
        print(f"  数量: {comp_dict}")
        print("=" * 60)
        print()

        # 按配位环境分组显示原子坐标
        print("  原子坐标 (按环境分组):")
        for el in sorted(env_map.keys()):
            for env_idx, g in enumerate(env_map[el], 1):
                print(f"  [{el}{env_idx}] {g.label} ({len(g.sites)} 个位点):")
                for site_idx in g.sites:
                    site = struct[site_idx]
                    sym = list(site.species.keys())[0].symbol
                    x, y, z = site.coords
                    print(f"      位点 {site_idx+1:<3} {sym:<3}"
                          f" {x:>10.6f} {y:>10.6f} {z:>10.6f}")
        print()

        print("  配位环境说明:")
        for el in sorted(env_map.keys()):
            for env_idx, g in enumerate(env_map[el], 1):
                site_nums = [s + 1 for s in g.sites]
                print(f"    {el}{env_idx}={g.label}: 位点 {site_nums}, "
                      f"近邻 {dict(g.neighbor_counts)}, "
                      f"氧化态 {g.avg_ox:.2f}, CN={g.avg_cn}")
        print()

        # 解析输入:
        #   N  = 替换全部 N（多环境加权综合排名）
        #   N1 = 只替换 N 的环境 1
        inp = input("  输入元素(全部) 或 元素+环境编号(单个环境): ").strip()
        if not inp:
            print("未输入", file=sys.stderr)
            sys.exit(1)

        el_only = inp.rstrip("0123456789")
        num_part = inp[len(el_only):]
        el_only = el_only.strip()

        if num_part:
            if el_only:
                # 元素 + 编号 → 环境模式 (如 N1 = N 的环境 1)
                if el_only in env_map:
                    env_num = int(num_part)
                    if 1 <= env_num <= len(env_map[el_only]):
                        g = env_map[el_only][env_num - 1]
                        args.site = [s + 1 for s in g.sites]
                    else:
                        print(f"错误: {el_only} 只有 "
                              f"{len(env_map[el_only])} 个环境 "
                              f"(1-{len(env_map[el_only])})",
                              file=sys.stderr)
                        sys.exit(1)
                else:
                    print(f"错误: 未找到元素 {el_only}",
                          file=sys.stderr)
                    sys.exit(1)
        else:
            # 纯元素 → 全部替换（加权综合）
            args.element = el_only

    # ── 交互式输入氧化态 ──
    if args.element and args.ox_state is None:
        try:
            inp_ox = input("  氧化态 (直接回车=自动计算): ").strip()
            if inp_ox:
                args.ox_state = float(inp_ox)
        except (EOFError, KeyboardInterrupt):
            pass
        except ValueError:
            args.ox_state = None  # 非数值 → 自动计算

    # ── 确定替换位点 ──
    if args.site:
        # 按位点：传统模式，单组
        sites = [i - 1 for i in args.site]
        for s in sites:
            if not (0 <= s < len(struct)):
                print(f"位点 {s+1} 超出范围 (1–{len(struct)})", file=sys.stderr)
                sys.exit(1)
        host_sym = list(struct[sites[0]].species.keys())[0].symbol
        host_el = Element(host_sym)

        ox = args.ox_state or guess_oxidation(struct, sites[0])
        cn = args.coord or block_cn(host_el.block)

        if args.candidates:
            cands = resolve_candidates(args.candidates)
        else:
            cands = auto_candidates(host_sym)
            print(f"  (自动生成了 {len(cands)} 个候选)")
            print()
        if args.exclude:
            xs = set(s.strip() for s in args.exclude.split(","))
            cands = [c for c in cands if c not in xs]
        cands = [c for c in cands if c != host_sym]

        w = args.weights
        hits = replace(struct, sites[0], host_sym, ox, cands, w, cn)

        print()
        print("=" * 65)
        print(f"  元素替换推荐 — {host_sym} in {fmt}")
        print("=" * 65)
        print(f"  POSCAR   : {args.poscar}")
        print(f"  位点     : {[s+1 for s in sites]}")
        print(f"  替换     : {host_sym}")
        print(f"  电子构型 : {host_el.block}-区 第{host_el.group}族")
        print(f"  氧化态   : {ox or '自动检测失败(用 --ox-state 指定)'}")
        print(f"  配位数   : {cn}")
        print()
        print(table(hits, args.top))
        print()

        if args.output_dir:
            paths = write_poscars(struct, sites, hits, args.top,
                                  args.output_dir, host_sym)
            print(f"  -> {len(paths)} 个 POSCAR 文件写入 {args.output_dir}/")
            for p in paths:
                print(f"    {p}")
            print()

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

    else:
        # ── 按元素：按配位环境分组 ──
        host_sym = args.element.strip()
        host_el = Element(host_sym)

        groups = classify_sites(struct, host_sym)
        if not groups:
            print(f"错误: POSCAR 中没有元素 {host_sym}", file=sys.stderr)
            sys.exit(1)

        print()
        print("=" * 55)
        print(f"  元素替换推荐 — {host_sym} in {fmt}")
        print("=" * 55)
        print(f"  检测到 {len(groups)} 种配位环境:")
        for env_idx, g in enumerate(groups, 1):
            print(f"    {host_sym}{env_idx}={g.label}: {len(g.sites)} 个位点, "
                  f"配位 {dict(g.neighbor_counts)}, "
                  f"有效氧化态 {g.avg_ox:.2f}, 平均配位数 {g.avg_cn}")
        print()

        # 候选列表（所有环境共用）
        if args.candidates:
            cands = resolve_candidates(args.candidates)
        else:
            cands = auto_candidates(host_sym)
            print(f"  (自动生成了 {len(cands)} 个候选)")
            print()
        if args.exclude:
            xs = set(s.strip() for s in args.exclude.split(","))
            cands = [c for c in cands if c not in xs]
        cands = [c for c in cands if c != host_sym]

        w = args.weights
        all_hits = {}
        for g in groups:
            ox = args.ox_state or g.avg_ox
            cn = args.coord or g.avg_cn or block_cn(host_el.block)
            hits = replace(struct, g.sites[0], host_sym, ox, cands, w, cn)
            all_hits[g.label] = (hits, ox, cn)

        # ── 输出：每个环境单独表格 ──
        for env_idx, g in enumerate(groups, 1):
            hits_g, ox_g, cn_g = all_hits[g.label]
            site_list = [s + 1 for s in g.sites]
            prefix = (
                f"  ── {host_sym}{env_idx}={g.label} (位点 {site_list}) ──\n"
                f"  配位: {dict(g.neighbor_counts)}  "
                f"氧化态: {ox_g:.2f}  配位数: {cn_g}"
            )
            print(table(hits_g, args.top, prefix))
            print()
            print()

        # ── 加权综合排名（多环境加权平均） ──
        if len(groups) > 1:
            total_sites = sum(len(g.sites) for g in groups)
            # 计算每个候选的综合加权分数
            combined = {}
            for g in groups:
                hits_g = all_hits[g.label][0]
                weight = len(g.sites) / total_sites
                for h in hits_g:
                    sym = h.symbol
                    if sym not in combined:
                        combined[sym] = {
                            "total": 0.0, "electronic": 0.0, "radius": 0.0,
                            "en": 0.0, "os": 0.0, "cand_group": h.cand_group,
                            "cand_block": h.cand_block, "note": h.note
                        }
                    combined[sym]["total"] += h.total * weight
                    combined[sym]["electronic"] += h.electronic * weight
                    combined[sym]["radius"] += h.radius * weight
                    combined[sym]["en"] += h.en * weight
                    combined[sym]["os"] += h.os * weight

            # 排序并输出
            ranked = sorted(combined.items(), key=lambda x: x[1]["total"], reverse=True)
            SEP = "-" * 95
            print("  " + "=" * 50)
            print("  加权综合排名 (按位点数量加权)")
            print("  " + "=" * 50)
            for env_idx, g in enumerate(groups, 1):
                print(f"    {host_sym}{env_idx}={g.label}: {len(g.sites)}/{total_sites} = "
                      f"{len(g.sites)/total_sites:.2f}")
            print()
            lines = ["  加权综合:"]
            fmt = "{rank:<4} {el:<7} {total:<7} {ec:<7} {r:<7} {en:<7} {os:<6}"
            lines.append(fmt.format(
                rank="Rank", el="元素", total="总分", ec="电子",
                r="半径", en="电负", os="氧化"
            ))
            lines.append(SEP)
            for i, (sym, d) in enumerate(ranked[:args.top], 1):
                lines.append(
                    f"{i:<4} {sym:<7} {d['total']:<7.3f} {d['electronic']:<7.3f} "
                    f"{d['radius']:<7.3f} {d['en']:<7.3f} {d['os']:<6.2f}"
                )
            print("\n".join(lines))
            print()
            print()

        # ── 输出 POSCAR ──
        if args.output_dir:
            for g in groups:
                hits_g = all_hits[g.label][0]
                paths = write_poscars(struct, g.sites, hits_g, args.top,
                                      os.path.join(args.output_dir, g.label), host_sym)
                print(f"  -> {len(paths)} 个 POSCAR 文件写入 {args.output_dir}/{g.label}/")
                for p in paths:
                    print(f"    {p}")
                print()

        # ── JSON ──
        if args.json:
            combined = {
                "poscar": args.poscar, "formula": fmt,
                "host": host_sym, "block": host_el.block,
                "group": host_el.group,
                "num_candidates": len(cands),
                "top_n": args.top,
                "weights": dict(zip(
                    ["radius","en","os","struct","electronic"], w)),
                "environments": [],
            }
            for g in groups:
                hits_g, ox_g, cn_g = all_hits[g.label]
                combined["environments"].append({
                    "label": g.label,
                    "sites": [s + 1 for s in g.sites],
                    "neighbor_counts": g.neighbor_counts,
                    "oxidation": ox_g,
                    "coordination": cn_g,
                    "hits": [{"rank": i + 1, **h.as_dict}
                             for i, h in enumerate(hits_g[:args.top])],
                })
            with open(args.json, "w") as f:
                json.dump(combined, f, indent=2, ensure_ascii=False)
            print(f"  -> JSON 结果保存至 {args.json}")
            print()


if __name__ == "__main__":
    main()
