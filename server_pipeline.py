#!/usr/bin/env python3
"""
VASP + phonopy 自动化流水线（服务器端）
========================================
从启发式筛选结果或自定义方案出发，自动生成 VASP 弛豫 + 声子计算作业。

用法:

  1. 生成候选 POSCAR:
     python screen.py POSCAR --element V --candidates Nb,Ta --output-dir ./candidates

  2. 为候选生成 VASP 作业:
     python server_pipeline.py setup ./candidates -r REF_DIR -o ./jobs

  3. 分析声子结果:
     python server_pipeline.py analyze ./phonon_dirs

快速示例:
  python server_pipeline.py setup ./candidates -r  ./ref_opt -p ./ref_phonon -o ./msx_jobs

依赖:
  - Python 3.8+, pymatgen
  - VASP, phonopy (服务器上)
"""

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from string import Template

try:
    from pymatgen.core import Structure
    from pymatgen.io.vasp import Poscar, Incar, Kpoints
except ImportError:
    print("Need pymatgen: pip install pymatgen", file=sys.stderr)
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════
#  Templates
# ═══════════════════════════════════════════════════════════════

INCAR_RELAX_TEMPLATE = """ICHARG = 2
ISTART = 0
ISYM   = 2
GGA    = PE
ISPIN  = 2
PREC   = Normal
ENCUT  = 500
ALGO   = Fast
EDIFF  = 1E-4
EDIFFG = -0.01
LREAL  = Auto
ISIF   = 3
NELM   = 200
NELMIN = 5
NSW    = 1000
IBRION = 2
ISMEAR = 0
SIGMA  = 0.05
NWRITE = 2
LWAVE  = .FALSE.
LCHARG = .FALSE.
"""

INCAR_PHONON_TEMPLATE = """ICHARG = 2
ISTART = 0
ISYM   = 0
GGA    = PE
ISPIN  = 2
PREC   = Accurate
ENCUT  = 500
ALGO   = Fast
EDIFF  = 1E-6
EDIFFG = -0.001
LREAL  = Auto
ISIF   = 2
NELM   = 1000
NELMIN = 5
NSW    = 0
IBRION = -1
ADDGRID = .TRUE.
ISMEAR = 0
SIGMA  = 0.05
NWRITE = 2
LWAVE  = .FALSE.
LCHARG = .FALSE.
"""

KPOINTS_TEMPLATE = """Automatic mesh
0
Gamma
1 1 1
"""

PBS_RELAX_TEMPLATE = Template("""#!/bin/bash
#PBS -N relax-${label}
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=20
#PBS -q batch
#PBS -j oe
#PBS -o script.out
#PBS -l walltime=999:00:00

cd $$PBS_O_WORKDIR
NP=4
ulimit -s unlimited

mpirun --mca btl sm,self -n $$NP vasp541-gpu &> vasp.out
echo "Relax done for ${label}"
""")

PBS_PHONON_TEMPLATE = Template("""#!/bin/bash
#PBS -N phonon-${label}
#PBS -S /bin/bash
#PBS -l nodes=1:ppn=20
#PBS -q batch
#PBS -j oe
#PBS -o script.out
#PBS -l walltime=999:00:00

cd $$PBS_O_WORKDIR
NP=4
ulimit -s unlimited

# phonopy finite displacement
# assumes POSCAR is the relaxed supercell
# adjust DIM as needed

# Step 1: generate displacements
phonopy -d --dim "2 2 1" -c POSCAR

# Step 2: run VASP for each displaced POSCAR
ndisp=$$(ls POSCAR-* | wc -l)
for i in $$(seq 1 $${ndisp}); do
  dir="$${i}"
  mkdir -p $${dir}
  cp INCAR KPOINTS POTCAR $${dir}/
  cp "POSCAR-$$(printf '%03d' $${i})" $${dir}/POSCAR
  cd $${dir}
  mpirun --mca btl sm,self -n $$NP vasp541-gpu &> vasp.out
  cd ..
done

# Step 3: collect forces
phonopy -f {1..$${ndisp}}/vasprun.xml

# Step 4: band structure
cat > band.conf << 'BANDEOF'
ATOM_NAME = ${species}
DIM = 2 2 1
BAND = 0.0 0.0 0.0  0.5 0.0 0.0  0.5 0.5 0.0  0.0 0.0 0.0
BAND_POINTS = 505
FORCE_CONSTANTS = WRITE
BANDEOF

phonopy -p -s band.conf

echo "Phonon done for ${label}"
""")


# ═══════════════════════════════════════════════════════════════
#  Utilities
# ═══════════════════════════════════════════════════════════════

def find_poscars(directory: str) -> list:
    """Find all POSCAR files in a directory."""
    paths = []
    for f in os.listdir(directory):
        full = os.path.join(directory, f)
        if os.path.isfile(full) and 'POSCAR' in f:
            paths.append(full)
    # Also check subdirectories
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f == 'POSCAR' or f.startswith('POSCAR_'):
                paths.append(os.path.join(root, f))
    return sorted(set(paths))


def get_atom_species(poscar_path: str) -> str:
    """Extract species line from POSCAR for band.conf."""
    try:
        s = Structure.from_file(poscar_path)
        species = []
        seen = set()
        for site in s:
            sp = list(site.species.keys())[0].symbol
            if sp not in seen:
                species.append(sp)
                seen.add(sp)
        return " ".join(species)
    except Exception:
        return ""


def compose_label(poscar_path: str) -> str:
    """Create a short label from the POSCAR filename."""
    name = os.path.splitext(os.path.basename(poscar_path))[0]
    # Extract meaningful part
    name = name.replace('POSCAR_', '').replace('POSCAR-', '')
    return name


def _parse_band_yaml(path):
    """
    Parse band.yaml with q-point tracking.
    Returns (freqs_by_q, nqpoint, natom, error).
    freqs_by_q: list of (q_position_index, freq_list) per q-point
    """
    try:
        import yaml
    except ImportError:
        return None, 0, "yaml not available (pip install pyyaml)"
    try:
        with open(path) as f:
            data = yaml.safe_load(f)
    except Exception as e:
        return None, 0, str(e)

    nqpoint = data.get('nqpoint', 0)
    natom = data.get('natom', 0)
    nbands = 3 * natom
    freqs_by_q = []
    for i, phonon in enumerate(data.get('phonon', [])):
        bands = phonon.get('band', [])
        qfreqs = [b['frequency'] for b in bands]
        freqs_by_q.append((i, qfreqs))
    return freqs_by_q, nqpoint, natom, None


def _parse_band_dat(path):
    """
    Parse band.dat with q-point tracking.
    Returns (freqs_by_q, nqpoint, None|error).
    band.dat format: q_distance freq1 freq2 ... freqN
    or single-column: q_distance freq
    """
    freqs_by_q = []
    try:
        with open(path) as f:
            lines = [l.strip() for l in f
                     if not l.startswith('#') and l.strip()]
    except Exception as e:
        return None, 0, str(e)

    # band.dat: each line = qdist freq
    if not lines:
        return None, 0, "empty band.dat"

    # Group by q-point: assume frequencies repeat in same order per q
    # Actually band.dat has one frequency per line, not grouped by q
    # Let's just treat it as a flat list
    freqs = []
    for line in lines:
        parts = line.split()
        if len(parts) >= 2:
            try:
                freqs.append(float(parts[1]))
            except ValueError:
                continue

    if not freqs:
        return None, 0, "no parseable data"

    return [(0, freqs)], 1, None


def check_phonon_stability(phonon_dir):
    """
    Analyze phonon results for imaginary modes.

    Core principle (from experimental experience):
    - Gamma-point imaginary modes → can be corrected
      (acoustic sum rule / rotational sum rules via hiphive)
      → considered STABLE
    - Imaginary modes at ANY non-Gamma q-point → REAL instability
    - The magnitude of Gamma-point imaginary modes is irrelevant
    - The number of imaginary mode points is irrelevant as long as
      they are all concentrated at Gamma

    Returns:
        dict with stability analysis
    """
    dir_path = Path(phonon_dir)
    result = {
        "dir": str(dir_path),
        "min_freq_thz": None,
        "stable": None,
        "imaginary_modes": 0,
        "gamma_imaginary": False,
        "non_gamma_imaginary": False,
        "gamma_freq_thz": None,
        "min_non_gamma_freq": None,
        "source": None,
        "natom": 0,
    }

    # Find data sources
    sources = []
    for fn in ("band.yaml", "band.dat"):
        p = dir_path / fn
        if p.exists():
            sources.append(p)

    if not sources:
        result["stable"] = None
        result["error"] = "no band.yaml or band.dat found"
        return result

    # Parse data
    freqs_by_q = None
    nqpoint = 0
    natom = 0
    for src in sources:
        if src.suffix == ".yaml":
            freqs_by_q, nqpoint, natom, err = _parse_band_yaml(str(src))
            result["source"] = "band.yaml"
        else:
            freqs_by_q, nqpoint, err = _parse_band_dat(str(src))
            natom = 0
            result["source"] = "band.dat"
        if freqs_by_q and err is None:
            break

    if freqs_by_q is None or not freqs_by_q:
        result["error"] = err or "no parseable data"
        return result

    result["natom"] = natom

    # Analyze per q-point: which q-points have imaginary modes?
    gamma_has_imag = False
    non_gamma_has_imag = False
    gamma_min = None
    non_gamma_min = None
    total_imag = 0
    gamma_imag_count = 0
    non_gamma_imag_count = 0

    for qi, qfreqs in freqs_by_q:
        negs = [f for f in qfreqs if f < 0]
        n_neg = len(negs)
        if n_neg == 0:
            continue
        total_imag += n_neg
        min_f = min(negs)

        if qi == 0:
            # q = 0 (Gamma)
            gamma_has_imag = True
            gamma_min = min(gamma_min, min_f) if gamma_min is not None else min_f
            gamma_imag_count = n_neg
        else:
            # q > 0 (non-Gamma)
            non_gamma_has_imag = True
            non_gamma_min = min(non_gamma_min, min_f) if non_gamma_min is not None else min_f
            non_gamma_imag_count += n_neg

    result["imaginary_modes"] = total_imag
    if gamma_has_imag:
        result["gamma_imaginary"] = True
        result["gamma_freq_thz"] = round(gamma_min, 4)
        result["min_freq_thz"] = round(gamma_min, 4)

    if non_gamma_has_imag:
        result["non_gamma_imaginary"] = True
        result["min_non_gamma_freq"] = round(non_gamma_min, 4)
        result["min_freq_thz"] = min(
            result.get("min_freq_thz") or 0, non_gamma_min)
        result["min_freq_thz"] = round(result["min_freq_thz"], 4)

    # ── Decision (based on user's experimental experience) ──
    if not gamma_has_imag and not non_gamma_has_imag:
        # No imaginary modes at all → perfectly stable
        result["stable"] = True
    elif gamma_has_imag and not non_gamma_has_imag:
        # Imaginary modes ONLY at Gamma → correctable artifact
        # Even large magnitude (-6 THz) or many modes are fine
        # as long as they're ONLY at Gamma
        result["stable"] = True
        result["gamma_only_artifact"] = True
    elif non_gamma_has_imag:
        # Imaginary modes exist away from Gamma → REAL instability
        # These CANNOT be fixed by ASR / rotational sum rule correction
        result["stable"] = False
    else:
        result["stable"] = None

    return result


# ═══════════════════════════════════════════════════════════════
#  Setup: generate VASP jobs from candidate POSCARs
# ═══════════════════════════════════════════════════════════════

def cmd_setup(args):
    """Generate VASP relax + phonon job directories from candidate POSCARs."""
    cand_dir = args.candidates
    poscars = find_poscars(cand_dir)

    if not poscars:
        print(f"No POSCAR files found in {cand_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(poscars)} candidate POSCARs in {cand_dir}")
    print()

    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)
    ref_relax = args.ref_relax
    ref_phonon = args.ref_phonon

    # Read reference POTCAR
    potcar_path = None
    if ref_relax and os.path.exists(os.path.join(ref_relax, 'POTCAR')):
        potcar_path = os.path.join(ref_relax, 'POTCAR')
    elif os.path.exists(os.path.join(cand_dir, 'POTCAR')):
        potcar_path = os.path.join(cand_dir, 'POTCAR')

    job_summary = []

    for p in poscars:
        label = compose_label(p)
        job_dir = os.path.join(out_dir, label)
        os.makedirs(job_dir, exist_ok=True)

        # Relax directory
        relax_dir = os.path.join(job_dir, 'relax')
        os.makedirs(relax_dir, exist_ok=True)

        # Copy/modify POSCAR
        try:
            s = Structure.from_file(p)
            Poscar(s).write_file(os.path.join(relax_dir, 'POSCAR'))
        except Exception as e:
            print(f"  SKIP {label}: POSCAR error: {e}", file=sys.stderr)
            continue

        # INCAR
        with open(os.path.join(relax_dir, 'INCAR'), 'w') as f:
            f.write(INCAR_RELAX_TEMPLATE)

        # KPOINTS
        kp = Kpoints.from_str(KPOINTS_TEMPLATE)
        kp.write_file(os.path.join(relax_dir, 'KPOINTS'))

        # POTCAR
        if potcar_path:
            shutil.copy2(potcar_path, os.path.join(relax_dir, 'POTCAR'))

        # PBS script
        pbs_path = os.path.join(relax_dir, 'run_relax.pbs')
        with open(pbs_path, 'w') as f:
            f.write(PBS_RELAX_TEMPLATE.substitute(label=label))

        # Relax done. Generate phonon directories later
        job_summary.append({
            "label": label,
            "poscar": p,
            "relax_dir": relax_dir,
            "has_potcar": potcar_path is not None,
        })
        print(f"  [OK] {label} -> {relax_dir}")

    # Write summary
    summary_path = os.path.join(out_dir, 'job_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(job_summary, f, indent=2)

    print()
    print(f"Created {len(job_summary)} job directories in {out_dir}/")
    print(f"Summary saved to {summary_path}")
    print()

    if args.no_phonon:
        print("NOTE: phonon directories not generated (--no-phonon). Run phonon setup separately.")
    else:
        print("After relax completes, run:")
        print(f"  python server_pipeline.py phonon-setup {out_dir} --species '...'")
        print("to generate phonon job directories.")


def cmd_phonon_setup(args):
    """Generate phonon job directories from relaxed CONTCARs."""
    jobs_dir = args.jobs_dir
    species_override = args.species

    # Find all relax directories
    relax_dirs = sorted(glob.glob(os.path.join(jobs_dir, '*/relax')))
    if not relax_dirs:
        print(f"No relax directories found in {jobs_dir}/*/relax", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(relax_dirs)} relax directories")
    print()

    for rd in relax_dirs:
        label = os.path.basename(os.path.dirname(rd))

        # Source POSCAR: use CONTCAR if available, else POSCAR
        contcar = os.path.join(rd, 'CONTCAR')
        poscar_src = contcar if os.path.exists(contcar) else os.path.join(rd, 'POSCAR')

        if not os.path.exists(poscar_src):
            print(f"  SKIP {label}: no POSCAR/CONTCAR in {rd}", file=sys.stderr)
            continue

        phonon_dir = os.path.join(os.path.dirname(rd), 'phonon')
        os.makedirs(phonon_dir, exist_ok=True)

        # Copy structure
        shutil.copy2(poscar_src, os.path.join(phonon_dir, 'POSCAR'))

        # INCAR
        with open(os.path.join(phonon_dir, 'INCAR'), 'w') as f:
            f.write(INCAR_PHONON_TEMPLATE)

        # KPOINTS
        kp = Kpoints.from_str(KPOINTS_TEMPLATE)
        kp.write_file(os.path.join(phonon_dir, 'KPOINTS'))

        # POTCAR from relax dir
        potcar_src = os.path.join(rd, 'POTCAR')
        if os.path.exists(potcar_src):
            shutil.copy2(potcar_src, os.path.join(phonon_dir, 'POTCAR'))

        # Determine species
        if species_override:
            species = species_override
        else:
            species = get_atom_species(poscar_src)

        # PBS script — this one handles the full phonon workflow
        pbs_path = os.path.join(phonon_dir, 'run_phonon.pbs')
        with open(pbs_path, 'w') as f:
            f.write(PBS_PHONON_TEMPLATE.substitute(label=label, species=species))

        print(f"  [OK] {label} -> {phonon_dir}  (species: {species})")

    print()
    print("Done. Submit each phonon directory with:")
    print("  cd <dir>/phonon && qsub run_phonon.pbs")


# ═══════════════════════════════════════════════════════════════
#  Analyze: check phonon results
# ═══════════════════════════════════════════════════════════════

def cmd_analyze(args):
    """Analyze phonon results for imaginary modes."""
    base_dir = args.directory

    # Find phonon directories (contain band.yaml or band.dat)
    phonon_dirs = {}
    for root, dirs, files in os.walk(base_dir):
        for marker in ('band.yaml', 'band.dat'):
            if marker in files:
                phonon_dirs[root] = marker

    if not phonon_dirs:
        print(f"No phonon data (band.yaml / band.dat) found under {base_dir}",
              file=sys.stderr)
        sys.exit(1)

    results = []
    for d in sorted(phonon_dirs):
        label = os.path.basename(d)
        r = check_phonon_stability(d)
        r["label"] = label
        results.append(r)

    # Sort: unstable first
    def sort_key(r):
        if r["stable"] is False:
            return r.get("min_non_gamma_freq") or r.get("min_freq_thz") or 0
        return 9999

    results.sort(key=sort_key)

    # Count
    artifact_count = sum(1 for r in results if r.get("gamma_only_artifact"))
    stable_count = sum(1 for r in results if r["stable"] is True
                       and not r.get("gamma_only_artifact"))
    unstable_count = sum(1 for r in results if r["stable"] is False)
    unknown_count = sum(1 for r in results if r["stable"] is None)

    print()
    print("=" * 80)
    print("  Phonon Stability Analysis")
    print("=" * 80)
    print(f"  Scanned  : {len(results)}  |  STABLE: {stable_count}  "
          f"|  ARTIFACT: {artifact_count}  |  UNSTABLE: {unstable_count}")
    print()
    print("  Gamma-point imaginary modes = correctable artifact (hiphive ASR /")
    print("  rotational sum rule). Only non-Gamma imaginary = real instability.")
    print()
    print(f"  {'System':<25} {'Min(THz)':<12} {'Non-G':<10} {'#Imag':<8} Status")
    print("  " + "-" * 65)

    for r in results:
        label = r["label"][:24]
        min_f = f"{r['min_freq_thz']:.2f}" if r["min_freq_thz"] is not None else "N/A"
        ng = f"{r.get('min_non_gamma_freq', 0):.2f}" if r.get("non_gamma_imaginary") else "--"
        nimag = r.get("imaginary_modes", 0)
        if r["stable"] is False:
            st = "UNSTABLE"
        elif r.get("gamma_only_artifact"):
            st = "STABLE(Gamma-artifact)"
        elif r["stable"] is True:
            st = "STABLE"
        else:
            st = f"ERR: {r.get('error','?')[:20]}"
        print(f"  {label:<25} {min_f:<12} {ng:<10} {nimag:<8} {st}")

    print()

    unstable = [r for r in results if r["stable"] is False]
    if unstable:
        print("--- Unstable Systems (non-Gamma imaginary modes) ---")
        for r in unstable:
            print(f"  {r['label']}: min={r.get('min_non_gamma_freq','?'):} THz "
                  f"(non-Gamma), {r['imaginary_modes']} total imaginary modes")
        print()

    if args.json:
        with open(args.json, 'w') as f:
            json.dump({
                "total": len(results),
                "stable": stable_count,
                "artifact": artifact_count,
                "unstable": unstable_count,
                "gamma_artifact_note": "Gamma-point imaginary modes are correctable (ASR/rotational sum rules)",
                "results": results,
            }, f, indent=2)
        print(f"  Results saved to {args.json}")


# ═══════════════════════════════════════════════════════════════
#  Generate bulk substitution from pattern
# ═══════════════════════════════════════════════════════════════

def cmd_bulk(args):
    """Generate substituted POSCARs for a grid of sites + elements."""
    poscar_path = args.poscar
    if not os.path.exists(poscar_path):
        print(f"POSCAR not found: {poscar_path}", file=sys.stderr)
        sys.exit(1)

    try:
        struct = Structure.from_file(poscar_path)
    except Exception as e:
        print(f"Read error: {e}", file=sys.stderr)
        sys.exit(1)

    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    # Parse substitution patterns: "V@Nb,Ta" or "Cl@F,Br,I"
    patterns = args.pattern
    total = 0

    for pat in patterns:
        if '@' not in pat:
            print(f"  Invalid pattern: {pat} (expected e.g. V@Nb,Ta)", file=sys.stderr)
            continue
        host_sym, cand_list = pat.split('@', 1)
        host_sym = host_sym.strip()
        candidates = [c.strip() for c in cand_list.split(',') if c.strip()]

        # Find sites
        sites = [i for i, site in enumerate(struct)
                 if any(sp.symbol == host_sym for sp in site.species)]
        if not sites:
            print(f"  Element {host_sym} not found in POSCAR", file=sys.stderr)
            continue

        print(f"\n  {host_sym} -> {candidates} ({len(sites)} sites)")

        for cand in candidates:
            new_struct = struct.copy()
            for idx in sites:
                try:
                    from pymatgen.core import Element
                    new_struct[idx] = {Element(cand): 1.0}
                except Exception as e:
                    print(f"    Error substituting {cand}: {e}", file=sys.stderr)
                    continue

            fname = f"POSCAR_{host_sym}to{cand}"
            fpath = os.path.join(out_dir, fname)
            Poscar(new_struct).write_file(fpath)
            total += 1
            print(f"    -> {fname}")

    print(f"\nTotal: {total} POSCAR files written to {out_dir}/")


# ═══════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="VASP + phonopy Automation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── setup ──
    p_setup = sub.add_parser("setup", help="Generate VASP relax jobs from candidate POSCARs")
    p_setup.add_argument("candidates", help="Directory with candidate POSCAR files")
    p_setup.add_argument("-o", "--output", required=True, help="Output job directory")
    p_setup.add_argument("-r", "--ref-relax", help="Reference relax dir (for POTCAR)")
    p_setup.add_argument("-p", "--ref-phonon", help="Reference phonon dir")
    p_setup.add_argument("--no-phonon", action="store_true", help="Skip phonon job generation")

    # ── phonon-setup ──
    p_phonon = sub.add_parser("phonon-setup", help="Generate phonon jobs from relaxed CONTCARs")
    p_phonon.add_argument("jobs_dir", help="Job directory (from setup command)")
    p_phonon.add_argument("--species", help="Override atom species (for band.conf, space-separated)")

    # ── analyze ──
    p_analyze = sub.add_parser("analyze", help="Analyze phonon results")
    p_analyze.add_argument("directory", help="Directory with phonon results (band.yaml/band.dat)")
    p_analyze.add_argument("--json", help="Save results as JSON")

    # ── bulk ──
    p_bulk = sub.add_parser("bulk", help="Generate substituted POSCARs from patterns")
    p_bulk.add_argument("poscar", help="Base POSCAR file")
    p_bulk.add_argument("-p", "--pattern", required=True, nargs="+",
                       help="Substitution patterns, e.g. V@Nb,Ta  Cl@F,Br,I")
    p_bulk.add_argument("-o", "--output", default="./bulk_candidates",
                       help="Output directory (default: ./bulk_candidates)")

    args = parser.parse_args()

    if args.command == "setup":
        cmd_setup(args)
    elif args.command == "phonon-setup":
        cmd_phonon_setup(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "bulk":
        cmd_bulk(args)


if __name__ == "__main__":
    main()
