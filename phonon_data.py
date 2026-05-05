"""
Phonon Stability Database
=========================
Compiled from 125+ DFT+phonopy calculations across three parent structures.

Each entry records which element substitutions were phonon-stable or unstable.
"""

# ──────────────────────────────────────────────────────────────
# Structure type: V₄S₉Br₄ (P4/nmm #129)
#   Sites: M (metal, 4/c), S (sulfur, 9), X (halogen, 4)
#   MSX 1:1:1 composition also tested for many M
# ──────────────────────────────────────────────────────────────

V4S9Br4_TYPE = {
    "name": "V₄S₉Br₄-type",
    "space_group": "P4/nmm (#129)",
    "crystal_system": "tetragonal",
    "n_atoms": 34,
    "sites": [
        {"label": "M",  "role": "metal",    "count": 4, "common": "V"},
        {"label": "S",  "role": "chalcogen","count": 9, "common": "S"},
        {"label": "X",  "role": "halogen",  "count": 4, "common": "Br"},
    ],
    "lattice_range": {"a": (10.8, 13.2), "c": (24.0, 26.0)},
    # M-site: metal substitutions tested in M₄S₉X₄ form
    "M_substitutions": {
        "stable": [
            "Co", "Cr", "Fe", "Hf", "Mn", "Nb", "Ni",
            "Re", "Ta", "Tc", "Ti", "V",  "Y",
        ],
        "unstable": [
            "Bi", "Cu", "Ir", "Mo", "Os", "Pd", "Pt",
            "Rh", "Ru", "Sb", "Sc", "Sn", "W",  "Zn", "Zr",
        ],
    },
    # X-site: halogen substitutions tested in M₄S₉X₄ form
    "X_substitutions": {
        "stable": ["Cl", "F", "Br"],
        "unstable": ["I"],
    },
    # Substitutions only tested in MSX 1:1:1 form (all unstable)
    "MSX_only": [
        "Co", "Fe", "Ir", "Mn", "Mo", "Ni", "Os",
        "Pd", "Pt", "Re", "Rh", "Ru", "Tc", "W",
    ],
    # Paired stability: some metals depend on the halogen
    "paired_stability": {
        "Cr": {"stable_with": ["Cl", "F"],            "unstable_with": ["Br"]},
        "Hf": {"stable_with": ["Br", "F"],            "unstable_with": ["Cl", "I"]},
        "Nb": {"stable_with": ["Br"],                 "unstable_with": ["Cl", "F", "I"]},
        "Ta": {"stable_with": ["Br", "Cl"],           "unstable_with": ["F", "I"]},
        "Ti": {"stable_with": ["Cl"],                 "unstable_with": ["Br", "F", "I"]},
        "Y":  {"stable_with": ["Cl", "F"],            "unstable_with": ["Br", "I"]},
        "V":  {"stable_with": ["Br", "Cl", "F", "I"], "unstable_with": []},
    },
}

# ──────────────────────────────────────────────────────────────
# Structure type: W₆CCl₁₆
#   Sites: M (metal), C (carbon), X (halogen)
#   Also tested in MCX 1:1:1 form
# ──────────────────────────────────────────────────────────────

W6CCl16_TYPE = {
    "name": "W₆CCl₁₆-type",
    "space_group": "unknown",
    "crystal_system": "unknown",
    "n_atoms": 6 + 1 + 16,
    "sites": [
        {"label": "M", "role": "metal",    "count": 6, "common": "W"},
        {"label": "C", "role": "carbon",   "count": 1, "common": "C"},
        {"label": "X", "role": "halogen",  "count": 16, "common": "Cl"},
    ],
    "M_substitutions": {
        "stable": ["Ir", "Mo", "W"],
        "unstable": ["Au", "Co", "Hf", "Os", "Rh", "Ta"],
    },
    "paired_stability": {
        "Ir": {"stable_with": ["Br", "I"], "unstable_with": []},
        "W":  {"stable_with": ["Br", "Cl", "I"], "unstable_with": ["Si"]},
    },
}

# ──────────────────────────────────────────────────────────────
# Structure type: PbNV
#   Formula: MNY (M, Y = transition metals, N = nitrogen)
# ──────────────────────────────────────────────────────────────

PbNV_TYPE = {
    "name": "PbNV-type",
    "space_group": "unknown",
    "crystal_system": "unknown",
    "n_atoms": 3,
    "sites": [
        {"label": "M", "role": "metal",   "count": 1, "common": "Pb"},
        {"label": "N", "role": "nitrogen","count": 1, "common": "N"},
        {"label": "Y", "role": "metal",   "count": 1, "common": "V"},
    ],
    "M_substitutions": {
        "stable": ["Cd", "Ir", "Pb", "W", "Zn"],
        "unstable": ["Ir"],
    },
    "Y_substitutions": {
        "stable": ["Ta", "V"],
        "unstable": ["Nb"],
    },
}

# ──────────────────────────────────────────────────────────────
# Combined database for lookup
# ──────────────────────────────────────────────────────────────

ALL_STRUCTURES = [V4S9Br4_TYPE, W6CCl16_TYPE, PbNV_TYPE]


def lookup_substitutions(parent_struct, site_label):
    """Get stable/unstable lists for a site in a parent structure."""
    stable_key = f"{site_label}_substitutions"
    for struct in ALL_STRUCTURES:
        for s in struct["sites"]:
            if s["label"] == site_label:
                if stable_key in struct:
                    return struct[stable_key]
    return None


def get_parent_by_name(name):
    """Find parent structure by name keyword."""
    for struct in ALL_STRUCTURES:
        if name.lower() in struct["name"].lower():
            return struct
    return None
