"""
surveyor.iso6346 — offline ISO 6346 / BIC container-code intelligence.

Everything in this module is pure deterministic logic (no ML, no network) so it
is always available and forms the backbone of "decode the container after the
number is read".

Two distinct codes are decoded:

1. **Container number** (BIC code), 11 characters, e.g. ``MSCU1234567``::
       AAA  U  NNNNNN  C
       │    │  │       └ check digit (ISO 6346 modulo-11)
       │    │  └ 6-digit serial number
       │    └ equipment category (U/J/Z + legacy R)
       └ 3-letter owner / operator prefix (BIC registered)

2. **ISO size/type code**, 4 characters stamped near the number, e.g. ``22G1``::
       2  2  G1
       │  │  └ type code (G=general, R=reefer, U=open-top, P=flat-rack, T=tank…)
       │  └ height & width code  (5/E/F… ⇒ 9'6" High-Cube)
       └ length code             (2=20', 4=40', L=45'…)

Optional online enrichment (shipping line / leasing company / last status) is
left to :mod:`surveyor.lookup`; this module guarantees a useful result offline.
"""

from __future__ import annotations

import re
from typing import Optional

# ── ISO 6346 check-digit machinery ──────────────────────────────────────────

# Letter -> numeric value.  Values run 10..38 skipping every multiple of 11
# (11, 22, 33), which is why L=23 (not 22) and V=34 (not 33).
_CHAR_VALUE = {
    "A": 10, "B": 12, "C": 13, "D": 14, "E": 15, "F": 16, "G": 17, "H": 18,
    "I": 19, "J": 20, "K": 21, "L": 23, "M": 24, "N": 25, "O": 26, "P": 27,
    "Q": 28, "R": 29, "S": 30, "T": 31, "U": 32, "V": 34, "W": 35, "X": 36,
    "Y": 37, "Z": 38,
}

_OWNER_RE = re.compile(r"^[A-Z]{3}$")
_FULL_RE = re.compile(r"^[A-Z]{3}[A-Z][0-9]{6}[0-9]$")
_EQUIPMENT_CATEGORIES = {
    "U": "Freight container",
    "J": "Detachable freight-related equipment",
    "Z": "Trailer / chassis",
    "R": "Reefer (legacy category letter)",
}


def check_digit(number: str) -> Optional[int]:
    """Compute the ISO 6346 check digit for the first 10 characters.

    Accepts a 10- or 11-character string (the 11th, if present, is ignored).
    Returns the expected check digit (0-9) or ``None`` if the input is malformed.
    """
    if not number:
        return None
    s = re.sub(r"[^A-Za-z0-9]", "", number).upper()
    if len(s) < 10:
        return None
    s = s[:10]
    if not s[:4].isalpha() or not s[4:10].isdigit():
        return None
    total = 0
    for i, ch in enumerate(s):
        val = _CHAR_VALUE[ch] if ch.isalpha() else int(ch)
        total += val * (2 ** i)
    # modulo 11, and 10 maps back to 0.
    return (total % 11) % 10


def is_valid_container_number(number: str) -> bool:
    """True when *number* is a well-formed 11-char code with a correct check digit."""
    if not number:
        return False
    s = re.sub(r"[^A-Za-z0-9]", "", number).upper()
    if not _FULL_RE.match(s):
        return False
    expected = check_digit(s)
    return expected is not None and expected == int(s[10])


# ── BIC owner-prefix directory (offline subset of common operators) ─────────
# Not exhaustive (the BIC registry has ~10k entries); covers the high-frequency
# lines/lessors so offline decoding still names the operator for most traffic.
# Replace/extend via update_owner_directory() or a CSV import in surveyor.lookup.
_OWNER_DIRECTORY = {
    "MSC": "Mediterranean Shipping Company", "MSD": "Mediterranean Shipping Company",
    "MSK": "Maersk Line", "MRK": "Maersk Line", "MAE": "Maersk Line",
    "MSU": "Maersk Line", "PON": "Maersk Line (P&O Nedlloyd)",
    "CMA": "CMA CGM", "CMU": "CMA CGM", "CGM": "CMA CGM",
    "COS": "COSCO Shipping", "CBH": "COSCO Shipping", "CCL": "COSCO Shipping Lines",
    "HLC": "Hapag-Lloyd", "HLX": "Hapag-Lloyd", "HPL": "Hapag-Lloyd",
    "OOL": "OOCL (Orient Overseas Container Line)", "OOC": "OOCL",
    "EIS": "Evergreen Marine", "EGH": "Evergreen Marine", "EMC": "Evergreen Marine",
    "EGS": "Evergreen Marine", "EISU": "Evergreen Marine",
    "ONE": "Ocean Network Express", "ONEU": "Ocean Network Express",
    "YML": "Yang Ming Line", "YMM": "Yang Ming Line", "YML": "Yang Ming Line",
    "HMM": "HMM (Hyundai Merchant Marine)", "HDM": "HMM",
    "ZIM": "ZIM Integrated Shipping", "ZCS": "ZIM",
    "APL": "APL (American President Lines)", "APH": "APL",
    "HJM": "Hanjin Shipping", "WHL": "Wan Hai Lines", "WHS": "Wan Hai Lines",
    "PIL": "Pacific International Lines", "PCI": "Pacific International Lines",
    "KKF": "K Line", "KKT": "K Line",
    "NYK": "NYK Line", "MOL": "Mitsui O.S.K. Lines", "MOA": "Mitsui O.S.K. Lines",
    # Leasing companies
    "TCL": "Triton Container", "TLL": "Triton Container", "TCN": "Triton Container",
    "TRI": "Triton Container", "TGH": "Textainer", "TEX": "Textainer",
    "GES": "GE SeaCo / SeaCube", "SEG": "SeaCube Containers",
    "FBL": "Florens Container", "FCI": "Florens Container",
    "BMO": "Beacon Intermodal", "BEA": "Beacon Intermodal",
    "CAI": "CAI International", "CAX": "CAI International",
    "GAT": "GATX / Gateway", "DRY": "Dong Fang International",
    "UES": "UES International", "SUD": "Hamburg Süd", "SUZ": "Hamburg Süd",
}


def update_owner_directory(mapping: dict[str, str]) -> None:
    """Merge additional owner prefixes (e.g. loaded from the full BIC CSV)."""
    _OWNER_DIRECTORY.update({k.upper(): v for k, v in mapping.items()})


def lookup_owner(owner_code: str) -> Optional[str]:
    code = (owner_code or "").upper()
    return _OWNER_DIRECTORY.get(code) or _OWNER_DIRECTORY.get(code[:3])


# ── Container-number decode ─────────────────────────────────────────────────

def decode_container_number(number: str) -> dict:
    """Decode an 11-char container number into its components.

    Returns a dict with: ``raw``, ``normalized``, ``valid_format``,
    ``owner_code``, ``owner_name``, ``equipment_category``,
    ``equipment_category_desc``, ``serial_number``, ``check_digit``,
    ``check_digit_valid``, ``computed_check_digit``.
    """
    raw = number or ""
    s = re.sub(r"[^A-Za-z0-9]", "", raw).upper()
    out: dict = {
        "raw": raw,
        "normalized": s,
        "valid_format": False,
        "owner_code": None,
        "owner_name": None,
        "equipment_category": None,
        "equipment_category_desc": None,
        "serial_number": None,
        "check_digit": None,
        "computed_check_digit": None,
        "check_digit_valid": False,
    }
    if len(s) < 10 or not s[:4].isalpha() or not s[4:10].isdigit():
        return out

    out["owner_code"] = s[:3]
    out["owner_name"] = lookup_owner(s[:3])
    cat = s[3]
    out["equipment_category"] = cat
    out["equipment_category_desc"] = _EQUIPMENT_CATEGORIES.get(cat, "Unknown category")
    out["serial_number"] = s[4:10]
    out["computed_check_digit"] = check_digit(s)

    if len(s) >= 11 and s[10].isdigit():
        out["check_digit"] = int(s[10])
        out["check_digit_valid"] = out["check_digit"] == out["computed_check_digit"]
        out["valid_format"] = bool(_FULL_RE.match(s)) and out["check_digit_valid"]
    return out


# ── ISO size/type code tables ───────────────────────────────────────────────

# First character → length.
_LENGTH = {
    "1": (10, "10 ft", 2991),
    "2": (20, "20 ft", 6058),
    "3": (30, "30 ft", 9125),
    "4": (40, "40 ft", 12192),
    "A": (None, "Reserved (A)", None),
    "B": (24, "24 ft", 7150),
    "C": (24.5, "24.5 ft", 7430),
    "G": (41, "41 ft", 12500),
    "H": (43, "43 ft", 13106),
    "K": (44.5, "44.5 ft", 13600),
    "L": (45, "45 ft", 13716),
    "M": (48, "48 ft", 14630),
    "N": (49, "49 ft", 14935),
    "P": (53, "53 ft", 16154),
}

# Second character → (height_mm, height_label, high_cube?, width_class)
# Width class "STD" ≈ 2438 mm; "WIDE" indicates >2438 mm (specials).
_HEIGHT_WIDTH = {
    "0": (2438, "8'0\"", False, "STD"),
    "2": (2591, "8'6\"", False, "STD"),
    "4": (2743, "9'0\"", False, "STD"),
    "5": (2896, "9'6\" (High-Cube)", True, "STD"),
    "6": (2438, ">9'6\"", True, "STD"),
    "8": (1295, "4'3\"", False, "STD"),
    "9": (1219, "≤4'0\"", False, "STD"),
    # Wide (>2438mm) variants
    "C": (2591, "8'6\" wide", False, "WIDE"),
    "D": (2743, "9'0\" wide", False, "WIDE"),
    "E": (2895, "9'6\" wide (High-Cube)", True, "WIDE"),
    "F": (2438, ">9'6\" wide", True, "WIDE"),
    "L": (2591, "8'6\" wide", False, "WIDE"),
    "M": (2743, "9'0\" wide", False, "WIDE"),
    "N": (2895, "9'6\" wide (High-Cube)", True, "WIDE"),
    "P": (2438, ">9'6\" wide", True, "WIDE"),
}

# Type code (first char of the 2-char type group) → family.
_TYPE_FAMILY = {
    "G": "Dry / General Purpose",
    "V": "Ventilated",
    "B": "Dry Bulk",
    "S": "Named Cargo (e.g. livestock, automobile)",
    "R": "Refrigerated (Reefer)",
    "H": "Insulated / Refrigerated (external unit)",
    "U": "Open Top",
    "P": "Platform / Flat Rack",
    "T": "Tank",
    "A": "Air / Surface",
    "K": "Tank (named)",
}

# Fully-spelled common type codes for nicer reports.
_TYPE_DETAIL = {
    "G0": "General purpose, vents at upper part",
    "G1": "General purpose, passive vents",
    "G2": "General purpose, openings at one or both ends",
    "G3": "General purpose, full openings at one or both ends + sides",
    "V0": "Ventilated, non-mechanical vents lower & upper",
    "R0": "Reefer, mechanically refrigerated",
    "R1": "Reefer, mechanically refrigerated & heated",
    "R2": "Reefer, self-powered",
    "H0": "Insulated, refrigerated/heated (removable equipment)",
    "U0": "Open top, openings at one or both ends",
    "U1": "Open top, removable top members",
    "P0": "Platform (flat, no superstructure)",
    "P1": "Flat rack, fixed complete ends",
    "P3": "Flat rack, folding complete ends",
    "T0": "Tank, non-dangerous liquids",
    "T1": "Tank, non-dangerous liquids (low pressure)",
    "T4": "Tank, dangerous liquids",
    "T7": "Tank, gases",
    "B0": "Dry bulk, closed",
    "S0": "Named cargo, livestock carrier",
    "S1": "Named cargo, automobile carrier",
}


def decode_size_type_code(code: str) -> dict:
    """Decode a 4-character ISO size/type code (e.g. ``22G1`` / ``45R1``).

    Returns dict with: ``raw``, ``valid``, ``length_ft``, ``length_label``,
    ``length_mm``, ``height_mm``, ``height_label``, ``high_cube``, ``width_mm``,
    ``width_class``, ``type_code``, ``type_family``, ``type_detail``,
    ``container_type``.
    """
    raw = (code or "").strip().upper()
    s = re.sub(r"[^A-Z0-9]", "", raw)
    out: dict = {
        "raw": raw,
        "valid": False,
        "length_ft": None, "length_label": None, "length_mm": None,
        "height_mm": None, "height_label": None, "high_cube": None,
        "width_mm": None, "width_class": None,
        "type_code": None, "type_family": None, "type_detail": None,
        "container_type": None,
    }
    if len(s) != 4:
        return out

    L = _LENGTH.get(s[0])
    HW = _HEIGHT_WIDTH.get(s[1])
    type_code = s[2:4]
    family = _TYPE_FAMILY.get(s[2])

    if L:
        out["length_ft"], out["length_label"], out["length_mm"] = L
    if HW:
        h_mm, h_lbl, hc, wclass = HW
        out["height_mm"] = h_mm
        out["height_label"] = h_lbl
        out["high_cube"] = hc
        out["width_class"] = wclass
        out["width_mm"] = 2438 if wclass == "STD" else 2500  # nominal
    out["type_code"] = type_code
    out["type_family"] = family
    out["type_detail"] = _TYPE_DETAIL.get(type_code, family)
    out["container_type"] = family or "Unknown"
    out["valid"] = bool(L and HW and family)
    return out


# Map a type family to the canonical surveyor "container type" label.
def container_type_label(size_type_decoded: dict) -> str:
    """A short human label like 'Dry High-Cube 40ft' from a size/type decode."""
    if not size_type_decoded or not size_type_decoded.get("type_family"):
        return "Unknown"
    fam = size_type_decoded["type_family"].split(" /")[0].split(" (")[0]
    length = size_type_decoded.get("length_label") or ""
    hc = " High-Cube" if size_type_decoded.get("high_cube") else ""
    return f"{fam}{hc} {length}".strip()
