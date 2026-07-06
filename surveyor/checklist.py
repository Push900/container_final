"""
surveyor.checklist — the surveyor verification checklist.

Encodes the full list of items a human surveyor signs off, grouped by section,
and turns the inspection evidence gathered by the orchestrator into a per-item
status so the report and dashboard can render a complete sheet.

Statuses::
    PASS         verified OK
    FAIL         verified and defective
    ATTENTION    present but needs review (e.g. moderate damage)
    NOT_VISIBLE  not observable from the supplied views
    PENDING      not yet evaluated (no model/evidence)
"""

from __future__ import annotations

PASS, FAIL, ATTENTION, NOT_VISIBLE, PENDING = (
    "PASS", "FAIL", "ATTENTION", "NOT_VISIBLE", "PENDING")

# section -> list of checklist item keys (human labels)
CHECKLIST_SECTIONS = {
    "Identity & Markings": [
        "Container number", "ISO size/type code", "Owner / operator",
        "Shipping line logo", "CSC safety-approval plate", "Weight markings",
        "Hazard / placard labels",
    ],
    "Seal & Doors": [
        "Seal present", "Seal condition", "Seal number", "Door condition",
        "Door alignment", "Locking bar condition", "Bent locking rods",
        "Hinges", "Handles",
    ],
    "Structure": [
        "Corner castings", "Roof damage", "Side wall dents", "End wall damage",
        "Structural deformation", "Floor damage", "Holes", "Cracks",
    ],
    "Surface & Condition": [
        "Rust", "Corrosion", "Paint damage", "Previous repair patches",
        "Water leakage evidence",
    ],
    "Integrity & Compliance": [
        "Missing components", "Loose components", "Unauthorized modifications",
        "Evidence of tampering", "Container cleanliness", "Foreign objects",
    ],
}

ALL_ITEMS = [item for items in CHECKLIST_SECTIONS.values() for item in items]


def _component_for(item: str) -> str:
    m = {
        "Roof damage": "roof", "Side wall dents": "wall", "End wall damage": "wall",
        "Floor damage": "floor", "Door condition": "door", "Door alignment": "door",
        "Hinges": "door", "Handles": "door", "Locking bar condition": "door",
    }
    return m.get(item, "body")


def build_checklist(evidence: dict) -> dict:
    """Map gathered evidence to a checklist of {section: [{item, status, detail}]}.

    ``evidence`` keys (all optional) the orchestrator provides:
        container_decoded   dict from iso6346.decode_container_number
        size_type_decoded   dict from iso6346.decode_size_type_code
        seal                SealResult.as_dict()
        damage              DamageReport.as_dict()
        detected_markings   set/list of marking keywords found by OCR
                            (e.g. {"csc", "max_gr", "tare", "hazard", "logo"})
    """
    cont = evidence.get("container_decoded") or {}
    st = evidence.get("size_type_decoded") or {}
    seal = evidence.get("seal") or {}
    damage = evidence.get("damage") or {}
    markings = set(evidence.get("detected_markings") or [])
    defects = damage.get("defects", [])

    # index defects by component + type for quick lookup
    def defect_status(*type_keys, components=None):
        hit = None
        for d in defects:
            if components and d.get("component") not in components:
                continue
            if any(k in d.get("defect_type", "") for k in type_keys):
                if hit is None or _sev_rank(d["severity"]) > _sev_rank(hit["severity"]):
                    hit = d
        if hit is None:
            return PASS, "No defect detected"
        sev = hit["severity"]
        status = FAIL if sev in ("severe", "critical") else ATTENTION
        return status, f"{hit['defect_type']} ({sev})"

    results: dict[str, list] = {}
    for section, items in CHECKLIST_SECTIONS.items():
        rows = []
        for item in items:
            status, detail = _evaluate(item, cont, st, seal, markings, defect_status)
            rows.append({"item": item, "status": status, "detail": detail})
        results[section] = rows
    return results


def _sev_rank(sev: str) -> int:
    return {"cosmetic": 1, "minor": 2, "moderate": 3, "severe": 4, "critical": 5}.get(sev, 0)


def _evaluate(item, cont, st, seal, markings, defect_status):
    if item == "Container number":
        if cont.get("normalized"):
            ok = cont.get("check_digit_valid")
            return (PASS if ok else ATTENTION,
                    f"{cont['normalized']} ({'check digit OK' if ok else 'check digit FAILED'})")
        return NOT_VISIBLE, "Number not read"
    if item == "ISO size/type code":
        if st.get("valid"):
            return PASS, f"{st['raw']} → {st.get('container_type')} {st.get('length_label') or ''}"
        return NOT_VISIBLE if not st.get("raw") else ATTENTION, st.get("raw") or "Not read"
    if item == "Owner / operator":
        return (PASS, cont["owner_name"]) if cont.get("owner_name") \
            else (ATTENTION if cont.get("owner_code") else NOT_VISIBLE,
                  cont.get("owner_code") or "Unknown")
    if item == "Shipping line logo":
        return (PASS, "Logo region detected") if "logo" in markings else (NOT_VISIBLE, "Not detected")
    if item == "CSC safety-approval plate":
        return (PASS, "CSC plate text detected") if "csc" in markings else (NOT_VISIBLE, "Not detected")
    if item == "Weight markings":
        return (PASS, "MAX GR / TARE detected") if (markings & {"max_gr", "tare", "mgw"}) \
            else (NOT_VISIBLE, "Not detected")
    if item == "Hazard / placard labels":
        return (ATTENTION, "Hazard placard detected — verify class") if "hazard" in markings \
            else (PASS, "No hazard placard")
    # Seal
    if item == "Seal present":
        s = seal.get("status")
        if s == "MISSING":
            return FAIL, "No seal where expected"
        if s in ("INTACT", "BROKEN"):
            return PASS, "Seal detected"
        return PENDING, "Insufficient evidence"
    if item == "Seal condition":
        s = seal.get("status")
        if s == "INTACT":
            return PASS, f"Intact (conf {seal.get('status_confidence', 0):.2f})"
        if s == "BROKEN":
            return FAIL, "Broken / cut"
        if s == "MISSING":
            return FAIL, "Missing"
        return PENDING, "Not determined"
    if item == "Seal number":
        return (PASS, seal["seal_number"]) if seal.get("seal_number") \
            else (NOT_VISIBLE, "Not read")
    # Door / structure / surface items → defect lookup
    mapping = {
        "Door condition": (("dent", "hole", "deform"), {"door"}),
        "Door alignment": (("deform", "bent"), {"door"}),
        "Locking bar condition": (("bent", "deform", "rust"), {"door"}),
        "Bent locking rods": (("bent_rod", "bent", "deform"), {"door"}),
        "Hinges": (("rust", "deform", "broken"), {"door"}),
        "Handles": (("deform", "broken", "rust"), {"door"}),
        "Corner castings": (("crack", "deform", "rust"), None),
        "Roof damage": (("dent", "hole", "deform", "rust"), {"roof"}),
        "Side wall dents": (("dent", "deform"), {"wall"}),
        "End wall damage": (("dent", "hole", "deform"), {"wall"}),
        "Structural deformation": (("deform",), None),
        "Floor damage": (("hole", "crack", "rot"), {"floor"}),
        "Holes": (("hole", "puncture"), None),
        "Cracks": (("crack",), None),
        "Rust": (("rust",), None),
        "Corrosion": (("corros", "rust"), None),
        "Paint damage": (("paint", "scratch"), None),
        "Previous repair patches": (("patch", "repair"), None),
        "Water leakage evidence": (("water", "stain", "leak"), None),
    }
    if item in mapping:
        keys, comps = mapping[item]
        if item == "Floor damage":
            return defect_status(*keys, components=comps) if comps else defect_status(*keys)
        return defect_status(*keys, components=comps) if comps else defect_status(*keys)
    # Integrity items — not detectable without dedicated models yet.
    if item in ("Missing components", "Loose components", "Unauthorized modifications",
                "Evidence of tampering", "Foreign objects"):
        return PENDING, "Requires dedicated model"
    if item == "Container cleanliness":
        return PENDING, "Requires interior view"
    return PENDING, "Not evaluated"
