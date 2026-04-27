"""
vocab.py — Canonical vocabularies for the M2 policy graph.

Three module-level constants:
  DATA_TYPES  : canonical data-type identifiers (Google Play DSL categories)
  PURPOSES    : canonical purpose identifiers (Google Play DSL purposes)
  THIRD_PARTIES: deduplicated set of company/tracker names from TrackerControl
                 xray-blacklist and Yale Privacy Lab tracker profiles.

Side effect on import:  nothing heavy.
Run as __main__ to regenerate data/processed/third_parties.json.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import List

# ---------------------------------------------------------------------------
# Canonical data types — aligned with Google Play Data Safety DSL categories
# plus common extras found in OPP-115 and Princeton PPC corpus.
# ---------------------------------------------------------------------------

DATA_TYPES: List[str] = [
    # Personal info
    "name",
    "email_address",
    "phone_number",
    "address",
    "user_ids",
    "date_of_birth",
    "race_ethnicity",
    "political_or_religious_beliefs",
    "sexual_orientation",
    # Financial
    "user_payment_info",
    "purchase_history",
    "credit_score",
    "financial_info",
    # Location
    "precise_location",
    "approximate_location",
    # Contacts / communications
    "contacts",
    "emails",
    "sms_mms",
    "calendar_events",
    # Device / usage
    "device_id",
    "advertising_id",
    "installed_apps",
    "app_interactions",
    "web_browsing_history",
    "in_app_search_history",
    # Files / media
    "photos",
    "videos",
    "files_and_docs",
    "voice_or_sound_recordings",
    "music_files",
    # Health / biometric
    "health_info",
    "fitness_info",
    "biometric_data",
    # Diagnostics / performance
    "crash_logs",
    "diagnostics",
    "other_app_performance_data",
    # Misc
    "other_info",
    "other_user_generated_content",
]

# ---------------------------------------------------------------------------
# Canonical purposes — aligned with Google Play DSL `purpose` column values,
# normalised to snake_case.
# ---------------------------------------------------------------------------

PURPOSES: List[str] = [
    "app_functionality",
    "analytics",
    "advertising_marketing",
    "personalization",
    "account_management",
    "developer_communications",
    "fraud_prevention_security",
    "compliance_legal",
]

# ---------------------------------------------------------------------------
# Third-party company names — built from two datasets at import time via
# the helper below; populated when build_third_parties() is called.
# Module-level constant is populated at the bottom of this file.
# ---------------------------------------------------------------------------


def _load_xray_companies(xray_path: Path) -> List[str]:
    """Extract unique company names from TrackerControl xray-blacklist.json."""
    with open(xray_path, encoding="utf-8") as fh:
        entries = json.load(fh)  # list of dicts with owner_name, parent, root_parent

    names: set[str] = set()
    for entry in entries:
        for field in ("owner_name", "parent", "root_parent"):
            val = entry.get(field, "")
            if val and isinstance(val, str) and val.strip():
                names.add(val.strip())
    return list(names)


def _load_yale_companies(yale_path: Path) -> List[str]:
    """Extract unique tracker/owner names from Yale Privacy Lab trackers_parsed.json."""
    with open(yale_path, encoding="utf-8") as fh:
        entries = json.load(fh)  # list of dicts with tracker_name, owner, ...

    names: set[str] = set()
    for entry in entries:
        for field in ("tracker_name", "owner"):
            val = entry.get(field, "")
            if val and isinstance(val, str) and val.strip():
                # Skip placeholder / header rows
                if val.strip().lower() in ("tracker name", "owner", "n/a", ""):
                    continue
                names.add(val.strip())
    return list(names)


def _normalize(name: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for deduplication key."""
    return re.sub(r"[^a-z0-9 ]", "", name.lower()).strip()


def build_third_parties(
    xray_path: Path,
    yale_path: Path,
    output_path: Path | None = None,
) -> List[str]:
    """
    Load company names from both sources, deduplicate by normalised name,
    and optionally save to output_path as JSON.

    Returns a sorted list of canonical company name strings (original casing
    from first occurrence wins).
    """
    xray = _load_xray_companies(xray_path)
    yale = _load_yale_companies(yale_path)

    seen: dict[str, str] = {}  # norm_key -> original_name
    for name in xray + yale:
        key = _normalize(name)
        if key and key not in seen:
            seen[key] = name

    deduped = sorted(seen.values(), key=lambda n: n.lower())

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(deduped, fh, indent=2, ensure_ascii=False)

    return deduped


# ---------------------------------------------------------------------------
# Populate THIRD_PARTIES at module import time (reads small JSON files).
# Falls back to an empty list if data is not available (e.g., unit tests).
# ---------------------------------------------------------------------------

def _resolve_data_root() -> Path:
    """Walk up from this file to find the project root data/ directory."""
    here = Path(__file__).resolve()
    for parent in [here.parent, here.parent.parent, here.parent.parent.parent]:
        candidate = parent / "data"
        if candidate.is_dir():
            return candidate
        candidate = parent.parent / "data"
        if candidate.is_dir():
            return candidate
    # Last resort — relative to cwd
    return Path("data")


def _try_load_third_parties() -> List[str]:
    data_root = _resolve_data_root()
    xray_path = data_root / "raw" / "trackercontrol" / "xray-blacklist.json"
    yale_path = data_root / "raw" / "yale_pl" / "trackers_parsed.json"
    processed_path = data_root / "processed" / "third_parties.json"

    # If already built, load from cache
    if processed_path.exists():
        with open(processed_path, encoding="utf-8") as fh:
            return json.load(fh)

    if xray_path.exists() and yale_path.exists():
        return build_third_parties(xray_path, yale_path, output_path=processed_path)

    return []


THIRD_PARTIES: List[str] = _try_load_third_parties()


# ---------------------------------------------------------------------------
# CLI entry point — regenerate processed/third_parties.json
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    data_root = _resolve_data_root()
    xray_path = data_root / "raw" / "trackercontrol" / "xray-blacklist.json"
    yale_path = data_root / "raw" / "yale_pl" / "trackers_parsed.json"
    out_path = data_root / "processed" / "third_parties.json"

    companies = build_third_parties(xray_path, yale_path, output_path=out_path)
    print(f"Built THIRD_PARTIES: {len(companies)} unique companies")
    print(f"Saved to {out_path}")
    print("Sample (first 10):", companies[:10])
