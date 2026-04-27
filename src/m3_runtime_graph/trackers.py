"""
trackers.py — Unified tracker registry from Exodus, Yale Privacy Lab, and TrackerControl.

Loads three raw tracker datasets and merges them into a single SDK registry
saved at data/processed/sdk_registry.json.

Public API
----------
get_sdk_registry() -> list[dict]
    Return the full list of unified tracker records.

get_sdks_for_category(category) -> list[dict]
    Filter registry to a specific Exodus category (e.g. 'Analytics').

infer_sdks_for_app(genreId, declared_purposes) -> list[dict]
    Heuristic: return SDKs whose canonical purpose overlaps with the app's
    declared purposes, ranked by prevalence (lower tracker_id = older = more
    widespread).  Capped at 5 per (genreId, purpose) combination.

Exodus category → canonical purpose mapping
-------------------------------------------
  Advertisement  → advertising_marketing
  Analytics      → analytics
  Profiling      → personalization
  Location       → app_functionality
  Crash reporting→ app_functionality  (diagnostic / app-stability purpose)
  Identification → app_functionality  (device ID / account binding)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _resolve_data_root() -> Path:
    here = Path(__file__).resolve()
    for parent in [here.parent, here.parent.parent, here.parent.parent.parent]:
        candidate = parent / "data"
        if candidate.is_dir():
            return candidate
        candidate = parent.parent / "data"
        if candidate.is_dir():
            return candidate
    return Path("data")


DATA_ROOT = _resolve_data_root()
EXODUS_PATH   = DATA_ROOT / "raw" / "exodus"   / "trackers.json"
YALE_PATH     = DATA_ROOT / "raw" / "yale_pl"  / "trackers_parsed.json"
XRAY_PATH     = DATA_ROOT / "raw" / "trackercontrol" / "xray-blacklist.json"
REGISTRY_PATH = DATA_ROOT / "processed" / "sdk_registry.json"

# ---------------------------------------------------------------------------
# Exodus category → canonical purpose
# ---------------------------------------------------------------------------

# Category → DataType priors: fallback when Yale cross-ref is missing.
# Also used to augment Yale-derived edges (union/dedup).
CATEGORY_DATATYPE_PRIORS: dict[str, list[str]] = {
    "Advertisement":   ["advertising_id", "device_id", "installed_apps", "approximate_location"],
    "Analytics":       ["device_id", "diagnostics", "crash_logs", "installed_apps"],
    "Profiling":       ["advertising_id", "device_id", "email_address", "name"],
    "Location":        ["precise_location", "approximate_location", "device_id"],
    "Crash reporting": ["crash_logs", "diagnostics", "device_id"],
    "Identification":  ["device_id", "advertising_id", "user_ids"],
}

CATEGORY_PURPOSE_MAP: dict[str, str] = {
    "Advertisement":   "advertising_marketing",
    "Analytics":       "analytics",
    "Profiling":       "personalization",
    "Location":        "app_functionality",
    "Crash reporting": "app_functionality",
    "Identification":  "app_functionality",
}

# ---------------------------------------------------------------------------
# Yale data-type strings → canonical DATA_TYPES (best-effort)
# Yale uses free-text phrases, so we do keyword matching.
# ---------------------------------------------------------------------------

_YALE_DT_KEYWORDS: list[tuple[str, str]] = [
    ("location",            "precise_location"),
    ("gps",                 "precise_location"),
    ("geolocation",         "precise_location"),
    ("ibeacon",             "precise_location"),
    ("indoor location",     "precise_location"),
    ("ultrasonic",          "precise_location"),  # proximity = location signal
    ("footfall",            "precise_location"),
    ("phone number",        "phone_number"),
    ("email",               "email_address"),
    ("contact",             "contacts"),
    ("device id",           "device_id"),
    ("advertising id",      "advertising_id"),
    ("idfa",                "advertising_id"),
    ("gaid",                "advertising_id"),
    ("install",             "installed_apps"),
    ("crash",               "crash_logs"),
    ("diagnostic",          "diagnostics"),
    ("health",              "health_info"),
    ("fitness",             "fitness_info"),
    ("biometric",           "biometric_data"),
    ("browsing",            "web_browsing_history"),
    ("search history",      "in_app_search_history"),
    ("photo",               "photos"),
    ("video",               "videos"),
    ("audio",               "voice_or_sound_recordings"),
    ("voice",               "voice_or_sound_recordings"),
    ("payment",             "user_payment_info"),
    ("purchase",            "purchase_history"),
    ("e-commerce",          "purchase_history"),
    # Behavioral analytics phrases → app_interactions (user behaviour data)
    ("behavioral analytics","app_interactions"),
    ("behavorial analytics","app_interactions"),  # typo in Yale data
    ("data collection",     "other_info"),
    ("usage statistics",    "app_interactions"),
    ("market research",     "other_info"),
    ("customer data",       "user_ids"),
    ("mobile advertising",  "advertising_id"),
    ("targeted advertising","advertising_id"),
    ("personalized advertising", "advertising_id"),
    ("personali",           "advertising_id"),
    ("online marketing",    "advertising_id"),
    # User verification
    ("user verification",   "user_ids"),
    ("name",                "name"),
    ("age",                 "date_of_birth"),
    ("gender",              "race_ethnicity"),
]


def _yale_dt_to_canonical(raw: str) -> Optional[str]:
    """Keyword-based mapping from Yale free-text data type to canonical."""
    lower = raw.lower()
    for keyword, canonical in _YALE_DT_KEYWORDS:
        if keyword in lower:
            return canonical
    return None


# ---------------------------------------------------------------------------
# Domain → company lookup from xray-blacklist
# ---------------------------------------------------------------------------

def _build_domain_company_map(xray: list[dict]) -> dict[str, str]:
    """Build domain → root_parent company name mapping."""
    dm: dict[str, str] = {}
    for entry in xray:
        company = (entry.get("root_parent") or entry.get("parent") or
                   entry.get("owner_name") or "")
        company = company.strip()
        for dom in entry.get("doms", []):
            if dom and company:
                dm[dom.strip().lower()] = company
    return dm


def _extract_domains_from_signature(sig: str) -> list[str]:
    """Pull domain-like tokens from a network_signature regex string."""
    if not sig:
        return []
    # Remove regex metacharacters and split on whitespace / pipes
    cleaned = re.sub(r"[\\()\[\]{}?+*^$|]", " ", sig)
    tokens = cleaned.split()
    domains = [t.strip(".").lower() for t in tokens if "." in t and len(t) > 3]
    return domains


# ---------------------------------------------------------------------------
# Build unified registry
# ---------------------------------------------------------------------------

def _build_registry(
    exodus_path: Path = EXODUS_PATH,
    yale_path: Path   = YALE_PATH,
    xray_path: Path   = XRAY_PATH,
) -> list[dict]:
    """
    Merge Exodus, Yale, and xray-blacklist into a single SDK record list.

    Each record:
    {
      tracker_id       : int,
      name             : str,
      category         : str,           # primary Exodus category or ''
      canonical_purpose: str,           # mapped from category
      owner_company    : str,
      network_domains  : list[str],
      collects_data_types: list[str],   # canonical, from Yale cross-reference
      purposes         : list[str],     # free-text from Yale
      jurisdiction     : str,
    }
    """
    # Load raw data
    with open(exodus_path, encoding="utf-8") as fh:
        exodus_raw = json.load(fh)["trackers"]   # dict keyed by str int

    with open(yale_path, encoding="utf-8") as fh:
        yale_list: list[dict] = json.load(fh)

    with open(xray_path, encoding="utf-8") as fh:
        xray_list: list[dict] = json.load(fh)

    domain_company = _build_domain_company_map(xray_list)

    # Build Yale lookup by normalised tracker name
    yale_by_name: dict[str, dict] = {}
    for entry in yale_list:
        key = entry.get("tracker_name", "").strip().lower()
        if key:
            yale_by_name[key] = entry

    registry: list[dict] = []

    for str_id, tracker in exodus_raw.items():
        t_id    = tracker.get("id", int(str_id))
        name    = tracker.get("name", "").strip()
        cats    = tracker.get("categories", [])
        primary_cat = cats[0] if cats else ""
        canonical_purpose = CATEGORY_PURPOSE_MAP.get(primary_cat, "app_functionality")

        # Network domains from network_signature
        net_sig = tracker.get("network_signature", "")
        domains = _extract_domains_from_signature(net_sig)

        # Owner company: try domain → xray lookup, else parse description
        owner = ""
        for dom in domains:
            if dom in domain_company:
                owner = domain_company[dom]
                break
        if not owner:
            # Heuristic: grab "## Ownership\n<owner>" from description
            desc = tracker.get("description", "")
            m = re.search(r"##\s+Ownership\s*\n([^\n#]+)", desc)
            if m:
                owner = m.group(1).strip()

        # Cross-reference with Yale
        yale_key = name.lower()
        yale_entry = yale_by_name.get(yale_key)
        if yale_entry is None:
            # Try partial match
            for yk, ye in yale_by_name.items():
                if yk in yale_key or yale_key in yk:
                    yale_entry = ye
                    break

        collects_data_types: list[str] = []
        dt_confidence: dict[str, str] = {}   # data_type -> 'yale' | 'category_prior'
        purposes_free: list[str] = []
        jurisdiction = ""

        has_yale = False
        if yale_entry:
            has_yale = True
            raw_dts = yale_entry.get("data_types_collected", [])
            for raw_dt in raw_dts:
                canonical_dt = _yale_dt_to_canonical(raw_dt)
                if canonical_dt and canonical_dt not in collects_data_types:
                    collects_data_types.append(canonical_dt)
                    dt_confidence[canonical_dt] = "yale"
            purposes_free = yale_entry.get("purposes", [])
            juris_list = yale_entry.get("jurisdictions", [])
            jurisdiction = juris_list[0] if juris_list else ""

        # Apply category-prior DataType edges: always add missing ones (union/dedup).
        # Trackers with an empty category fall back to the 'Analytics' prior
        # (device_id + diagnostics) as a minimal conservative default.
        effective_cat = primary_cat if primary_cat else "Analytics"
        category_priors = CATEGORY_DATATYPE_PRIORS.get(effective_cat, ["device_id", "diagnostics"])
        has_category_prior = False
        for prior_dt in category_priors:
            if prior_dt not in collects_data_types:
                collects_data_types.append(prior_dt)
                dt_confidence[prior_dt] = "category_prior"
                has_category_prior = True
            # Yale data already present: keep 'yale' confidence (don't downgrade)

        registry.append({
            "tracker_id":           t_id,
            "name":                 name,
            "category":             primary_cat,
            "canonical_purpose":    canonical_purpose,
            "owner_company":        owner,
            "network_domains":      domains,
            "collects_data_types":  collects_data_types,
            "dt_confidence":        dt_confidence,
            "purposes":             purposes_free,
            "jurisdiction":         jurisdiction,
            "has_yale":             has_yale,
            "has_category_prior":   has_category_prior,
        })

    # Sort by tracker_id (ascending = older/more widespread first)
    registry.sort(key=lambda r: r["tracker_id"])
    log.info("Built SDK registry: %d trackers", len(registry))
    return registry


def _load_or_build_registry() -> list[dict]:
    """Load cached registry or rebuild from raw sources."""
    if REGISTRY_PATH.exists():
        with open(REGISTRY_PATH, encoding="utf-8") as fh:
            return json.load(fh)

    reg = _build_registry()
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w", encoding="utf-8") as fh:
        json.dump(reg, fh, indent=2, ensure_ascii=False)
    log.info("Saved SDK registry to %s", REGISTRY_PATH)
    return reg


# Module-level cache — populated lazily on first use
_REGISTRY: list[dict] | None = None


def get_sdk_registry() -> list[dict]:
    """Return the full unified SDK registry (432 Exodus trackers)."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _load_or_build_registry()
    return _REGISTRY


def get_sdks_for_category(category: str) -> list[dict]:
    """Return trackers whose primary Exodus category matches `category`."""
    return [r for r in get_sdk_registry() if r["category"] == category]


# ---------------------------------------------------------------------------
# Heuristic SDK inference for an app
# ---------------------------------------------------------------------------

_MAX_SDKS_PER_PURPOSE = 2  # tuned down from 5 to reduce false-positive CONTAINS_SDK edges (see M3 notes)


def infer_sdks_for_app(
    genreId: str,
    declared_purposes: set[str],
) -> list[dict]:
    """
    Estimate which SDKs an app likely contains based on its declared purposes.

    Strategy (categorical-prior approach)
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Without per-app Exodus reports (API key required), we use a category-level
    prior: if an app declares purpose P, infer that it uses SDKs whose Exodus
    category maps to P.  Trackers are ranked by tracker_id (lower = older =
    historically more widespread in the ecosystem).  We return at most
    _MAX_SDKS_PER_PURPOSE SDKs per purpose to avoid over-inflation.

    Parameters
    ----------
    genreId : str
        Play Store genre (e.g. "EDUCATION", "GAME_ACTION").  Reserved for
        future genre-specific priors; currently used only for logging.
    declared_purposes : set[str]
        Canonical purpose strings (from m2 vocab.py) declared by the app.

    Returns
    -------
    list[dict]
        De-duplicated list of inferred SDK records, ordered by tracker_id.
    """
    registry = get_sdk_registry()
    seen_ids: set[int] = set()
    results: list[dict] = []

    for purpose in declared_purposes:
        # Find trackers whose canonical_purpose matches
        matching = [
            r for r in registry
            if r["canonical_purpose"] == purpose
        ]
        # Already sorted by tracker_id ascending (prevalence proxy)
        added = 0
        for sdk in matching:
            if sdk["tracker_id"] not in seen_ids:
                seen_ids.add(sdk["tracker_id"])
                results.append(sdk)
                added += 1
                if added >= _MAX_SDKS_PER_PURPOSE:
                    break

    # Final sort by tracker_id
    results.sort(key=lambda r: r["tracker_id"])
    log.debug(
        "infer_sdks_for_app(genreId=%s, purposes=%s) → %d SDKs",
        genreId, declared_purposes, len(results)
    )
    return results


# ---------------------------------------------------------------------------
# CLI — build and inspect registry
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    # Force rebuild
    if REGISTRY_PATH.exists():
        REGISTRY_PATH.unlink()

    reg = get_sdk_registry()
    print(f"Registry built: {len(reg)} trackers")

    from collections import Counter
    cat_counts = Counter(r["category"] for r in reg)
    print("By category:", dict(cat_counts))

    purpose_counts = Counter(r["canonical_purpose"] for r in reg)
    print("By canonical purpose:", dict(purpose_counts))

    # Enrichment summary
    n_yale_only = sum(1 for r in reg if r.get("has_yale") and not r.get("has_category_prior"))
    n_prior_only = sum(1 for r in reg if not r.get("has_yale") and r.get("has_category_prior"))
    n_both = sum(1 for r in reg if r.get("has_yale") and r.get("has_category_prior"))
    n_with_dt = sum(1 for r in reg if r.get("collects_data_types"))
    print(f"\nDataType enrichment summary:")
    print(f"  Yale data only      : {n_yale_only}")
    print(f"  Category-prior only : {n_prior_only}")
    print(f"  Both Yale + prior   : {n_both}")
    print(f"  Total with DT edges : {n_with_dt} / {len(reg)}")

    print("\nSample analytics SDKs (first 5):")
    for sdk in get_sdks_for_category("Analytics")[:5]:
        print(f"  [{sdk['tracker_id']}] {sdk['name']:30s} domains={sdk['network_domains'][:2]}")

    print("\nInfer for EDUCATION app declaring analytics + app_functionality:")
    inferred = infer_sdks_for_app("EDUCATION", {"analytics", "app_functionality"})
    for sdk in inferred:
        print(f"  [{sdk['tracker_id']}] {sdk['name']}")
