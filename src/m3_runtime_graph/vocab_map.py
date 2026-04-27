"""
vocab_map.py — Map Play DSL raw string values to canonical M2 vocabularies.

The Play DSL uses human-readable strings like "Email address" or
"Advertising or marketing".  This module maps those to the canonical
snake_case identifiers defined in m2_policy_graph.vocab.

Public API
----------
DATATYPE_MAP : dict[str, str]   raw → canonical
PURPOSE_MAP  : dict[str, str]   raw → canonical
map_data_type(raw) -> str | None
map_purpose(raw)   -> str | None
"""

from __future__ import annotations

import logging
from typing import Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DATA TYPE MAP
# Raw Play DSL dataType values → canonical DATA_TYPES from m2 vocab.py
# Covers every value seen in the 5K sample.  Non-data sentinel values
# (e.g. "No data collected") map to None via map_data_type().
# ---------------------------------------------------------------------------

DATATYPE_MAP: dict[str, str] = {
    # ---- Personal info ----
    "Name":                         "name",
    "Email address":                 "email_address",
    "Phone number":                  "phone_number",
    "Address":                       "address",
    "User IDs":                      "user_ids",
    "Date of birth":                 "date_of_birth",
    "Race and ethnicity":            "race_ethnicity",
    "Political or religious beliefs":"political_or_religious_beliefs",
    "Sexual orientation":            "sexual_orientation",

    # ---- Financial ----
    "User payment info":             "user_payment_info",
    "Purchase history":              "purchase_history",
    "Credit score":                  "credit_score",
    "Other financial info":          "financial_info",

    # ---- Location ----
    "Precise location":              "precise_location",
    "Approximate location":          "approximate_location",

    # ---- Contacts / communications ----
    "Contacts":                      "contacts",
    "Emails":                        "emails",
    "SMS or MMS":                    "sms_mms",
    "Calendar events":               "calendar_events",
    "Other in-app messages":         "emails",   # closest canonical analog

    # ---- Device / usage ----
    "Device or other IDs":           "device_id",
    "Installed apps":                "installed_apps",
    "App interactions":              "app_interactions",
    "Other actions":                 "app_interactions",   # catch-all for in-app actions
    "Web browsing history":          "web_browsing_history",
    "In-app search history":         "in_app_search_history",

    # ---- Files / media ----
    "Photos":                        "photos",
    "Videos":                        "videos",
    "Files and docs":                "files_and_docs",
    "Voice or sound recordings":     "voice_or_sound_recordings",
    "Other audio files":             "voice_or_sound_recordings",  # same canonical bucket
    "Music files":                   "music_files",

    # ---- Health / biometric ----
    "Health info":                   "health_info",
    "Fitness info":                  "fitness_info",

    # ---- Diagnostics / performance ----
    "Crash logs":                    "crash_logs",
    "Diagnostics":                   "diagnostics",
    "Other app performance data":    "other_app_performance_data",

    # ---- Misc / other ----
    "Other info":                    "other_info",
    "Other user-generated content":  "other_user_generated_content",
}

# Sentinel / security-practice rows that do not represent data types.
# map_data_type() returns None for these.
_DATATYPE_SENTINELS: set[str] = {
    "No data collected",
    "No data provided",
    "No data shared with third parties",
    "Data is encrypted in transit",
    "Data isn't encrypted",
    # Unicode smart-quote variants (seen in sample)
    "Data isn’t encrypted",
    "Data can't be deleted",
    "Data can’t be deleted",
    "You can request that data be deleted",
    "Committed to follow the Play Families Policy",
}

# ---------------------------------------------------------------------------
# PURPOSE MAP
# Raw Play DSL purpose values → canonical PURPOSES from m2 vocab.py
# ---------------------------------------------------------------------------

PURPOSE_MAP: dict[str, str] = {
    "App functionality":                            "app_functionality",
    "Analytics":                                    "analytics",
    "Advertising or marketing":                     "advertising_marketing",
    "Personalization":                              "personalization",
    "Account management":                           "account_management",
    "Developer communications":                     "developer_communications",
    "Fraud prevention, security, and compliance":   "fraud_prevention_security",
    # Variants / partial matches seen in data
    "Fraud prevention":                             "fraud_prevention_security",
    "Security":                                     "fraud_prevention_security",
    "Compliance":                                   "fraud_prevention_security",
    # N/A / No data are not purposes
    "N/A":                                          None,
    "No data":                                      None,
}

# ---------------------------------------------------------------------------
# Lookup functions
# ---------------------------------------------------------------------------

def map_data_type(raw: str) -> Optional[str]:
    """
    Map a raw Play DSL dataType string to a canonical DATA_TYPE identifier.

    Returns None for sentinel rows (e.g. "No data collected") and logs a
    warning for any value not in DATATYPE_MAP.
    """
    if raw in _DATATYPE_SENTINELS:
        return None
    canonical = DATATYPE_MAP.get(raw)
    if canonical is None:
        log.warning("Unknown dataType %r — no canonical mapping", raw)
    return canonical


def map_purpose(raw: str) -> Optional[str]:
    """
    Map a raw Play DSL purpose string to a canonical PURPOSE identifier.

    Returns None for 'N/A', 'No data', and logs a warning for unmapped values.
    """
    # Handle explicit None-mapped values
    if raw in PURPOSE_MAP:
        result = PURPOSE_MAP[raw]
        if result is None:
            return None
        return result
    log.warning("Unknown purpose %r — no canonical mapping", raw)
    return None


# ---------------------------------------------------------------------------
# CLI — coverage stats
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    from pathlib import Path
    from load_play_dsl import load_play_dsl

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    df = load_play_dsl()

    # DataType coverage
    dt_vals = df["dataType"].dropna().unique()
    dt_mapped = sum(1 for v in dt_vals if map_data_type(v) is not None or v in _DATATYPE_SENTINELS)
    print(f"DataType coverage: {dt_mapped}/{len(dt_vals)} "
          f"({100 * dt_mapped / len(dt_vals):.1f}%)")

    # Purpose coverage
    pu_vals = df["purpose"].dropna().unique()
    pu_mapped = sum(1 for v in pu_vals if v in PURPOSE_MAP)
    print(f"Purpose coverage:  {pu_mapped}/{len(pu_vals)} "
          f"({100 * pu_mapped / len(pu_vals):.1f}%)")

    # Unmapped values
    dt_unmapped = [v for v in dt_vals if map_data_type(v) is None and v not in _DATATYPE_SENTINELS]
    if dt_unmapped:
        print("UNMAPPED dataTypes:", dt_unmapped)
    pu_unmapped = [v for v in pu_vals if v not in PURPOSE_MAP]
    if pu_unmapped:
        print("UNMAPPED purposes:", pu_unmapped)
