"""
canonical_entities.py — Deterministic canonical-ID generator for M4 graph fusion.

The same DataType "email_address" appearing in a policy graph (M2) and a
label/runtime graph (M3) must map to ONE node in the fused graph.  This
module provides a single `canonical_id` function that produces a stable,
normalized string key for every node type.

Collapsing rules
----------------
Shared across apps (collapsed by name):
  DataType, Purpose, ThirdParty

Unique per app (scoped by appId):
  App, Policy, PrivacyLabel, SDK, PolicySegment, Endpoint

Public API
----------
canonical_id(node_type, name, app_id=None) -> str
normalize(text) -> str
"""

from __future__ import annotations

import re
from typing import Optional

# ---------------------------------------------------------------------------
# Node types that are SHARED across apps (collapsed by canonical name alone)
# ---------------------------------------------------------------------------
_SHARED_TYPES = frozenset({"DataType", "Purpose", "ThirdParty"})

# Node types scoped per app (need appId in their ID)
_APP_SCOPED_TYPES = frozenset(
    {"Policy", "PrivacyLabel", "SDK", "PolicySegment", "Endpoint"}
)

# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize(text: str) -> str:
    """
    Normalize a free-form name to a canonical key:
    - lowercase
    - replace whitespace and hyphens with underscores
    - remove characters that are not alphanumeric or underscore
    - collapse multiple underscores

    Examples
    --------
    >>> normalize("Email Address")
    'email_address'
    >>> normalize("Google Analytics!")
    'google_analytics'
    >>> normalize("3rd-Party SDK")
    '3rd_party_sdk'
    """
    text = text.lower().strip()
    # Replace whitespace and hyphens with underscores
    text = re.sub(r"[\s\-]+", "_", text)
    # Remove everything that isn't alphanumeric, underscore, or dot
    text = re.sub(r"[^a-z0-9_.]", "", text)
    # Collapse repeated underscores
    text = re.sub(r"_+", "_", text)
    # Strip leading/trailing underscores
    text = text.strip("_")
    return text


# ---------------------------------------------------------------------------
# Canonical ID generator
# ---------------------------------------------------------------------------

def canonical_id(
    node_type: str,
    name: str,
    app_id: Optional[str] = None,
) -> str:
    """
    Generate a deterministic, stable node identifier for graph fusion.

    Parameters
    ----------
    node_type : str
        One of the schema node types: App, Policy, PolicySegment, DataType,
        Purpose, ThirdParty, PrivacyLabel, SDK, Endpoint.
    name : str
        The raw name or identifier for this node.
    app_id : str, optional
        Required for app-scoped node types (Policy, PrivacyLabel, SDK,
        PolicySegment, Endpoint).  Not used for shared types or App itself.

    Returns
    -------
    str
        A canonical identifier of the form ``{node_type}::{key}`` where key
        is normalized and scoped appropriately.

    Examples
    --------
    >>> canonical_id("DataType", "Email Address")
    'DataType::email_address'

    >>> canonical_id("DataType", "email_address")
    'DataType::email_address'

    >>> canonical_id("App", "com.example.myapp")
    'App::com.example.myapp'

    >>> canonical_id("Policy", "com.example.myapp_policy", app_id="com.example.myapp")
    'Policy::com.example.myapp::com_example_myapp_policy'

    >>> canonical_id("SDK", "firebase_analytics", app_id="com.example.myapp")
    'SDK::com.example.myapp::firebase_analytics'
    """
    norm_name = normalize(name)

    if node_type == "App":
        # App nodes are keyed by package name (appId); preserve dots
        return f"App::{name}"

    if node_type in _SHARED_TYPES:
        # Shared vocabulary nodes: collapsed across all apps by normalized name
        return f"{node_type}::{norm_name}"

    if node_type in _APP_SCOPED_TYPES:
        # Per-app nodes must be scoped to avoid ID collisions across apps
        if app_id is None:
            raise ValueError(
                f"app_id is required for node_type={node_type!r} but was None"
            )
        norm_app = normalize(app_id)
        return f"{node_type}::{norm_app}::{norm_name}"

    # Unknown type: fall back to simple scoped ID
    norm_app = normalize(app_id) if app_id else "unknown"
    return f"{node_type}::{norm_app}::{norm_name}"


# ---------------------------------------------------------------------------
# Convenience: re-map an existing M2/M3 node ID to its canonical form
# ---------------------------------------------------------------------------

# M2 node-ID prefixes → (node_type, strip_prefix)
_M2_PREFIX_MAP = {
    "policy:": ("Policy", True),
    "segment:": ("PolicySegment", True),
    "datatype:": ("DataType", True),
    "purpose:": ("Purpose", True),
    "thirdparty:": ("ThirdParty", True),
}

# M3 node-ID prefixes → (node_type, strip_prefix)
_M3_PREFIX_MAP = {
    "app:": ("App", True),
    "label:": ("PrivacyLabel", True),
    "datatype:": ("DataType", True),
    "purpose:": ("Purpose", True),
    "sdk:": ("SDK", True),
    "thirdparty:": ("ThirdParty", True),
}


def remap_node_id(
    old_id: str,
    prefix_map: dict,
    app_id: str,
) -> tuple[str, str]:
    """
    Convert a raw M2/M3 node ID to a canonical ID using the prefix map.

    Returns
    -------
    (canonical_id, node_type)
    """
    for prefix, (nt, do_strip) in prefix_map.items():
        if old_id.startswith(prefix):
            raw_name = old_id[len(prefix):] if do_strip else old_id
            cid = canonical_id(nt, raw_name, app_id=app_id)
            return cid, nt
    # Fallback: treat as unknown
    return f"Unknown::{normalize(old_id)}", "Unknown"


def remap_m2_node(old_id: str, app_id: str) -> tuple[str, str]:
    """Remap an M2 policy-graph node ID to its canonical form."""
    return remap_node_id(old_id, _M2_PREFIX_MAP, app_id)


def remap_m3_node(old_id: str, app_id: str) -> tuple[str, str]:
    """Remap an M3 label/runtime-graph node ID to its canonical form."""
    return remap_node_id(old_id, _M3_PREFIX_MAP, app_id)
