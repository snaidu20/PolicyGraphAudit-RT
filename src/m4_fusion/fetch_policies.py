"""
fetch_policies.py — Policy document fetcher and cache manager for M4.

Reads the Play DSL sample, collects (appId, privacyPolicyURL) for all apps
with ≥3 declaration rows, fetches and strips HTML content, and caches results
under data/interim/policies/.

Public API
----------
fetch_all_policies(min_rows, timeout, delay, max_workers) -> dict
    Fetch and cache all policy texts.  Returns {ok, fail, skipped}.

load_policies_index() -> dict
    Load the policies_index.json mapping appId -> {path, status}.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _resolve_data_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [
        here.parent.parent.parent / "data",
        here.parent.parent.parent.parent / "data",
        Path("data"),
    ]:
        if candidate.is_dir():
            return candidate
    return Path("data")


_DATA_ROOT = _resolve_data_root()
_POLICIES_DIR = _DATA_ROOT / "interim" / "policies"
_INDEX_PATH = _DATA_ROOT / "interim" / "policies_index.json"
_SAMPLE_PATH = _DATA_ROOT / "raw" / "play_data_safety" / "sample_5000.json"

# Minimum policy text length to be considered non-trivial
_MIN_POLICY_CHARS = 500

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha1(url: str) -> str:
    return hashlib.sha1(url.encode()).hexdigest()


def _strip_html(html: str) -> str:
    """Parse HTML and extract clean text, removing boilerplate tags."""
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    return soup.get_text(separator="\n\n")


def _load_index() -> Dict[str, dict]:
    if _INDEX_PATH.exists():
        with open(_INDEX_PATH, encoding="utf-8") as f:
            return json.load(f)
    return {}


def _save_index(index: Dict[str, dict]) -> None:
    _INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)


def _collect_app_urls(min_rows: int = 3) -> Dict[str, str]:
    """
    Return dict of appId -> privacyPolicyURL for apps with >= min_rows rows.
    One URL per app (uses first non-empty URL found).
    """
    if not _SAMPLE_PATH.exists():
        raise FileNotFoundError(f"Play DSL sample not found at {_SAMPLE_PATH}")

    row_counts: Dict[str, int] = {}
    app_urls: Dict[str, str] = {}

    with open(_SAMPLE_PATH, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            app_id = row.get("appId", "")
            if not app_id:
                continue
            row_counts[app_id] = row_counts.get(app_id, 0) + 1
            url = row.get("privacyPolicy", "")
            if url and app_id not in app_urls:
                app_urls[app_id] = url

    # Filter to apps with enough rows
    qualifying = {
        app_id: url
        for app_id, url in app_urls.items()
        if row_counts.get(app_id, 0) >= min_rows
    }
    log.info(
        "Collected %d qualifying apps (≥%d rows) with policy URLs",
        len(qualifying), min_rows,
    )
    return qualifying


# ---------------------------------------------------------------------------
# Per-domain delay tracking (polite fetching)
# ---------------------------------------------------------------------------

_last_fetch_time: Dict[str, float] = {}


def _polite_sleep(url: str, delay: float) -> None:
    """Sleep if we fetched from the same domain recently."""
    domain = urlparse(url).netloc
    last = _last_fetch_time.get(domain, 0.0)
    elapsed = time.time() - last
    if elapsed < delay:
        time.sleep(delay - elapsed)
    _last_fetch_time[domain] = time.time()


def _check_robots(url: str, timeout: float = 5.0) -> bool:
    """
    Return True if crawling is permitted by robots.txt.
    Returns True on any error (fail-open for robustness).
    """
    try:
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        rp = RobotFileParser()
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch(_HEADERS["User-Agent"], url)
    except Exception:
        return True  # fail-open


# ---------------------------------------------------------------------------
# Single-URL fetch
# ---------------------------------------------------------------------------

def _fetch_one(
    app_id: str,
    url: str,
    timeout: float,
    delay: float,
) -> Tuple[str, str]:
    """
    Fetch one privacy policy URL.

    Returns
    -------
    (status, text_or_error_msg)
      status: 'ok' | 'timeout' | 'http_error' | 'empty'
    """
    _polite_sleep(url, delay)

    try:
        resp = requests.get(
            url,
            headers=_HEADERS,
            timeout=timeout,
            allow_redirects=True,
        )
        resp.raise_for_status()
    except requests.Timeout:
        return "timeout", f"timeout after {timeout}s"
    except requests.HTTPError as e:
        return "http_error", str(e)
    except Exception as e:
        return "http_error", str(e)

    # Strip HTML
    content_type = resp.headers.get("Content-Type", "")
    if "html" in content_type or resp.text.lstrip().startswith("<"):
        text = _strip_html(resp.text)
    else:
        text = resp.text

    text = text.strip()
    if len(text) < _MIN_POLICY_CHARS:
        return "empty", f"only {len(text)} chars after stripping"

    return "ok", text


# ---------------------------------------------------------------------------
# Main fetch function
# ---------------------------------------------------------------------------

def fetch_all_policies(
    min_rows: int = 3,
    timeout: float = 10.0,
    delay: float = 1.0,
    force_refetch: bool = False,
) -> Dict[str, int]:
    """
    Fetch and cache all privacy policies for qualifying apps.

    Parameters
    ----------
    min_rows : int
        Minimum Play DSL rows for an app to qualify.
    timeout : float
        Per-request timeout in seconds.
    delay : float
        Minimum delay between requests to the same domain.
    force_refetch : bool
        Re-fetch even if already cached.

    Returns
    -------
    dict with keys 'ok', 'fail', 'skipped'.
    """
    _POLICIES_DIR.mkdir(parents=True, exist_ok=True)

    app_urls = _collect_app_urls(min_rows=min_rows)
    index = _load_index() if not force_refetch else {}

    counts = {"ok": 0, "fail": 0, "skipped": 0}

    apps = sorted(app_urls.items())
    with tqdm(total=len(apps), desc="Fetching policies", unit="app") as pbar:
        for app_id, url in apps:
            sha = _sha1(url)
            cache_path = _POLICIES_DIR / f"{sha}.txt"

            # Already cached and indexed?
            if app_id in index and not force_refetch:
                existing = index[app_id]
                if existing.get("status") == "ok" and cache_path.exists():
                    counts["skipped"] += 1
                    pbar.update(1)
                    pbar.set_postfix(counts)
                    continue
                # If previously failed, retry
                if existing.get("status") in ("timeout", "http_error", "empty"):
                    # skip retry to save time during smoke test (already tried)
                    counts["skipped"] += 1
                    pbar.update(1)
                    pbar.set_postfix(counts)
                    continue

            # Attempt fetch
            log.debug("Fetching %s → %s", app_id, url)
            status, payload = _fetch_one(app_id, url, timeout=timeout, delay=delay)

            entry: dict = {
                "app_id": app_id,
                "url": url,
                "status": status,
                "cached_path": str(cache_path) if status == "ok" else None,
            }

            if status == "ok":
                cache_path.write_text(payload, encoding="utf-8")
                counts["ok"] += 1
                log.debug("Cached %s (%d chars)", app_id, len(payload))
            else:
                entry["error"] = payload
                counts["fail"] += 1
                log.warning("Failed %s [%s]: %s", app_id, status, payload[:100])

            index[app_id] = entry
            _save_index(index)

            pbar.update(1)
            pbar.set_postfix(counts)

    log.info(
        "Policy fetch complete: ok=%d, fail=%d, skipped=%d",
        counts["ok"], counts["fail"], counts["skipped"],
    )
    return counts


# ---------------------------------------------------------------------------
# Index loader
# ---------------------------------------------------------------------------

def load_policies_index() -> Dict[str, dict]:
    """Load the policies_index.json. Returns {} if not yet generated."""
    return _load_index()


def load_policy_text(app_id: str) -> Optional[str]:
    """
    Return cached policy text for an app, or None if not available.
    """
    index = _load_index()
    entry = index.get(app_id)
    if not entry or entry.get("status") != "ok":
        return None
    path = entry.get("cached_path")
    if not path or not Path(path).exists():
        return None
    return Path(path).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    counts = fetch_all_policies()
    total = sum(counts.values())
    print(f"\nFetch results: {counts}")
    print(f"Success rate: {counts['ok']}/{total} ({100*counts['ok']/max(total,1):.1f}%)")
    if counts["ok"] >= 150:
        print("✓ Prototype threshold (≥150) reached.")
    else:
        print(f"✗ Below prototype threshold (150 needed, got {counts['ok']}).")
