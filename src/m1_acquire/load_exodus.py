"""
Load Exodus Privacy tracker data and print stats.
Source: https://reports.exodus-privacy.eu.org/api/trackers
Public API, no auth required. 432 trackers as of acquisition date.
Note: App reports require an API key (skipped for now).
"""
import json
import os

RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw/exodus"))

def main():
    print("=" * 60)
    print("Exodus Privacy Tracker Dataset Loader")
    print("=" * 60)

    trackers_path = os.path.join(RAW_DIR, "trackers.json")
    if not os.path.exists(trackers_path):
        print("  ERROR: trackers.json not found")
        return

    with open(trackers_path) as f:
        data = json.load(f)

    trackers = data.get('trackers', {})
    print(f"\n[1] Total trackers: {len(trackers)}")

    # Category distribution
    from collections import Counter
    cats = Counter()
    for t in trackers.values():
        for cat in t.get('categories', []):
            cats[cat] += 1
    
    print(f"\n[2] Top categories:")
    for cat, cnt in cats.most_common(10):
        print(f"    {cat}: {cnt}")

    # Key fields
    sample = list(trackers.values())[0]
    print(f"\n[3] Fields: {list(sample.keys())}")

    print(f"\n[4] Sample trackers:")
    for t in list(trackers.values())[:3]:
        print(f"    [{t['id']}] {t['name']}")
        print(f"      Code sig: {t.get('code_signature','')[:50]}")
        print(f"      Website: {t.get('website','')[:50]}")
        print(f"      Categories: {t.get('categories', [])}")

    print(f"\n[5] Access notes:")
    print(f"    Public API: https://reports.exodus-privacy.eu.org/api/trackers (no auth)")
    print(f"    App reports API: requires API key (not acquired yet)")
    print(f"    License: AGPL-3.0")
    print("=" * 60)

if __name__ == "__main__":
    main()
