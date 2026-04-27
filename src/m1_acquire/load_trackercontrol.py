"""
Load TrackerControl / X-Ray tracker datasets and print stats.
Sources:
  - xray-blacklist.json: TrackerControl/tracker-control-android (domain → company)
  - trackers.json: TC's Exodus-mirror tracker list
  - disconnect-blacklist.json: Disconnect.me blacklist (reversed)
"""
import json
import os

RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw/trackercontrol"))

def load_json(fname):
    fpath = os.path.join(RAW_DIR, fname)
    if not os.path.exists(fpath):
        print(f"  ERROR: {fname} not found")
        return None
    with open(fpath) as f:
        return json.load(f)

def main():
    print("=" * 60)
    print("TrackerControl + X-Ray Tracker Dataset Loader")
    print("=" * 60)

    # 1. X-Ray blacklist
    print("\n[1] xray-blacklist.json (domain → tracker company)")
    xray = load_json("xray-blacklist.json")
    if xray is not None:
        if isinstance(xray, dict):
            domains = list(xray.keys())
            companies = set(xray.values())
            print(f"    Domains: {len(domains)}")
            print(f"    Unique companies: {len(companies)}")
            print(f"    Sample: {domains[:3]} → {[xray[d] for d in domains[:3]]}")
        elif isinstance(xray, list):
            print(f"    Entries: {len(xray)}")
            print(f"    Sample: {xray[:2]}")

    # 2. Trackers.json (Exodus mirror)
    print("\n[2] trackers.json (Exodus-format tracker list)")
    trackers = load_json("trackers.json")
    if trackers is not None:
        if isinstance(trackers, dict) and 'trackers' in trackers:
            t = trackers['trackers']
            print(f"    Trackers: {len(t)}")
            sample_keys = list(t.keys())[:3]
            for k in sample_keys:
                print(f"    [{k}] {t[k].get('name','')} | cats: {t[k].get('categories',[])}")
        elif isinstance(trackers, list):
            print(f"    Trackers: {len(trackers)}")

    # 3. Disconnect blacklist (decoded version - original was byte-reversed)
    print("\n[3] disconnect-blacklist-decoded.json")
    disconnect = load_json("disconnect-blacklist-decoded.json")
    if disconnect is not None:
        cats = disconnect.get('categories', {})
        print(f"    Categories: {list(cats.keys())}")
        total = sum(len(v) for v in cats.values() if isinstance(v, list))
        print(f"    Total company entries: {total}")
        sample_cat = list(cats.keys())[0] if cats else None
        if sample_cat:
            print(f"    Sample ({sample_cat}): {str(cats[sample_cat][:1])[:100]}")

    print(f"\n[4] Source: https://github.com/TrackerControl/tracker-control-android")
    print(f"    License: GPLv3")
    print(f"    Note: PlatformControl repo had no data/ directory (iOS signatures not available)")
    print("=" * 60)

if __name__ == "__main__":
    main()
