"""
Load Yale Privacy Lab tracker profiles and print stats.
Source: https://github.com/YalePrivacyLab/tracker-profiles
77 tracker profiles in markdown, parsed to JSON.
"""
import json
import os

RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw/yale_pl"))

def main():
    print("=" * 60)
    print("Yale Privacy Lab Tracker Profiles Loader")
    print("=" * 60)

    parsed_path = os.path.join(RAW_DIR, "trackers_parsed.json")
    if not os.path.exists(parsed_path):
        print("  ERROR: trackers_parsed.json not found")
        return

    with open(parsed_path) as f:
        trackers = json.load(f)

    print(f"\n[1] Total trackers: {len(trackers)}")
    
    # Stats
    with_data_types = [t for t in trackers if t['data_types_collected']]
    with_exodus = [t for t in trackers if t.get('exodus_code_rule') or t.get('exodus_network_rule')]
    with_desc = [t for t in trackers if t.get('description')]
    
    print(f"    With data types/purposes: {len(with_data_types)}")
    print(f"    With Exodus detection rules: {len(with_exodus)}")
    print(f"    With descriptions: {len(with_desc)}")

    # All unique data types
    all_types = set()
    for t in trackers:
        all_types.update(t.get('data_types_collected', []))
    print(f"\n[2] Unique data type categories: {len(all_types)}")
    print(f"    Sample: {list(all_types)[:5]}")

    print(f"\n[3] Sample trackers:")
    for t in trackers[:5]:
        print(f"    {t['tracker_name']}")
        print(f"      Types: {t['data_types_collected'][:3]}")
        print(f"      Exodus code: {t.get('exodus_code_rule','N/A')[:40]}")

    tracker_dir = os.path.join(RAW_DIR, "trackers")
    md_count = len([f for f in os.listdir(tracker_dir) if f.endswith('.md')]) if os.path.isdir(tracker_dir) else 0
    print(f"\n[4] Source markdown files: {md_count}")
    print(f"    Source: https://github.com/YalePrivacyLab/tracker-profiles")
    print(f"    License: CC BY-SA 4.0 (assumed)")
    print("=" * 60)

if __name__ == "__main__":
    main()
