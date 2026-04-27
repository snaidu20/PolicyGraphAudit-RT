"""
Load iOS App Store privacy label data and print stats.
Source: Keeping-Privacy-Labels-Honest/privacyLabels (GitHub)
22,000 apps across 22 App Store categories (top 1000 per category)
German App Store locale ('de'). 2022 snapshot.
"""
import json
import os

RAW_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/raw/ios_labels"))

def main():
    print("=" * 60)
    print("iOS Privacy Labels Dataset Loader")
    print("=" * 60)

    # App inventory
    inv_path = os.path.join(RAW_DIR, "app_inventory.json")
    if not os.path.exists(inv_path):
        print("  ERROR: app_inventory.json not found")
        return

    with open(inv_path) as f:
        inventory = json.load(f)

    total_apps = sum(v['app_count'] for v in inventory.values())
    print(f"\n[1] App Inventory: {len(inventory)} categories, {total_apps:,} total apps")
    for cat, data in inventory.items():
        print(f"    {cat}: {data['app_count']} apps")

    # Sample privacy labels
    sample_path = os.path.join(RAW_DIR, "sample_privacy_labels.json")
    if os.path.exists(sample_path):
        with open(sample_path) as f:
            samples = json.load(f)
        
        with_data = [s for s in samples if s['privacy_types_count'] > 0]
        print(f"\n[2] Privacy labels sample: {len(samples)} apps")
        print(f"    Apps with disclosed privacy types: {len(with_data)} ({100*len(with_data)/len(samples):.0f}%)")
        
        print(f"\n[3] Sample apps with privacy labels:")
        for s in with_data[:5]:
            print(f"    {s['app_name']} ({s['bundle_id']}) [{s['category']}]")
            print(f"      Privacy types: {s['privacy_types_count']}")
            for pt in s['privacy_types'][:2]:
                print(f"        - {pt.get('privacyType', '')}")

    print(f"\n[4] Source: https://github.com/Keeping-Privacy-Labels-Honest/privacyLabels")
    print(f"    Research: https://github.com/Keeping-Privacy-Labels-Honest/Main")
    print(f"    Locale: German App Store (de)")
    print(f"    Note: Labels are self-reported by developers (not audited)")
    print("=" * 60)

if __name__ == "__main__":
    main()
